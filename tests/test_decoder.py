"""Tests for ObservationAnchoredDecoder."""

import pytest
import torch

from model.decoder import (
    ObservationAnchoredDecoder,
    TemporalAttentionPooling,
    TemporalAttentionAnalysis,
    create_decoder,
)
from model.config import SeasonalViTConfig


def make_target_tensors(batch_size: int, device: torch.device = None):
    """Create target_season and target_year tensors for testing."""
    target_season = torch.randint(0, 4, (batch_size,), device=device)
    target_year = torch.randint(0, 35, (batch_size,), device=device)
    return target_season, target_year


class TestObservationAnchoredDecoder:
    """Tests for ObservationAnchoredDecoder."""

    def test_output_shape(self, model_config: SeasonalViTConfig) -> None:
        """Test that output has correct shapes."""
        decoder = ObservationAnchoredDecoder(model_config)
        batch_size = 2
        num_timesteps = 8
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        temporal_features = torch.randn(
            batch_size, num_timesteps, num_patches, model_config.embed_dim
        )
        anchor = torch.randn(batch_size, 1, H, W)
        target_season, target_year = make_target_tensors(batch_size)

        output = decoder(temporal_features, anchor, target_season, target_year)

        assert output["prediction"].shape == (batch_size, 1, H, W)
        assert output["delta"].shape == (batch_size, 1, H, W)
        assert output["anchor"].shape == (batch_size, 1, H, W)

    def test_prediction_is_anchor_plus_delta(
        self, model_config: SeasonalViTConfig
    ) -> None:
        """Test that prediction = anchor + delta (before clamping)."""
        decoder = ObservationAnchoredDecoder(model_config)
        decoder.eval()
        batch_size = 2
        num_timesteps = 8
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        temporal_features = torch.randn(
            batch_size, num_timesteps, num_patches, model_config.embed_dim
        )
        # Use small anchor values to avoid clamping effects
        anchor = torch.randn(batch_size, 1, H, W) * 0.1
        target_season, target_year = make_target_tensors(batch_size)

        with torch.no_grad():
            output = decoder(temporal_features, anchor, target_season, target_year)

        # Prediction should equal anchor + delta (may differ due to clamping)
        expected = anchor + output["delta"]
        expected = expected.clamp(-1, 1)
        assert torch.allclose(output["prediction"], expected)

    def test_delta_is_bounded(self, model_config: SeasonalViTConfig) -> None:
        """Test that delta values are approximately bounded.

        The per-patch delta is bounded by Tanh * delta_scale, but the
        progressive upsampling conv layers can slightly exceed this.
        Check that delta stays within a reasonable range.
        """
        decoder = ObservationAnchoredDecoder(model_config)
        decoder.eval()
        batch_size = 4
        num_timesteps = 8
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        temporal_features = torch.randn(
            batch_size, num_timesteps, num_patches, model_config.embed_dim
        )
        anchor = torch.randn(batch_size, 1, H, W)
        target_season, target_year = make_target_tensors(batch_size)

        with torch.no_grad():
            output = decoder(temporal_features, anchor, target_season, target_year)

        # Delta should stay small (within ~2x of delta_scale after conv upsampling)
        max_delta = model_config.delta_scale * 2
        assert output["delta"].min() >= -max_delta
        assert output["delta"].max() <= max_delta

    def test_anchored_delta_is_hard_clamped(
        self, model_config: SeasonalViTConfig,
    ) -> None:
        """Post-upsample delta in _predict_anchored must be clamped to [-1, 1].

        Without this clamp, a randomly-initialized ConvTranspose2d upsample
        stack can produce out-of-range activations (>|1|) which, under
        bf16 and large target-anchor gaps (e.g. `last_step` anchor mode),
        cause multiplicative weight drift and NaN training within ~100
        steps. Symmetry with `_predict_absolute`, which already clamps.
        """
        decoder = ObservationAnchoredDecoder(model_config)
        batch_size = 2
        num_timesteps = 8
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        # Inject extreme upstream activations so the upsample stack has to
        # amplify them somewhere. Pre-clamp buffer would let this overflow.
        temporal_features = torch.randn(
            batch_size, num_timesteps, num_patches, model_config.embed_dim,
        ) * 50.0
        anchor = torch.zeros(batch_size, 1, H, W)
        target_season, target_year = make_target_tensors(batch_size)

        with torch.no_grad():
            output = decoder(temporal_features, anchor, target_season, target_year)

        assert output["delta"].min() >= -1.0 - 1e-4
        assert output["delta"].max() <= 1.0 + 1e-4

    def test_prediction_is_clamped(self, model_config: SeasonalViTConfig) -> None:
        """Test that prediction is clamped to [-1, 1]."""
        decoder = ObservationAnchoredDecoder(model_config)
        batch_size = 2
        num_timesteps = 8
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        temporal_features = torch.randn(
            batch_size, num_timesteps, num_patches, model_config.embed_dim
        )
        # Use extreme anchor values to trigger clamping
        anchor = torch.ones(batch_size, 1, H, W) * 0.9
        target_season, target_year = make_target_tensors(batch_size)

        output = decoder(temporal_features, anchor, target_season, target_year)

        assert output["prediction"].min() >= -1
        assert output["prediction"].max() <= 1

    def test_gradient_flow(self, model_config: SeasonalViTConfig) -> None:
        """Test that gradients flow through the decoder."""
        decoder = ObservationAnchoredDecoder(model_config)
        batch_size = 2
        num_timesteps = 8
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        temporal_features = torch.randn(
            batch_size, num_timesteps, num_patches, model_config.embed_dim,
            requires_grad=True
        )
        anchor = torch.randn(batch_size, 1, H, W)
        target_season, target_year = make_target_tensors(batch_size)

        output = decoder(temporal_features, anchor, target_season, target_year)
        loss = output["prediction"].sum()
        loss.backward()

        assert temporal_features.grad is not None
        assert not torch.isnan(temporal_features.grad).any()

    def test_deterministic_output(self, model_config: SeasonalViTConfig) -> None:
        """Test that same input produces same output in eval mode."""
        decoder = ObservationAnchoredDecoder(model_config)
        decoder.eval()
        batch_size = 2
        num_timesteps = 8
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        temporal_features = torch.randn(
            batch_size, num_timesteps, num_patches, model_config.embed_dim
        )
        anchor = torch.randn(batch_size, 1, H, W)
        target_season, target_year = make_target_tensors(batch_size)

        with torch.no_grad():
            output1 = decoder(temporal_features, anchor, target_season, target_year)
            output2 = decoder(temporal_features, anchor, target_season, target_year)

        assert torch.allclose(output1["prediction"], output2["prediction"])

    def test_temporal_attention_pooling(self, model_config: SeasonalViTConfig) -> None:
        """Test temporal attention pooling produces correct output shape."""
        decoder = ObservationAnchoredDecoder(model_config)
        batch_size = 2
        num_timesteps = 8
        num_patches = model_config.num_spatial_patches

        temporal_features = torch.randn(
            batch_size, num_timesteps, num_patches, model_config.embed_dim
        )

        pooled, weights = decoder.temporal_pool(temporal_features, return_weights=True)

        assert pooled.shape == (batch_size, num_patches, model_config.embed_dim)
        assert weights.shape == (batch_size, num_patches, num_timesteps)
        # Attention weights should sum to 1 for each spatial location
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, num_patches), atol=1e-5)

    def test_attention_analysis(self, model_config: SeasonalViTConfig) -> None:
        """Test attention analysis returns correct metrics."""
        decoder = ObservationAnchoredDecoder(model_config)
        decoder.eval()
        batch_size = 2
        num_timesteps = 140  # Full 35 years
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        temporal_features = torch.randn(
            batch_size, num_timesteps, num_patches, model_config.embed_dim
        )
        anchor = torch.randn(batch_size, 1, H, W)
        target_season, target_year = make_target_tensors(batch_size)

        with torch.no_grad():
            output = decoder(
                temporal_features, anchor, target_season, target_year,
                return_attention_analysis=True
            )

        assert "attention_analysis" in output
        analysis = output["attention_analysis"]
        assert isinstance(analysis, TemporalAttentionAnalysis)

        # Check shapes
        assert analysis.attention_weights.shape == (batch_size, num_timesteps)
        assert analysis.effective_history_90.shape == (batch_size,)
        assert analysis.effective_history_50.shape == (batch_size,)
        assert analysis.recent_5yr_weight.shape == (batch_size,)
        assert analysis.attention_entropy.shape == (batch_size,)
        assert analysis.peak_timestep.shape == (batch_size,)

        # Check value ranges
        assert (analysis.effective_history_90 >= 1).all()
        assert (analysis.effective_history_90 <= num_timesteps).all()
        assert (analysis.recent_5yr_weight >= 0).all()
        assert (analysis.recent_5yr_weight <= 1).all()
        assert (analysis.peak_timestep >= 0).all()
        assert (analysis.peak_timestep < num_timesteps).all()

    def test_attention_analysis_summary_stats(self, model_config: SeasonalViTConfig) -> None:
        """Test attention analysis summary statistics."""
        decoder = ObservationAnchoredDecoder(model_config)
        decoder.eval()
        batch_size = 4
        num_timesteps = 140
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        temporal_features = torch.randn(
            batch_size, num_timesteps, num_patches, model_config.embed_dim
        )
        anchor = torch.randn(batch_size, 1, H, W)
        target_season, target_year = make_target_tensors(batch_size)

        with torch.no_grad():
            output = decoder(
                temporal_features, anchor, target_season, target_year,
                return_attention_analysis=True
            )

        analysis = output["attention_analysis"]
        stats = analysis.summary_stats()

        assert "effective_history_90_mean" in stats
        assert "effective_history_50_mean" in stats
        assert "recent_5yr_weight_mean" in stats
        assert "attention_entropy_mean" in stats
        assert all(isinstance(v, float) for v in stats.values())

    def test_upsample_delta(self, model_config: SeasonalViTConfig) -> None:
        """Test _upsample_delta produces correct output shape."""
        decoder = ObservationAnchoredDecoder(model_config)
        batch_size = 2
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        delta_patches = torch.randn(batch_size, num_patches, 1)

        upsampled = decoder._upsample_delta(delta_patches)

        assert upsampled.shape == (batch_size, 1, H, W)

    def test_get_delta_statistics(self, model_config: SeasonalViTConfig) -> None:
        """Test get_delta_statistics computes correct values."""
        decoder = ObservationAnchoredDecoder(model_config)
        H, W = model_config.image_size

        delta = torch.randn(2, 1, H, W)
        stats = decoder.get_delta_statistics(delta)

        assert "delta_mean" in stats
        assert "delta_std" in stats
        assert "delta_min" in stats
        assert "delta_max" in stats
        assert "delta_abs_mean" in stats

        # Verify values are reasonable
        assert abs(stats["delta_mean"] - delta.mean().item()) < 1e-6
        assert abs(stats["delta_std"] - delta.std().item()) < 1e-6

    def test_factory_function(self) -> None:
        """Test create_decoder factory function."""
        config = SeasonalViTConfig(swin_pretrained=False)
        decoder = create_decoder(config)

        assert isinstance(decoder, ObservationAnchoredDecoder)

    def test_factory_function_default_config(self) -> None:
        """Test factory function with default config."""
        decoder = create_decoder()

        assert isinstance(decoder, ObservationAnchoredDecoder)
        assert decoder.grid_size == 8


class TestDecoderEdgeCases:
    """Edge case tests for decoder."""

    def test_single_batch(self, model_config: SeasonalViTConfig) -> None:
        """Test with single batch."""
        decoder = ObservationAnchoredDecoder(model_config)
        num_timesteps = 8
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        temporal_features = torch.randn(1, num_timesteps, num_patches, model_config.embed_dim)
        anchor = torch.randn(1, 1, H, W)
        target_season, target_year = make_target_tensors(1)

        output = decoder(temporal_features, anchor, target_season, target_year)

        assert output["prediction"].shape == (1, 1, H, W)

    def test_single_timestep(self, model_config: SeasonalViTConfig) -> None:
        """Test with single timestep."""
        decoder = ObservationAnchoredDecoder(model_config)
        batch_size = 2
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        temporal_features = torch.randn(
            batch_size, 1, num_patches, model_config.embed_dim
        )
        anchor = torch.randn(batch_size, 1, H, W)
        target_season, target_year = make_target_tensors(batch_size)

        output = decoder(temporal_features, anchor, target_season, target_year)

        assert output["prediction"].shape == (batch_size, 1, H, W)

    def test_zero_anchor(self, model_config: SeasonalViTConfig) -> None:
        """Test with zero anchor (prediction equals delta clamped)."""
        decoder = ObservationAnchoredDecoder(model_config)
        batch_size = 2
        num_timesteps = 8
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        temporal_features = torch.randn(
            batch_size, num_timesteps, num_patches, model_config.embed_dim
        )
        anchor = torch.zeros(batch_size, 1, H, W)
        target_season, target_year = make_target_tensors(batch_size)

        output = decoder(temporal_features, anchor, target_season, target_year)

        # With zero anchor, prediction should be clamped delta
        expected = output["delta"].clamp(-1, 1)
        assert torch.allclose(output["prediction"], expected)

    def test_extreme_features(self, model_config: SeasonalViTConfig) -> None:
        """Test with extreme feature values."""
        decoder = ObservationAnchoredDecoder(model_config)
        decoder.eval()
        batch_size = 2
        num_timesteps = 8
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        # Large feature values
        temporal_features = torch.randn(
            batch_size, num_timesteps, num_patches, model_config.embed_dim
        ) * 10
        anchor = torch.randn(batch_size, 1, H, W)
        target_season, target_year = make_target_tensors(batch_size)

        output = decoder(temporal_features, anchor, target_season, target_year)

        # Output should still be valid
        assert not torch.isnan(output["prediction"]).any()
        assert not torch.isinf(output["prediction"]).any()
        # Delta should be bounded by hard clamp at [-1, 1] (post scale)
        assert output["delta"].abs().max() <= 1.0 + 0.01

    def test_ndvi_value_range(self, model_config: SeasonalViTConfig) -> None:
        """Test with NDVI value range [-1, 1] anchor."""
        decoder = ObservationAnchoredDecoder(model_config)
        batch_size = 2
        num_timesteps = 8
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        temporal_features = torch.randn(
            batch_size, num_timesteps, num_patches, model_config.embed_dim
        )
        # Valid NDVI range
        anchor = torch.rand(batch_size, 1, H, W) * 2 - 1
        target_season, target_year = make_target_tensors(batch_size)

        output = decoder(temporal_features, anchor, target_season, target_year)

        # Prediction should be in valid NDVI range
        assert output["prediction"].min() >= -1
        assert output["prediction"].max() <= 1

    def test_delta_scale_affects_output(
        self, model_config: SeasonalViTConfig
    ) -> None:
        """Test that delta_scale parameter affects predictions."""
        # Create two decoders with different delta scales
        config1 = SeasonalViTConfig(delta_scale=0.1, swin_pretrained=False)
        config2 = SeasonalViTConfig(delta_scale=0.9, swin_pretrained=False)
        decoder1 = ObservationAnchoredDecoder(config1)
        decoder2 = ObservationAnchoredDecoder(config2)

        # Copy weights from decoder1 to decoder2 (except delta_scale)
        state_dict = decoder1.state_dict()
        del state_dict["delta_scale"]
        decoder2.load_state_dict(state_dict, strict=False)

        decoder1.eval()
        decoder2.eval()

        batch_size = 2
        num_timesteps = 8
        num_patches = config1.num_spatial_patches
        H, W = config1.image_size

        temporal_features = torch.randn(
            batch_size, num_timesteps, num_patches, config1.embed_dim
        )
        anchor = torch.zeros(batch_size, 1, H, W)
        target_season, target_year = make_target_tensors(batch_size)

        with torch.no_grad():
            output1 = decoder1(temporal_features, anchor, target_season, target_year)
            output2 = decoder2(temporal_features, anchor, target_season, target_year)

        # Different scales should produce different delta magnitudes
        assert output1["delta"].abs().mean() < output2["delta"].abs().mean()

    def test_different_targets_produce_different_outputs(
        self, model_config: SeasonalViTConfig
    ) -> None:
        """Test that different target conditioning produces different outputs."""
        decoder = ObservationAnchoredDecoder(model_config)
        decoder.eval()
        batch_size = 2
        num_timesteps = 8
        num_patches = model_config.num_spatial_patches
        H, W = model_config.image_size

        temporal_features = torch.randn(
            batch_size, num_timesteps, num_patches, model_config.embed_dim
        )
        anchor = torch.randn(batch_size, 1, H, W)

        # Two different targets
        target_season1 = torch.tensor([0, 1])  # spring, summer
        target_year1 = torch.tensor([34, 34])  # year 34
        target_season2 = torch.tensor([2, 3])  # fall, winter
        target_year2 = torch.tensor([30, 30])  # year 30

        with torch.no_grad():
            output1 = decoder(temporal_features, anchor, target_season1, target_year1)
            output2 = decoder(temporal_features, anchor, target_season2, target_year2)

        # Different targets should produce different predictions
        assert not torch.allclose(output1["prediction"], output2["prediction"])
