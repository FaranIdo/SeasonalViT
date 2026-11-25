"""Tests for the ablation study.

Covers the three configurable ablation knobs:
- temporal_pathway: {both, cross_season, cross_year}
- anchor_mode: {same_season, last_step, none}
- (from-scratch fine-tuning is exercised by existing tests; no new model flag
  is required.)

All tests verify shapes, gradient flow, and backward compatibility (default
config behavior unchanged).
"""

from dataclasses import replace

import pytest
import torch

from model.config import FinetuningConfig, SeasonalViTConfig
from model.seasonal_vit import SeasonalViT
from model.temporal_transformer import DualScaleTemporalTransformer
from train.finetune import FinetuningTrainer


def _make_batch(config: SeasonalViTConfig, num_timesteps: int = 8):
    """Build a small input batch matching config dims."""
    batch_size = 2
    H, W = config.image_size
    images = torch.randn(batch_size, num_timesteps, H, W).clamp(-1, 1)
    season_ids = torch.tensor(
        [[s % 4 for s in range(num_timesteps)]] * batch_size,
    )
    year_indices = torch.tensor(
        [[s // 4 for s in range(num_timesteps)]] * batch_size,
    )
    lat = torch.tensor([45.0, -30.0])
    lon = torch.tensor([-122.0, 150.0])
    target_season = torch.tensor([0, 0])
    target_year = torch.tensor([0, 0])
    return images, season_ids, year_indices, lat, lon, target_season, target_year


# =============================================================================
# Backward compatibility: defaults unchanged
# =============================================================================


class TestBackwardCompatibility:
    """Default config must produce the same module tree and behavior."""

    def test_default_config_values(self) -> None:
        config = SeasonalViTConfig(swin_pretrained=False)
        assert config.anchor_mode == "same_season"
        assert config.temporal_pathway == "both"

    def test_default_transformer_has_both_pathways(
        self, model_config: SeasonalViTConfig,
    ) -> None:
        transformer = DualScaleTemporalTransformer(model_config)
        assert hasattr(transformer, "cross_season_transformer")
        assert hasattr(transformer, "cross_year_transformer")
        assert hasattr(transformer, "cross_attn")
        assert hasattr(transformer, "fusion_mlp")

    def test_default_model_outputs_anchor(
        self, small_model_config: SeasonalViTConfig,
    ) -> None:
        """Default anchor_mode='same_season' must produce non-zero anchor."""
        model = SeasonalViT(small_model_config)
        batch = _make_batch(small_model_config)
        output = model(*batch)
        # Anchor should not be all zeros (it's a real observation)
        assert output["anchor"].abs().sum() > 0

    def test_checkpoint_round_trip_default(
        self, small_model_config: SeasonalViTConfig,
    ) -> None:
        """A default model's state_dict must load into a fresh default model.

        Guards against future refactors breaking pre-trained weight loading.
        """
        src = SeasonalViT(small_model_config)
        dst = SeasonalViT(small_model_config)
        missing, unexpected = dst.load_state_dict(src.state_dict(), strict=True)
        assert not missing
        assert not unexpected

    def test_checkpoint_cross_mode_loads_non_strict(
        self, small_model_config: SeasonalViTConfig,
    ) -> None:
        """Default-trained weights must load (non-strict) into ablation variants.

        Single-pathway and anchor-free variants have fewer parameters than the
        default; strict=False should accept the mismatch with only 'unexpected'
        keys (weights that ablation doesn't need).
        """
        src = SeasonalViT(small_model_config)  # default: both pathways, same_season
        dst_cfg = replace(small_model_config, temporal_pathway="cross_season")
        dst = SeasonalViT(dst_cfg)
        missing, unexpected = dst.load_state_dict(src.state_dict(), strict=False)
        # cross_season variant has the season pathway (under a fresh re-init since
        # layer count changed). We only assert that loading doesn't error —
        # users would then fine-tune from these (partially loaded) weights.
        # Fusion and cross_year keys should be reported as unexpected.
        assert any("cross_year" in k or "fusion" in k for k in unexpected)


# =============================================================================
# Temporal pathway ablations
# =============================================================================


class TestTemporalPathwayAblation:
    """Test cross_season-only and cross_year-only pathway modes."""

    @pytest.mark.parametrize("pathway", ["both", "cross_season", "cross_year"])
    def test_output_shape(
        self, small_model_config: SeasonalViTConfig, pathway: str,
    ) -> None:
        config = replace(small_model_config, temporal_pathway=pathway)
        transformer = DualScaleTemporalTransformer(config)
        B, T = 2, 8
        N = config.num_spatial_patches
        D = config.embed_dim

        spatial = torch.randn(B, T, N, D)
        temporal = torch.randn(B, T, D)
        output = transformer(spatial, temporal)
        assert output.shape == (B, T, N, D)

    def test_cross_season_only_has_no_year_module(
        self, small_model_config: SeasonalViTConfig,
    ) -> None:
        config = replace(small_model_config, temporal_pathway="cross_season")
        transformer = DualScaleTemporalTransformer(config)
        assert hasattr(transformer, "cross_season_transformer")
        assert not hasattr(transformer, "cross_year_transformer")
        assert not hasattr(transformer, "fusion_mlp")

    def test_cross_year_only_has_no_season_module(
        self, small_model_config: SeasonalViTConfig,
    ) -> None:
        config = replace(small_model_config, temporal_pathway="cross_year")
        transformer = DualScaleTemporalTransformer(config)
        assert hasattr(transformer, "cross_year_transformer")
        assert not hasattr(transformer, "cross_season_transformer")
        assert not hasattr(transformer, "fusion_mlp")

    @pytest.mark.parametrize("pathway", ["cross_season", "cross_year"])
    def test_gradient_flow_single_pathway(
        self, small_model_config: SeasonalViTConfig, pathway: str,
    ) -> None:
        config = replace(small_model_config, temporal_pathway=pathway)
        transformer = DualScaleTemporalTransformer(config)
        B, T = 2, 8
        N = config.num_spatial_patches
        D = config.embed_dim

        spatial = torch.randn(B, T, N, D, requires_grad=True)
        temporal = torch.randn(B, T, D)
        output = transformer(spatial, temporal)
        output.sum().backward()
        assert spatial.grad is not None
        assert not torch.isnan(spatial.grad).any()

    def test_end_to_end_single_pathway(
        self, small_model_config: SeasonalViTConfig,
    ) -> None:
        """SeasonalViT forward pass works with cross_season-only pathway."""
        config = replace(small_model_config, temporal_pathway="cross_season")
        model = SeasonalViT(config)
        batch = _make_batch(config)
        output = model(*batch)
        H, W = config.image_size
        assert output["prediction"].shape == (2, 1, H, W)


# =============================================================================
# Anchor mode ablations
# =============================================================================


class TestAnchorModeAblation:
    """Test last_step and none anchor modes."""

    def test_last_step_anchor_equals_last_timestep_no_nodata(
        self, small_model_config: SeasonalViTConfig,
    ) -> None:
        """last_step mode with no nodata: anchor == images[:, -1:]."""
        config = replace(small_model_config, anchor_mode="last_step")
        model = SeasonalViT(config)
        batch = _make_batch(config)
        images = batch[0]
        output = model(*batch)
        expected = images[:, -1:, :, :]
        assert torch.allclose(output["anchor"], expected)

    def test_last_step_anchor_skips_nodata(
        self, small_model_config: SeasonalViTConfig,
    ) -> None:
        """last_step mode: nodata pixels at the end fall back to prior valid."""
        config = replace(small_model_config, anchor_mode="last_step")
        model = SeasonalViT(config)
        images, season_ids, year_indices, lat, lon, ts, ty = _make_batch(config)
        # Make the last timestep entirely nodata, leaving T-2 as the true last valid
        nodata = config.nodata_value
        images = images.clone()
        expected_anchor = images[:, -2:-1, :, :].clone()
        images[:, -1, :, :] = nodata

        output = model(images, season_ids, year_indices, lat, lon, ts, ty)
        assert torch.allclose(output["anchor"], expected_anchor)

    def test_last_step_anchor_all_nodata_pixel(
        self, small_model_config: SeasonalViTConfig,
    ) -> None:
        """Pixels that are nodata for every timestep stay nodata in anchor."""
        config = replace(small_model_config, anchor_mode="last_step")
        model = SeasonalViT(config)
        images, season_ids, year_indices, lat, lon, ts, ty = _make_batch(config)
        nodata = config.nodata_value
        images = images.clone()
        # Make pixel (0, 0) nodata across all timesteps for sample 0
        images[0, :, 0, 0] = nodata

        output = model(images, season_ids, year_indices, lat, lon, ts, ty)
        assert output["anchor"][0, 0, 0, 0].item() == nodata

    def test_none_anchor_is_zero(
        self, small_model_config: SeasonalViTConfig,
    ) -> None:
        """anchor_mode='none': decoder returns zero anchor, prediction ≈ delta."""
        config = replace(small_model_config, anchor_mode="none")
        model = SeasonalViT(config)
        batch = _make_batch(config)
        output = model(*batch)
        assert torch.allclose(output["anchor"], torch.zeros_like(output["anchor"]))
        # delta and prediction coincide in absolute mode
        assert torch.allclose(output["delta"], output["prediction"])

    def test_none_anchor_prediction_in_valid_range(
        self, small_model_config: SeasonalViTConfig,
    ) -> None:
        config = replace(small_model_config, anchor_mode="none")
        model = SeasonalViT(config)
        batch = _make_batch(config)
        output = model(*batch)
        assert output["prediction"].min() >= -1.0
        assert output["prediction"].max() <= 1.0

    @pytest.mark.parametrize("mode", ["same_season", "last_step", "none"])
    def test_gradient_flow_all_modes(
        self, small_model_config: SeasonalViTConfig, mode: str,
    ) -> None:
        config = replace(small_model_config, anchor_mode=mode)
        model = SeasonalViT(config)
        batch = _make_batch(config)
        output = model(*batch)
        loss = output["prediction"].mean()
        loss.backward()
        # Decoder params should receive gradient in all modes
        decoder_grads = [
            p.grad for p in model.decoder.parameters() if p.grad is not None
        ]
        assert len(decoder_grads) > 0
        assert not any(torch.isnan(g).any() for g in decoder_grads)

    def test_same_season_vs_last_step_differ(
        self, small_model_config: SeasonalViTConfig,
    ) -> None:
        """Sanity: two anchor modes should produce different anchors."""
        batch = _make_batch(small_model_config)

        cfg_a = replace(small_model_config, anchor_mode="same_season")
        cfg_b = replace(small_model_config, anchor_mode="last_step")
        model_a = SeasonalViT(cfg_a)
        model_b = SeasonalViT(cfg_b)
        # Sync weights so only anchor branch differs
        model_b.load_state_dict(model_a.state_dict())

        with torch.no_grad():
            out_a = model_a(*batch)
            out_b = model_b(*batch)
        # Anchors will differ whenever target_season doesn't match the last
        # timestep's season. Our batch has target_season=0 and last-step
        # season_id = (T-1) % 4 = 3, so they must differ.
        assert not torch.allclose(out_a["anchor"], out_b["anchor"])


# =============================================================================
# Combined ablations
# =============================================================================


class TestTrainerAnchorFree:
    """Trainer must auto-disable delta losses when anchor_mode='none'."""

    def test_delta_losses_auto_disabled(
        self, tmp_path, small_model_config: SeasonalViTConfig,
    ) -> None:
        config = replace(small_model_config, anchor_mode="none")
        model = SeasonalViT(config)
        train_cfg = FinetuningConfig(
            delta_supervision_weight=1.0,  # user-supplied, should be forced to 0
            delta_reg_weight=0.5,           # user-supplied, should be forced to 0
        )
        trainer = FinetuningTrainer(
            model=model,
            config=train_cfg,
            device=torch.device("cpu"),
            output_dir=tmp_path,
        )
        assert trainer.loss_fn.delta_supervision_weight == 0.0
        assert trainer.loss_fn.delta_reg_weight == 0.0
        assert trainer._need_delta_std_estimation is False

    def test_delta_losses_preserved_in_anchored_mode(
        self, tmp_path, small_model_config: SeasonalViTConfig,
    ) -> None:
        model = SeasonalViT(small_model_config)  # default same_season
        train_cfg = FinetuningConfig(
            delta_supervision_weight=1.0,
            delta_reg_weight=0.5,
            delta_std=0.025,
        )
        trainer = FinetuningTrainer(
            model=model,
            config=train_cfg,
            device=torch.device("cpu"),
            output_dir=tmp_path,
        )
        assert trainer.loss_fn.delta_supervision_weight == 1.0
        assert trainer.loss_fn.delta_reg_weight == 0.5


class TestCombinedAblations:
    """Ablations can be combined without breaking shape contracts."""

    @pytest.mark.parametrize("pathway", ["cross_season", "cross_year"])
    @pytest.mark.parametrize("anchor_mode", ["last_step", "none"])
    def test_combined(
        self,
        small_model_config: SeasonalViTConfig,
        pathway: str,
        anchor_mode: str,
    ) -> None:
        config = replace(
            small_model_config,
            temporal_pathway=pathway,
            anchor_mode=anchor_mode,
        )
        model = SeasonalViT(config)
        batch = _make_batch(config)
        output = model(*batch)
        H, W = config.image_size
        assert output["prediction"].shape == (2, 1, H, W)
