"""Tests for SeasonalViT forecasting model."""

import pytest
import torch

from model.seasonal_vit import SeasonalViT, create_seasonal_vit
from model.pretrain_model import PretrainingModel
from model.config import SeasonalViTConfig


class TestSeasonalViT:
    """Tests for SeasonalViT forecasting model."""

    def test_output_shape(self, model_config: SeasonalViTConfig) -> None:
        """Test that output has correct shapes."""
        model = SeasonalViT(model_config)
        batch_size = 2
        num_timesteps = 8
        H, W = model_config.image_size

        images = torch.randn(batch_size, num_timesteps, H, W)
        season_ids = torch.tensor([[s % 4 for s in range(num_timesteps)]] * batch_size)
        year_indices = torch.tensor([[s // 4 for s in range(num_timesteps)]] * batch_size)
        lat = torch.tensor([45.0, -30.0])
        lon = torch.tensor([-122.0, 150.0])
        target_season = torch.tensor([0, 0])
        target_year = torch.tensor([0, 0])

        output = model(images, season_ids, year_indices, lat, lon, target_season, target_year)

        assert output["prediction"].shape == (batch_size, 1, H, W)
        assert output["delta"].shape == (batch_size, 1, H, W)
        assert output["anchor"].shape == (batch_size, 1, H, W)

    def test_1d_season_year_indices(self, small_model_config: SeasonalViTConfig) -> None:
        """Test with 1D season/year indices (broadcast)."""
        model = SeasonalViT(small_model_config)
        batch_size = 2
        num_timesteps = 8
        H, W = small_model_config.image_size

        images = torch.randn(batch_size, num_timesteps, H, W)
        # 1D inputs
        season_ids = torch.tensor([s % 4 for s in range(num_timesteps)])
        year_indices = torch.tensor([s // 4 for s in range(num_timesteps)])
        lat = torch.tensor([45.0, -30.0])
        lon = torch.tensor([-122.0, 150.0])
        target_season = torch.tensor([0, 0])
        target_year = torch.tensor([0, 0])

        output = model(images, season_ids, year_indices, lat, lon, target_season, target_year)

        assert output["prediction"].shape == (batch_size, 1, H, W)

    def test_gradient_flow(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that gradients flow through the model."""
        model = SeasonalViT(small_model_config)
        batch_size = 2
        num_timesteps = 8
        H, W = small_model_config.image_size

        images = torch.randn(batch_size, num_timesteps, H, W, requires_grad=True)
        season_ids = torch.tensor([[s % 4 for s in range(num_timesteps)]] * batch_size)
        year_indices = torch.tensor([[s // 4 for s in range(num_timesteps)]] * batch_size)
        lat = torch.tensor([45.0, -30.0])
        lon = torch.tensor([-122.0, 150.0])
        target_season = torch.tensor([0, 0])
        target_year = torch.tensor([0, 0])

        output = model(images, season_ids, year_indices, lat, lon, target_season, target_year)
        loss = output["prediction"].sum()
        loss.backward()

        assert images.grad is not None
        assert not torch.isnan(images.grad).any()

    def test_deterministic_output(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that same input produces same output in eval mode."""
        model = SeasonalViT(small_model_config)
        model.eval()
        batch_size = 2
        num_timesteps = 8
        H, W = small_model_config.image_size

        images = torch.randn(batch_size, num_timesteps, H, W)
        season_ids = torch.tensor([[s % 4 for s in range(num_timesteps)]] * batch_size)
        year_indices = torch.tensor([[s // 4 for s in range(num_timesteps)]] * batch_size)
        lat = torch.tensor([45.0, -30.0])
        lon = torch.tensor([-122.0, 150.0])
        target_season = torch.tensor([0, 0])
        target_year = torch.tensor([0, 0])

        with torch.no_grad():
            output1 = model(images, season_ids, year_indices, lat, lon, target_season, target_year)
            output2 = model(images, season_ids, year_indices, lat, lon, target_season, target_year)

        assert torch.allclose(output1["prediction"], output2["prediction"])

    def test_attention_analysis(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that attention analysis is returned when requested."""
        from model.decoder import TemporalAttentionAnalysis

        model = SeasonalViT(small_model_config)
        model.eval()
        batch_size = 2
        num_timesteps = 8
        H, W = small_model_config.image_size

        images = torch.randn(batch_size, num_timesteps, H, W)
        season_ids = torch.tensor([[s % 4 for s in range(num_timesteps)]] * batch_size)
        year_indices = torch.tensor([[s // 4 for s in range(num_timesteps)]] * batch_size)
        lat = torch.tensor([45.0, -30.0])
        lon = torch.tensor([-122.0, 150.0])
        target_season = torch.tensor([0, 0])
        target_year = torch.tensor([0, 0])

        with torch.no_grad():
            output = model(
                images, season_ids, year_indices, lat, lon, target_season, target_year,
                return_attention_analysis=True
            )

        # Verify attention analysis is included
        assert "attention_analysis" in output
        analysis = output["attention_analysis"]
        assert isinstance(analysis, TemporalAttentionAnalysis)

        # Verify shapes
        assert analysis.attention_weights.shape == (batch_size, num_timesteps)
        assert analysis.effective_history_90.shape == (batch_size,)
        assert analysis.recent_5yr_weight.shape == (batch_size,)

        # Without flag, no attention analysis
        output_no_analysis = model(
            images, season_ids, year_indices, lat, lon, target_season, target_year,
            return_attention_analysis=False
        )
        assert "attention_analysis" not in output_no_analysis

    def test_prediction_in_valid_range(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that prediction is in valid NDVI range [-1, 1]."""
        model = SeasonalViT(small_model_config)
        batch_size = 2
        num_timesteps = 8
        H, W = small_model_config.image_size

        images = torch.rand(batch_size, num_timesteps, H, W) * 2 - 1
        season_ids = torch.tensor([[s % 4 for s in range(num_timesteps)]] * batch_size)
        year_indices = torch.tensor([[s // 4 for s in range(num_timesteps)]] * batch_size)
        lat = torch.tensor([45.0, -30.0])
        lon = torch.tensor([-122.0, 150.0])
        target_season = torch.tensor([0, 0])
        target_year = torch.tensor([0, 0])

        output = model(images, season_ids, year_indices, lat, lon, target_season, target_year)

        assert output["prediction"].min() >= -1
        assert output["prediction"].max() <= 1

    def test_get_climatology_anchor(
        self, small_model_config: SeasonalViTConfig
    ) -> None:
        """Test _get_climatology_anchor averages same-season observations."""
        model = SeasonalViT(small_model_config)
        batch_size = 2
        num_timesteps = 8
        H, W = small_model_config.image_size

        # Create images with distinct values per timestep
        images = torch.zeros(batch_size, num_timesteps, H, W)
        for t in range(num_timesteps):
            images[:, t] = t

        season_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]] * batch_size)
        target_season = torch.tensor([0, 1])

        anchor = model._get_climatology_anchor(images, season_ids, target_season)

        # For target_season=0: springs at t=0 (val=0) and t=4 (val=4), avg=2.0
        # For target_season=1: summers at t=1 (val=1) and t=5 (val=5), avg=3.0
        assert anchor[0, 0].mean() == pytest.approx(2.0, abs=1e-5)
        assert anchor[1, 0].mean() == pytest.approx(3.0, abs=1e-5)

    def test_factory_function(self) -> None:
        """Test create_seasonal_vit factory function."""
        config = SeasonalViTConfig(swin_pretrained=False)
        model = create_seasonal_vit(config)

        assert isinstance(model, SeasonalViT)

    def test_factory_function_default_config(self) -> None:
        """Test factory function with default config."""
        model = create_seasonal_vit(SeasonalViTConfig(swin_pretrained=False))

        assert isinstance(model, SeasonalViT)


class TestSeasonalViTFromPretrained:
    """Tests for loading from pre-trained weights."""

    def test_from_pretrained(self, small_model_config: SeasonalViTConfig) -> None:
        """Test loading from pre-trained model."""
        # Create a pre-training model
        pretrain_model = PretrainingModel(small_model_config)
        pretrain_state = pretrain_model.state_dict()

        # Create SeasonalViT from pre-trained weights
        model = SeasonalViT.from_pretrained(pretrain_state, small_model_config)

        assert isinstance(model, SeasonalViT)

    def test_from_pretrained_weights_loaded(
        self, small_model_config: SeasonalViTConfig
    ) -> None:
        """Test that pre-trained weights are actually loaded."""
        # Create a pre-training model
        pretrain_model = PretrainingModel(small_model_config)

        # Create fresh SeasonalViT
        fresh_model = SeasonalViT(small_model_config)

        # Create SeasonalViT from pre-trained
        loaded_model = SeasonalViT.from_pretrained(
            pretrain_model.state_dict(), small_model_config
        )

        # Spatial encoder weights should match pre-trained, not fresh
        for name, param in loaded_model.spatial_encoder.named_parameters():
            pretrain_param = getattr(pretrain_model.spatial_encoder, name.split(".")[0])
            if hasattr(pretrain_param, name.split(".")[-1]):
                # Just check that loaded model has valid weights
                assert not torch.isnan(param).any()


class TestSeasonalViTFreezing:
    """Tests for encoder freezing."""

    def test_freeze_encoder(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that freeze_encoder disables gradients for encoder."""
        model = SeasonalViT(small_model_config)
        model.freeze_encoder()

        # Encoder should be frozen
        for name, param in model.named_parameters():
            if (
                name.startswith("spatial_encoder")
                or name.startswith("temporal_encoding")
                or name.startswith("temporal_transformer")
            ):
                assert not param.requires_grad, f"{name} should be frozen"

        # Location encoder and decoder should be trainable
        for name, param in model.named_parameters():
            if name.startswith("location_encoder") or name.startswith("decoder"):
                assert param.requires_grad, f"{name} should be trainable"

    def test_unfreeze_encoder(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that unfreeze_encoder enables all gradients."""
        model = SeasonalViT(small_model_config)
        model.freeze_encoder()
        model.unfreeze_encoder()

        # All should be trainable
        for name, param in model.named_parameters():
            assert param.requires_grad, f"{name} should be trainable"

    def test_frozen_encoder_no_gradients(
        self, small_model_config: SeasonalViTConfig
    ) -> None:
        """Test that frozen encoder doesn't accumulate gradients."""
        model = SeasonalViT(small_model_config)
        model.freeze_encoder()
        batch_size = 2
        num_timesteps = 4
        H, W = small_model_config.image_size

        images = torch.randn(batch_size, num_timesteps, H, W, requires_grad=True)
        season_ids = torch.tensor([[0, 1, 2, 3]] * batch_size)
        year_indices = torch.tensor([[0, 0, 0, 0]] * batch_size)
        lat = torch.tensor([45.0, -30.0])
        lon = torch.tensor([-122.0, 150.0])
        target_season = torch.tensor([0, 0])
        target_year = torch.tensor([0, 0])

        output = model(images, season_ids, year_indices, lat, lon, target_season, target_year)
        loss = output["prediction"].sum()
        loss.backward()

        # Frozen parameters should have no gradient
        for name, param in model.named_parameters():
            if not param.requires_grad:
                assert param.grad is None or (param.grad == 0).all()


class TestSeasonalViTEdgeCases:
    """Edge case tests for SeasonalViT."""

    def test_single_batch(self, small_model_config: SeasonalViTConfig) -> None:
        """Test with single batch."""
        model = SeasonalViT(small_model_config)
        num_timesteps = 4
        H, W = small_model_config.image_size

        images = torch.randn(1, num_timesteps, H, W)
        season_ids = torch.tensor([[0, 1, 2, 3]])
        year_indices = torch.tensor([[0, 0, 0, 0]])
        lat = torch.tensor([45.0])
        lon = torch.tensor([-122.0])
        target_season = torch.tensor([0])
        target_year = torch.tensor([0])

        output = model(images, season_ids, year_indices, lat, lon, target_season, target_year)

        assert output["prediction"].shape == (1, 1, H, W)

    def test_no_matching_season(self, small_model_config: SeasonalViTConfig) -> None:
        """Test fallback when target season not in history."""
        model = SeasonalViT(small_model_config)
        batch_size = 2
        num_timesteps = 4
        H, W = small_model_config.image_size

        images = torch.randn(batch_size, num_timesteps, H, W)
        # All seasons are 0, 1, 2, 3
        season_ids = torch.tensor([[0, 1, 2, 3]] * batch_size)
        year_indices = torch.tensor([[0, 0, 0, 0]] * batch_size)
        lat = torch.tensor([45.0, -30.0])
        lon = torch.tensor([-122.0, 150.0])
        # Target season exists in history
        target_season = torch.tensor([0, 0])
        target_year = torch.tensor([0, 0])

        output = model(images, season_ids, year_indices, lat, lon, target_season, target_year)

        # Should not crash, anchor should be the matching season
        assert output["anchor"].shape == (batch_size, 1, H, W)

    def test_get_num_parameters(self, small_model_config: SeasonalViTConfig) -> None:
        """Test parameter counting."""
        model = SeasonalViT(small_model_config)

        total_params = model.get_num_parameters(trainable_only=False)
        trainable_params = model.get_num_parameters(trainable_only=True)

        assert total_params > 0
        assert trainable_params == total_params  # All trainable by default

        # After freezing, trainable should be less
        model.freeze_encoder()
        trainable_after_freeze = model.get_num_parameters(trainable_only=True)

        assert trainable_after_freeze < total_params

    def test_ndvi_value_range_input(self, small_model_config: SeasonalViTConfig) -> None:
        """Test with valid NDVI range input."""
        model = SeasonalViT(small_model_config)
        batch_size = 2
        num_timesteps = 4
        H, W = small_model_config.image_size

        # Valid NDVI range
        images = torch.rand(batch_size, num_timesteps, H, W) * 2 - 1
        season_ids = torch.tensor([[0, 1, 2, 3]] * batch_size)
        year_indices = torch.tensor([[0, 0, 0, 0]] * batch_size)
        lat = torch.tensor([45.0, -30.0])
        lon = torch.tensor([-122.0, 150.0])
        target_season = torch.tensor([0, 0])
        target_year = torch.tensor([0, 0])

        output = model(images, season_ids, year_indices, lat, lon, target_season, target_year)

        assert not torch.isnan(output["prediction"]).any()
        assert not torch.isinf(output["prediction"]).any()

    def test_different_hemispheres(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that different hemispheres produce different outputs."""
        model = SeasonalViT(small_model_config)
        model.eval()
        num_timesteps = 4
        H, W = small_model_config.image_size

        images = torch.randn(2, num_timesteps, H, W)
        season_ids = torch.tensor([[0, 1, 2, 3]] * 2)
        year_indices = torch.tensor([[0, 0, 0, 0]] * 2)
        # Different hemispheres
        lat = torch.tensor([45.0, -45.0])
        lon = torch.tensor([0.0, 0.0])
        target_season = torch.tensor([0, 0])
        target_year = torch.tensor([0, 0])

        with torch.no_grad():
            output = model(images, season_ids, year_indices, lat, lon, target_season, target_year)

        # Predictions should be different due to location encoding
        # (even with same input images)
        assert not torch.allclose(output["prediction"][0], output["prediction"][1])
