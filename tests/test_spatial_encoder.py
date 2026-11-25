"""Tests for HierarchicalSpatialEncoder."""

import pytest
import torch

from model.encoders.spatial import HierarchicalSpatialEncoder, create_spatial_encoder
from model.config import SeasonalViTConfig


class TestHierarchicalSpatialEncoder:
    """Test suite for HierarchicalSpatialEncoder."""

    def test_output_shape(self, model_config: SeasonalViTConfig) -> None:
        """Test that output has correct shape."""
        encoder = HierarchicalSpatialEncoder(model_config)
        batch_size = 4
        x = torch.randn(batch_size, 1, 256, 256)

        output = encoder(x)

        assert output.shape == (batch_size, 64, 768)

    def test_output_shape_large_batch(self, small_model_config: SeasonalViTConfig) -> None:
        """Test with larger batch simulating B*T flattening."""
        encoder = HierarchicalSpatialEncoder(small_model_config)
        # Simulate 2 batches × 14 timesteps (validates B*T flattening logic)
        batch_size = 2 * 14
        H, W = small_model_config.image_size
        x = torch.randn(batch_size, 1, H, W)

        output = encoder(x)

        assert output.shape == (batch_size, small_model_config.num_spatial_patches, small_model_config.embed_dim)

    def test_gradient_flow(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that gradients flow through the encoder."""
        H, W = small_model_config.image_size
        encoder = HierarchicalSpatialEncoder(small_model_config)
        x = torch.randn(2, 1, H, W, requires_grad=True)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.isnan(x.grad).any()

    def test_input_proj_shape(self, small_model_config: SeasonalViTConfig) -> None:
        """Test input projection from 1 to 3 channels."""
        H, W = small_model_config.image_size
        encoder = HierarchicalSpatialEncoder(small_model_config)
        x = torch.randn(2, 1, H, W)

        projected = encoder.input_proj(x)

        assert projected.shape == (2, 3, H, W)

    def test_deterministic_output(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that same input produces same output in eval mode."""
        H, W = small_model_config.image_size
        encoder = HierarchicalSpatialEncoder(small_model_config)
        encoder.eval()
        x = torch.randn(2, 1, H, W)

        with torch.no_grad():
            output1 = encoder(x)
            output2 = encoder(x)

        assert torch.allclose(output1, output2)

    def test_get_output_shape(self, small_model_config: SeasonalViTConfig) -> None:
        """Test get_output_shape helper method."""
        encoder = HierarchicalSpatialEncoder(small_model_config)
        batch_size = 10

        shape = encoder.get_output_shape(batch_size)

        assert shape == (10, small_model_config.num_spatial_patches, small_model_config.embed_dim)

    def test_factory_function_default_config(self) -> None:
        """Test create_spatial_encoder with custom config (no pretrained for CI)."""
        # Use non-pretrained config to avoid network issues in CI
        config = SeasonalViTConfig(swin_pretrained=False)
        encoder = create_spatial_encoder(config)

        assert isinstance(encoder, HierarchicalSpatialEncoder)
        assert encoder.config.image_size == (256, 256)

    def test_factory_function_custom_config(self) -> None:
        """Test create_spatial_encoder with custom config."""
        config = SeasonalViTConfig(swin_pretrained=False)
        encoder = create_spatial_encoder(config)

        assert encoder.config.swin_pretrained is False

    def test_no_pretrained_weights(self) -> None:
        """Test encoder works without pretrained weights."""
        config = SeasonalViTConfig(swin_pretrained=False)
        encoder = HierarchicalSpatialEncoder(config)
        x = torch.randn(2, 1, 256, 256)

        output = encoder(x)

        assert output.shape == (2, 64, 768)
        assert not torch.isnan(output).any()

    def test_ndvi_value_range(self, small_model_config: SeasonalViTConfig) -> None:
        """Test encoder handles NDVI value range [-1, 1]."""
        H, W = small_model_config.image_size
        encoder = HierarchicalSpatialEncoder(small_model_config)
        x = torch.rand(2, 1, H, W) * 2 - 1  # [-1, 1]

        output = encoder(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestSpatialEncoderEdgeCases:
    """Edge case tests for spatial encoder."""

    def test_single_image(self, small_model_config: SeasonalViTConfig) -> None:
        """Test with single image batch."""
        H, W = small_model_config.image_size
        encoder = HierarchicalSpatialEncoder(small_model_config)
        x = torch.randn(1, 1, H, W)

        output = encoder(x)

        assert output.shape == (1, small_model_config.num_spatial_patches, small_model_config.embed_dim)

    def test_extreme_values(self, small_model_config: SeasonalViTConfig) -> None:
        """Test with extreme NDVI values."""
        H, W = small_model_config.image_size
        encoder = HierarchicalSpatialEncoder(small_model_config)
        encoder.eval()

        # Test with all zeros
        x_zeros = torch.zeros(2, 1, H, W)
        output_zeros = encoder(x_zeros)
        assert not torch.isnan(output_zeros).any()

        # Test with all ones
        x_ones = torch.ones(2, 1, H, W)
        output_ones = encoder(x_ones)
        assert not torch.isnan(output_ones).any()

        # Test with all negative ones
        x_neg = -torch.ones(2, 1, H, W)
        output_neg = encoder(x_neg)
        assert not torch.isnan(output_neg).any()
