"""Tests for DualScaleTemporalTransformer."""

import pytest
import torch

from model.temporal_transformer import (
    DualScaleTemporalTransformer,
    create_temporal_transformer,
)
from model.config import SeasonalViTConfig


class TestDualScaleTemporalTransformer:
    """Test suite for DualScaleTemporalTransformer."""

    def test_output_shape(self, model_config: SeasonalViTConfig) -> None:
        """Test that output has correct shape."""
        transformer = DualScaleTemporalTransformer(model_config)
        batch_size = 2
        num_timesteps = 16  # 4 years; full T=140 covered by test_full_sequence_140_timesteps
        num_patches = model_config.num_spatial_patches  # 64
        embed_dim = model_config.embed_dim  # 768

        spatial_features = torch.randn(batch_size, num_timesteps, num_patches, embed_dim)
        temporal_encoding = torch.randn(batch_size, num_timesteps, embed_dim)

        output = transformer(spatial_features, temporal_encoding)

        assert output.shape == (batch_size, num_timesteps, num_patches, embed_dim)

    def test_output_shape_small(self, model_config: SeasonalViTConfig) -> None:
        """Test with smaller temporal sequence (8 timesteps = 2 years)."""
        transformer = DualScaleTemporalTransformer(model_config)
        batch_size = 2
        num_timesteps = 8  # 2 years × 4 seasons
        num_patches = model_config.num_spatial_patches

        spatial_features = torch.randn(
            batch_size, num_timesteps, num_patches, model_config.embed_dim
        )
        temporal_encoding = torch.randn(
            batch_size, num_timesteps, model_config.embed_dim
        )

        output = transformer(spatial_features, temporal_encoding)

        assert output.shape == spatial_features.shape

    def test_gradient_flow(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that gradients flow through the transformer."""
        transformer = DualScaleTemporalTransformer(small_model_config)
        batch_size = 2
        num_timesteps = 8
        num_patches = small_model_config.num_spatial_patches

        spatial_features = torch.randn(
            batch_size, num_timesteps, num_patches, small_model_config.embed_dim,
            requires_grad=True
        )
        temporal_encoding = torch.randn(
            batch_size, num_timesteps, small_model_config.embed_dim
        )

        output = transformer(spatial_features, temporal_encoding)
        loss = output.sum()
        loss.backward()

        assert spatial_features.grad is not None
        assert spatial_features.grad.shape == spatial_features.shape
        assert not torch.isnan(spatial_features.grad).any()

    def test_deterministic_output(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that same input produces same output in eval mode."""
        transformer = DualScaleTemporalTransformer(small_model_config)
        transformer.eval()
        batch_size = 2
        num_timesteps = 8
        num_patches = small_model_config.num_spatial_patches

        spatial_features = torch.randn(
            batch_size, num_timesteps, num_patches, small_model_config.embed_dim
        )
        temporal_encoding = torch.randn(
            batch_size, num_timesteps, small_model_config.embed_dim
        )

        with torch.no_grad():
            output1 = transformer(spatial_features, temporal_encoding)
            output2 = transformer(spatial_features, temporal_encoding)

        assert torch.allclose(output1, output2)

    def test_temporal_encoding_affects_output(
        self, small_model_config: SeasonalViTConfig
    ) -> None:
        """Test that different temporal encodings produce different outputs."""
        transformer = DualScaleTemporalTransformer(small_model_config)
        transformer.eval()
        batch_size = 2
        num_timesteps = 8
        num_patches = small_model_config.num_spatial_patches

        spatial_features = torch.randn(
            batch_size, num_timesteps, num_patches, small_model_config.embed_dim
        )
        temporal_encoding1 = torch.randn(
            batch_size, num_timesteps, small_model_config.embed_dim
        )
        temporal_encoding2 = torch.randn(
            batch_size, num_timesteps, small_model_config.embed_dim
        )

        with torch.no_grad():
            output1 = transformer(spatial_features, temporal_encoding1)
            output2 = transformer(spatial_features, temporal_encoding2)

        assert not torch.allclose(output1, output2)

    def test_factory_function(self) -> None:
        """Test create_temporal_transformer factory function."""
        config = SeasonalViTConfig(swin_pretrained=False)
        transformer = create_temporal_transformer(config)

        assert isinstance(transformer, DualScaleTemporalTransformer)
        assert transformer.config.embed_dim == config.embed_dim

    def test_factory_function_default_config(self) -> None:
        """Test factory function with default config."""
        transformer = create_temporal_transformer()

        assert isinstance(transformer, DualScaleTemporalTransformer)
        assert transformer.config.embed_dim == 768

    def test_get_output_shape(self, model_config: SeasonalViTConfig) -> None:
        """Test get_output_shape method."""
        transformer = DualScaleTemporalTransformer(model_config)

        shape = transformer.get_output_shape(
            batch_size=4,
            num_timesteps=140,
            num_patches=64,
        )

        assert shape == (4, 140, 64, model_config.embed_dim)


class TestDualScalePathways:
    """Tests for individual pathways in the transformer."""

    def test_cross_season_pathway_processes_within_years(
        self, small_model_config: SeasonalViTConfig
    ) -> None:
        """Test that cross-season (phenological) pathway processes each year."""
        transformer = DualScaleTemporalTransformer(small_model_config)
        transformer.eval()
        batch_size = 2
        n_years = 3
        num_timesteps = n_years * 4
        num_patches = small_model_config.num_spatial_patches

        # Create input with distinct year patterns
        x = torch.randn(
            batch_size, n_years, 4, num_patches, small_model_config.embed_dim
        )

        with torch.no_grad():
            cross_season_out = transformer._process_cross_season(x, n_years)

        # Output should have same shape
        assert cross_season_out.shape == x.shape

    def test_cross_year_pathway_processes_across_years(
        self, small_model_config: SeasonalViTConfig
    ) -> None:
        """Test that cross-year (climatological) pathway processes same season."""
        transformer = DualScaleTemporalTransformer(small_model_config)
        transformer.eval()
        batch_size = 2
        n_years = 3
        num_patches = small_model_config.num_spatial_patches

        x = torch.randn(
            batch_size, n_years, 4, num_patches, small_model_config.embed_dim
        )

        with torch.no_grad():
            cross_year_out = transformer._process_cross_year(x, n_years)

        # Output should have same shape
        assert cross_year_out.shape == x.shape

    def test_fusion_combines_pathways(
        self, small_model_config: SeasonalViTConfig
    ) -> None:
        """Test that fusion combines cross-season and cross-year features."""
        transformer = DualScaleTemporalTransformer(small_model_config)
        transformer.eval()
        batch_size = 2
        num_timesteps = 8
        num_patches = small_model_config.num_spatial_patches

        cross_season = torch.randn(
            batch_size, num_timesteps, num_patches, small_model_config.embed_dim
        )
        cross_year = torch.randn(
            batch_size, num_timesteps, num_patches, small_model_config.embed_dim
        )

        with torch.no_grad():
            fused = transformer._fuse_pathways(cross_season, cross_year)

        assert fused.shape == cross_season.shape
        # Fused should be different from both inputs
        assert not torch.allclose(fused, cross_season)
        assert not torch.allclose(fused, cross_year)


class TestTemporalTransformerEdgeCases:
    """Edge case tests for temporal transformer."""

    def test_single_batch(self, small_model_config: SeasonalViTConfig) -> None:
        """Test with single batch."""
        transformer = DualScaleTemporalTransformer(small_model_config)
        num_timesteps = 8
        num_patches = small_model_config.num_spatial_patches

        spatial_features = torch.randn(1, num_timesteps, num_patches, small_model_config.embed_dim)
        temporal_encoding = torch.randn(1, num_timesteps, small_model_config.embed_dim)

        output = transformer(spatial_features, temporal_encoding)

        assert output.shape == (1, num_timesteps, num_patches, small_model_config.embed_dim)
        assert not torch.isnan(output).any()

    def test_minimal_sequence(self, small_model_config: SeasonalViTConfig) -> None:
        """Test with minimal sequence (1 year = 4 timesteps)."""
        transformer = DualScaleTemporalTransformer(small_model_config)
        batch_size = 2
        num_timesteps = 4  # 1 year
        num_patches = small_model_config.num_spatial_patches

        spatial_features = torch.randn(
            batch_size, num_timesteps, num_patches, small_model_config.embed_dim
        )
        temporal_encoding = torch.randn(
            batch_size, num_timesteps, small_model_config.embed_dim
        )

        output = transformer(spatial_features, temporal_encoding)

        assert output.shape == spatial_features.shape
        assert not torch.isnan(output).any()

    def test_extreme_values(self, small_model_config: SeasonalViTConfig) -> None:
        """Test with extreme input values."""
        transformer = DualScaleTemporalTransformer(small_model_config)
        transformer.eval()
        batch_size = 2
        num_timesteps = 8
        num_patches = small_model_config.num_spatial_patches

        # Test with zeros
        spatial_zeros = torch.zeros(
            batch_size, num_timesteps, num_patches, small_model_config.embed_dim
        )
        temporal_zeros = torch.zeros(batch_size, num_timesteps, small_model_config.embed_dim)
        output_zeros = transformer(spatial_zeros, temporal_zeros)
        assert not torch.isnan(output_zeros).any()

        # Test with small values
        spatial_small = torch.randn(
            batch_size, num_timesteps, num_patches, small_model_config.embed_dim
        ) * 0.01
        temporal_small = torch.randn(
            batch_size, num_timesteps, small_model_config.embed_dim
        ) * 0.01
        output_small = transformer(spatial_small, temporal_small)
        assert not torch.isnan(output_small).any()

    def test_full_sequence_140_timesteps(
        self, small_model_config: SeasonalViTConfig
    ) -> None:
        """Test with full 140-timestep sequence (35 years)."""
        transformer = DualScaleTemporalTransformer(small_model_config)
        batch_size = 1  # Small batch to save memory
        num_timesteps = small_model_config.num_timesteps  # 140
        num_patches = small_model_config.num_spatial_patches

        spatial_features = torch.randn(
            batch_size, num_timesteps, num_patches, small_model_config.embed_dim
        )
        temporal_encoding = torch.randn(
            batch_size, num_timesteps, small_model_config.embed_dim
        )

        output = transformer(spatial_features, temporal_encoding)

        assert output.shape == spatial_features.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_different_configs(self) -> None:
        """Test with different model configurations."""
        config_small = SeasonalViTConfig(
            temporal_depth=2,
            temporal_heads=4,
            swin_pretrained=False,
        )
        transformer_small = DualScaleTemporalTransformer(config_small)

        batch_size = 2
        num_timesteps = 8
        num_patches = config_small.num_spatial_patches

        spatial_features = torch.randn(
            batch_size, num_timesteps, num_patches, config_small.embed_dim
        )
        temporal_encoding = torch.randn(
            batch_size, num_timesteps, config_small.embed_dim
        )

        output = transformer_small(spatial_features, temporal_encoding)

        assert output.shape == spatial_features.shape
        assert not torch.isnan(output).any()

    def test_gradient_checkpointing(self) -> None:
        """Test that gradient checkpointing works and gradients flow correctly."""
        config = SeasonalViTConfig(
            temporal_depth=2,
            temporal_heads=4,
            swin_pretrained=False,
            use_gradient_checkpointing=True,
        )
        transformer = DualScaleTemporalTransformer(config)
        transformer.train()

        batch_size = 2
        num_timesteps = 8
        num_patches = config.num_spatial_patches

        spatial_features = torch.randn(
            batch_size, num_timesteps, num_patches, config.embed_dim,
            requires_grad=True,
        )
        temporal_encoding = torch.randn(
            batch_size, num_timesteps, config.embed_dim,
        )

        output = transformer(spatial_features, temporal_encoding)
        loss = output.sum()
        loss.backward()

        # Verify gradients flow through
        assert spatial_features.grad is not None
        assert spatial_features.grad.shape == spatial_features.shape
        assert not torch.isnan(spatial_features.grad).any()

        # Verify checkpointing flag is set
        assert transformer.use_checkpointing is True
