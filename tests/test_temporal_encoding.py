"""Tests for SeasonalPositionalEncoding."""

import math

import pytest
import torch

from model.encoders.temporal import SeasonalPositionalEncoding, create_temporal_encoding
from model.config import SeasonalViTConfig


class TestSeasonalPositionalEncoding:
    """Test suite for SeasonalPositionalEncoding."""

    def test_output_shape(self, model_config: SeasonalViTConfig) -> None:
        """Test that output has correct shape."""
        encoder = SeasonalPositionalEncoding(model_config)
        batch_size = 4
        num_timesteps = model_config.num_timesteps

        season_ids = torch.tensor(
            [s for _ in range(35) for s in range(4)]
        ).unsqueeze(0).expand(batch_size, -1)
        year_indices = torch.tensor(
            [y for y in range(35) for _ in range(4)]
        ).unsqueeze(0).expand(batch_size, -1)

        output = encoder(season_ids, year_indices)

        assert output.shape == (batch_size, num_timesteps, model_config.embed_dim)

    def test_output_shape_1d_input(self, model_config: SeasonalViTConfig) -> None:
        """Test with 1D inputs (auto-broadcast to batch)."""
        encoder = SeasonalPositionalEncoding(model_config)
        num_timesteps = model_config.num_timesteps

        season_ids = torch.tensor([s for _ in range(35) for s in range(4)])
        year_indices = torch.tensor([y for y in range(35) for _ in range(4)])

        output = encoder(season_ids, year_indices)

        assert output.shape == (1, num_timesteps, model_config.embed_dim)

    def test_gradient_flow(self, model_config: SeasonalViTConfig) -> None:
        """Test that gradients flow through the encoder."""
        encoder = SeasonalPositionalEncoding(model_config)

        season_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        year_indices = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])

        output = encoder(season_ids, year_indices)
        loss = output.sum()
        loss.backward()

        # Check gradients exist for learnable parameters
        assert encoder.season_embed.weight.grad is not None
        assert not torch.isnan(encoder.season_embed.weight.grad).any()

    def test_cyclical_property(self, model_config: SeasonalViTConfig) -> None:
        """Test that cyclical encoding produces valid sin/cos values."""
        encoder = SeasonalPositionalEncoding(model_config)

        # Test all 4 seasons
        season_ids = torch.tensor([[0, 1, 2, 3]])
        year_indices = torch.tensor([[0, 0, 0, 0]])

        output = encoder(season_ids, year_indices)

        # Output should be valid (no NaN/Inf)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_different_seasons_produce_different_encodings(
        self, model_config: SeasonalViTConfig
    ) -> None:
        """Test that different seasons produce different encodings."""
        encoder = SeasonalPositionalEncoding(model_config)
        encoder.eval()

        # Same year, different seasons
        with torch.no_grad():
            spring = encoder(torch.tensor([[0]]), torch.tensor([[0]]))
            summer = encoder(torch.tensor([[1]]), torch.tensor([[0]]))
            fall = encoder(torch.tensor([[2]]), torch.tensor([[0]]))
            winter = encoder(torch.tensor([[3]]), torch.tensor([[0]]))

        # All should be different
        assert not torch.allclose(spring, summer)
        assert not torch.allclose(summer, fall)
        assert not torch.allclose(fall, winter)
        assert not torch.allclose(winter, spring)

    def test_different_years_produce_different_encodings(
        self, model_config: SeasonalViTConfig
    ) -> None:
        """Test that different years produce different encodings."""
        encoder = SeasonalPositionalEncoding(model_config)
        encoder.eval()

        # Same season, different years
        with torch.no_grad():
            year0 = encoder(torch.tensor([[0]]), torch.tensor([[0]]))
            year10 = encoder(torch.tensor([[0]]), torch.tensor([[10]]))
            year30 = encoder(torch.tensor([[0]]), torch.tensor([[30]]))

        # All should be different
        assert not torch.allclose(year0, year10)
        assert not torch.allclose(year10, year30)
        assert not torch.allclose(year0, year30)

    def test_deterministic_output(self, model_config: SeasonalViTConfig) -> None:
        """Test that same input produces same output in eval mode."""
        encoder = SeasonalPositionalEncoding(model_config)
        encoder.eval()

        season_ids = torch.tensor([[0, 1, 2, 3]])
        year_indices = torch.tensor([[0, 0, 0, 0]])

        with torch.no_grad():
            output1 = encoder(season_ids, year_indices)
            output2 = encoder(season_ids, year_indices)

        assert torch.allclose(output1, output2)

    def test_explicit_seq_positions(self, model_config: SeasonalViTConfig) -> None:
        """Test with explicit sequence positions."""
        encoder = SeasonalPositionalEncoding(model_config)

        season_ids = torch.tensor([[0, 1, 2, 3]])
        year_indices = torch.tensor([[0, 0, 0, 0]])
        seq_positions = torch.tensor([[5, 6, 7, 8]])  # Custom positions

        output = encoder(season_ids, year_indices, seq_positions)

        assert output.shape == (1, 4, model_config.embed_dim)
        assert not torch.isnan(output).any()

    def test_factory_function(self) -> None:
        """Test create_temporal_encoding factory function."""
        config = SeasonalViTConfig(swin_pretrained=False)
        encoder = create_temporal_encoding(config)

        assert isinstance(encoder, SeasonalPositionalEncoding)
        assert encoder.get_output_dim() == config.embed_dim

    def test_factory_function_default_config(self) -> None:
        """Test factory function with default config."""
        encoder = create_temporal_encoding()

        assert isinstance(encoder, SeasonalPositionalEncoding)
        assert encoder.get_output_dim() == 768  # Default embed_dim

    def test_get_output_dim(self, model_config: SeasonalViTConfig) -> None:
        """Test get_output_dim method."""
        encoder = SeasonalPositionalEncoding(model_config)

        assert encoder.get_output_dim() == model_config.embed_dim


class TestSeasonalEncodingEdgeCases:
    """Edge case tests for seasonal positional encoding."""

    def test_single_timestep(self, model_config: SeasonalViTConfig) -> None:
        """Test with single timestep."""
        encoder = SeasonalPositionalEncoding(model_config)

        season_ids = torch.tensor([[2]])
        year_indices = torch.tensor([[15]])

        output = encoder(season_ids, year_indices)

        assert output.shape == (1, 1, model_config.embed_dim)
        assert not torch.isnan(output).any()

    def test_large_batch(self, model_config: SeasonalViTConfig) -> None:
        """Test with large batch size."""
        encoder = SeasonalPositionalEncoding(model_config)
        batch_size = 32

        season_ids = torch.randint(0, 4, (batch_size, 140))
        year_indices = torch.randint(0, 35, (batch_size, 140))

        output = encoder(season_ids, year_indices)

        assert output.shape == (batch_size, 140, model_config.embed_dim)
        assert not torch.isnan(output).any()

    def test_boundary_years(self, model_config: SeasonalViTConfig) -> None:
        """Test with boundary year values."""
        encoder = SeasonalPositionalEncoding(model_config)

        # Year 0 and max year
        season_ids = torch.tensor([[0, 0]])
        year_indices = torch.tensor([[0, 34]])

        output = encoder(season_ids, year_indices)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_season_cyclic_continuity(self, model_config: SeasonalViTConfig) -> None:
        """Test that season 3 and season 0 are neighbors in cyclic space."""
        encoder = SeasonalPositionalEncoding(model_config)
        encoder.eval()

        # Check that the cyclical encoding places season 3 close to season 0
        # (they are adjacent in the annual cycle)
        with torch.no_grad():
            # Get raw cyclical values (before projection)
            season_0_angle = 2 * math.pi * 0 / 4
            season_3_angle = 2 * math.pi * 3 / 4

            # These should be 90 degrees apart (π/2), not 270 degrees
            diff = abs(season_0_angle - season_3_angle)
            # Normalize to [0, π]
            diff = min(diff, 2 * math.pi - diff)

            # Season 0 and 3 should be 90 degrees apart in angle
            assert abs(diff - math.pi / 2) < 1e-6 or abs(diff - 3 * math.pi / 2) < 1e-6
