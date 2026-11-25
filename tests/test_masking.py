"""Tests for 3-level masking strategy."""

import pytest
import torch

from train.masking import (
    generate_mask,
    generate_patch_mask,
    generate_temporal_mask,
    generate_image_mask,
    compute_mask_stats,
    MaskOutput,
)
from model.config import PretrainingConfig


class TestPatchMasking:
    """Tests for patch-level masking."""

    def test_patch_mask_shape(self) -> None:
        """Test patch mask has correct shape."""
        batch_size, num_timesteps, num_patches = 4, 140, 64
        mask = generate_patch_mask(batch_size, num_timesteps, num_patches, mask_ratio=0.75)

        assert mask.shape == (batch_size, num_timesteps, num_patches)
        assert mask.dtype == torch.bool

    def test_patch_mask_ratio(self) -> None:
        """Test that mask ratio is approximately correct."""
        batch_size, num_timesteps, num_patches = 8, 140, 64
        mask_ratio = 0.75

        mask = generate_patch_mask(batch_size, num_timesteps, num_patches, mask_ratio)

        # Calculate actual ratio
        actual_ratio = mask.float().mean().item()
        # Should be close to target (within 1% for large batches)
        assert abs(actual_ratio - mask_ratio) < 0.01

    def test_patch_mask_per_image(self) -> None:
        """Test that each image has correct number of masked patches."""
        batch_size, num_timesteps, num_patches = 2, 10, 64
        mask_ratio = 0.75
        expected_masked = int(num_patches * mask_ratio)

        mask = generate_patch_mask(batch_size, num_timesteps, num_patches, mask_ratio)

        # Each image should have exactly expected_masked patches masked
        masked_per_image = mask.sum(dim=-1)
        assert (masked_per_image == expected_masked).all()

    def test_patch_mask_randomness(self) -> None:
        """Test that masks are different between calls."""
        batch_size, num_timesteps, num_patches = 2, 10, 64

        mask1 = generate_patch_mask(batch_size, num_timesteps, num_patches, 0.75)
        mask2 = generate_patch_mask(batch_size, num_timesteps, num_patches, 0.75)

        # Masks should be different (with very high probability)
        assert not torch.equal(mask1, mask2)


class TestTemporalMasking:
    """Tests for temporal-level masking."""

    def test_temporal_mask_shape(self) -> None:
        """Test temporal mask has correct shape."""
        batch_size, num_timesteps = 4, 140
        mask = generate_temporal_mask(batch_size, num_timesteps, mask_ratio=0.25)

        assert mask.shape == (batch_size, num_timesteps)
        assert mask.dtype == torch.bool

    def test_temporal_mask_ratio(self) -> None:
        """Test that temporal mask ratio is approximately correct."""
        batch_size, num_timesteps = 32, 140
        mask_ratio = 0.25

        mask = generate_temporal_mask(batch_size, num_timesteps, mask_ratio)

        actual_ratio = mask.float().mean().item()
        assert abs(actual_ratio - mask_ratio) < 0.02

    def test_temporal_mask_per_sample(self) -> None:
        """Test that each sample has correct number of masked timesteps."""
        batch_size, num_timesteps = 4, 140
        mask_ratio = 0.25
        expected_masked = int(num_timesteps * mask_ratio)

        mask = generate_temporal_mask(batch_size, num_timesteps, mask_ratio)

        masked_per_sample = mask.sum(dim=-1)
        assert (masked_per_sample == expected_masked).all()


class TestImageMasking:
    """Tests for image-level masking."""

    def test_image_mask_shape(self) -> None:
        """Test image mask has correct shape."""
        batch_size, num_timesteps = 4, 140
        mask = generate_image_mask(batch_size, num_timesteps, mask_prob=0.1)

        assert mask.shape == (batch_size, num_timesteps)
        assert mask.dtype == torch.bool

    def test_image_mask_probability(self) -> None:
        """Test that image mask probability is approximately correct."""
        batch_size, num_timesteps = 64, 140
        mask_prob = 0.1

        mask = generate_image_mask(batch_size, num_timesteps, mask_prob)

        actual_prob = mask.float().mean().item()
        # Probabilistic masking has higher variance
        assert abs(actual_prob - mask_prob) < 0.05


class TestCombinedMasking:
    """Tests for combined 3-level masking."""

    def test_generate_mask_returns_mask_output(self) -> None:
        """Test that generate_mask returns MaskOutput."""
        batch_size, num_timesteps, num_patches = 4, 140, 64

        result = generate_mask(batch_size, num_timesteps, num_patches)

        assert isinstance(result, MaskOutput)
        assert result.patch_mask.shape == (batch_size, num_timesteps, num_patches)
        assert result.temporal_mask.shape == (batch_size, num_timesteps)
        assert result.combined_mask.shape == (batch_size, num_timesteps, num_patches)

    def test_combined_mask_includes_temporal(self) -> None:
        """Test that combined mask includes temporal mask."""
        batch_size, num_timesteps, num_patches = 4, 140, 64

        result = generate_mask(batch_size, num_timesteps, num_patches)

        # Where temporal_mask is True, all patches in that timestep should be masked
        for b in range(batch_size):
            for t in range(num_timesteps):
                if result.temporal_mask[b, t]:
                    assert result.combined_mask[b, t].all()

    def test_combined_mask_ratio_higher_than_patch(self) -> None:
        """Test that combined mask ratio >= patch mask ratio."""
        batch_size, num_timesteps, num_patches = 16, 140, 64

        result = generate_mask(batch_size, num_timesteps, num_patches)

        patch_ratio = result.patch_mask.float().mean().item()
        combined_ratio = result.combined_mask.float().mean().item()

        # Combined should have at least as many masked as patch alone
        assert combined_ratio >= patch_ratio - 0.01

    def test_custom_config(self) -> None:
        """Test masking with custom config."""
        config = PretrainingConfig(
            patch_mask_ratio=0.5,
            temporal_mask_ratio=0.1,
            image_mask_prob=0.05,
        )
        batch_size, num_timesteps, num_patches = 8, 140, 64

        result = generate_mask(batch_size, num_timesteps, num_patches, config=config)

        # Patch mask should be close to 50%
        patch_ratio = result.patch_mask.float().mean().item()
        assert abs(patch_ratio - 0.5) < 0.02

    def test_device_placement(self) -> None:
        """Test that masks are created on correct device."""
        batch_size, num_timesteps, num_patches = 4, 140, 64
        device = torch.device("cpu")

        result = generate_mask(batch_size, num_timesteps, num_patches, device=device)

        assert result.patch_mask.device == device
        assert result.temporal_mask.device == device
        assert result.combined_mask.device == device


class TestMaskStats:
    """Tests for mask statistics computation."""

    def test_compute_mask_stats(self) -> None:
        """Test that stats are computed correctly."""
        batch_size, num_timesteps, num_patches = 8, 140, 64

        result = generate_mask(batch_size, num_timesteps, num_patches)
        stats = compute_mask_stats(result)

        assert "combined_mask_ratio" in stats
        assert "temporal_mask_ratio" in stats
        assert "patch_mask_ratio" in stats
        assert "num_fully_masked_timesteps" in stats

        # Ratios should be in valid range
        assert 0 <= stats["combined_mask_ratio"] <= 1
        assert 0 <= stats["temporal_mask_ratio"] <= 1
        assert 0 <= stats["patch_mask_ratio"] <= 1

    def test_stats_match_actual_values(self) -> None:
        """Test that stats match actual mask values."""
        batch_size, num_timesteps, num_patches = 4, 140, 64

        result = generate_mask(batch_size, num_timesteps, num_patches)
        stats = compute_mask_stats(result)

        # Manual computation
        expected_combined = result.combined_mask.float().mean().item()
        expected_temporal = result.temporal_mask.float().mean().item()
        expected_patch = result.patch_mask.float().mean().item()

        assert abs(stats["combined_mask_ratio"] - expected_combined) < 1e-6
        assert abs(stats["temporal_mask_ratio"] - expected_temporal) < 1e-6
        assert abs(stats["patch_mask_ratio"] - expected_patch) < 1e-6


class TestMaskingEdgeCases:
    """Edge case tests for masking."""

    def test_zero_mask_ratio(self) -> None:
        """Test with zero mask ratio."""
        batch_size, num_timesteps, num_patches = 2, 10, 64

        mask = generate_patch_mask(batch_size, num_timesteps, num_patches, mask_ratio=0.0)

        assert mask.sum() == 0

    def test_full_mask_ratio(self) -> None:
        """Test with full mask ratio."""
        batch_size, num_timesteps, num_patches = 2, 10, 64

        mask = generate_patch_mask(batch_size, num_timesteps, num_patches, mask_ratio=1.0)

        assert mask.all()

    def test_single_batch(self) -> None:
        """Test with single batch."""
        result = generate_mask(1, 140, 64)

        assert result.combined_mask.shape == (1, 140, 64)

    def test_small_sequence(self) -> None:
        """Test with small sequence (4 timesteps)."""
        result = generate_mask(batch_size=2, num_timesteps=4, num_patches=64)

        assert result.combined_mask.shape == (2, 4, 64)
        assert not torch.isnan(result.combined_mask.float()).any()

    def test_small_patches(self) -> None:
        """Test with small number of patches."""
        result = generate_mask(batch_size=2, num_timesteps=10, num_patches=4)

        assert result.combined_mask.shape == (2, 10, 4)
