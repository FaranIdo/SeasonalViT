"""Tests for PretrainingModel."""

import pytest
import torch

from model.pretrain_model import (
    PretrainingModel,
    ReconstructionHead,
    create_pretrain_model,
)
from model.config import SeasonalViTConfig, PretrainingConfig
from train.masking import generate_mask, generate_efficient_temporal_mask
from train.pretrain import _sample_year_subset


class TestReconstructionHead:
    """Tests for ReconstructionHead."""

    def test_output_shape(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that output has correct shape."""
        head = ReconstructionHead(small_model_config)
        batch_size = 2
        num_timesteps = 8
        num_patches = small_model_config.num_spatial_patches

        features = torch.randn(batch_size, num_timesteps, num_patches, small_model_config.embed_dim)

        output = head(features)

        # Output should be (B, T, N, patch_pixels)
        assert output.shape == (batch_size, num_timesteps, num_patches, head.patch_pixels)

    def test_patch_pixels_correct(self, model_config: SeasonalViTConfig) -> None:
        """Test that patch_pixels is computed correctly."""
        head = ReconstructionHead(model_config)

        # 256x256 image with 8x8 patches → each patch is 32x32 = 1024 pixels
        assert head.patch_pixels == 32 * 32


class TestPretrainingModel:
    """Tests for PretrainingModel."""

    def test_output_shape(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that output has correct shapes."""
        model = PretrainingModel(small_model_config)
        batch_size = 2
        num_timesteps = 8  # 2 years × 4 seasons
        H, W = small_model_config.image_size

        images = torch.randn(batch_size, num_timesteps, H, W)
        season_ids = torch.tensor([[s % 4 for s in range(num_timesteps)]] * batch_size)
        year_indices = torch.tensor([[s // 4 for s in range(num_timesteps)]] * batch_size)

        output = model(images, season_ids, year_indices)

        # Check all output shapes
        assert output["reconstruction"].shape == (
            batch_size, num_timesteps, small_model_config.num_spatial_patches, 1024
        )
        assert output["target"].shape == output["reconstruction"].shape
        assert output["features"].shape == (
            batch_size, num_timesteps, small_model_config.num_spatial_patches, small_model_config.embed_dim
        )
        assert output["mask"] is None

    def test_output_with_mask(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that masking works correctly."""
        model = PretrainingModel(small_model_config)
        batch_size = 2
        num_timesteps = 8
        H, W = small_model_config.image_size

        images = torch.randn(batch_size, num_timesteps, H, W)
        season_ids = torch.tensor([[s % 4 for s in range(num_timesteps)]] * batch_size)
        year_indices = torch.tensor([[s // 4 for s in range(num_timesteps)]] * batch_size)

        # Generate mask
        mask_output = generate_mask(
            batch_size, num_timesteps, small_model_config.num_spatial_patches
        )
        mask = mask_output.combined_mask

        output = model(images, season_ids, year_indices, mask=mask)

        # Check mask is returned
        assert output["mask"] is not None
        assert torch.equal(output["mask"], mask)

    def test_gradient_flow(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that gradients flow through the model."""
        model = PretrainingModel(small_model_config)
        batch_size = 2
        num_timesteps = 8
        H, W = small_model_config.image_size

        images = torch.randn(batch_size, num_timesteps, H, W, requires_grad=True)
        season_ids = torch.tensor([[s % 4 for s in range(num_timesteps)]] * batch_size)
        year_indices = torch.tensor([[s // 4 for s in range(num_timesteps)]] * batch_size)

        output = model(images, season_ids, year_indices)
        loss = output["reconstruction"].sum()
        loss.backward()

        assert images.grad is not None
        assert not torch.isnan(images.grad).any()

    def test_deterministic_output(self, small_model_config: SeasonalViTConfig) -> None:
        """Test that same input produces same output in eval mode."""
        model = PretrainingModel(small_model_config)
        model.eval()
        batch_size = 2
        num_timesteps = 8
        H, W = small_model_config.image_size

        images = torch.randn(batch_size, num_timesteps, H, W)
        season_ids = torch.tensor([[s % 4 for s in range(num_timesteps)]] * batch_size)
        year_indices = torch.tensor([[s // 4 for s in range(num_timesteps)]] * batch_size)

        with torch.no_grad():
            output1 = model(images, season_ids, year_indices)
            output2 = model(images, season_ids, year_indices)

        assert torch.allclose(output1["reconstruction"], output2["reconstruction"])

    def test_patchify(self, model_config: SeasonalViTConfig) -> None:
        """Test _patchify produces correct output shape."""
        model = PretrainingModel(model_config)
        batch_size = 2
        num_timesteps = 4
        H, W = model_config.image_size

        images = torch.randn(batch_size, num_timesteps, H, W)
        patches = model._patchify(images)

        # Should be (B, T, 64, 1024) - 64 patches of 32x32 pixels
        assert patches.shape == (batch_size, num_timesteps, 64, 1024)

    def test_apply_mask(self, small_model_config: SeasonalViTConfig) -> None:
        """Test _apply_mask replaces masked positions."""
        model = PretrainingModel(small_model_config)
        batch_size = 2
        num_timesteps = 4
        num_patches = small_model_config.num_spatial_patches

        features = torch.randn(batch_size, num_timesteps, num_patches, small_model_config.embed_dim)
        mask = torch.zeros(batch_size, num_timesteps, num_patches, dtype=torch.bool)
        mask[0, 0, 0] = True  # Mask one position

        masked_features = model._apply_mask(features, mask)

        # Unmasked positions should be unchanged
        assert torch.allclose(masked_features[0, 0, 1], features[0, 0, 1])
        assert torch.allclose(masked_features[1, 0, 0], features[1, 0, 0])

        # Masked position should be different (mask token)
        assert not torch.allclose(masked_features[0, 0, 0], features[0, 0, 0])

    def test_get_encoder_state_dict(self, small_model_config: SeasonalViTConfig) -> None:
        """Test get_encoder_state_dict excludes reconstruction head."""
        model = PretrainingModel(small_model_config)

        encoder_state = model.get_encoder_state_dict()

        # Should not contain reconstruction_head or mask_token
        for key in encoder_state.keys():
            assert not key.startswith("reconstruction_head")
            assert key != "mask_token"

        # Should contain encoder components
        has_spatial = any("spatial_encoder" in k for k in encoder_state.keys())
        has_temporal_enc = any("temporal_encoding" in k for k in encoder_state.keys())
        has_temporal_trans = any("temporal_transformer" in k for k in encoder_state.keys())

        assert has_spatial
        assert has_temporal_enc
        assert has_temporal_trans

    def test_factory_function(self) -> None:
        """Test create_pretrain_model factory function."""
        config = SeasonalViTConfig(swin_pretrained=False)
        model = create_pretrain_model(config)

        assert isinstance(model, PretrainingModel)
        assert model.config.embed_dim == config.embed_dim

    def test_factory_function_default_config(self) -> None:
        """Test factory function with default config."""
        model = create_pretrain_model(SeasonalViTConfig(swin_pretrained=False))

        assert isinstance(model, PretrainingModel)


class TestPretrainingModelEdgeCases:
    """Edge case tests for pre-training model."""

    def test_single_batch(self, small_model_config: SeasonalViTConfig) -> None:
        """Test with single batch."""
        model = PretrainingModel(small_model_config)
        num_timesteps = 4
        H, W = small_model_config.image_size

        images = torch.randn(1, num_timesteps, H, W)
        season_ids = torch.tensor([[0, 1, 2, 3]])
        year_indices = torch.tensor([[0, 0, 0, 0]])

        output = model(images, season_ids, year_indices)

        assert output["reconstruction"].shape[0] == 1

    def test_1d_season_ids(self, small_model_config: SeasonalViTConfig) -> None:
        """Test with 1D season/year indices (broadcast)."""
        model = PretrainingModel(small_model_config)
        batch_size = 2
        num_timesteps = 4
        H, W = small_model_config.image_size

        images = torch.randn(batch_size, num_timesteps, H, W)
        # 1D inputs should be broadcast
        season_ids = torch.tensor([0, 1, 2, 3])
        year_indices = torch.tensor([0, 0, 0, 0])

        output = model(images, season_ids, year_indices)

        assert output["reconstruction"].shape[0] == batch_size

    def test_full_mask(self, small_model_config: SeasonalViTConfig) -> None:
        """Test with fully masked input."""
        model = PretrainingModel(small_model_config)
        model.eval()
        batch_size = 2
        num_timesteps = 4
        H, W = small_model_config.image_size

        images = torch.randn(batch_size, num_timesteps, H, W)
        season_ids = torch.tensor([[0, 1, 2, 3]] * batch_size)
        year_indices = torch.tensor([[0, 0, 0, 0]] * batch_size)

        # Full mask
        mask = torch.ones(batch_size, num_timesteps, small_model_config.num_spatial_patches, dtype=torch.bool)

        output = model(images, season_ids, year_indices, mask=mask)

        # Should still produce valid output
        assert not torch.isnan(output["reconstruction"]).any()

    def test_no_mask(self, small_model_config: SeasonalViTConfig) -> None:
        """Test with no mask (unmasked autoencoding)."""
        model = PretrainingModel(small_model_config)
        batch_size = 2
        num_timesteps = 4
        H, W = small_model_config.image_size

        images = torch.randn(batch_size, num_timesteps, H, W)
        season_ids = torch.tensor([[0, 1, 2, 3]] * batch_size)
        year_indices = torch.tensor([[0, 0, 0, 0]] * batch_size)

        output = model(images, season_ids, year_indices, mask=None)

        assert output["mask"] is None
        assert not torch.isnan(output["reconstruction"]).any()

    def test_ndvi_value_range(self, small_model_config: SeasonalViTConfig) -> None:
        """Test with NDVI value range [-1, 1]."""
        model = PretrainingModel(small_model_config)
        batch_size = 2
        num_timesteps = 4
        H, W = small_model_config.image_size

        # NDVI range
        images = torch.rand(batch_size, num_timesteps, H, W) * 2 - 1
        season_ids = torch.tensor([[0, 1, 2, 3]] * batch_size)
        year_indices = torch.tensor([[0, 0, 0, 0]] * batch_size)

        output = model(images, season_ids, year_indices)

        assert not torch.isnan(output["reconstruction"]).any()
        assert not torch.isinf(output["reconstruction"]).any()


class TestSampleYearSubset:
    """Tests for _sample_year_subset helper."""

    def test_output_shapes(self) -> None:
        """Test that subset has correct shapes for K=4 years."""
        B, T, H, W = 2, 140, 64, 64
        K = 4  # sample 4 years -> 16 timesteps
        images = torch.randn(B, T, H, W)
        season_ids = torch.tensor([[t % 4 for t in range(T)]] * B)
        year_indices = torch.tensor([[t // 4 for t in range(T)]] * B)
        nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

        out_img, out_sid, out_yid, out_nd, seq_pos = _sample_year_subset(
            images, season_ids, year_indices, nodata_mask, num_years=K
        )

        assert out_img.shape == (B, K * 4, H, W)
        assert out_sid.shape == (B, K * 4)
        assert out_yid.shape == (B, K * 4)
        assert out_nd.shape == (B, K * 4, H, W)
        assert seq_pos.shape == (B, K * 4)

    def test_none_nodata_mask(self) -> None:
        """Test that None nodata_mask passes through."""
        B, T, H, W = 2, 140, 64, 64
        images = torch.randn(B, T, H, W)
        season_ids = torch.tensor([[t % 4 for t in range(T)]] * B)
        year_indices = torch.tensor([[t // 4 for t in range(T)]] * B)

        _, _, _, out_nd, _ = _sample_year_subset(
            images, season_ids, year_indices, None, num_years=8
        )

        assert out_nd is None

    def test_preserves_season_structure(self) -> None:
        """Each sampled year must have exactly 4 seasons in order (0,1,2,3)."""
        B, T = 2, 140
        K = 10
        images = torch.randn(B, T, 32, 32)  # Small for speed
        season_ids = torch.tensor([[t % 4 for t in range(T)]] * B)
        year_indices = torch.tensor([[t // 4 for t in range(T)]] * B)

        _, out_sid, out_yid, _, _ = _sample_year_subset(
            images, season_ids, year_indices, None, num_years=K
        )

        # Check each group of 4 has seasons 0,1,2,3
        for i in range(K):
            start = i * 4
            assert out_sid[0, start:start + 4].tolist() == [0, 1, 2, 3]

        # Check year indices are constant within each group of 4
        for i in range(K):
            start = i * 4
            year_group = out_yid[0, start:start + 4]
            assert year_group[0] == year_group[1] == year_group[2] == year_group[3]

    def test_seq_positions_are_original_indices(self) -> None:
        """seq_positions must be the original timestep indices, not 0..T_sub-1."""
        B, T = 2, 140
        K = 5
        images = torch.randn(B, T, 32, 32)
        season_ids = torch.tensor([[t % 4 for t in range(T)]] * B)
        year_indices = torch.tensor([[t // 4 for t in range(T)]] * B)

        _, _, out_yid, _, seq_pos = _sample_year_subset(
            images, season_ids, year_indices, None, num_years=K
        )

        # seq_positions should equal year_index * 4 + season_offset
        for i in range(K):
            start = i * 4
            year = out_yid[0, start].item()
            expected_positions = [year * 4 + s for s in range(4)]
            assert seq_pos[0, start:start + 4].tolist() == expected_positions

    def test_data_integrity(self) -> None:
        """Subsetted data must match original at the correct indices."""
        B, T, H, W = 1, 140, 32, 32
        K = 8
        # Use unique values per timestep so we can verify correctness
        images = torch.arange(T).float().view(1, T, 1, 1).expand(B, T, H, W)
        season_ids = torch.tensor([[t % 4 for t in range(T)]] * B)
        year_indices = torch.tensor([[t // 4 for t in range(T)]] * B)

        out_img, _, _, _, seq_pos = _sample_year_subset(
            images, season_ids, year_indices, None, num_years=K
        )

        # Each output timestep's value should match its seq_position
        for t in range(K * 4):
            expected_val = seq_pos[0, t].item()
            assert out_img[0, t, 0, 0].item() == expected_val

    def test_same_subset_across_batch(self) -> None:
        """All samples in batch must use the same year subset."""
        B, T = 4, 140
        K = 6
        images = torch.randn(B, T, 32, 32)
        season_ids = torch.tensor([[t % 4 for t in range(T)]] * B)
        year_indices = torch.tensor([[t // 4 for t in range(T)]] * B)

        _, _, _, _, seq_pos = _sample_year_subset(
            images, season_ids, year_indices, None, num_years=K
        )

        # All batch elements should have same seq_positions
        for b in range(1, B):
            assert torch.equal(seq_pos[0], seq_pos[b])


class TestForwardWithTemporalSubset:
    """Test model forward pass with subsetted temporal input."""

    def test_efficient_forward_with_subset(self, small_model_config: SeasonalViTConfig) -> None:
        """Model forward_efficient works with T=48 (12 years) and seq_positions."""
        model = PretrainingModel(small_model_config)
        model.eval()
        B, K = 2, 12
        T_sub = K * 4
        H, W = small_model_config.image_size

        images = torch.randn(B, T_sub, H, W)
        season_ids = torch.tensor([[t % 4 for t in range(T_sub)]] * B)
        # Use non-contiguous year indices to simulate subsampling
        sampled_years = [0, 3, 5, 8, 12, 15, 18, 21, 24, 27, 30, 33]
        year_indices = torch.tensor(
            [[y for y in sampled_years for _ in range(4)]] * B
        )
        seq_positions = torch.tensor(
            [[y * 4 + s for y in sampled_years for s in range(4)]] * B
        )

        config = PretrainingConfig()
        efficient_mask = generate_efficient_temporal_mask(
            B, T_sub, config=config, device=images.device
        )

        with torch.no_grad():
            output = model.forward_efficient(
                images, season_ids, year_indices, efficient_mask,
                seq_positions=seq_positions,
            )

        assert output["reconstruction"].shape == (
            B, T_sub, small_model_config.num_spatial_patches, 1024
        )
        assert not torch.isnan(output["reconstruction"]).any()

    def test_legacy_forward_with_subset(self, small_model_config: SeasonalViTConfig) -> None:
        """Model forward works with T=16 (4 years) and seq_positions."""
        model = PretrainingModel(small_model_config)
        model.eval()
        B, K = 2, 4
        T_sub = K * 4
        H, W = small_model_config.image_size

        images = torch.randn(B, T_sub, H, W)
        season_ids = torch.tensor([[t % 4 for t in range(T_sub)]] * B)
        sampled_years = [2, 10, 20, 30]
        year_indices = torch.tensor(
            [[y for y in sampled_years for _ in range(4)]] * B
        )
        seq_positions = torch.tensor(
            [[y * 4 + s for y in sampled_years for s in range(4)]] * B
        )

        with torch.no_grad():
            output = model(
                images, season_ids, year_indices,
                seq_positions=seq_positions,
            )

        assert output["reconstruction"].shape == (
            B, T_sub, small_model_config.num_spatial_patches, 1024
        )
        assert not torch.isnan(output["reconstruction"]).any()
