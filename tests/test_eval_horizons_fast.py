"""Equivalence tests for the fast horizon eval path.

The fast script (scripts/eval_horizons_fast.py) computes the encoder
once per batch and only the decoder per horizon, instead of running the
full SeasonalViT.forward() for every (batch, horizon) pair. The
load-bearing claim is that calling encoder + decoder separately is
mathematically equivalent to calling forward() atomically. If it isn't,
the per-horizon TSV is wrong.

These tests pin that equivalence with a small synthetic batch and
random weights. They run on CPU in seconds and don't need a real
checkpoint.
"""

import torch

from model.config import SeasonalViTConfig
from model.seasonal_vit import SeasonalViT
from scripts.eval_horizons_fast import (
    encode_batch_once,
    decode_for_horizon,
    horizon_to_index,
)


def _build_test_model() -> SeasonalViT:
    """Small SeasonalViT for fast equivalence checking on CPU."""
    config = SeasonalViTConfig(
        image_size=(64, 64),
        spatial_grid_size=2,
        num_spatial_patches=4,
        temporal_depth=2,
        temporal_heads=4,
        swin_pretrained=False,
    )
    return SeasonalViT(config).eval()


def _build_test_batch(B: int = 2, T_in: int = 132, H: int = 64, W: int = 64):
    """Synthetic batch matching the fast script's expectations.

    Input window is T_in=132 (1990 spring..2022 winter), targets live at
    idx >= T_in. Returns the input slice plus the season/year ids
    needed for the encoder.
    """
    torch.manual_seed(0)
    full_T = T_in + 8  # room for h=1..8 targets at idx 132..139
    images = torch.randn(B, full_T, H, W).clamp(-1, 1)
    season_ids = torch.tensor([t % 4 for t in range(full_T)], dtype=torch.long)
    year_indices = torch.tensor([t // 4 for t in range(full_T)], dtype=torch.long)
    lat = torch.tensor([45.0, -10.0])
    lon = torch.tensor([10.0, 120.0])
    return images, season_ids, year_indices, lat, lon


def _full_forward(model, images, season_ids, year_indices, lat, lon, target_idx):
    """Reference path: model.forward() over the input window only."""
    B = images.shape[0]
    input_images = images[:, : target_idx]  # last_input_idx = target_idx - 1
    input_season = season_ids[:target_idx].unsqueeze(0).expand(B, -1)
    input_year = year_indices[:target_idx].unsqueeze(0).expand(B, -1)
    target_season = torch.full((B,), target_idx % 4, dtype=torch.long)
    target_year = torch.full((B,), target_idx // 4, dtype=torch.long)
    return model(
        input_images, input_season, input_year, lat, lon,
        target_season, target_year,
    )


def _split_forward(model, images, season_ids, year_indices, lat, lon, target_idx, last_input_idx):
    """Fast path: encode_batch_once + decode_for_horizon, with FIXED window."""
    input_len = last_input_idx + 1
    B = images.shape[0]
    input_images = images[:, :input_len]
    input_season = season_ids[:input_len].unsqueeze(0).expand(B, -1)
    input_year = year_indices[:input_len].unsqueeze(0).expand(B, -1)
    temporal_features = encode_batch_once(
        model, input_images, input_season, input_year, lat, lon,
    )
    h = target_idx - last_input_idx
    return decode_for_horizon(
        model, temporal_features, input_images, input_season,
        horizon_to_index(h, last_input_idx),
    )


class TestEncoderDecoderSplitEquivalence:
    """Verify the fast path matches model.forward() bit-for-bit (within fp tolerance)."""

    def test_h1_matches_full_forward(self):
        """h=1 (target_idx=132) is the cleanest case: input window matches in both paths."""
        model = _build_test_model()
        images, season_ids, year_indices, lat, lon = _build_test_batch()
        last_input_idx = 131  # 2022 winter
        target_idx = 132  # 2023 spring

        ref = _full_forward(
            model, images, season_ids, year_indices, lat, lon, target_idx,
        )
        fast = _split_forward(
            model, images, season_ids, year_indices, lat, lon,
            target_idx, last_input_idx,
        )

        assert torch.allclose(ref["prediction"], fast["prediction"], atol=1e-5), (
            f"prediction mismatch: max diff = "
            f"{(ref['prediction'] - fast['prediction']).abs().max()}"
        )
        assert torch.allclose(ref["anchor"], fast["anchor"], atol=1e-5)
        assert torch.allclose(ref["delta"], fast["delta"], atol=1e-5)

class TestFixedInputWindow:
    """Verify the fast path always uses [0:132], independent of target_idx."""

    def test_encoder_features_identical_across_horizons(self):
        """encode_batch_once depends only on input_images, not on target.

        So calling it twice with the same input_images must produce the
        same temporal_features regardless of which horizon we're about
        to decode for.
        """
        model = _build_test_model()
        images, season_ids, year_indices, lat, lon = _build_test_batch()
        input_images = images[:, :132]
        input_season = season_ids[:132].unsqueeze(0).expand(2, -1)
        input_year = year_indices[:132].unsqueeze(0).expand(2, -1)

        feat1 = encode_batch_once(
            model, input_images, input_season, input_year, lat, lon,
        )
        feat2 = encode_batch_once(
            model, input_images, input_season, input_year, lat, lon,
        )
        assert torch.allclose(feat1, feat2, atol=1e-6)

    def test_anchor_at_h8_uses_input_window_only(self):
        """Same-season anchor at h=8 (winter 2024) should pick winter 2022
        from the input window, NOT winter 2023 (which is at idx 135 and
        would require seeing past the input window).
        """
        model = _build_test_model()
        images, season_ids, year_indices, lat, lon = _build_test_batch()
        # Mark winter 2022 (idx 131) and winter 2023 (idx 135) with
        # distinct sentinel values so we can tell which one the anchor
        # selects.
        images[:, 131] = 0.5  # winter 2022 sentinel
        images[:, 135] = -0.5  # winter 2023 sentinel (must NOT leak)

        # Run the fast split path for h=8 (target winter 2024 = idx 139)
        fast = _split_forward(
            model, images, season_ids, year_indices, lat, lon,
            target_idx=139, last_input_idx=131,
        )

        # Anchor must be winter 2022 (mean ~0.5), not winter 2023 (-0.5)
        anchor_mean = fast["anchor"].mean().item()
        assert anchor_mean > 0.0, (
            f"Anchor mean is {anchor_mean}, expected ~0.5 from winter 2022. "
            "If negative, winter 2023 leaked into the input window."
        )
