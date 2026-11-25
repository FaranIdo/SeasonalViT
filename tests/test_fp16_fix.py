"""Verify that linear_trend works correctly with float32 input (no float16 cast)."""
import torch
from baselines.statistical import LinearTrendBaseline


def test_linear_trend_no_overflow_with_33_years():
    """OLS with 33 same-season observations should not overflow."""
    baseline = LinearTrendBaseline()

    B, T, H, W = 4, 140, 8, 8
    images = torch.randn(B, T, H, W)
    nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

    # Predict spring 2023 (target_idx=132), context up to index 131
    pred, valid = baseline.predict(images, 132, nodata_mask, context_end_idx=131)

    assert valid.all(), f"Expected all valid, got {valid.float().mean():.2%}"
    assert not torch.isnan(pred).any(), "Predictions contain NaN"
    assert not torch.isinf(pred).any(), "Predictions contain inf"
