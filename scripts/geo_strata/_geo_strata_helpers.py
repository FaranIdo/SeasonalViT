"""Pure tensor helpers for the OOD-s 2022 geographic-stratification gather.

Kept separate from the gather script so they can be unit-tested without
loading the model, the OOD-s shards, or any GPU. The gather script uses
these to build a per-tile TSV that the downstream map and table scripts
read.
"""
from __future__ import annotations

from typing import Dict

import torch

NODATA_VALUE = -9999.0
SEASON_NAMES = ["spring", "summer", "fall", "winter"]


def per_tile_stats_one_season(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Per-tile MAE, RMSE, bias, and mean(pred)/mean(gt) for one target season.

    Args:
        pred:  (B, H, W) float tensor of predicted NDVI for one season.
        gt:    (B, H, W) float tensor of ground-truth NDVI for the same season.
        valid: (B, H, W) bool tensor; True = pixel is valid (not NoData).

    Returns:
        Dict of per-tile (B,) tensors. Tiles with zero valid pixels get NaN
        in the float metrics and 0 in `n_valid`.
    """
    valid_f = valid.float()
    n_valid = valid_f.sum(dim=(1, 2))
    safe_n = n_valid.clamp(min=1)
    diff = (pred - gt) * valid_f
    mae = diff.abs().sum(dim=(1, 2)) / safe_n
    rmse = ((diff * diff).sum(dim=(1, 2)) / safe_n).sqrt()
    bias = diff.sum(dim=(1, 2)) / safe_n
    pred_mean = (pred * valid_f).sum(dim=(1, 2)) / safe_n
    gt_mean = (gt * valid_f).sum(dim=(1, 2)) / safe_n
    has_data = n_valid > 0
    nan = torch.full_like(mae, float("nan"))
    return {
        "mae": torch.where(has_data, mae, nan),
        "rmse": torch.where(has_data, rmse, nan),
        "bias": torch.where(has_data, bias, nan),
        "pred_mean": torch.where(has_data, pred_mean, nan),
        "gt_mean": torch.where(has_data, gt_mean, nan),
        "n_valid": n_valid.long(),
    }


def sigma_tile_temporal(
    images: torch.Tensor,
    nodata_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-tile temporal NDVI std over the input window.

    Aggregates across all valid (T, H, W) entries per tile to get a single
    scalar describing how variable NDVI is at this place across history.
    NoData is encoded both as -9999 in `images` and as True in `nodata_mask`;
    the function masks via `nodata_mask` AND additionally rejects any pixel
    whose value falls outside the physically valid NDVI range [-1, 1]. This
    second check is load-bearing: in practice the per-shard nodata_mask
    occasionally misses individual -9999 sentinels, and a single missed
    pixel inflates sigma by ~3.5 because (-9999 - mean)^2 dominates the sum.
    Computation runs in fp32 to avoid (-9999)^2 overflowing fp16.

    Args:
        images:      (B, T, H, W) float tensor of seasonal NDVI.
        nodata_mask: (B, T, H, W) bool tensor; True = invalid pixel.

    Returns:
        (B,) tensor of standard deviations. NaN for tiles with <2 valid entries.
    """
    images = images.float()
    out_of_range = (images < -1.0) | (images > 1.0)
    invalid = nodata_mask | out_of_range
    valid = (~invalid).float()
    n_valid = valid.sum(dim=(1, 2, 3))
    images_clean = torch.where(invalid, torch.zeros_like(images), images)
    mean = images_clean.sum(dim=(1, 2, 3)) / n_valid.clamp(min=1)
    sq_dev = ((images_clean - mean.view(-1, 1, 1, 1)) ** 2) * valid
    var = sq_dev.sum(dim=(1, 2, 3)) / (n_valid - 1).clamp(min=1)
    sigma = var.sqrt()
    nan = torch.full_like(sigma, float("nan"))
    return torch.where(n_valid >= 2, sigma, nan)


def format_tsv_value(v: float | int | str) -> str:
    """Format a value for TSV output. Floats use 6 decimal places; NaN is 'nan'."""
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.6f}"
    return str(v)
