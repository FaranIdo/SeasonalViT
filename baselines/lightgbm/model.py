"""LightGBM per-pixel baseline for NDVI prediction.

Features per pixel:
- Last 5 same-season values (zero-filled if unavailable)
- Mean and std of all same-season values
- Linear trend slope
- Season ID (0-3)

A single global model is trained across all pixels from training shards.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Feature names for interpretability
FEATURE_NAMES = [
    "recent_1", "recent_2", "recent_3", "recent_4", "recent_5",
    "same_season_mean", "same_season_std", "trend_slope", "season_id",
]
NUM_FEATURES = len(FEATURE_NAMES)


def extract_pixel_features(
    images: torch.Tensor,
    nodata_mask: torch.Tensor,
    target_idx: int,
    context_end_idx: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-pixel features from a batch of images.

    Args:
        images: NDVI images of shape (B, T, H, W).
        nodata_mask: Boolean mask where True = nodata, shape (B, T, H, W).
        target_idx: Index of timestep to predict.
        context_end_idx: Last usable context index. None = target_idx - 1.
        device: Optional GPU device for accelerated extraction.

    Returns:
        Tuple of (features, targets, valid_mask):
            features: shape (B*H*W, NUM_FEATURES) float32 array
            targets: shape (B*H*W,) float32 array
            valid_mask: shape (B*H*W,) bool array (True = valid pixel)
    """
    B, T, H, W = images.shape
    season = target_idx % 4
    end_idx = target_idx if context_end_idx is None else context_end_idx + 1

    same_season_indices = list(range(season, end_idx, 4))

    if device is not None and device.type == "cuda":
        features = _compute_features_gpu(
            images, nodata_mask, same_season_indices, season, B, H, W, device,
        )
    else:
        features = _compute_features_cpu(
            images, nodata_mask, same_season_indices, season, B, H, W,
        )

    target_vals = images[:, target_idx].reshape(B * H * W).float().numpy()
    target_valid = ~nodata_mask[:, target_idx].reshape(B * H * W).numpy()

    feature_valid = features[:, 5] != 0
    valid_mask = target_valid & feature_valid

    return features, target_vals, valid_mask


def _compute_features_cpu(
    images: torch.Tensor,
    nodata_mask: torch.Tensor,
    same_season_indices: List[int],
    season: int,
    B: int,
    H: int,
    W: int,
) -> np.ndarray:
    """CPU single-pass feature extraction with O(n_pixels) memory."""
    n_pixels = B * H * W
    features = np.zeros((n_pixels, NUM_FEATURES), dtype=np.float32)

    if not same_season_indices:
        features[:, 8] = season
        return features

    n_obs = len(same_season_indices)
    x_vals = np.array(
        [idx // 4 for idx in same_season_indices], dtype=np.float32
    )

    # Precompute valid masks to avoid repeated boolean negation
    valid_bands = ~nodata_mask[:, same_season_indices]

    count = np.zeros(n_pixels, dtype=np.float32)
    sum_y = np.zeros(n_pixels, dtype=np.float32)
    sum_y2 = np.zeros(n_pixels, dtype=np.float32)
    sum_x = np.zeros(n_pixels, dtype=np.float32)
    sum_x2 = np.zeros(n_pixels, dtype=np.float32)
    sum_xy = np.zeros(n_pixels, dtype=np.float32)

    for i in range(n_obs):
        obs = images[:, same_season_indices[i]].float().reshape(n_pixels).numpy()
        valid = valid_bands[:, i].reshape(n_pixels).numpy()
        mask = valid.astype(np.float32)
        y = obs * mask
        xi = x_vals[i]

        count += mask
        sum_y += y
        sum_y2 += y * y
        sum_x += xi * mask
        sum_x2 += (xi * xi) * mask
        sum_xy += xi * y

        recency = n_obs - 1 - i
        if recency < 5:
            features[:, recency] = y

    _fill_stats_and_slope(features, count, sum_y, sum_y2,
                          sum_x, sum_x2, sum_xy, season)
    return features


def _compute_features_gpu(
    images: torch.Tensor,
    nodata_mask: torch.Tensor,
    same_season_indices: List[int],
    season: int,
    B: int,
    H: int,
    W: int,
    device: torch.device,
) -> np.ndarray:
    """GPU-accelerated vectorized feature extraction."""
    n_pixels = B * H * W

    if not same_season_indices:
        features = np.zeros((n_pixels, NUM_FEATURES), dtype=np.float32)
        features[:, 8] = season
        return features

    n_obs = len(same_season_indices)

    # Move only same-season bands to GPU
    obs = images[:, same_season_indices].to(device, dtype=torch.float32)
    valid = (~nodata_mask[:, same_season_indices]).to(device)

    # Reshape to (n_pixels, n_obs)
    obs = obs.permute(0, 2, 3, 1).reshape(n_pixels, -1)
    valid = valid.permute(0, 2, 3, 1).reshape(n_pixels, -1)

    mask = valid.float()
    y = obs * mask

    # Stats
    count = mask.sum(dim=1)
    sum_y = y.sum(dim=1)
    sum_y2 = (y * y).sum(dim=1)

    # OLS
    x_vals = torch.tensor(
        [idx // 4 for idx in same_season_indices],
        dtype=torch.float32, device=device,
    )
    x_masked = x_vals.unsqueeze(0) * mask
    sum_x = x_masked.sum(dim=1)
    sum_x2 = (x_masked * x_masked).sum(dim=1)
    sum_xy = (x_masked * y).sum(dim=1)

    # Recent 5
    features_gpu = torch.zeros(n_pixels, NUM_FEATURES, device=device)
    for i in range(min(5, n_obs)):
        src = n_obs - 1 - i
        features_gpu[:, i] = torch.where(valid[:, src], obs[:, src], torch.zeros(1, device=device))

    # Mean and std
    safe_count = torch.clamp(count, min=1.0)
    mean = sum_y / safe_count
    variance = torch.clamp(sum_y2 / safe_count - mean * mean, min=0.0)
    features_gpu[:, 5] = mean
    features_gpu[:, 6] = torch.sqrt(variance)

    # Slope
    denom = count * sum_x2 - sum_x * sum_x
    slope_valid = (count >= 2) & (denom.abs() > 1e-10)
    features_gpu[slope_valid, 7] = (
        (count[slope_valid] * sum_xy[slope_valid]
         - sum_x[slope_valid] * sum_y[slope_valid])
        / denom[slope_valid]
    )

    # Zero stats where no valid obs
    no_valid = count == 0
    features_gpu[no_valid, 5] = 0.0
    features_gpu[no_valid, 6] = 0.0

    features_gpu[:, 8] = season

    return features_gpu.cpu().numpy()


def _fill_stats_and_slope(
    features: np.ndarray,
    count: np.ndarray,
    sum_y: np.ndarray,
    sum_y2: np.ndarray,
    sum_x: np.ndarray,
    sum_x2: np.ndarray,
    sum_xy: np.ndarray,
    season: int,
) -> None:
    """Fill mean, std, slope, and season into features array."""
    safe_count = np.maximum(count, 1.0)
    mean = sum_y / safe_count
    variance = np.maximum(sum_y2 / safe_count - mean * mean, 0.0)
    features[:, 5] = mean
    features[:, 6] = np.sqrt(variance)

    denom = count * sum_x2 - sum_x * sum_x
    slope_valid = (count >= 2) & (np.abs(denom) > 1e-10)
    features[slope_valid, 7] = (
        (count[slope_valid] * sum_xy[slope_valid]
         - sum_x[slope_valid] * sum_y[slope_valid])
        / denom[slope_valid]
    )

    no_valid = count == 0
    features[no_valid, 5] = 0.0
    features[no_valid, 6] = 0.0
    features[:, 8] = season


class LightGBMPredictor:
    """Wrapper for trained LightGBM model for evaluation pipeline compatibility."""

    def __init__(self, model_path: Path, device: Optional[torch.device] = None):
        """Load trained LightGBM model.

        Args:
            model_path: Path to saved model file (.txt or .pkl).
            device: Optional GPU device for accelerated feature extraction.
        """
        import lightgbm as lgb

        if str(model_path).endswith(".txt"):
            self.model = lgb.Booster(model_file=str(model_path))
        else:
            import pickle
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

        self.device = device
        logger.info(f"Loaded LightGBM model from {model_path} (device={device})")

    def predict_batch(
        self,
        images: torch.Tensor,
        nodata_mask: torch.Tensor,
        target_idx: int,
        context_end_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate predictions for a batch.

        Args:
            images: NDVI images (B, T, H, W).
            nodata_mask: Nodata mask (B, T, H, W).
            target_idx: Target timestep index.
            context_end_idx: Last context index.

        Returns:
            Tuple of (predictions, valid_mask), each (B, H, W).
        """
        B, T, H, W = images.shape
        features, _, valid_mask = extract_pixel_features(
            images, nodata_mask, target_idx, context_end_idx, self.device,
        )

        preds_flat = self.model.predict(features).astype(np.float32)
        predictions = torch.from_numpy(preds_flat).reshape(B, H, W)
        valid = torch.from_numpy(valid_mask).reshape(B, H, W)

        return predictions, valid
