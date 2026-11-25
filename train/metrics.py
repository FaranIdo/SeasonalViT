"""Evaluation metrics for SeasonalViT.

Provides metrics for evaluating NDVI prediction quality:

- **MAE**: Mean Absolute Error - mean(|pred - target|)
  Average prediction error in NDVI units. Range: [0, ∞)

- **RMSE**: Root Mean Squared Error - sqrt(mean((pred - target)²))
  Penalizes large errors more than MAE. Range: [0, ∞)

- **R²**: Squared Pearson correlation coefficient - corr(pred, target)²
  Measures linear relationship strength, ignores bias/scale. Range: [0, 1]

- **NSE**: Nash-Sutcliffe Efficiency - 1 - MSE/Var(target)
  Skill relative to predicting the mean. Penalizes bias and scale errors.
  NSE=1: perfect, NSE=0: same as mean, NSE<0: worse than mean. Range: (-∞, 1]

- **bias**: Absolute bias - |mean(pred) - mean(target)|
  Systematic over/under-prediction. Range: [0, ∞)

Note: R² and NSE are different metrics despite similar names in some literature.
R² only measures correlation; NSE penalizes systematic errors.
"""

from typing import Dict, List, Optional

import torch


def compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute evaluation metrics for NDVI predictions.

    Args:
        prediction: Predicted NDVI of shape (..., H, W) or (B, 1, H, W).
        target: Target NDVI of same shape as prediction.
        mask: Optional boolean mask where True = valid (include in metrics).

    Returns:
        Dictionary with metric values.
    """
    # Flatten tensors for metric computation
    pred_flat = prediction.flatten()
    target_flat = target.flatten()

    if mask is not None:
        mask_flat = mask.flatten()
        pred_flat = pred_flat[mask_flat]
        target_flat = target_flat[mask_flat]

    if pred_flat.numel() == 0:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "nse": 0.0,
            "n_valid": 0,
        }

    mae = compute_mae(pred_flat, target_flat)
    mse = compute_mse(pred_flat, target_flat)
    rmse = compute_rmse(pred_flat, target_flat)
    r2 = compute_r2(pred_flat, target_flat)
    nse = compute_nse(pred_flat, target_flat)

    return {
        "mae": mae.item(),
        "mse": mse.item(),
        "rmse": rmse.item(),
        "r2": r2.item(),
        "nse": nse.item(),
        "n_valid": pred_flat.numel(),
    }


def compute_mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Mean Absolute Error.

    Args:
        prediction: Predicted values.
        target: Target values.

    Returns:
        MAE value.
    """
    return (prediction - target).abs().mean()


def compute_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Mean Squared Error.

    Args:
        prediction: Predicted values.
        target: Target values.

    Returns:
        MSE value.
    """
    return ((prediction - target) ** 2).mean()


def compute_rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Root Mean Squared Error.

    Args:
        prediction: Predicted values.
        target: Target values.

    Returns:
        RMSE value.
    """
    return compute_mse(prediction, target).sqrt()


def compute_r2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute R² as squared Pearson correlation coefficient.

    R² = corr(prediction, target)²

    This measures linear relationship strength, ignoring bias and scale.
    Always in range [0, 1].

    Args:
        prediction: Predicted values.
        target: Target values.

    Returns:
        R² value in [0, 1].
    """
    pred_mean = prediction.mean()
    target_mean = target.mean()

    pred_centered = prediction - pred_mean
    target_centered = target - target_mean

    cov = (pred_centered * target_centered).sum()
    pred_std = pred_centered.pow(2).sum().sqrt()
    target_std = target_centered.pow(2).sum().sqrt()

    if pred_std == 0 or target_std == 0:
        return torch.tensor(0.0)

    corr = cov / (pred_std * target_std)
    return corr ** 2


def compute_nse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Nash-Sutcliffe Efficiency (NSE).

    NSE = 1 - SS_res / SS_tot
    where SS_res = sum((obs - pred)²) and SS_tot = sum((obs - obs_mean)²)

    Standard metric in environmental/hydrological modeling.
    - NSE = 1: Perfect prediction
    - NSE = 0: Model performs as well as predicting the mean
    - NSE < 0: Model performs worse than predicting the mean

    Note: Different from R² (Pearson correlation squared). NSE penalizes
    bias and scale errors, while R² only measures correlation strength.

    Args:
        prediction: Predicted values.
        target: Target (observed) values.

    Returns:
        NSE value.
    """
    ss_res = ((target - prediction) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()

    if ss_tot == 0:
        return torch.tensor(1.0) if ss_res == 0 else torch.tensor(float("-inf"))

    return 1 - ss_res / ss_tot


def compute_bias(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute absolute bias: |mean(prediction) - mean(target)|.

    Measures systematic over/under-prediction.

    Args:
        prediction: Predicted values.
        target: Target values.

    Returns:
        Absolute bias value (>= 0).
    """
    return (prediction.mean() - target.mean()).abs()


def compute_per_season_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    season_ids: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics separately for each season.

    Args:
        prediction: Predicted NDVI of shape (B, T, H, W).
        target: Target NDVI of same shape.
        season_ids: Season indices of shape (B, T) or (T,), values 0-3.
        mask: Optional boolean mask of shape (B, T, H, W).

    Returns:
        Dictionary with metrics for each season (0=spring, 1=summer, 2=fall, 3=winter).
    """
    B, T, H, W = prediction.shape

    # Handle 1D season_ids
    if season_ids.dim() == 1:
        season_ids = season_ids.unsqueeze(0).expand(B, -1)

    season_names = ["spring", "summer", "fall", "winter"]
    results = {}

    for season_idx, season_name in enumerate(season_names):
        # Find timesteps for this season
        season_mask = season_ids == season_idx

        # Gather predictions and targets for this season
        pred_season = []
        target_season = []

        for b in range(B):
            for t in range(T):
                if season_mask[b, t]:
                    if mask is not None:
                        valid = mask[b, t]
                        pred_season.append(prediction[b, t][valid])
                        target_season.append(target[b, t][valid])
                    else:
                        pred_season.append(prediction[b, t].flatten())
                        target_season.append(target[b, t].flatten())

        if pred_season:
            pred_concat = torch.cat(pred_season)
            target_concat = torch.cat(target_season)
            results[season_name] = compute_metrics(pred_concat, target_concat)
        else:
            results[season_name] = {
                "mae": 0.0, "mse": 0.0, "rmse": 0.0, "r2": 0.0, "nse": 0.0,
                "n_valid": 0,
            }

    return results


class MetricsTracker:
    """Track metrics over multiple batches.

    Accumulates predictions and targets to compute metrics at the end.

    Attributes:
        predictions: List of prediction tensors.
        targets: List of target tensors.
        masks: List of mask tensors (optional).
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self.predictions = []
        self.targets = []
        self.masks = []

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Add batch to tracker.

        Args:
            prediction: Predicted values.
            target: Target values.
            mask: Optional validity mask.
        """
        self.predictions.append(prediction.detach().cpu())
        self.targets.append(target.detach().cpu())
        if mask is not None:
            self.masks.append(mask.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute metrics over all accumulated batches.

        Returns:
            Dictionary with metric values.
        """
        if not self.predictions:
            return {
                "mae": 0.0, "mse": 0.0, "rmse": 0.0, "r2": 0.0, "nse": 0.0,
                "n_valid": 0,
            }

        pred_all = torch.cat(self.predictions)
        target_all = torch.cat(self.targets)

        mask_all = None
        if self.masks:
            mask_all = torch.cat(self.masks)

        return compute_metrics(pred_all, target_all, mask_all)

    def reset(self) -> None:
        """Clear accumulated data."""
        self.predictions = []
        self.targets = []
        self.masks = []


class PerSampleMetricsTracker:
    """Track metrics with per-sample averaging.

    Computes MAE, R², bias per sample, then averages across samples.
    This matches the baseline evaluation approach and gives meaningful
    R² values (not inflated by between-sample variance).
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self.sample_metrics: List[Dict[str, Optional[float]]] = []
        self.n_samples = 0
        self.n_valid_pixels = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Add batch of samples, computing metrics per sample on-device.

        Computes MAE, R², bias on GPU and extracts only scalar .item() values.
        This avoids thousands of .detach().cpu() tensor allocations per
        validation epoch that fragment the Windows heap.

        Args:
            prediction: Shape (B, 1, H, W) or (B, H, W).
            target: Same shape as prediction.
            mask: Optional boolean mask, True = valid.
        """
        pred = prediction.detach()
        tgt = target.detach()
        if mask is not None:
            mask = mask.detach()

        # Ensure 3D: (B, 1, H, W) -> (B, H, W)
        if pred.dim() == 4:
            pred = pred.squeeze(1)
            tgt = tgt.squeeze(1)
            if mask is not None:
                mask = mask.squeeze(1)

        B = pred.shape[0]

        for b in range(B):
            self._compute_sample_metrics(pred[b], tgt[b], mask[b] if mask is not None else None)

    def _compute_sample_metrics(
        self,
        pred_2d: torch.Tensor,
        target_2d: torch.Tensor,
        mask_2d: Optional[torch.Tensor],
    ) -> None:
        """Compute and store metrics for a single sample on-device.

        Args:
            pred_2d: Shape (H, W).
            target_2d: Shape (H, W).
            mask_2d: Optional boolean mask (H, W), True = valid.
        """
        # Upcast to float32 to avoid overflow in sum-of-squares with float16
        pred_b = pred_2d.flatten().float()
        target_b = target_2d.flatten().float()

        if mask_2d is not None:
            valid = mask_2d.flatten()
            pred_b = pred_b[valid]
            target_b = target_b[valid]

        n = pred_b.numel()
        if n == 0:
            return

        self.n_valid_pixels += n
        self.n_samples += 1

        error = pred_b - target_b
        mae = error.abs().mean().item()
        rmse = error.pow(2).mean().sqrt().item()
        bias = (pred_b.mean() - target_b.mean()).abs().item()

        r2 = self._compute_sample_r2(pred_b, target_b)
        R2_det = self._compute_sample_R2_determination(pred_b, target_b)

        self.sample_metrics.append({
            "mae": mae, "rmse": rmse, "r2": r2, "R2_det": R2_det, "bias": bias,
        })

    @staticmethod
    def _compute_sample_r2(
        pred_b: torch.Tensor, target_b: torch.Tensor,
    ) -> Optional[float]:
        """Compute R² (squared Pearson correlation) for a single sample.

        Returns None when either prediction or target has zero variance.
        """
        pred_c = pred_b - pred_b.mean()
        target_c = target_b - target_b.mean()
        cov = (pred_c * target_c).sum()
        pred_std = pred_c.pow(2).sum().sqrt()
        target_std = target_c.pow(2).sum().sqrt()

        if pred_std > 0 and target_std > 0:
            corr = cov / (pred_std * target_std)
            return (corr**2).item()
        return None

    @staticmethod
    def _compute_sample_R2_determination(
        pred_b: torch.Tensor, target_b: torch.Tensor,
    ) -> Optional[float]:
        """Compute R² coefficient of determination for a single sample.

        R² = 1 - SS_res / SS_tot, where SS_res = sum((y - y_hat)²)
        and SS_tot = sum((y - y_mean)²). Can be negative if the model
        is worse than predicting the mean.

        Returns None when target has zero variance.
        """
        ss_res = (target_b - pred_b).pow(2).sum()
        ss_tot = (target_b - target_b.mean()).pow(2).sum()

        if ss_tot > 0:
            return (1.0 - ss_res / ss_tot).item()
        return None

    def compute(self) -> Dict[str, float]:
        """Compute averaged metrics across all samples.

        Returns:
            Dictionary with mae, rmse, r2, R2_det, bias, n_samples, n_valid_pixels.
        """
        if not self.sample_metrics:
            return {
                "mae": 0.0,
                "rmse": 0.0,
                "r2": 0.0,
                "R2_det": 0.0,
                "bias": 0.0,
                "n_samples": 0,
                "n_valid_pixels": 0,
            }

        def avg(key: str) -> float:
            vals = [m[key] for m in self.sample_metrics if m[key] is not None]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "mae": avg("mae"),
            "rmse": avg("rmse"),
            "r2": avg("r2"),
            "R2_det": avg("R2_det"),
            "bias": avg("bias"),
            "n_samples": self.n_samples,
            "n_valid_pixels": self.n_valid_pixels,
        }

    def reset(self) -> None:
        """Clear accumulated data."""
        self.sample_metrics = []
        self.n_samples = 0
        self.n_valid_pixels = 0
