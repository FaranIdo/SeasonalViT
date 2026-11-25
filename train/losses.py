"""Loss functions for SeasonalViT training.

Provides loss functions for both stages:
- Pre-training: Masked reconstruction loss (MSE on masked positions)
- Fine-tuning: Forecasting loss (Smooth L1 + correlation + delta regularization)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def pretrain_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    nodata_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    normalize_target: bool = False,
) -> torch.Tensor:
    """Compute masked reconstruction loss for pre-training.

    Only computes loss on masked positions, following MAE-style training.
    This encourages the model to learn meaningful representations.

    When nodata_mask is provided, positions that were originally nodata
    (and filled via interpolation) are excluded from loss computation
    to avoid training against synthetic targets.

    Args:
        prediction: Predicted patches of shape (B, T, N, P) where P is pixels per patch.
        target: Target patches of shape (B, T, N, P).
        mask: Boolean mask of shape (B, T, N) where True indicates masked positions.
        nodata_mask: Optional boolean mask. Can be:
            - Shape (B, T, N): patch-level, True = patch has nodata
            - Shape (B, T, H, W): pixel-level, will be converted to patch-level
        reduction: Loss reduction mode ('mean', 'sum', 'none').

    Returns:
        Scalar loss value (or unreduced tensor if reduction='none').
    """
    # Convert pixel-level nodata_mask to patch-level if needed
    patch_nodata_mask = None
    if nodata_mask is not None:
        if nodata_mask.dim() == 4:
            # Pixel-level (B, T, H, W) -> patch-level (B, T, N)
            patch_nodata_mask = _patchify_nodata_mask(nodata_mask)
        else:
            # Already patch-level (B, T, N)
            patch_nodata_mask = nodata_mask

    # Combine masks: compute loss where (masked for reconstruction) AND (not nodata)
    compute_loss_mask = mask
    if patch_nodata_mask is not None:
        compute_loss_mask = mask & ~patch_nodata_mask

    # Expand mask to match prediction shape
    mask_expanded = compute_loss_mask.unsqueeze(-1).expand_as(prediction)

    # Normalize targets per patch (MAE-style)
    if normalize_target:
        target = _normalize_patches(target)

    # Extract positions for loss computation
    pred_masked = prediction[mask_expanded]
    target_masked = target[mask_expanded]

    if pred_masked.numel() == 0:
        return torch.tensor(0.0, device=prediction.device, requires_grad=True)

    return F.mse_loss(pred_masked, target_masked, reduction=reduction)


def pretrain_loss_efficient(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temporal_mask: torch.Tensor,
    nodata_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    normalize_target: bool = False,
) -> torch.Tensor:
    """Compute reconstruction loss for efficient temporal masking.

    In efficient mode, entire timesteps are masked (not individual patches).
    Loss is computed on ALL patches of masked timesteps, excluding only
    patches with nodata (not entire timesteps).

    Args:
        prediction: Predicted patches of shape (B, T, N, P).
        target: Target patches of shape (B, T, N, P).
        temporal_mask: Boolean mask of shape (B, T) where True = masked timestep.
        nodata_mask: Optional boolean mask. Can be:
            - Shape (B, T): timestep-level, True = timestep has nodata
            - Shape (B, T, N): patch-level, True = patch has nodata
            - Shape (B, T, H, W): pixel-level, will be converted to patch-level
        reduction: Loss reduction mode ('mean', 'sum', 'none').

    Returns:
        Scalar loss value.
    """
    B, T, N, P = prediction.shape

    # Convert nodata_mask to patch-level (B, T, N) for fine-grained exclusion
    # This ensures we only exclude bad patches, not entire timesteps
    patch_nodata_mask = None
    if nodata_mask is not None:
        if nodata_mask.dim() == 4:
            # Pixel-level (B, T, H, W) -> patch-level (B, T, N)
            patch_nodata_mask = _patchify_nodata_mask(nodata_mask)
        elif nodata_mask.dim() == 3:
            # Already patch-level (B, T, N)
            patch_nodata_mask = nodata_mask
        else:
            # Timestep-level (B, T) -> expand to patch-level (B, T, N)
            patch_nodata_mask = nodata_mask.unsqueeze(-1).expand(B, T, N)

    # Expand temporal_mask from (B, T) to (B, T, N) to combine with patch-level nodata
    temporal_mask_expanded = temporal_mask.unsqueeze(-1).expand(B, T, N)

    # Compute loss where (masked for reconstruction) AND (not nodata)
    compute_loss_mask = temporal_mask_expanded  # (B, T, N)
    if patch_nodata_mask is not None:
        compute_loss_mask = temporal_mask_expanded & ~patch_nodata_mask

    # Normalize targets per patch (MAE-style)
    if normalize_target:
        target = _normalize_patches(target)

    # Expand mask to match prediction shape: (B, T, N) -> (B, T, N, P)
    mask_expanded = compute_loss_mask.unsqueeze(-1).expand_as(prediction)

    # Extract positions for loss computation
    pred_masked = prediction[mask_expanded]
    target_masked = target[mask_expanded]

    if pred_masked.numel() == 0:
        return torch.tensor(0.0, device=prediction.device, requires_grad=True)

    return F.mse_loss(pred_masked, target_masked, reduction=reduction)


def _patchify_nodata_mask(nodata_mask: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
    """Convert pixel-level nodata mask to patch-level mask.

    A patch is considered nodata if ANY pixel in the patch is nodata.

    Args:
        nodata_mask: Boolean mask of shape (B, T, H, W) where True = nodata.
        patch_size: Size of each patch (default 32 for Swin-T).

    Returns:
        Patch-level mask of shape (B, T, N) where True = patch has nodata.
    """
    B, T, H, W = nodata_mask.shape
    grid_size = H // patch_size

    # Reshape to patches: (B, T, grid, patch, grid, patch)
    patches = nodata_mask.view(B, T, grid_size, patch_size, grid_size, patch_size)
    # Permute to (B, T, grid, grid, patch, patch)
    patches = patches.permute(0, 1, 2, 4, 3, 5)
    # Reshape to (B, T, N, patch_pixels)
    patches = patches.reshape(B, T, grid_size * grid_size, patch_size * patch_size)

    # A patch is nodata if ANY pixel is nodata
    return patches.any(dim=-1)


def _normalize_patches(patches: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each patch to zero mean and unit variance (MAE-style).

    Following He et al. (2022), normalizing targets per patch shifts the
    reconstruction task from predicting absolute values to predicting
    relative spatial patterns within each patch.

    Args:
        patches: Tensor of shape (..., P) where P is pixels per patch.
        eps: Small constant for numerical stability.

    Returns:
        Normalized patches with the same shape as input.
    """
    mean = patches.mean(dim=-1, keepdim=True)
    var = patches.var(dim=-1, keepdim=True)
    return (patches - mean) / torch.sqrt(var + eps)


def forecast_loss(
    output: Dict[str, torch.Tensor],
    target: torch.Tensor,
    nodata_mask: Optional[torch.Tensor] = None,
    delta_reg_weight: float = 0.01,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute forecasting loss for fine-tuning.

    Combines:
    1. Reconstruction loss: MSE between prediction and target (excluding nodata)
    2. Delta regularization: Penalize large delta values for stability

    Args:
        output: Model output dict with 'prediction' and 'delta' keys.
        target: Target NDVI of shape (B, 1, H, W).
        nodata_mask: Boolean mask of shape (B, 1, H, W) where True = nodata.
        delta_reg_weight: Weight for delta regularization term.
        reduction: Loss reduction mode ('mean', 'sum').

    Returns:
        Tuple of (total_loss, loss_components_dict).
    """
    prediction = output["prediction"]
    delta = output["delta"]

    # Create valid mask (where we have data)
    if nodata_mask is not None:
        valid_mask = ~nodata_mask
    else:
        valid_mask = torch.ones_like(prediction, dtype=torch.bool)

    # Reconstruction loss on valid pixels
    if valid_mask.any():
        pred_valid = prediction[valid_mask]
        target_valid = target[valid_mask]
        recon_loss = F.mse_loss(pred_valid, target_valid, reduction=reduction)
    else:
        recon_loss = torch.tensor(0.0, device=prediction.device, requires_grad=True)

    # Delta regularization (encourage small changes, excluding nodata)
    if valid_mask.any():
        delta_reg = (delta[valid_mask] ** 2).mean()
    else:
        delta_reg = torch.tensor(0.0, device=delta.device)

    # Total loss
    total_loss = recon_loss + delta_reg_weight * delta_reg

    loss_components = {
        "recon_loss": recon_loss,
        "delta_reg": delta_reg,
        "total_loss": total_loss,
    }

    return total_loss, loss_components


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """General MSE loss with optional mask.

    Computes MSE only where mask is True (valid positions).

    Args:
        prediction: Predicted tensor.
        target: Target tensor (same shape as prediction).
        mask: Boolean mask where True = valid position (compute loss).
            If None, compute loss everywhere.
        reduction: Loss reduction mode ('mean', 'sum', 'none').

    Returns:
        Loss value.
    """
    if mask is None:
        return F.mse_loss(prediction, target, reduction=reduction)

    if not mask.any():
        return torch.tensor(0.0, device=prediction.device, requires_grad=True)

    pred_valid = prediction[mask]
    target_valid = target[mask]

    return F.mse_loss(pred_valid, target_valid, reduction=reduction)


def per_sample_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute mean per-sample (1 - Pearson correlation) loss.

    For each sample in the batch, computes Pearson correlation between the
    predicted and target spatial maps, then returns mean(1 - r) across samples.
    Samples with zero-variance prediction or target are excluded.

    This directly optimizes for R² (squared Pearson correlation), which measures
    how well the model captures spatial patterns within each tile.

    Args:
        prediction: Shape (B, 1, H, W).
        target: Shape (B, 1, H, W).
        valid_mask: Boolean mask, True = valid pixel. Shape (B, 1, H, W).
        eps: Small constant for numerical stability.

    Returns:
        Scalar loss in [0, 2]. Returns 0 if no valid samples.
    """
    B = prediction.shape[0]
    corr_losses = []

    for b in range(B):
        valid = valid_mask[b].flatten()
        pred_b = prediction[b].flatten()[valid].float()
        target_b = target[b].flatten()[valid].float()

        corr = _pearson_correlation(pred_b, target_b, eps)
        if corr is not None:
            corr_losses.append(1.0 - corr)

    if not corr_losses:
        return torch.tensor(0.0, device=prediction.device, requires_grad=True)

    return torch.stack(corr_losses).mean()


def _pearson_correlation(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """Compute Pearson correlation between two flat tensors.

    Returns None if either tensor has near-zero variance.

    Args:
        pred: 1D tensor of predicted values.
        target: 1D tensor of target values.
        eps: Minimum standard deviation to consider non-degenerate.

    Returns:
        Scalar correlation in [-1, 1], or None if degenerate.
    """
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    pred_std = pred_c.pow(2).sum().sqrt()
    target_std = target_c.pow(2).sum().sqrt()

    if pred_std < eps or target_std < eps:
        return None

    return (pred_c * target_c).sum() / (pred_std * target_std)


def smooth_l1_forecast_loss(
    output: Dict[str, torch.Tensor],
    target: torch.Tensor,
    nodata_mask: Optional[torch.Tensor] = None,
    delta_reg_weight: float = 0.0,
    beta: float = 0.1,
    correlation_loss_weight: float = 0.0,
    delta_supervision_weight: float = 0.0,
    delta_std: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Forecasting loss using Smooth L1 + optional correlation + delta supervision.

    Combines:
    1. Smooth L1 reconstruction loss (robust to outliers)
    2. Per-sample correlation loss (optimizes spatial pattern / R²)
    3. Delta regularization (legacy, default disabled)
    4. Delta supervision loss: L1(delta/std, true_delta/std) — direct gradient to delta

    Args:
        output: Model output dict with 'prediction', 'delta', and 'anchor' keys.
        target: Target NDVI of shape (B, 1, H, W).
        nodata_mask: Boolean mask of shape (B, 1, H, W) where True = nodata.
        delta_reg_weight: Weight for delta regularization term (default 0, disabled).
        beta: Smooth L1 transition point.
        correlation_loss_weight: Weight for per-sample correlation loss.
        delta_supervision_weight: Weight for direct delta supervision loss.
        delta_std: Std of historical deltas for normalization. Higher = weaker signal.

    Returns:
        Tuple of (total_loss, loss_components_dict).
    """
    prediction = output["prediction"]
    delta = output["delta"]
    valid_mask = ~nodata_mask if nodata_mask is not None else torch.ones_like(prediction, dtype=torch.bool)

    recon_loss, delta_reg = _compute_recon_and_delta(prediction, target, delta, valid_mask, beta)

    corr_loss = torch.tensor(0.0, device=prediction.device)
    if correlation_loss_weight > 0 and valid_mask.any():
        corr_loss = per_sample_correlation_loss(prediction, target, valid_mask)

    delta_sup_loss = torch.tensor(0.0, device=prediction.device)
    if delta_supervision_weight > 0 and valid_mask.any():
        anchor = output.get("anchor")
        if anchor is not None:
            delta_sup_loss = _compute_delta_supervision(
                delta, target, anchor, valid_mask, delta_std,
            )

    total_loss = (
        recon_loss
        + delta_reg_weight * delta_reg
        + correlation_loss_weight * corr_loss
        + delta_supervision_weight * delta_sup_loss
    )

    return total_loss, {
        "recon_loss": recon_loss,
        "delta_reg": delta_reg,
        "corr_loss": corr_loss,
        "delta_sup_loss": delta_sup_loss,
        "total_loss": total_loss,
    }


def _compute_recon_and_delta(
    prediction: torch.Tensor,
    target: torch.Tensor,
    delta: torch.Tensor,
    valid_mask: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Smooth L1 reconstruction loss and delta regularization.

    Args:
        prediction: Predicted NDVI (B, 1, H, W).
        target: Target NDVI (B, 1, H, W).
        delta: Delta tensor (B, 1, H, W).
        valid_mask: Boolean mask, True = valid pixel.
        beta: Smooth L1 transition point.

    Returns:
        Tuple of (recon_loss, delta_reg).
    """
    zero = torch.tensor(0.0, device=prediction.device, requires_grad=True)
    if not valid_mask.any():
        return zero, torch.tensor(0.0, device=delta.device)

    pred_valid = prediction[valid_mask]
    target_valid = target[valid_mask]
    recon_loss = F.smooth_l1_loss(pred_valid, target_valid, beta=beta)
    delta_reg = (delta[valid_mask] ** 2).mean()
    return recon_loss, delta_reg


def _compute_delta_supervision(
    delta: torch.Tensor,
    target: torch.Tensor,
    anchor: torch.Tensor,
    valid_mask: torch.Tensor,
    delta_std: float,
) -> torch.Tensor:
    """Compute normalized delta supervision loss (L1 in normalized space).

    Directly supervises the delta to match (target - anchor), normalized by
    the historical standard deviation of deltas. This amplifies the gradient
    signal: a 0.02 NDVI delta with std=0.025 becomes 0.8 in normalized space.

    Uses L1 (not Smooth L1) for uniform gradient regardless of error magnitude.

    Args:
        delta: Predicted delta (B, 1, H, W).
        target: Target NDVI (B, 1, H, W).
        anchor: Anchor observation (B, 1, H, W).
        valid_mask: Boolean mask, True = valid pixel.
        delta_std: Std of historical deltas. Used for normalization.

    Returns:
        Scalar L1 loss in normalized space.
    """
    # Exclude pixels where the anchor itself is nodata. Some anchor modes
    # (e.g. `last_step`) write the nodata sentinel (-9999) to pixels whose
    # entire history is nodata. If those leak into the loss, (target-anchor)
    # becomes ~10000, the normalized L1 explodes, and bf16 training NaNs out
    # within ~50 steps. NDVI values are bounded in [-1, 1] by construction,
    # so any value outside that range is a sentinel or numerical artifact.
    valid_mask = valid_mask & (anchor.abs() <= 1.5)

    if not valid_mask.any():
        return torch.tensor(0.0, device=delta.device, requires_grad=True)

    true_delta = (target - anchor)[valid_mask]
    pred_delta = delta[valid_mask]

    # Normalize to unit variance for stronger gradient signal (GraphCast approach)
    if delta_std > 0:
        true_delta = true_delta / delta_std
        pred_delta = pred_delta / delta_std

    return F.l1_loss(pred_delta, true_delta)


class PretrainingLoss:
    """Pre-training loss module for MAE-style reconstruction.

    Attributes:
        reduction: Loss reduction mode.
        normalize_target: Whether to normalize targets per patch (MAE-style).
    """

    def __init__(self, reduction: str = "mean", normalize_target: bool = False) -> None:
        """Initialize loss function.

        Args:
            reduction: Loss reduction mode ('mean', 'sum').
            normalize_target: Whether to normalize targets per patch.
        """
        self.reduction = reduction
        self.normalize_target = normalize_target

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        nodata_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            prediction: Predicted patches (B, T, N, P).
            target: Target patches (B, T, N, P).
            mask: Boolean mask (B, T, N) where True = masked.
            nodata_mask: Optional boolean mask (B, T, N) where True = nodata.

        Returns:
            Scalar loss.
        """
        return pretrain_loss(
            prediction, target, mask, nodata_mask, self.reduction, self.normalize_target
        )


class ForecastingLoss:
    """Forecasting loss combining Smooth L1, correlation, delta supervision.

    Uses Smooth L1 (Huber) loss for pixel-level reconstruction, optional per-sample
    correlation loss for spatial pattern optimization (R²), and normalized delta
    supervision for direct gradient signal to the delta pathway.

    Attributes:
        delta_reg_weight: Weight for delta regularization (default 0, disabled).
        beta: Smooth L1 transition point between L1 and L2 behavior.
        correlation_loss_weight: Weight for per-sample correlation loss.
        delta_supervision_weight: Weight for direct delta supervision loss.
        delta_std: Std of historical deltas for normalization.
    """

    def __init__(
        self,
        delta_reg_weight: float = 0.0,
        beta: float = 0.1,
        correlation_loss_weight: float = 0.0,
        delta_supervision_weight: float = 0.0,
        delta_std: float = 1.0,
    ) -> None:
        self.delta_reg_weight = delta_reg_weight
        self.beta = beta
        self.correlation_loss_weight = correlation_loss_weight
        self.delta_supervision_weight = delta_supervision_weight
        self.delta_std = delta_std

    def __call__(
        self,
        output: Dict[str, torch.Tensor],
        target: torch.Tensor,
        nodata_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss.

        Args:
            output: Model output dict with 'prediction', 'delta', 'anchor'.
            target: Target NDVI (B, 1, H, W).
            nodata_mask: Optional nodata mask (B, 1, H, W).

        Returns:
            Tuple of (total_loss, loss_components).
        """
        return smooth_l1_forecast_loss(
            output, target, nodata_mask,
            self.delta_reg_weight, self.beta, self.correlation_loss_weight,
            self.delta_supervision_weight, self.delta_std,
        )
