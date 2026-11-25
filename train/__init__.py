"""Training utilities for SeasonalViT.

Components:
- masking: Configurable masking strategy for pre-training
  - "efficient": Levels 2+3 only (temporal masking before encoding, ~2x faster)
  - "full": All 3 levels (patch + temporal masking after encoding)
- losses: Loss functions for pre-training and forecasting
- metrics: Evaluation metrics

Imports are lazy to avoid errors during incremental development.
"""

__all__ = [
    # Masking
    "generate_mask",
    "generate_efficient_temporal_mask",
    "MaskOutput",
    "EfficientMaskOutput",
    # Losses
    "PretrainingLoss",
    "ForecastingLoss",
    "pretrain_loss",
    "pretrain_loss_efficient",
    "forecast_loss",
    "masked_mse_loss",
    # Metrics
    "compute_metrics",
    "compute_mae",
    "compute_rmse",
    "compute_r2",
    "compute_per_season_metrics",
    "MetricsTracker",
    # Training
    "PretrainingTrainer",
    "FinetuningTrainer",
]


def __getattr__(name: str):
    """Lazy import to allow incremental development."""
    # Masking
    if name == "generate_mask":
        from train.masking import generate_mask
        return generate_mask
    elif name == "generate_efficient_temporal_mask":
        from train.masking import generate_efficient_temporal_mask
        return generate_efficient_temporal_mask
    elif name == "MaskOutput":
        from train.masking import MaskOutput
        return MaskOutput
    elif name == "EfficientMaskOutput":
        from train.masking import EfficientMaskOutput
        return EfficientMaskOutput

    # Losses
    elif name == "PretrainingLoss":
        from train.losses import PretrainingLoss
        return PretrainingLoss
    elif name == "ForecastingLoss":
        from train.losses import ForecastingLoss
        return ForecastingLoss
    elif name == "pretrain_loss":
        from train.losses import pretrain_loss
        return pretrain_loss
    elif name == "pretrain_loss_efficient":
        from train.losses import pretrain_loss_efficient
        return pretrain_loss_efficient
    elif name == "forecast_loss":
        from train.losses import forecast_loss
        return forecast_loss
    elif name == "masked_mse_loss":
        from train.losses import masked_mse_loss
        return masked_mse_loss

    # Metrics
    elif name in (
        "compute_metrics", "compute_mae", "compute_rmse", "compute_r2",
        "compute_per_season_metrics", "MetricsTracker",
    ):
        from train import metrics
        return getattr(metrics, name)

    # Training
    elif name == "PretrainingTrainer":
        from train.pretrain import PretrainingTrainer
        return PretrainingTrainer
    elif name == "FinetuningTrainer":
        from train.finetune import FinetuningTrainer
        return FinetuningTrainer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
