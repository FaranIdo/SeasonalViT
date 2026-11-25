"""Visualization utilities for SeasonalViT predictions.

Creates visualizations for model predictions, comparisons, and analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    anchors: Optional[torch.Tensor] = None,
    deltas: Optional[torch.Tensor] = None,
    save_path: Optional[Path] = None,
    title: str = "NDVI Prediction",
) -> plt.Figure:
    """Visualize model predictions alongside targets.

    Creates a grid showing prediction, target, and optionally anchor/delta.

    Args:
        predictions: Predicted NDVI of shape (B, 1, H, W) or (H, W).
        targets: Target NDVI of same shape.
        anchors: Optional anchor observations.
        deltas: Optional predicted deltas.
        save_path: Optional path to save figure.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    # Convert to numpy and handle dimensions
    pred = _to_numpy(predictions)
    targ = _to_numpy(targets)

    # Determine layout based on available data
    has_anchor = anchors is not None
    has_delta = deltas is not None
    n_cols = 2 + int(has_anchor) + int(has_delta)

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    # Prediction
    im = axes[0].imshow(pred, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[0].set_title("Prediction")
    axes[0].axis("off")

    # Target
    axes[1].imshow(targ, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("Target")
    axes[1].axis("off")

    idx = 2
    if has_anchor:
        anch = _to_numpy(anchors)
        axes[idx].imshow(anch, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[idx].set_title("Anchor")
        axes[idx].axis("off")
        idx += 1

    if has_delta:
        delt = _to_numpy(deltas)
        axes[idx].imshow(delt, cmap="coolwarm", vmin=-0.5, vmax=0.5)
        axes[idx].set_title("Delta")
        axes[idx].axis("off")

    # Colorbar
    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
    fig.suptitle(title)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_error_map(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    save_path: Optional[Path] = None,
    title: str = "Prediction Error",
) -> plt.Figure:
    """Visualize prediction error map.

    Args:
        predictions: Predicted NDVI.
        targets: Target NDVI.
        save_path: Optional path to save figure.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    pred = _to_numpy(predictions)
    targ = _to_numpy(targets)
    error = pred - targ

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Prediction
    axes[0].imshow(pred, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[0].set_title("Prediction")
    axes[0].axis("off")

    # Target
    axes[1].imshow(targ, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("Target")
    axes[1].axis("off")

    # Error
    im = axes[2].imshow(error, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    axes[2].set_title("Error (Pred - Target)")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_temporal_sequence(
    sequence: torch.Tensor,
    timesteps: Optional[List[int]] = None,
    save_path: Optional[Path] = None,
    title: str = "NDVI Time Series",
) -> plt.Figure:
    """Visualize a temporal sequence of NDVI images.

    Args:
        sequence: NDVI sequence of shape (T, H, W) or (1, T, H, W).
        timesteps: Optional list of timestep indices to show.
        save_path: Optional path to save figure.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    seq = _to_numpy(sequence)
    if seq.ndim == 4:
        seq = seq[0]  # Remove batch dimension

    T = seq.shape[0]
    if timesteps is None:
        # Show evenly spaced timesteps (max 8)
        n_show = min(8, T)
        timesteps = np.linspace(0, T - 1, n_show, dtype=int).tolist()

    n_cols = len(timesteps)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols, 3))
    if n_cols == 1:
        axes = [axes]

    for ax, t in zip(axes, timesteps):
        ax.imshow(seq[t], cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_title(f"t={t}")
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Visualize training history curves.

    Args:
        history: Dictionary with lists of metric values per epoch.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    n_metrics = len(history)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, history.items()):
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_per_season_metrics(
    season_metrics: Dict[int, Dict[str, float]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Visualize metrics broken down by season.

    Args:
        season_metrics: Dictionary mapping season_id to metrics dict.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    season_names = ["Spring", "Summer", "Fall", "Winter"]
    metrics = list(next(iter(season_metrics.values())).keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    x = np.arange(len(season_names))
    width = 0.6

    for ax, metric in zip(axes, metrics):
        values = [season_metrics.get(i, {}).get(metric, 0) for i in range(4)]
        colors = plt.cm.Set2(np.linspace(0, 1, 4))
        ax.bar(x, values, width, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(season_names)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} by Season")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_prediction_grid(
    model_outputs: List[Dict[str, torch.Tensor]],
    targets: List[torch.Tensor],
    n_samples: int = 4,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create a grid of prediction samples.

    Args:
        model_outputs: List of model output dictionaries.
        targets: List of target tensors.
        n_samples: Number of samples to show.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    n_samples = min(n_samples, len(model_outputs))

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples), squeeze=False)

    for i in range(n_samples):
        pred = _to_numpy(model_outputs[i]["prediction"])
        targ = _to_numpy(targets[i])
        anchor = _to_numpy(model_outputs[i].get("anchor"))
        delta = _to_numpy(model_outputs[i].get("delta"))
        error = pred - targ

        axes[i, 0].imshow(anchor if anchor is not None else pred, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[i, 0].set_title("Anchor" if anchor is not None else "Pred")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[i, 1].set_title("Prediction")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(targ, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[i, 2].set_title("Target")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(error, cmap="coolwarm", vmin=-0.5, vmax=0.5)
        axes[i, 3].set_title(f"Error (MAE={np.abs(error).mean():.3f})")
        axes[i, 3].axis("off")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _to_numpy(tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    """Convert tensor to numpy array for visualization.

    Args:
        tensor: Input tensor.

    Returns:
        Numpy array or None.
    """
    if tensor is None:
        return None

    arr = tensor.detach().cpu().numpy()

    # Remove batch and channel dimensions if present
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]

    return arr
