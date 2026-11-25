"""Visualization functions for interesting sample analysis.

Creates multi-row comparison figures showing target vs. baseline predictions
with error maps, and a summary bar chart across all samples.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from baselines import get_baseline

logger = logging.getLogger(__name__)

SEASON_NAMES = ["spring", "summer", "fall", "winter"]
BASELINE_METHODS = ["recent_climatology", "previous_year", "persistence"]
BASELINE_LABELS = {
    "recent_climatology": "Recent Climatology",
    "previous_year": "Previous Year",
    "persistence": "Persistence",
}
# Use recent_climatology for error maps (primary baseline)
ERROR_METHOD = "recent_climatology"


def compute_all_baselines(
    images: torch.Tensor,
    nodata_mask: torch.Tensor,
    target_indices: List[int],
    context_end_idx: int,
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """Compute predictions for all baselines across target indices.

    Args:
        images: Single sample images (T, H, W).
        nodata_mask: Nodata mask (T, H, W).
        target_indices: List of 4 target timestep indices (one per season).
        context_end_idx: Last index usable as context (inclusive).

    Returns:
        Dict mapping method_name to list of (pred_np, valid_mask_np) tuples,
        one per target index.
    """
    images_batch = images.unsqueeze(0)  # (1, T, H, W)
    mask_batch = nodata_mask.unsqueeze(0)

    results: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    for method_name in BASELINE_METHODS:
        baseline = get_baseline(method_name)
        preds = []
        for target_idx in target_indices:
            pred, valid = baseline.predict(
                images_batch, target_idx, mask_batch, context_end_idx
            )
            preds.append((pred[0].numpy(), valid[0].numpy()))
        results[method_name] = preds

    return results


def _compute_mae(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    """Compute MAE over valid pixels."""
    if valid.sum() == 0:
        return float("nan")
    return float(np.abs(pred[valid] - target[valid]).mean())


def _prepare_masked_image(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set invalid pixels to NaN for clean rendering."""
    img = data.astype(np.float64).copy()
    img[~mask] = np.nan
    return img


def plot_sample_comparison(
    images: torch.Tensor,
    nodata_mask: torch.Tensor,
    baseline_preds: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    target_indices: List[int],
    metadata: dict,
    output_path: Path,
) -> None:
    """Create multi-row comparison figure for a single sample.

    Layout:
        Row 0: Target NDVI (4 seasons)
        Row 1: Recent Climatology predictions
        Row 2: Previous Year predictions
        Row 3: Persistence predictions
        Row 4: Error maps for Recent Climatology

    Args:
        images: Single sample (T, H, W).
        nodata_mask: Nodata mask (T, H, W).
        baseline_preds: Output of compute_all_baselines().
        target_indices: 4 target indices (one per season).
        metadata: Sample metadata from interesting_samples.json.
        output_path: Where to save the PNG.
    """
    n_rows = 5  # target + 3 baselines + error
    n_cols = 4  # 4 seasons

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows),
        gridspec_kw={"hspace": 0.25, "wspace": 0.05},
    )

    ndvi_kwargs = {"cmap": "RdYlGn", "vmin": -1, "vmax": 1}
    error_kwargs = {"cmap": "coolwarm", "vmin": -0.5, "vmax": 0.5}

    # Collect per-method MAE for sidebar
    method_maes = {}

    # Row 0: Target NDVI
    for col, target_idx in enumerate(target_indices):
        target = images[target_idx].numpy()
        target_valid = ~nodata_mask[target_idx].numpy()
        img = _prepare_masked_image(target, target_valid)
        axes[0, col].imshow(img, **ndvi_kwargs)
        season = SEASON_NAMES[target_idx % 4]
        year = 1990 + target_idx // 4
        axes[0, col].set_title(f"{season.title()} {year}", fontsize=10)

    axes[0, 0].set_ylabel("Target", fontsize=11, fontweight="bold")

    # Rows 1-3: Baseline predictions
    for row_offset, method_name in enumerate(BASELINE_METHODS):
        row = row_offset + 1
        preds = baseline_preds[method_name]
        label = BASELINE_LABELS[method_name]
        season_maes = []

        for col, target_idx in enumerate(target_indices):
            pred, valid = preds[col]
            target = images[target_idx].numpy()
            target_valid = ~nodata_mask[target_idx].numpy()
            combined_valid = valid & target_valid

            img = _prepare_masked_image(pred, valid)
            axes[row, col].imshow(img, **ndvi_kwargs)

            mae = _compute_mae(pred, target, combined_valid)
            season_maes.append(mae)
            axes[row, col].set_title(f"MAE={mae:.4f}", fontsize=8, color="gray")

        axes[row, 0].set_ylabel(label, fontsize=11, fontweight="bold")
        avg_mae = np.nanmean(season_maes)
        method_maes[method_name] = avg_mae

    # Row 4: Error maps for primary method
    error_preds = baseline_preds[ERROR_METHOD]
    for col, target_idx in enumerate(target_indices):
        pred, valid = error_preds[col]
        target = images[target_idx].numpy()
        target_valid = ~nodata_mask[target_idx].numpy()
        combined_valid = valid & target_valid

        error = np.full_like(pred, np.nan, dtype=np.float64)
        error[combined_valid] = pred[combined_valid] - target[combined_valid]
        axes[4, col].imshow(error, **error_kwargs)

    axes[4, 0].set_ylabel(
        f"Error\n({BASELINE_LABELS[ERROR_METHOD]})",
        fontsize=10, fontweight="bold",
    )

    # Remove tick marks from all axes
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Build title with metadata
    sid = metadata.get("sample_id", "?")
    category = metadata.get("category", "?").replace("_", " ").title()
    mean_ndvi = metadata.get("mean_ndvi", 0)
    title_lines = _build_title(sid, category, mean_ndvi, method_maes, metadata)
    fig.suptitle(title_lines, fontsize=12, fontweight="bold", y=1.02)

    # Colorbars
    _add_colorbars(fig, axes, ndvi_kwargs, error_kwargs)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def _build_title(
    sid: str,
    category: str,
    mean_ndvi: float,
    method_maes: Dict[str, float],
    metadata: dict,
) -> str:
    """Build the figure super-title string."""
    line1 = f"Sample {sid}  |  {category}"

    # Add lat/lon if available
    lat, lon = metadata.get("lat"), metadata.get("lon")
    if lat is not None:
        line1 += f"  |  ({lat}, {lon})"

    # Add land cover if available
    landcover = metadata.get("landcover_name")
    if landcover:
        line1 += f"  |  {landcover}"

    line1 += f"  |  Mean NDVI: {mean_ndvi:.3f}"

    # MAE line
    mae_strs = []
    for method_name in BASELINE_METHODS:
        label = BASELINE_LABELS[method_name]
        mae = method_maes.get(method_name, float("nan"))
        mae_strs.append(f"{label}: {mae:.4f}")
    line2 = "MAE  -  " + "  |  ".join(mae_strs)

    return f"{line1}\n{line2}"


def _add_colorbars(fig, axes, ndvi_kwargs, error_kwargs) -> None:
    """Add NDVI and error colorbars to the figure."""
    # NDVI colorbar (for rows 0-3)
    sm_ndvi = plt.cm.ScalarMappable(
        cmap=ndvi_kwargs["cmap"],
        norm=plt.Normalize(ndvi_kwargs["vmin"], ndvi_kwargs["vmax"]),
    )
    cbar_ndvi = fig.colorbar(
        sm_ndvi, ax=axes[:4, :].ravel().tolist(),
        orientation="vertical", fraction=0.015, pad=0.02,
    )
    cbar_ndvi.set_label("NDVI", fontsize=9)

    # Error colorbar (for row 4)
    sm_err = plt.cm.ScalarMappable(
        cmap=error_kwargs["cmap"],
        norm=plt.Normalize(error_kwargs["vmin"], error_kwargs["vmax"]),
    )
    cbar_err = fig.colorbar(
        sm_err, ax=axes[4, :].ravel().tolist(),
        orientation="vertical", fraction=0.015, pad=0.02,
    )
    cbar_err.set_label("Error (Pred - Target)", fontsize=9)


def plot_summary_bars(
    all_metadata: List[dict],
    output_path: Path,
) -> None:
    """Create grouped bar chart of MAE per method across all samples.

    Args:
        all_metadata: List of per-sample metadata dicts, each containing
            sample_id, category, and per-method MAE keys (rc_mae, py_mae, ps_mae).
        output_path: Where to save the summary PNG.
    """
    if not all_metadata:
        logger.warning("No metadata for summary plot")
        return

    n_samples = len(all_metadata)
    methods = [
        ("rc_mae", "Recent Climatology", "#2ecc71"),
        ("py_mae", "Previous Year", "#3498db"),
        ("ps_mae", "Persistence", "#9b59b6"),
        ("cl_mae", "Climatology", "#e74c3c"),
    ]

    x = np.arange(n_samples)
    width = 0.18
    offsets = np.arange(len(methods)) - (len(methods) - 1) / 2

    fig, ax = plt.subplots(figsize=(max(10, n_samples * 1.5), 6))

    for i, (key, label, color) in enumerate(methods):
        values = [m.get(key, 0) for m in all_metadata]
        ax.bar(x + offsets[i] * width, values, width, label=label, color=color)

    # X-axis labels: sample_id + category
    labels = _build_bar_labels(all_metadata)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    ax.set_ylabel("MAE", fontsize=12)
    ax.set_title("Baseline MAE per Interesting Sample", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved summary: {output_path}")


def _build_bar_labels(all_metadata: List[dict]) -> List[str]:
    """Build x-axis labels for the summary bar chart."""
    labels = []
    for m in all_metadata:
        sid = m.get("sample_id", "?")
        cat = m.get("category", "").replace("_", " ")
        # Truncate long category names
        if len(cat) > 20:
            cat = cat[:18] + ".."
        labels.append(f"{sid}\n{cat}")
    return labels
