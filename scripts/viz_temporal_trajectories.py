"""Generate temporal trajectory figure for IEEE JSTARS paper.

Plots tile-averaged NDVI over the full 35-year record (140 timesteps)
for 3 samples. Shows observed history, annual moving average, ground
truth, and predictions in a vertical 3-row layout.

Usage:
    # Default 3 samples
    python scripts/viz_temporal_trajectories.py

    # Custom samples
    python scripts/viz_temporal_trajectories.py \
        --inputs path/to/a.tif path/to/b.tif path/to/c.tif \
        --labels "(a) Site A" "(b) Site B" "(c) Site C"
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines import get_baseline

NODATA_VALUE = -9999.0
CROP_SIZE = 256
START_YEAR = 1990
N_YEARS = 35
N_SEASONS = 4
N_BANDS = N_YEARS * N_SEASONS  # 140
SEASON_LABELS = ["Spr", "Sum", "Aut", "Win"]

PROJECT_ROOT = Path(__file__).parent.parent
SAMPLES_DIR = PROJECT_ROOT / "results/rc_winners/samples"

SAMPLE_DEFS = [
    {
        "tif": "0269458_47d0699_-114d2363.tif",
        "label": "(a) Missoula County, USA \u2014 Tree Cover",
    },
    {
        "tif": "0666156_-27d1347_-63d6132.tif",
        "label": "(b) Santiago del Estero, Argentina \u2014 Shrubland",
    },
    {
        "tif": "0292994_-17d8235_-52d1906.tif",
        "label": "(c) Goi\u00e1s, Brazil \u2014 Cropland",
    },
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_paper_style():
    """Configure matplotlib for IEEE paper figures."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "text.usetex": False,
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal trajectory visualization")
    parser.add_argument(
        "--inputs", nargs="+", default=None,
        help="Paths to .tif samples (default: 3 built-in samples)",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Labels for each sample (default: built-in labels)",
    )
    parser.add_argument(
        "--output",
        default="results/rc_winners/temporal_trajectories.png",
        help="Output PNG path",
    )
    parser.add_argument("--method", default="recent_climatology")
    parser.add_argument(
        "--predict-years", type=int, nargs="+", default=[2022, 2023, 2024],
        help="Years to predict (default: 2022 2023 2024)",
    )
    parser.add_argument("--context-gap-years", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_tif(tif_path: Path) -> tuple:
    """Load TIF, center-crop to 256, return (images, nodata_mask) tensors."""
    with rasterio.open(tif_path) as src:
        data_raw = src.read()  # (140, H, W)
    nodata_full = data_raw == NODATA_VALUE
    data_filled = np.where(nodata_full, 0.0, data_raw)
    _, h, w = data_filled.shape
    sh = (h - CROP_SIZE) // 2
    sw = (w - CROP_SIZE) // 2
    images = data_filled[:, sh:sh + CROP_SIZE, sw:sw + CROP_SIZE]
    nodata = nodata_full[:, sh:sh + CROP_SIZE, sw:sw + CROP_SIZE]
    return (
        torch.from_numpy(images.astype(np.float32)),
        torch.from_numpy(nodata),
    )


def compute_predictions(images, nodata_mask, method_name, target_indices, context_end):
    """Run baseline prediction for each target season."""
    baseline = get_baseline(method_name)
    imgs_batch = images.unsqueeze(0)
    mask_batch = nodata_mask.unsqueeze(0)
    results = []
    for tidx in target_indices:
        pred, valid = baseline.predict(imgs_batch, tidx, mask_batch, context_end)
        results.append((pred[0].numpy(), valid[0].numpy()))
    return results


def resolve_inputs_and_labels(args):
    """Resolve TIF paths and labels from args or defaults."""
    if args.inputs:
        tif_paths = [Path(p) for p in args.inputs]
        labels = args.labels or [p.stem for p in tif_paths]
    else:
        tif_paths = [SAMPLES_DIR / s["tif"] for s in SAMPLE_DEFS]
        labels = [s["label"] for s in SAMPLE_DEFS]
    return tif_paths, labels


def compute_tile_averages(images, nodata_mask):
    """Compute tile-averaged NDVI per timestep. Returns (140,) array with NaN for nodata."""
    avgs = np.full(N_BANDS, np.nan)
    for t in range(N_BANDS):
        valid = ~nodata_mask[t].numpy()
        if valid.sum() > 0:
            avgs[t] = images[t].numpy()[valid].mean()
    return avgs


def compute_pred_averages(predictions):
    """Compute tile-averaged prediction per target season. Returns (4,) array."""
    avgs = np.full(len(predictions), np.nan)
    for i, (pred, valid) in enumerate(predictions):
        if valid.sum() > 0:
            avgs[i] = pred[valid].mean()
    return avgs


def make_time_axis():
    """Create x-axis values: year + season offset (0.125, 0.375, 0.625, 0.875)."""
    season_offsets = [0.125, 0.375, 0.625, 0.875]
    times = []
    for y in range(N_YEARS):
        for s_off in season_offsets:
            times.append(START_YEAR + y + s_off)
    return np.array(times)


def build_figure(all_data, labels, target_indices, predict_years, context_end):
    """Build the 3×1 vertical temporal trajectory figure."""
    n_samples = len(all_data)
    fig, axes = plt.subplots(
        n_samples, 1, figsize=(7.16, 5.0), sharex=False, sharey=False,
    )
    time_axis = make_time_axis()
    target_times = np.array([time_axis[ti] for ti in target_indices])

    for si, (images, nodata_mask, predictions) in enumerate(all_data):
        is_last = (si == n_samples - 1)
        plot_trajectory(
            axes[si], images, nodata_mask, predictions,
            time_axis, target_indices, target_times,
            context_end, predict_years, labels[si], is_last,
        )

    add_legend(fig, axes[0])
    fig.subplots_adjust(
        left=0.07, right=0.98, top=0.96, bottom=0.10,
        hspace=0.22,
    )
    return fig


def plot_trajectory(ax, images, nodata_mask, predictions,
                    time_axis, target_indices, target_times,
                    context_end, predict_years, label, is_last):
    """Plot NDVI trajectory for one sample, zoomed to recent years."""
    tile_avgs = compute_tile_averages(images, nodata_mask)
    pred_avgs = compute_pred_averages(predictions)
    gt_avgs = np.array([tile_avgs[ti] for ti in target_indices])

    # Show only recent ~15 years for clarity
    display_start_year = 2010
    display_start_idx = (display_start_year - START_YEAR) * N_SEASONS

    # Raw observed data
    ax.plot(time_axis[display_start_idx:], tile_avgs[display_start_idx:],
            color="#404040", linewidth=0.8, label="Observed")

    # Prediction zone highlight (spans all predict years)
    ax.axvspan(min(predict_years) - 0.1, max(predict_years) + 1.1,
               color="#fff3cd", alpha=0.5, zorder=0)

    # Context boundary
    context_time = time_axis[context_end]
    ax.axvline(context_time, color="#888888", linestyle="--",
               linewidth=0.6, alpha=0.8, label="Context end")

    plot_markers(ax, target_times, gt_avgs, pred_avgs)

    ax.set_ylabel("NDVI", fontsize=7)
    ax.text(
        0.01, 0.95, label, transform=ax.transAxes,
        fontsize=7, fontweight="bold", va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 2, "edgecolor": "none"},
    )
    ax.set_xlim(display_start_year - 0.3, max(predict_years) + 1.3)
    ax.set_ylim(bottom=-0.05)
    ax.tick_params(axis="both", which="both", length=2)
    ax.grid(axis="y", linewidth=0.3, alpha=0.3)

    ax.set_xticks(range(display_start_year, max(predict_years) + 2, 2))
    if is_last:
        ax.set_xlabel("Year", fontsize=7)


def plot_markers(ax, target_times, gt_avgs, pred_avgs):
    """Plot GT and prediction as connected lines with markers."""
    # Ground truth line + markers
    ax.plot(target_times, gt_avgs, color="#2ca02c", linewidth=1.5, zorder=6,
            marker="o", markersize=5, markeredgecolor="white", markeredgewidth=0.5,
            label="Ground truth")

    # Prediction line + markers
    ax.plot(target_times, pred_avgs, color="#ff7f0e", linewidth=1.5, zorder=6,
            marker="D", markersize=4.5, markeredgecolor="white", markeredgewidth=0.5,
            label="Prediction")


def add_legend(fig, first_ax):
    """Add a shared legend below the subplots."""
    handles, leg_labels = first_ax.get_legend_handles_labels()
    fig.legend(
        handles, leg_labels, loc="lower center",
        ncol=4, fontsize=6, frameon=False,
        bbox_to_anchor=(0.5, 0.0), columnspacing=3.0,
    )


def load_all_samples(tif_paths, method, target_indices, context_end):
    """Load TIFs and compute predictions for all samples."""
    all_data = []
    for tif_path in tif_paths:
        if not tif_path.exists():
            logger.error(f"Input file not found: {tif_path}")
            sys.exit(1)
        images, nodata_mask = load_tif(tif_path)
        logger.info(f"Loaded {tif_path.name}: shape={images.shape}")
        predictions = compute_predictions(
            images, nodata_mask, method, target_indices, context_end,
        )
        all_data.append((images, nodata_mask, predictions))
    return all_data


def main():
    args = parse_args()
    setup_paper_style()

    tif_paths, labels = resolve_inputs_and_labels(args)
    logger.info(f"Samples: {[p.name for p in tif_paths]}")

    predict_years = sorted(args.predict_years)
    target_indices = []
    for yr in predict_years:
        target_indices.extend([(yr - START_YEAR) * 4 + s for s in range(4)])
    context_gap = args.context_gap_years
    context_end = min(target_indices) - 1 - (context_gap - 1) * 4
    logger.info(
        f"Predict years: {predict_years}, targets: {target_indices}, "
        f"context_end: {context_end}, method: {args.method}"
    )

    all_data = load_all_samples(tif_paths, args.method, target_indices, context_end)

    fig = build_figure(all_data, labels, target_indices, predict_years, context_end)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
