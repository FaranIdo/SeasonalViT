"""Generate multi-sample spatial comparison figure for IEEE JSTARS paper.

Produces a 4-row × 9-column figure (3 samples × 3 columns each) showing
Ground Truth, Prediction, and Error for all 4 seasons. Designed for
paper figures at IEEE column width (7.16 inches).

Usage:
    # Default 3 samples (Bosnia, Argentina, Montana)
    python scripts/viz_spatial_comparison.py

    # Custom samples
    python scripts/viz_spatial_comparison.py \
        --inputs path/to/a.tif path/to/b.tif path/to/c.tif \
        --labels "(a) Bosnia" "(b) Argentina" "(c) Montana"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines import get_baseline

NODATA_VALUE = -9999.0
CROP_SIZE = 256
SEASON_NAMES = ["Spring", "Summer", "Fall", "Winter"]
START_YEAR = 1990

PROJECT_ROOT = Path(__file__).parent.parent
WINNERS_JSON = PROJECT_ROOT / "results/rc_winners/rc_winners.json"
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

NDVI_KW = {"cmap": "RdYlGn", "vmin": -0.2, "vmax": 0.9, "interpolation": "nearest"}
ERR_KW = {"cmap": "coolwarm", "vmin": -0.3, "vmax": 0.3, "interpolation": "nearest"}

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
    parser = argparse.ArgumentParser(description="Multi-sample spatial comparison")
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
        default="results/rc_winners/spatial_comparison.png",
        help="Output PNG path",
    )
    parser.add_argument("--method", default="recent_climatology")
    parser.add_argument("--predict-year", type=int, default=2023)
    parser.add_argument("--context-gap-years", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_filename(path: Path) -> dict:
    """Extract sample_id, lat, lon from filename like 0014254_44d4033_17d7914.tif."""
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return {"sample_id": stem, "lat": None, "lon": None}
    sample_id = parts[0]
    lat_str = parts[1].replace("d", ".")
    lon_str = parts[2].replace("d", ".")
    try:
        return {"sample_id": sample_id, "lat": float(lat_str), "lon": float(lon_str)}
    except ValueError:
        return {"sample_id": sample_id, "lat": None, "lon": None}


def lookup_metadata(sample_id: str) -> dict:
    """Look up land cover and location from rc_winners.json if available."""
    if not WINNERS_JSON.exists():
        return {}
    with open(WINNERS_JSON) as f:
        winners = json.load(f)
    for w in winners:
        if w["sample_id"] == sample_id:
            return {"landcover": w.get("landcover", ""), "rc_mae": w.get("rc_mae")}
    return {}


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


def build_multi_figure(all_data, labels, target_indices, predict_year):
    """Build the 4×9 multi-sample comparison figure with colorbars."""
    n_samples = len(all_data)
    # GridSpec: 11 cols = 3 + sep + 3 + sep + 3; 5 rows = 4 seasons + colorbar
    width_ratios = [1, 1, 1, 0.08, 1, 1, 1, 0.08, 1, 1, 1]
    height_ratios = [1, 1, 1, 1, 0.12]

    fig = plt.figure(figsize=(7.16, 4.5))
    gs = GridSpec(
        5, 11, figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=0.15, wspace=0.06,
        left=0.06, right=0.98, top=0.90, bottom=0.06,
    )

    col_offsets = [0, 4, 8]
    col_headers = ["GT", "Pred", "Error"]

    for si in range(n_samples):
        images, nodata_mask, predictions = all_data[si]
        offset = col_offsets[si]
        plot_sample_group(fig, gs, images, nodata_mask, predictions,
                          target_indices, offset, predict_year, is_first=(si == 0))
        add_sample_label(fig, gs, labels[si], offset)
        add_column_headers(fig, gs, col_headers, offset, row=0)

    add_colorbars(fig, gs)
    return fig


def plot_sample_group(fig, gs, images, nodata_mask, predictions,
                      target_indices, col_offset, predict_year, is_first):
    """Plot 4 rows × 3 cols for one sample at a given column offset."""
    for row in range(4):
        tidx = target_indices[row]
        pred, pvalid = predictions[row]
        gt = images[tidx].numpy()
        tvalid = ~nodata_mask[tidx].numpy()
        combined = pvalid & tvalid

        gt_display = np.where(tvalid, gt, np.nan)
        pred_display = np.where(pvalid, pred, np.nan)
        error = np.full_like(pred, np.nan, dtype=np.float64)
        error[combined] = pred[combined] - gt[combined]

        mae = float(np.abs(error[combined]).mean()) if combined.sum() > 0 else float("nan")

        panels = [gt_display, pred_display, error]
        kw_list = [NDVI_KW, NDVI_KW, ERR_KW]

        for ci in range(3):
            ax = fig.add_subplot(gs[row, col_offset + ci])
            ax.imshow(panels[ci], **kw_list[ci])
            ax.set_xticks([])
            ax.set_yticks([])
            if ci == 0 and is_first:
                ax.set_ylabel(
                    f"{SEASON_NAMES[tidx % 4]} {predict_year}", fontsize=7,
                )
            if ci == 2:
                ax.text(
                    0.97, 0.03, f"{mae:.3f}",
                    transform=ax.transAxes, fontsize=5, color="white",
                    ha="right", va="bottom",
                    bbox={"facecolor": "black", "alpha": 0.5,
                          "pad": 1.5, "edgecolor": "none"},
                )


def add_sample_label(fig, gs, label, col_offset):
    """Add sample label centered above the 3-column group."""
    left_ax = gs[0, col_offset]
    right_ax = gs[0, col_offset + 2]
    left_pos = left_ax.get_position(fig)
    right_pos = right_ax.get_position(fig)
    center_x = (left_pos.x0 + right_pos.x1) / 2
    top_y = left_pos.y1 + 0.04
    fig.text(center_x, top_y, label, ha="center", va="bottom",
             fontsize=7, fontweight="bold")


def add_column_headers(fig, gs, headers, col_offset, row):
    """Add GT/Pred/Error headers above row 0 for a sample group."""
    for ci, header in enumerate(headers):
        ax_pos = gs[row, col_offset + ci].get_position(fig)
        cx = (ax_pos.x0 + ax_pos.x1) / 2
        top_y = ax_pos.y1 + 0.01
        fig.text(cx, top_y, header, ha="center", va="bottom", fontsize=6)


def add_colorbars(fig, gs):
    """Add NDVI and error colorbars spanning the bottom row."""
    # NDVI colorbar: spans left ~2/3 (columns 0-6)
    left_pos = gs[4, 0].get_position(fig)
    mid_pos = gs[4, 6].get_position(fig)
    cax_ndvi = fig.add_axes([
        left_pos.x0, left_pos.y0,
        mid_pos.x1 - left_pos.x0, left_pos.y1 - left_pos.y0,
    ])
    sm_ndvi = plt.cm.ScalarMappable(
        cmap=NDVI_KW["cmap"],
        norm=plt.Normalize(NDVI_KW["vmin"], NDVI_KW["vmax"]),
    )
    fig.colorbar(sm_ndvi, cax=cax_ndvi, orientation="horizontal").set_label("NDVI")

    # Error colorbar: spans right ~1/3 (columns 8-10)
    right_left = gs[4, 8].get_position(fig)
    right_right = gs[4, 10].get_position(fig)
    cax_err = fig.add_axes([
        right_left.x0, right_left.y0,
        right_right.x1 - right_left.x0, right_left.y1 - right_left.y0,
    ])
    sm_err = plt.cm.ScalarMappable(
        cmap=ERR_KW["cmap"],
        norm=plt.Normalize(ERR_KW["vmin"], ERR_KW["vmax"]),
    )
    fig.colorbar(sm_err, cax=cax_err, orientation="horizontal").set_label(
        "Error = Pred \u2212 GT, MAE per panel",
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

    target_indices = [(args.predict_year - START_YEAR) * 4 + s for s in range(4)]
    context_gap = args.context_gap_years
    context_end = min(target_indices) - 1 - (context_gap - 1) * 4
    logger.info(
        f"Predict year: {args.predict_year}, targets: {target_indices}, "
        f"context_end: {context_end}, method: {args.method}"
    )

    all_data = load_all_samples(tif_paths, args.method, target_indices, context_end)

    fig = build_multi_figure(all_data, labels, target_indices, args.predict_year)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
