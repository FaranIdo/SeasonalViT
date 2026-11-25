"""Visualize interesting samples with baseline predictions.

Loads specific samples from TIF files or shards, computes baseline predictions,
and generates spatial NDVI comparison figures (target vs. baselines + error maps).

Usage:
    # From raw TIF files (recommended)
    python eval/visualize_interesting_samples.py \
        --data-dir /path/to/ndvi_35years_clean \
        --output-dir outputs/interesting_viz

    # From shard files
    python eval/visualize_interesting_samples.py \
        --shard-dir /path/to/shards \
        --output-dir outputs/interesting_viz

    # Single sample for quick testing
    python eval/visualize_interesting_samples.py \
        --data-dir /path/to/ndvi_35years_clean \
        --sample-ids 0515794
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.evaluate_baselines import LANDCOVER_NAMES, compute_target_indices
from eval.interesting_viz import (
    compute_all_baselines,
    plot_sample_comparison,
    plot_summary_bars,
)
from eval.sample_finder import (
    build_sample_index,
    load_samples_from_shards,
    load_samples_from_tifs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize interesting samples with baseline predictions"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data-dir",
        type=Path,
        help="Directory with raw TIF files ({sample_id}_{lat}_{lon}.tif)",
    )
    group.add_argument(
        "--shard-dir",
        type=Path,
        help="Directory with shard_*.safetensors files",
    )
    parser.add_argument(
        "--interesting-json",
        type=Path,
        default=Path("results/baselines/interesting_samples.json"),
        help="Path to interesting_samples.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/interesting_viz"),
        help="Directory to save output PNGs",
    )
    parser.add_argument(
        "--landcover-json",
        type=Path,
        default=None,
        help="Path to landcover_stats.json for land cover labels",
    )
    parser.add_argument(
        "--predict-year",
        type=int,
        default=2023,
        help="Year to predict (default: 2023)",
    )
    parser.add_argument(
        "--context-gap-years",
        type=int,
        default=1,
        help="Gap years between context end and prediction (default: 1)",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Optional subset of sample IDs to visualize",
    )
    return parser.parse_args()


def load_interesting_metadata(json_path: Path) -> dict:
    """Load interesting samples JSON and return dict keyed by sample_id.

    Args:
        json_path: Path to interesting_samples.json.

    Returns:
        Dict mapping sample_id to metadata dict.
    """
    with open(json_path) as f:
        samples = json.load(f)
    return {s["sample_id"]: s for s in samples}


def load_landcover_names(landcover_path: Path) -> dict:
    """Load land cover names keyed by sample_id.

    Args:
        landcover_path: Path to landcover_stats.json.

    Returns:
        Dict mapping sample_id to dominant land cover name.
    """
    with open(landcover_path) as f:
        data = json.load(f)

    result = {}
    for sid, info in data.items():
        code = info.get("dominant_class")
        result[sid] = LANDCOVER_NAMES.get(code, f"Class {code}") if code else "Unknown"
    return result


def enrich_metadata(metadata: dict, landcover_names: dict) -> None:
    """Add land cover name to metadata entries in-place."""
    for sid, meta in metadata.items():
        lc_name = landcover_names.get(sid)
        if lc_name:
            meta["landcover_name"] = lc_name


def load_samples(args: argparse.Namespace, sample_ids: list) -> dict:
    """Load samples from either TIF files or shards based on args.

    Returns:
        Dict mapping sample_id to {images, nodata_mask, filename}.
    """
    if args.data_dir:
        return load_samples_from_tifs(sample_ids, args.data_dir)
    else:
        logger.info("Building sample index from shards...")
        index = build_sample_index(args.shard_dir)
        return load_samples_from_shards(sample_ids, index)


def _parse_lat_lon(filename: str) -> tuple:
    """Parse lat/lon from filename like '0515794_19d2912_6d9579.tif'."""
    parts = filename.replace(".tif", "").split("_")
    if len(parts) >= 3:
        lat = parts[1].replace("d", ".")
        lon = parts[2].replace("d", ".")
        return lat, lon
    return None, None


def visualize_single_sample(
    sid: str,
    sample_data: dict,
    metadata: dict,
    target_indices: list,
    context_end_idx: int,
    output_dir: Path,
) -> None:
    """Compute baselines and plot comparison for one sample."""
    images = sample_data["images"]
    nodata_mask = sample_data["nodata_mask"]
    meta = metadata.get(sid, {"sample_id": sid, "category": "unknown"})

    # Add lat/lon from filename if not already present
    if "lat" not in meta:
        lat, lon = _parse_lat_lon(sample_data.get("filename", ""))
        if lat is not None:
            meta["lat"] = lat
            meta["lon"] = lon

    baseline_preds = compute_all_baselines(
        images, nodata_mask, target_indices, context_end_idx,
    )

    category = meta.get("category", "unknown")
    output_path = output_dir / f"{sid}_{category}.png"

    plot_sample_comparison(
        images, nodata_mask, baseline_preds,
        target_indices, meta, output_path,
    )


def main():
    """Orchestrate the full visualization pipeline."""
    args = parse_args()

    # Load interesting samples metadata
    if not args.interesting_json.exists():
        logger.error(f"Interesting samples JSON not found: {args.interesting_json}")
        sys.exit(1)

    metadata = load_interesting_metadata(args.interesting_json)
    logger.info(f"Loaded metadata for {len(metadata)} interesting samples")

    # Determine which samples to process
    if args.sample_ids:
        sample_ids = args.sample_ids
    else:
        sample_ids = list(metadata.keys())
    logger.info(f"Will visualize {len(sample_ids)} samples: {sample_ids}")

    # Load land cover names if provided
    if args.landcover_json and args.landcover_json.exists():
        landcover_names = load_landcover_names(args.landcover_json)
        enrich_metadata(metadata, landcover_names)
        logger.info(f"Loaded land cover data for {len(landcover_names)} samples")

    # Compute target indices and context boundary
    target_indices = compute_target_indices(
        predict_years=None,
        predict_year=args.predict_year,
        forecast_horizon=None,
    )
    context_end_idx = min(target_indices) - 1 - (args.context_gap_years - 1) * 4
    logger.info(
        f"Predicting year {args.predict_year}, "
        f"targets={target_indices}, context_end_idx={context_end_idx}"
    )

    # Load samples
    samples = load_samples(args, sample_ids)

    if not samples:
        logger.error("No samples loaded. Check data directory and sample IDs.")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate per-sample visualizations
    for sid in sample_ids:
        if sid not in samples:
            logger.warning(f"Skipping {sid}: not found")
            continue
        visualize_single_sample(
            sid, samples[sid], metadata,
            target_indices, context_end_idx, args.output_dir,
        )

    # Generate summary bar chart
    ordered_metadata = [metadata[sid] for sid in sample_ids if sid in metadata]
    summary_path = args.output_dir / "summary_mae_comparison.png"
    plot_summary_bars(ordered_metadata, summary_path)

    logger.info(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
