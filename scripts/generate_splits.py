#!/usr/bin/env python3
"""Generate research-grade splits and save to a JSON manifest.

This is a CLI wrapper around datasets/splits.py that REQUIRES access to
the actual TIF files. It extracts coordinates from filenames.

When to use this script:
    - You have local access to the TIF files
    - You want to generate splits directly from a data directory

When to use generate_splits_from_locations.py instead:
    - TIF files are on a remote machine
    - You only have a locations.json file with coordinates
    - You want to generate splits locally without transferring data

Related files:
    - datasets/splits.py: Core library with splitting functions (imported here)
    - scripts/generate_splits_from_locations.py: Alternative that works from
      locations.json without needing actual TIF files

This script creates the 5-way split structure for proper evaluation of
spatial and temporal generalization following GreenEarthNet (CVPR 2024).

No files are copied - the script just generates a JSON manifest that
records which files belong to each split. The dataloader reads this
manifest and loads files from the original location.

Usage:
    python scripts/generate_splits.py \\
        --data-dir /path/to/source/tifs \\
        --output data/splits.json

The script will:
1. Scan the data directory for *.tif files
2. Extract coordinates from filenames (format: {id}_{lat}_{lon}.tif)
3. Cluster locations into spatial blocks using K-means
4. Allocate blocks to Group A (train/val/ood-t) and Group B (ood-s/ood-st)
5. Save the split assignments to a JSON manifest
"""

import argparse
import logging
from pathlib import Path

from datasets.splits import (
    create_research_splits,
    save_splits_to_json,
    get_research_split_statistics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate research-grade splits with geographic + temporal separation"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing source NDVI GeoTIFF files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits.json"),
        help="Output path for splits manifest JSON (default: data/splits.json)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=158_000,
        help="Target number of training samples (default: 158000)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=8_000,
        help="Target number of validation samples (default: 8000)",
    )
    parser.add_argument(
        "--ood-t-size",
        type=int,
        default=20_000,
        help="Target number of temporal OOD samples (default: 20000, includes val_iid)",
    )
    parser.add_argument(
        "--ood-spatial-size",
        type=int,
        default=22_000,
        help="Target number of spatial OOD samples (default: 22000)",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=100,
        help="Number of K-means clusters for geographic blocking (default: 100)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate input directory
    if not args.data_dir.exists():
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return

    # Find all TIF files
    logger.info(f"Scanning {args.data_dir} for TIF files...")
    tif_files = sorted(args.data_dir.glob("*.tif"))
    logger.info(f"Found {len(tif_files)} TIF files")

    if len(tif_files) == 0:
        logger.error("No TIF files found in data directory")
        return

    # Create splits
    logger.info("Creating research splits...")
    logger.info(f"  K-means blocks: {args.n_blocks}")
    logger.info(f"  Random state: {args.random_state}")

    splits = create_research_splits(
        file_paths=tif_files,
        train_size=args.train_size,
        val_size=args.val_size,
        ood_t_size=args.ood_t_size,
        ood_spatial_size=args.ood_spatial_size,
        n_blocks=args.n_blocks,
        random_state=args.random_state,
    )

    # Print split statistics
    stats = get_research_split_statistics(splits)
    logger.info("\nSplit statistics:")
    for split_name, split_stats in stats.items():
        logger.info(f"  {split_name}:")
        logger.info(f"    Samples: {split_stats['n_samples']}")
        logger.info(f"    Input years: {split_stats['input_years']}")
        logger.info(f"    Predict years: {split_stats['predict_years']}")

    # Save to JSON manifest
    logger.info(f"\nSaving splits manifest to {args.output}...")
    save_splits_to_json(
        splits,
        args.output,
        n_blocks=args.n_blocks,
        random_state=args.random_state,
    )

    # Print verification info
    logger.info("\nSplit generation complete!")
    logger.info(f"Manifest saved to: {args.output}")
    logger.info("\nTo use these splits:")
    logger.info(f"  python scripts/convert_to_shards.py --input-dir TILES --output-dir data/shards --splits {args.output}")
    logger.info("  then pass the split shard directories to scripts/pretrain.py, finetune.py, and evaluate.py")


if __name__ == "__main__":
    main()
