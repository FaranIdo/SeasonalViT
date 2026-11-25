#!/usr/bin/env python3
"""Generate research-grade splits from locations.json file.

This is a STANDALONE script that does NOT require access to actual TIF files.
It reads coordinates from a locations.json file and generates the split manifest.

When to use this script:
    - TIF files are on a remote machine (e.g., Windows GPU server)
    - You only have a locations.json file locally
    - You want to generate splits without transferring large data files

When to use generate_splits.py instead:
    - You have local access to the actual TIF files
    - You want to generate splits directly from a data directory

Why this script duplicates some logic from datasets/splits.py:
    This script is intentionally self-contained so it can run without
    importing the datasets module. This makes it easy to copy to other
    machines or run in minimal environments. The K-means clustering and
    block allocation logic is duplicated here for this purpose.

Related files:
    - datasets/splits.py: Core library with splitting functions
    - scripts/generate_splits.py: CLI wrapper that requires actual TIF files

Workflow for remote data:
    1. Run this script locally with locations.json
    2. Transfer the generated splits.json to the remote machine
    3. Run sharding/training on remote using the splits manifest

Usage:
    python scripts/generate_splits_from_locations.py \\
        --locations data/locations_200k.json \\
        --output data/splits_200k.json \\
        --source-dir /path/to/GloSeaN/tiles

The --source-dir should be the path where TIF files are located on the
target machine where sharding/training will be performed.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _kmeans_cluster(coords, n_clusters, random_state, max_iter=300):
    """Simple K-means clustering implementation."""
    np.random.seed(random_state)
    n_samples = len(coords)

    indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = coords[indices].copy()
    labels = np.zeros(n_samples, dtype=np.int32)

    for _ in range(max_iter):
        # Compute distances
        coords_exp = coords[:, np.newaxis, :]
        centroids_exp = centroids[np.newaxis, :, :]
        diff = coords_exp - centroids_exp
        distances = np.sqrt((diff**2).sum(axis=2))

        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(labels, new_labels):
            break

        labels = new_labels

        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centroids[k] = coords[mask].mean(axis=0)

    return labels


def _allocate_blocks_to_groups(block_labels, target_group_b_size, random_state):
    """Allocate spatial blocks to Group A and Group B."""
    np.random.seed(random_state + 1)

    unique_blocks = np.unique(block_labels)
    block_sizes = {b: (block_labels == b).sum() for b in unique_blocks}

    shuffled_blocks = list(unique_blocks)
    np.random.shuffle(shuffled_blocks)

    group_b_blocks = set()
    group_b_count = 0

    for block in shuffled_blocks:
        if group_b_count >= target_group_b_size:
            break
        group_b_blocks.add(block)
        group_b_count += block_sizes[block]

    group_a_blocks = set(unique_blocks) - group_b_blocks

    return group_a_blocks, group_b_blocks


def format_coordinate(value):
    """Format coordinate as used in filenames (e.g., 36.3617 -> '36d3617')."""
    sign = "" if value >= 0 else "-"
    abs_val = abs(value)
    int_part = int(abs_val)
    # Get 4 decimal places
    dec_part = int(round((abs_val - int_part) * 10000))
    return f"{sign}{int_part}d{dec_part:04d}"


def generate_filename(sample_id, lat, lon):
    """Generate filename from sample_id and coordinates."""
    lat_str = format_coordinate(lat)
    lon_str = format_coordinate(lon)
    return f"{sample_id}_{lat_str}_{lon_str}.tif"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate research-grade splits from locations.json"
    )

    parser.add_argument(
        "--locations",
        type=Path,
        default=Path("data/locations_200k.json"),
        help="Path to locations JSON file (default: data/locations_200k.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits_200k.json"),
        help="Output path for splits manifest JSON (default: data/splits_200k.json)",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="",
        help="Source directory where TIF files are located (for manifest)",
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

    # Load locations
    logger.info(f"Loading locations from {args.locations}...")
    with open(args.locations) as f:
        locations = json.load(f)

    n_samples = len(locations)
    logger.info(f"Loaded {n_samples:,} locations")

    # Build coordinates array and filename list
    sample_ids = list(locations.keys())
    coords = np.array([locations[sid] for sid in sample_ids])
    filenames = [
        generate_filename(sid, coords[i, 0], coords[i, 1])
        for i, sid in enumerate(sample_ids)
    ]

    # Validate sizes
    total_requested = (
        args.train_size + args.ood_t_size + args.ood_spatial_size
    )
    # Note: val_iid is subset of ood_t, so not counted separately
    if total_requested > n_samples:
        logger.warning(
            f"Requested total ({total_requested:,}) exceeds available "
            f"samples ({n_samples:,}). Sizes will be adjusted."
        )

    # Adjust n_blocks
    n_blocks = min(args.n_blocks, n_samples // 10)
    n_blocks = max(n_blocks, 5)

    np.random.seed(args.random_state)

    # Step 1: Cluster into spatial blocks
    logger.info(f"Clustering into {n_blocks} spatial blocks...")
    block_labels = _kmeans_cluster(coords, n_blocks, args.random_state)

    # Step 2: Allocate blocks to Group A and Group B
    logger.info("Allocating blocks to Group A (train/val) and Group B (ood_spatial)...")
    group_a_blocks, group_b_blocks = _allocate_blocks_to_groups(
        block_labels, args.ood_spatial_size, args.random_state
    )

    # Step 3: Get sample indices for each group
    group_a_indices = np.where(np.isin(block_labels, list(group_a_blocks)))[0]
    group_b_indices = np.where(np.isin(block_labels, list(group_b_blocks)))[0]

    logger.info(f"Group A: {len(group_a_indices):,} samples")
    logger.info(f"Group B: {len(group_b_indices):,} samples")

    # Step 4: Within Group A, split into train and val/ood_t subset
    np.random.shuffle(group_a_indices)
    val_ood_size = args.ood_t_size  # ood_t includes val_iid
    val_ood_indices = group_a_indices[:val_ood_size]
    train_indices = group_a_indices[val_ood_size:]

    # Limit train to requested size
    if len(train_indices) > args.train_size:
        train_indices = train_indices[: args.train_size]

    # Step 5: Split val/ood_t into val_iid and ood_t
    val_iid_indices = val_ood_indices[: args.val_size]
    ood_t_indices = val_ood_indices  # ood_t uses all val_ood locations

    # Step 6: Group B becomes ood_s and ood_st (same locations, different years)
    ood_s_indices = group_b_indices
    ood_st_indices = group_b_indices

    # Build result
    splits = {
        "train": {
            "files": [filenames[i] for i in train_indices],
            "input_years": [1990, 2021],
            "predict_years": [2022, 2022],
        },
        "val_iid": {
            "files": [filenames[i] for i in val_iid_indices],
            "input_years": [1990, 2021],
            "predict_years": [2022, 2022],
        },
        "ood_t": {
            "files": [filenames[i] for i in ood_t_indices],
            "input_years": [1990, 2022],
            "predict_years": [2023, 2024],
        },
        "ood_s": {
            "files": [filenames[i] for i in ood_s_indices],
            "input_years": [1990, 2021],
            "predict_years": [2022, 2022],
        },
        "ood_st": {
            "files": [filenames[i] for i in ood_st_indices],
            "input_years": [1990, 2022],
            "predict_years": [2023, 2024],
        },
    }

    # Print statistics
    logger.info("\nSplit statistics:")
    for split_name, split_info in splits.items():
        n_files = len(split_info["files"])
        input_yrs = split_info["input_years"]
        pred_yrs = split_info["predict_years"]
        logger.info(
            f"  {split_name}: {n_files:,} samples | "
            f"input: {input_yrs[0]}-{input_yrs[1]} | "
            f"predict: {pred_yrs[0]}-{pred_yrs[1]}"
        )

    # Calculate unique total
    unique_total = (
        len(splits["train"]["files"])
        + len(splits["ood_t"]["files"])  # includes val_iid
        + len(splits["ood_s"]["files"])
    )
    logger.info(f"\nTotal unique samples: {unique_total:,}")

    # Build manifest
    manifest = {
        "created": datetime.now().isoformat(),
        "source_dir": args.source_dir,
        "n_blocks": n_blocks,
        "random_state": args.random_state,
        "total_unique_samples": unique_total,
        "splits": {},
    }

    for split_name, split_info in splits.items():
        manifest["splits"][split_name] = {
            "files": split_info["files"],
            "count": len(split_info["files"]),
            "input_years": split_info["input_years"],
            "predict_years": split_info["predict_years"],
        }

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\nSaved splits manifest to: {args.output}")
    logger.info("\nNext steps:")
    logger.info(
        "  1. Run create_unique_shard_lists.py to generate non-overlapping shard lists"
    )
    logger.info("  2. Transfer shard lists to remote machine")
    logger.info("  3. Run convert_to_shards.py for each shard list")


if __name__ == "__main__":
    main()
