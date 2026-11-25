#!/usr/bin/env python3
"""Create unique shard lists from splits JSON (no file duplication).

This script reads a splits JSON file and generates 4 non-overlapping
shard list files that can be used with convert_to_shards.py:

1. train: Training samples (unique)
2. val_iid: IID validation samples (unique)
3. ood_t_extra: ood_t - val_iid (temporal OOD samples not in val_iid)
4. ood_spatial: Group B samples for ood_s/ood_st (same files, year config at model level)

Usage:
    python scripts/create_unique_shard_lists.py \
        --splits data/splits_200k.json \
        --output-dir data/shard_lists

Output:
    data/shard_lists/shard_list_train.json
    data/shard_lists/shard_list_val_iid.json
    data/shard_lists/shard_list_ood_t_extra.json
    data/shard_lists/shard_list_ood_spatial.json
"""

import argparse
import json
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create unique shard lists from splits JSON"
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/splits_200k.json"),
        help="Path to splits JSON file (default: data/splits_200k.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/shard_lists"),
        help="Output directory for shard list files (default: data/shard_lists)",
    )
    return parser.parse_args()


def create_shard_list(source_dir: str, split_name: str, files: list) -> dict:
    """Create a shard list JSON structure."""
    return {
        "source_dir": source_dir,
        "splits": {
            split_name: {
                "files": files,
                "count": len(files),
            }
        },
    }


def main():
    """Generate unique shard lists from splits."""
    args = parse_args()

    print(f"Loading splits from {args.splits}...")
    with open(args.splits) as f:
        splits_data = json.load(f)

    source_dir = splits_data["source_dir"]
    splits = splits_data["splits"]

    # Extract file lists
    train_files = splits["train"]["files"]
    val_iid_files = set(splits["val_iid"]["files"])
    ood_t_files = set(splits["ood_t"]["files"])
    ood_s_files = splits["ood_s"]["files"]

    # Compute ood_t_extra = ood_t - val_iid (avoid duplication)
    ood_t_extra_files = sorted(ood_t_files - val_iid_files)

    print(f"\nSource directory: {source_dir}")
    print(f"\nOriginal splits:")
    print(f"  train:   {len(train_files):,} files")
    print(f"  val_iid: {len(val_iid_files):,} files")
    print(f"  ood_t:   {len(ood_t_files):,} files")
    print(f"  ood_s:   {len(ood_s_files):,} files")

    print(f"\nUnique shard groups (no duplication):")
    print(f"  train:       {len(train_files):,} files")
    print(f"  val_iid:     {len(val_iid_files):,} files")
    print(f"  ood_t_extra: {len(ood_t_extra_files):,} files (ood_t - val_iid)")
    print(f"  ood_spatial: {len(ood_s_files):,} files")

    total_unique = (
        len(train_files)
        + len(val_iid_files)
        + len(ood_t_extra_files)
        + len(ood_s_files)
    )
    print(f"\nTotal unique files: {total_unique:,}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build shard lists
    shard_lists = {
        "train": create_shard_list(source_dir, "train", train_files),
        "val_iid": create_shard_list(source_dir, "val_iid", sorted(val_iid_files)),
        "ood_t_extra": create_shard_list(source_dir, "ood_t_extra", ood_t_extra_files),
        "ood_spatial": create_shard_list(source_dir, "ood_spatial", ood_s_files),
    }

    # Save each shard list
    for name, data in shard_lists.items():
        output_path = args.output_dir / f"shard_list_{name}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {output_path}")

    print(f"\nDone! Shard lists saved to {args.output_dir}/")
    print("\nUsage with convert_to_shards.py:")
    print("  python scripts/convert_to_shards.py \\")
    print("      --input-dir <source_dir> \\")
    print("      --output-dir <output_dir>/train \\")
    print(f"      --splits {args.output_dir}/shard_list_train.json \\")
    print("      --samples-per-shard 1024")


if __name__ == "__main__":
    main()
