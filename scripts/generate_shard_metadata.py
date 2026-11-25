#!/usr/bin/env python3
"""Generate metadata.json for existing shard files.

Scans a directory for shard_*.safetensors files and creates metadata.json with:
- List of shards and their sample counts
- Total number of samples

Usage:
    python scripts/generate_shard_metadata.py --shard-dir /path/to/shards
"""

import argparse
import json
from pathlib import Path

import torch

from datasets.shard_io import load_shard


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate metadata.json for shard directory"
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        required=True,
        help="Directory containing shard_*.safetensors files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for metadata.json (default: shard-dir/metadata.json)",
    )
    return parser.parse_args()


def generate_metadata(shard_dir: Path) -> dict:
    """Scan shards and generate metadata dictionary.

    Args:
        shard_dir: Directory containing shard_*.safetensors files.

    Returns:
        Metadata dictionary with shard info and total samples.
    """
    shard_files = sorted(shard_dir.glob("shard_*.safetensors"))

    if not shard_files:
        raise ValueError(f"No shard_*.safetensors files found in {shard_dir}")

    print(f"Found {len(shard_files)} shard files")

    shards = []
    total_samples = 0

    for shard_path in shard_files:
        print(f"Processing {shard_path.name}...", end=" ")

        # Load shard to count samples
        shard = load_shard(shard_path)
        num_samples = len(shard["files"])
        total_samples += num_samples

        shards.append({
            "path": shard_path.name,
            "num_samples": num_samples,
        })

        print(f"{num_samples} samples")

    metadata = {
        "shards": shards,
        "total_samples": total_samples,
    }

    return metadata


def main():
    """Main entry point."""
    args = parse_args()

    if not args.shard_dir.exists():
        print(f"Error: Shard directory does not exist: {args.shard_dir}")
        return

    # Generate metadata
    metadata = generate_metadata(args.shard_dir)

    # Determine output path
    output_path = args.output or (args.shard_dir / "metadata.json")

    # Write metadata
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata written to {output_path}")
    print(f"Total shards: {len(metadata['shards'])}")
    print(f"Total samples: {metadata['total_samples']}")


if __name__ == "__main__":
    main()
