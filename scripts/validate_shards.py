#!/usr/bin/env python3
"""Validate shard files for corruption using ZIP integrity check."""

import argparse
import sys
import zipfile
from pathlib import Path


def validate_shard(path: Path) -> tuple[str, bool, str]:
    """Validate shard can be opened as ZIP (fast, no data read)."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            entries = zf.namelist()
            return path.name, True, f"OK ({len(entries)} entries)"
    except Exception as e:
        return path.name, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Validate shard files")
    parser.add_argument("--shard-dir", type=str, required=True)
    parser.add_argument("--shard-range", type=str, help="e.g., 0:15")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N shards")
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    if not shard_dir.exists():
        print(f"Directory not found: {shard_dir}")
        sys.exit(1)

    # Find all shard files
    shard_files = sorted(shard_dir.glob("shard_*.pt"))
    print(f"Found {len(shard_files)} shard files in {shard_dir}")

    # Filter by range if specified
    if args.shard_range:
        start, end = map(int, args.shard_range.split(":"))
        shard_files = [f for f in shard_files if start <= int(f.stem.split("_")[1]) < end]
        print(f"Filtered to {len(shard_files)} shards in range [{start}, {end})")

    # Skip first N shards
    if args.skip > 0:
        shard_files = shard_files[args.skip:]
        print(f"Skipping first {args.skip}, {len(shard_files)} remaining")

    print("Validating ZIP integrity (fast mode)...")

    # Validate shards using ZIP integrity check (no tensor loading)
    failed = []
    for shard_path in shard_files:
        name, valid, msg = validate_shard(shard_path)
        status = "OK" if valid else "FAIL"
        print(f"  [{status}] {name}: {msg}", flush=True)
        if not valid:
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} corrupted shard(s):")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nAll shards valid!")


if __name__ == "__main__":
    main()
