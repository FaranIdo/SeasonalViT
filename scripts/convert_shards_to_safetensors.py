#!/usr/bin/env python3
"""Batch-convert shard_*.pt files to safetensors format.

Converts each shard into:
  shard_XXXXX.safetensors  - tensors (images, nodata_masks, lats, lons)
  shard_XXXXX.files.json   - filenames list (JSON sidecar)

Usage:
  python scripts/convert_shards_to_safetensors.py --input-dir D:\\data\\shards\\train
  python scripts/convert_shards_to_safetensors.py --input-dir data/train --workers 4 --verify
  python scripts/convert_shards_to_safetensors.py --input-dir data/train --shard-range 0:100
"""

import argparse
import gc
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file


def parse_shard_range(range_str: str) -> tuple[int, int]:
    """Parse 'start:end' into (start, end)."""
    parts = range_str.split(":")
    return int(parts[0]), int(parts[1])


def convert_one_shard(
    pt_path: Path, verify: bool, output_dir: Path | None = None
) -> dict:
    """Convert a single .pt shard to safetensors + JSON sidecar.

    Returns a result dict with status info.
    """
    base_dir = output_dir if output_dir else pt_path.parent
    st_path = base_dir / pt_path.with_suffix(".safetensors").name
    json_path = base_dir / pt_path.with_suffix(".files.json").name

    if st_path.exists() and json_path.exists():
        return {"path": pt_path.name, "status": "skipped"}

    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        tensors = _extract_tensors(data)
        save_file(tensors, st_path)
        _write_files_json(data["files"], json_path)

        result = {
            "path": pt_path.name,
            "status": "ok",
            "pt_size": pt_path.stat().st_size,
            "st_size": st_path.stat().st_size,
        }

        if verify:
            _verify_shard(data, st_path, json_path)

        del data, tensors
        gc.collect()
        return result

    except Exception as e:
        return {"path": pt_path.name, "status": "error", "error": str(e)}


def _extract_tensors(data: dict) -> dict[str, torch.Tensor]:
    """Extract tensor dict from loaded shard data.

    Casts bool tensors to uint8 if safetensors doesn't support bool.
    """
    nodata = data["nodata_masks"]
    if nodata.dtype == torch.bool:
        nodata = nodata.to(torch.uint8)

    return {
        "images": data["images"],
        "nodata_masks": nodata,
        "lats": data["lats"],
        "lons": data["lons"],
    }


def _write_files_json(files_list: list[str], json_path: Path) -> None:
    """Write filenames list to JSON sidecar."""
    with open(json_path, "w") as f:
        json.dump(files_list, f)


def _verify_shard(
    original: dict, st_path: Path, json_path: Path
) -> None:
    """Verify converted shard matches original data."""
    loaded = load_file(st_path)
    with open(json_path) as f:
        loaded_files = json.load(f)

    assert torch.equal(original["images"], loaded["images"]), "images mismatch"
    orig_masks = original["nodata_masks"]
    loaded_masks = loaded["nodata_masks"]
    if orig_masks.dtype == torch.bool:
        loaded_masks = loaded_masks.to(torch.bool)
    assert torch.equal(orig_masks, loaded_masks), "nodata_masks mismatch"
    assert torch.equal(original["lats"], loaded["lats"]), "lats mismatch"
    assert torch.equal(original["lons"], loaded["lons"]), "lons mismatch"
    assert original["files"] == loaded_files, "files list mismatch"

    del loaded
    gc.collect()


def discover_shards(input_dir: Path, shard_range: str | None) -> list[Path]:
    """Find shard_*.pt files, optionally filtered by range."""
    shards = sorted(input_dir.glob("shard_*.pt"))
    if shard_range:
        start, end = parse_shard_range(shard_range)
        shards = shards[start:end]
    return shards


def run_conversion(
    shards: list[Path], workers: int, verify: bool,
    output_dir: Path | None = None,
) -> list[dict]:
    """Convert shards using subprocess workers to avoid torch.load memory leak.

    Each worker process loads one shard then exits, fully reclaiming memory.
    max_tasks_per_child=1 ensures a fresh process per shard.
    """
    results = []
    with ProcessPoolExecutor(
        max_workers=workers, max_tasks_per_child=1
    ) as pool:
        futures = {
            pool.submit(convert_one_shard, p, verify, output_dir): p
            for p in shards
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            _print_progress(result, i, len(shards))
            results.append(result)
    return results


def _print_progress(result: dict, current: int, total: int) -> None:
    """Print progress line for one shard."""
    name = result["path"]
    status = result["status"]
    if status == "ok":
        pt_mb = result["pt_size"] / 1024 / 1024
        st_mb = result["st_size"] / 1024 / 1024
        print(f"  [{current}/{total}] {name} -> .safetensors "
              f"({pt_mb:.1f} MB -> {st_mb:.1f} MB)")
    elif status == "skipped":
        print(f"  [{current}/{total}] {name} (skipped, already exists)")
    else:
        print(f"  [{current}/{total}] {name} ERROR: {result.get('error')}")


def print_summary(results: list[dict], elapsed: float) -> None:
    """Print final conversion summary."""
    converted = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]

    total_pt = sum(r["pt_size"] for r in converted)
    total_st = sum(r["st_size"] for r in converted)

    print(f"\n{'='*50}")
    print(f"Conversion complete in {elapsed:.1f}s")
    print(f"  Converted: {len(converted)}")
    print(f"  Skipped:   {len(skipped)}")
    print(f"  Errors:    {len(errors)}")
    if converted:
        ratio = total_st / total_pt * 100 if total_pt > 0 else 0
        print(f"  .pt total:  {total_pt / 1024**3:.2f} GB")
        print(f"  .safetensors total: {total_st / 1024**3:.2f} GB "
              f"({ratio:.1f}% of original)")
    for r in errors:
        print(f"  FAILED: {r['path']} - {r.get('error')}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert shard_*.pt files to safetensors format"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing shard_*.pt files",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Number of parallel conversion workers (default: 2)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify converted shards match originals",
    )
    parser.add_argument(
        "--shard-range", type=str, default=None,
        help="Convert subset of shards, e.g. '0:100'",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for safetensors files (default: same as input)",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory")
        return

    shards = discover_shards(args.input_dir, args.shard_range)
    if not shards:
        print(f"No shard_*.pt files found in {args.input_dir}")
        return

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {args.output_dir}")

    print(f"Found {len(shards)} shard(s) in {args.input_dir}")
    if args.verify:
        print("Verification enabled")
    print()

    t0 = time.time()
    results = run_conversion(shards, args.workers, args.verify, args.output_dir)
    print_summary(results, time.time() - t0)


if __name__ == "__main__":
    main()
