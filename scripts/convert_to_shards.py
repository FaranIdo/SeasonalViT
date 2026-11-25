"""Convert GeoTIFF NDVI data to pre-processed float16 shards for fast loading.

This script pre-processes all GeoTIFF files into efficient PyTorch tensor shards:
- Center crops from 264×264 to 256×256
- Fills nodata (-9999) with zeros
- Converts to float16 (halves storage)
- Packs into shards for sequential reading
- Creates split-aware subdirectories (train/, val_iid/, etc.) when --splits provided

Usage:
    # With splits (recommended):
    python scripts/convert_to_shards.py \
        --input-dir /path/to/geotiffs \
        --output-dir /path/to/shards \
        --splits /path/to/splits.json \
        --samples-per-shard 1000 \
        --workers 8 \
        --parallel-shards 4

    # Without splits (all files in one directory):
    python scripts/convert_to_shards.py \
        --input-dir /path/to/geotiffs \
        --output-dir /path/to/shards \
        --samples-per-shard 1000 \
        --workers 8 \
        --parallel-shards 4
"""

import argparse
import gc
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure GDAL environment BEFORE importing rasterio
# This must happen at module level to take effect reliably
os.environ["GDAL_NUM_THREADS"] = "1"  # Single-threaded GDAL per worker
os.environ["GDAL_CACHEMAX"] = "32"  # Limit block cache to 32MB per worker
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"  # Don't scan directories
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"  # Limit extensions

import numpy as np
import torch
from tqdm import tqdm

# How often to run gc.collect() in worker processes
GC_INTERVAL = 20


# Regex to parse filename: {sample_id}_{lat}_{lon}.tif
FILENAME_PATTERN = re.compile(r"^(\d+)_(-?\d+)d(\d+)_(-?\d+)d(\d+)\.tif$")


def parse_coordinates(filename: str) -> Tuple[float, float]:
    """Parse latitude and longitude from filename."""
    match = FILENAME_PATTERN.match(filename)
    if match is None:
        raise ValueError(f"Invalid filename format: {filename}")

    lat_int, lat_dec, lon_int, lon_dec = match.groups()[1:]
    lat = float(f"{lat_int}.{lat_dec}")
    lon = float(f"{lon_int}.{lon_dec}")
    return lat, lon


# Process-local counter for gc.collect() scheduling
_worker_file_count = 0

# Global for pipelined saving - tracks pending save thread
_pending_save_thread = None
_pending_save_lock = threading.Lock()


def load_and_preprocess(path_str: str, crop_size: int = 256, nodata_value: float = -9999.0) -> Optional[Tuple[np.ndarray, np.ndarray, float, float, str]]:
    """Load GeoTIFF, preprocess, and return numpy arrays.

    Returns numpy arrays instead of tensors to avoid pickling issues with multiprocessing.

    Returns:
        Tuple of (images, nodata_mask, lat, lon, filename) or None if error
        - images: (140, 256, 256) float16 numpy array
        - nodata_mask: (140, 256, 256) bool numpy array
        - lat, lon: coordinates
        - filename: original filename
    """
    global _worker_file_count
    import rasterio

    path = Path(path_str)
    src = None
    try:
        # Open and read data
        src = rasterio.open(path)
        data_raw = src.read()  # (140, 264, 264)
        src.close()  # Explicitly close before processing
        src = None

        # Create nodata mask before filling
        nodata_mask_full = data_raw == nodata_value

        # Fill nodata with zeros
        data_filled = np.where(nodata_mask_full, 0.0, data_raw)
        del data_raw  # Free original array

        # Center crop
        _, h, w = data_filled.shape
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        data_cropped = data_filled[:, start_h:start_h + crop_size, start_w:start_w + crop_size].copy()
        nodata_mask = nodata_mask_full[:, start_h:start_h + crop_size, start_w:start_w + crop_size].copy()
        del data_filled, nodata_mask_full  # Free uncropped arrays

        # Convert to float16
        data = data_cropped.astype(np.float16)
        del data_cropped  # Free float32 version

        # Parse coordinates
        lat, lon = parse_coordinates(path.name)

        # Periodic garbage collection to prevent memory buildup
        _worker_file_count += 1
        if _worker_file_count % GC_INTERVAL == 0:
            gc.collect()

        return data, nodata_mask, lat, lon, path.name

    except Exception as e:
        print(f"\nError processing {path}: {e}")
        return None
    finally:
        # Ensure file handle is closed even on error
        if src is not None:
            try:
                src.close()
            except Exception:
                pass


def _wait_for_pending_save():
    """Wait for any pending background save to complete."""
    global _pending_save_thread
    with _pending_save_lock:
        if _pending_save_thread is not None:
            _pending_save_thread.join()
            _pending_save_thread = None


def _save_shard_background(shard_data, shard_path, shard_idx, num_samples, progress_lock, progress_counter):
    """Save shard to disk (runs in background thread)."""
    t0 = time.perf_counter()
    torch.save(shard_data, shard_path)
    t_save = time.perf_counter() - t0
    print(f"Shard {shard_idx:03d}:   Save: {t_save:.2f}s")

    with progress_lock:
        progress_counter["completed"] += 1
        progress_counter["samples"] += num_samples
        print(f"Shard {shard_idx:03d}: {num_samples} samples saved")


def process_shard(
    shard_idx: int,
    batch_paths: List[str],
    output_dir: Path,
    num_workers: int,
    progress_lock: threading.Lock,
    progress_counter: Dict[str, int],
) -> Optional[Dict]:
    """Process a single shard: load files, preprocess, save.

    Uses pipelined saving: saves run in background while next shard loads.

    Args:
        shard_idx: Index of this shard.
        batch_paths: List of file paths for this shard.
        output_dir: Output directory.
        num_workers: Workers for file loading.
        progress_lock: Lock for thread-safe progress updates.
        progress_counter: Shared counter for progress tracking.

    Returns:
        Shard metadata dict or None if shard already exists.
    """
    global _pending_save_thread

    shard_path = output_dir / f"shard_{shard_idx:05d}.pt"

    # Skip if already exists (resume support)
    if shard_path.exists():
        with progress_lock:
            progress_counter["skipped"] += 1
            print(f"Shard {shard_idx:03d}: skipped (exists)")
        return {"path": shard_path.name, "num_samples": len(batch_paths), "existed": True}

    print(f"Shard {shard_idx:03d}: starting load...")

    shard_images = []
    shard_masks = []
    shard_lats = []
    shard_lons = []
    shard_files = []

    # Process files in parallel within this shard
    # max_tasks_per_child=None disables worker recycling to avoid Windows pipe issues
    # Workers are fresh per shard anyway since ProcessPoolExecutor is created per shard
    t_load_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers, max_tasks_per_child=None) as executor:
        futures = {executor.submit(load_and_preprocess, p): p for p in batch_paths}

        # Simple progress printing (no tqdm \r overwriting)
        done = 0
        total = len(batch_paths)
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                images, nodata_mask, lat, lon, filename = result
                shard_images.append(images)
                shard_masks.append(nodata_mask)
                shard_lats.append(lat)
                shard_lons.append(lon)
                shard_files.append(filename)
            done += 1
            if done % 100 == 0 or done == total:
                print(f"Shard {shard_idx:03d}: {done}/{total} ({100*done//total}%)")
    t_load = time.perf_counter() - t_load_start

    if len(shard_images) == 0:
        return None

    print(f"Shard {shard_idx:03d}: Loaded {len(shard_images)} files in {t_load:.1f}s")

    # Wait for previous shard's save AFTER loading (pipelined I/O)
    _wait_for_pending_save()

    print(f"Shard {shard_idx:03d}: post-processing...")

    # Sort by filename to maintain order
    t0 = time.perf_counter()
    sorted_indices = sorted(range(len(shard_files)), key=lambda i: shard_files[i])
    shard_images = [shard_images[i] for i in sorted_indices]
    shard_masks = [shard_masks[i] for i in sorted_indices]
    shard_lats = [shard_lats[i] for i in sorted_indices]
    shard_lons = [shard_lons[i] for i in sorted_indices]
    shard_files = [shard_files[i] for i in sorted_indices]
    t_sort = time.perf_counter() - t0
    print(f"Shard {shard_idx:03d}:   Sort: {t_sort:.2f}s")

    # Stack arrays into tensors
    t0 = time.perf_counter()
    images_tensor = torch.from_numpy(np.stack(shard_images))
    masks_tensor = torch.from_numpy(np.stack(shard_masks))
    t_stack = time.perf_counter() - t0
    print(f"Shard {shard_idx:03d}:   Stack: {t_stack:.2f}s")

    # Save count before freeing lists
    num_samples = len(shard_images)

    # Free lists to reduce memory before save
    del shard_images, shard_masks

    # Prepare shard data for saving
    shard_data = {
        "images": images_tensor,
        "nodata_masks": masks_tensor,
        "lats": torch.tensor(shard_lats, dtype=torch.float32),
        "lons": torch.tensor(shard_lons, dtype=torch.float32),
        "files": shard_files,
    }

    # Start save in background thread (pipelined with next shard's loading)
    print(f"Shard {shard_idx:03d}:   Starting background save...")
    with _pending_save_lock:
        _pending_save_thread = threading.Thread(
            target=_save_shard_background,
            args=(shard_data, shard_path, shard_idx, num_samples, progress_lock, progress_counter),
        )
        _pending_save_thread.start()

    return {"path": shard_path.name, "num_samples": num_samples, "existed": False}


def convert_single_split(
    tif_files: List[Path],
    output_dir: Path,
    samples_per_shard: int,
    num_workers: int,
    parallel_shards: int = 4,
    split_name: Optional[str] = None,
) -> None:
    """Convert a single split's GeoTIFFs to sharded format.

    Args:
        tif_files: List of GeoTIFF file paths to process.
        output_dir: Output directory for shards and metadata.
        samples_per_shard: Number of samples per shard file.
        num_workers: Number of parallel workers per shard.
        parallel_shards: Number of shards to process concurrently.
        split_name: Optional name of split (for logging).
    """
    num_files = len(tif_files)
    split_label = f" ({split_name})" if split_name else ""
    print(f"\n{'='*60}")
    print(f"Converting {num_files} files{split_label}")
    print(f"Output: {output_dir}")
    print(f"Workers per shard: {num_workers}")
    print(f"Parallel shards: {parallel_shards}")
    print(f"{'='*60}")

    if num_files == 0:
        print("No files found!")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate shard assignments
    num_shards = (num_files + samples_per_shard - 1) // samples_per_shard
    print(f"Creating {num_shards} shards with up to {samples_per_shard} samples each\n")

    # Convert paths to strings
    tif_paths = [str(p) for p in tif_files]

    # Build list of (shard_idx, batch_paths) tuples
    shard_batches = []
    for shard_idx in range(num_shards):
        batch_start = shard_idx * samples_per_shard
        batch_end = min(batch_start + samples_per_shard, num_files)
        batch_paths = tif_paths[batch_start:batch_end]
        shard_batches.append((shard_idx, batch_paths))

    # Shared progress tracking
    progress_lock = threading.Lock()
    progress_counter = {"completed": 0, "skipped": 0, "samples": 0}

    # Process shards in parallel using threads
    shard_results = [None] * num_shards

    with ThreadPoolExecutor(max_workers=parallel_shards) as executor:
        futures = {
            executor.submit(
                process_shard,
                shard_idx,
                batch_paths,
                output_dir,
                num_workers,
                progress_lock,
                progress_counter,
            ): shard_idx
            for shard_idx, batch_paths in shard_batches
        }

        for future in as_completed(futures):
            shard_idx = futures[future]
            try:
                result = future.result()
                shard_results[shard_idx] = result
            except Exception as e:
                print(f"Error processing shard {shard_idx}: {e}")

    # Wait for final pipelined save to complete
    print("Waiting for final save to complete...")
    _wait_for_pending_save()

    # Build metadata from results
    total_samples = 0
    shards_metadata = []
    for shard_idx, result in enumerate(shard_results):
        if result is not None:
            shards_metadata.append({
                "path": result["path"],
                "num_samples": result["num_samples"],
                "start_idx": total_samples,
            })
            total_samples += result["num_samples"]

    metadata = {
        "num_samples": total_samples,
        "num_shards": len(shards_metadata),
        "samples_per_shard": samples_per_shard,
        "shape": [140, 256, 256],
        "dtype": "float16",
        "shards": shards_metadata,
    }

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nConversion complete{split_label}!")
    print(f"  Shards: {output_dir}")
    print(f"  Samples: {total_samples}")
    print(f"  Metadata: {metadata_path}")

    # Calculate size
    total_size = sum(f.stat().st_size for f in output_dir.glob("shard_*.pt"))
    print(f"  Total size: {total_size / 1e9:.2f} GB")


def convert_to_shards(
    input_dir: Path,
    output_dir: Path,
    splits_path: Optional[Path],
    split_only: Optional[str] = None,
    samples_per_shard: int = 1000,
    num_workers: int = 8,
    parallel_shards: int = 4,
) -> None:
    """Convert all GeoTIFFs to sharded format, with optional split-aware organization.

    Args:
        input_dir: Directory containing GeoTIFF files.
        output_dir: Output directory for shards.
        splits_path: Optional path to splits.json for split-aware sharding.
        split_only: Optional name of single split to process (e.g., "train").
        samples_per_shard: Number of samples per shard file.
        num_workers: Number of parallel workers per shard.
        parallel_shards: Number of shards to process concurrently.
    """
    if splits_path is not None:
        # Split-aware conversion
        print(f"Loading splits from {splits_path}")
        with open(splits_path) as f:
            splits_data = json.load(f)

        # Get source directory from splits (use input_dir as fallback)
        source_dir = Path(splits_data.get("source_dir") or input_dir)

        # Filter to single split if requested
        splits_to_process = splits_data["splits"]
        if split_only:
            if split_only not in splits_to_process:
                print(f"Split '{split_only}' not found. Available: {list(splits_to_process.keys())}")
                sys.exit(1)
            splits_to_process = {split_only: splits_to_process[split_only]}

        total_splits = len(splits_to_process)
        for idx, (split_name, split_info) in enumerate(splits_to_process.items(), 1):
            print(f"\nProcessing split {idx}/{total_splits}: {split_name}")

            # Build full paths from filenames
            split_files = [
                source_dir / filename
                for filename in split_info["files"]
            ]

            # Filter to only existing files
            existing_files = [f for f in split_files if f.exists()]
            if len(existing_files) < len(split_files):
                missing = len(split_files) - len(existing_files)
                print(f"Warning: {missing} files not found in {source_dir}")

            if len(existing_files) == 0:
                print(f"Skipping {split_name}: no files found")
                continue

            # Create split-specific output directory
            split_output_dir = output_dir / split_name

            # Convert this split
            convert_single_split(
                tif_files=sorted(existing_files),
                output_dir=split_output_dir,
                samples_per_shard=samples_per_shard,
                num_workers=num_workers,
                parallel_shards=parallel_shards,
                split_name=split_name,
            )

        print(f"\n{'='*60}")
        print("All splits converted successfully!")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")

    else:
        # Convert all files without split organization
        tif_files = sorted(input_dir.glob("*.tif"))
        print(f"Found {len(tif_files)} GeoTIFF files in {input_dir}")

        if len(tif_files) == 0:
            print("No files found!")
            return

        convert_single_split(
            tif_files=tif_files,
            output_dir=output_dir,
            samples_per_shard=samples_per_shard,
            num_workers=num_workers,
            parallel_shards=parallel_shards,
            split_name=None,
        )


def main():
    parser = argparse.ArgumentParser(description="Convert GeoTIFFs to sharded format")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with GeoTIFF files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for shards")
    parser.add_argument("--splits", type=Path, help="Path to splits.json for split-aware sharding")
    parser.add_argument("--split-only", type=str, help="Only process this split (e.g., train)")
    parser.add_argument("--samples-per-shard", type=int, default=1000, help="Samples per shard file")
    parser.add_argument("--workers", type=int, default=8, help="Workers per shard for file loading")
    parser.add_argument("--parallel-shards", type=int, default=4, help="Number of shards to process concurrently")

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    if args.splits and not args.splits.exists():
        print(f"Splits file not found: {args.splits}")
        sys.exit(1)

    convert_to_shards(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        splits_path=args.splits,
        split_only=args.split_only,
        samples_per_shard=args.samples_per_shard,
        num_workers=args.workers,
        parallel_shards=args.parallel_shards,
    )


if __name__ == "__main__":
    main()
