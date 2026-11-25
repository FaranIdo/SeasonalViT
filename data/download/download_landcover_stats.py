#!/usr/bin/env python3
"""
Download ESA WorldCover land cover statistics for NDVI tile locations.

Computes per-tile land cover statistics server-side in GEE (no imagery download).
Uses batched parallel processing for efficiency - processes multiple locations
per GEE request.

Output: Single JSON file with statistics for all locations.

ESA WorldCover v200 (2021) classes:
    10: Tree cover
    20: Shrubland
    30: Grassland
    40: Cropland
    50: Built-up
    60: Bare/sparse vegetation
    70: Snow/ice
    80: Permanent water bodies
    90: Herbaceous wetland
    95: Mangroves
    100: Moss and lichen

Usage:
    # Test with 100 locations
    python download_landcover_stats.py --project PROJECT_ID --locations locations.json --test 100

    # Full download (all 250K locations)
    python download_landcover_stats.py --project PROJECT_ID --locations locations.json

    # Resume interrupted download
    python download_landcover_stats.py --project PROJECT_ID --locations locations.json --resume
"""

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import ee

# Tile configuration (must match NDVI download settings)
TILE_SIZE_PIXELS = 264
SCALE = 30  # NDVI resolution
WORLDCOVER_SCALE = 10  # ESA WorldCover resolution

# Batch processing configuration
BATCH_SIZE = 200  # Locations per GEE request
WORKERS = 20  # Parallel workers for batch processing
MAX_RETRIES = 3

# ESA WorldCover class definitions
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow/ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# Vegetation classes (for computing vegetation_fraction)
VEGETATION_CLASSES = {10, 20, 30, 40, 90, 95, 100}


def create_tile_geometry(lat: float, lon: float) -> ee.Geometry:
    """Create tile geometry matching NDVI tiles (264x264 pixels at 30m)."""
    half_size_deg = (TILE_SIZE_PIXELS * SCALE) / 111320.0 / 2
    return ee.Geometry.Rectangle([
        lon - half_size_deg,
        lat - half_size_deg,
        lon + half_size_deg,
        lat + half_size_deg
    ])


def compute_batch_stats(
    batch: dict[str, tuple[float, float]],
    worldcover: ee.Image
) -> dict:
    """
    Compute land cover statistics for a batch of locations.

    Args:
        batch: Dictionary mapping sample_id to (lat, lon) tuples
        worldcover: ESA WorldCover image

    Returns:
        Dictionary mapping sample_id to statistics
    """
    # Create FeatureCollection from batch
    features = []
    for sample_id, (lat, lon) in batch.items():
        geom = create_tile_geometry(lat, lon)
        feature = ee.Feature(geom, {
            'sample_id': sample_id,
            'lat': lat,
            'lon': lon
        })
        features.append(feature)

    fc = ee.FeatureCollection(features)

    # Map histogram computation over all features
    def compute_histogram(feature):
        histogram = worldcover.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=feature.geometry(),
            scale=WORLDCOVER_SCALE,
            maxPixels=1e8
        )
        return feature.set('histogram', histogram.get('Map'))

    stats_fc = fc.map(compute_histogram)

    # Get results from server
    results = stats_fc.getInfo()

    # Process results into output format
    output = {}
    for feature in results['features']:
        props = feature['properties']
        sample_id = props['sample_id']
        lat = props['lat']
        lon = props['lon']
        histogram = props.get('histogram', {})

        if not histogram:
            # No data for this location
            output[sample_id] = {
                'dominant_class': None,
                'dominant_name': None,
                'fractions': {},
                'vegetation_fraction': 0.0,
                'lat': lat,
                'lon': lon,
                'error': 'No WorldCover data'
            }
            continue

        # Convert histogram to fractions
        total_pixels = sum(histogram.values())
        fractions = {}
        for class_code_str, count in histogram.items():
            class_code = int(class_code_str)
            fractions[class_code] = round(count / total_pixels, 4)

        # Find dominant class
        dominant_class = max(histogram.keys(), key=lambda k: histogram[k])
        dominant_class = int(dominant_class)
        dominant_name = WORLDCOVER_CLASSES.get(dominant_class, "Unknown")

        # Compute vegetation fraction
        vegetation_fraction = sum(
            fractions.get(c, 0) for c in VEGETATION_CLASSES
        )

        output[sample_id] = {
            'dominant_class': dominant_class,
            'dominant_name': dominant_name,
            'fractions': fractions,
            'vegetation_fraction': round(vegetation_fraction, 4),
            'lat': lat,
            'lon': lon
        }

    return output


def process_batch_with_retry(
    batch: dict[str, tuple[float, float]],
    worldcover: ee.Image,
    batch_idx: int,
    total_batches: int
) -> tuple[int, dict, Optional[str]]:
    """
    Process a batch with retry logic.

    Returns:
        Tuple of (batch_idx, results_dict, error_message_or_None)
    """
    for attempt in range(MAX_RETRIES):
        try:
            results = compute_batch_stats(batch, worldcover)
            return (batch_idx, results, None)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return (batch_idx, {}, str(e))

    return (batch_idx, {}, "Max retries exceeded")


def load_locations(locations_path: str) -> dict:
    """Load locations from JSON file."""
    with open(locations_path, 'r') as f:
        return json.load(f)


def save_progress(
    progress_file: Path,
    completed_batches: set,
    all_stats: dict,
    failed_batches: dict
):
    """Save progress for resume capability."""
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=progress_file.parent,
            prefix='.progress_tmp_',
            suffix='.json',
            delete=False
        ) as f:
            json.dump({
                'completed_batches': list(completed_batches),
                'stats': all_stats,
                'failed_batches': failed_batches,
                'timestamp': datetime.now().isoformat()
            }, f)
            f.flush()
            os.fsync(f.fileno())
            temp_path = f.name
        os.replace(temp_path, progress_file)
    except Exception:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise


def load_progress(progress_file: Path) -> tuple[set, dict, dict]:
    """Load previous progress if exists."""
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return (
                set(data.get('completed_batches', [])),
                data.get('stats', {}),
                data.get('failed_batches', {})
            )
    return set(), {}, {}


def create_batches(
    locations: dict,
    batch_size: int
) -> list[dict[str, tuple[float, float]]]:
    """Split locations into batches."""
    items = list(locations.items())
    batches = []
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        batch = {sid: tuple(coords) for sid, coords in batch_items}
        batches.append(batch)
    return batches


def run_download(
    locations_path: str,
    output_path: str,
    batch_size: int = BATCH_SIZE,
    workers: int = WORKERS,
    limit: Optional[int] = None,
    resume: bool = False
):
    """Run the land cover statistics download."""
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("Note: Install tqdm for progress bar")

    # Setup paths
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file = output_file.parent / '_progress_landcover.json'

    # Load locations
    print(f"\nLoading locations from: {locations_path}")
    locations = load_locations(locations_path)
    total_locations = len(locations)
    print(f"Total locations: {total_locations:,}")

    # Apply limit if specified
    if limit:
        items = list(locations.items())[:limit]
        locations = dict(items)
        print(f"Limited to: {len(locations):,} locations")

    # Load previous progress if resuming
    if resume:
        completed_batches, all_stats, failed_batches = load_progress(progress_file)
        print(f"Resuming: {len(completed_batches):,} batches completed")
        print(f"Stats collected: {len(all_stats):,} locations")
    else:
        completed_batches = set()
        all_stats = {}
        failed_batches = {}

    # Create batches
    batches = create_batches(locations, batch_size)
    total_batches = len(batches)
    print(f"\nBatch configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches: {total_batches:,}")
    print(f"  Workers: {workers}")

    # Filter out completed batches
    batches_to_process = [
        (idx, batch) for idx, batch in enumerate(batches)
        if idx not in completed_batches
    ]

    if not batches_to_process:
        print("\nAll batches already completed!")
        # Write final output
        with open(output_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"Output saved to: {output_file}")
        return

    print(f"  Batches to process: {len(batches_to_process):,}")

    # Initialize WorldCover image
    print("\nLoading ESA WorldCover v200...")
    worldcover = ee.ImageCollection('ESA/WorldCover/v200').first()
    print("  Ready")

    # Process batches in parallel
    print(f"\nProcessing {len(batches_to_process):,} batches...")
    start_time = time.time()
    success_count = 0
    fail_count = 0

    if use_tqdm:
        pbar = tqdm(total=len(batches_to_process), desc="Batches", unit="batch")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_batch_with_retry,
                batch,
                worldcover,
                idx,
                total_batches
            ): idx
            for idx, batch in batches_to_process
        }

        for future in as_completed(futures):
            batch_idx, results, error = future.result()

            if error:
                fail_count += 1
                failed_batches[batch_idx] = error
                print(f"\n[FAILED] Batch {batch_idx}: {error[:100]}")
            else:
                success_count += 1
                all_stats.update(results)
                completed_batches.add(batch_idx)

            if use_tqdm:
                pbar.update(1)
                pbar.set_postfix({
                    'ok': success_count,
                    'fail': fail_count,
                    'locs': len(all_stats)
                })

            # Save progress periodically
            processed = success_count + fail_count
            if processed % 10 == 0:
                save_progress(progress_file, completed_batches, all_stats, failed_batches)

    if use_tqdm:
        pbar.close()

    # Final save
    save_progress(progress_file, completed_batches, all_stats, failed_batches)

    # Write final output
    print(f"\nWriting output to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(all_stats, f, indent=2)

    # Report
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Batches processed: {success_count:,}")
    print(f"  Batches failed: {fail_count:,}")
    print(f"  Locations with stats: {len(all_stats):,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Rate: {len(all_stats) / elapsed:.1f} locations/sec")
    print(f"  Output: {output_file}")

    if fail_count > 0:
        print(f"\nFailed batches saved to: {progress_file}")
        print("Re-run with --resume to retry failed batches.")

    # Print class distribution summary
    print(f"\n{'=' * 60}")
    print("LAND COVER DISTRIBUTION")
    print(f"{'=' * 60}")

    class_counts = {}
    for stats in all_stats.values():
        dominant = stats.get('dominant_class')
        if dominant is not None:
            class_counts[dominant] = class_counts.get(dominant, 0) + 1

    for class_code in sorted(class_counts.keys()):
        count = class_counts[class_code]
        pct = 100 * count / len(all_stats) if all_stats else 0
        name = WORLDCOVER_CLASSES.get(class_code, "Unknown")
        print(f"  {class_code:3d} ({name:25s}): {count:6,} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Download ESA WorldCover land cover statistics for NDVI locations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with 1000 locations
    python download_landcover_stats.py --project PROJECT_ID --locations locations.json --test 1000

    # Full download
    python download_landcover_stats.py --project PROJECT_ID --locations locations.json

    # Resume interrupted download
    python download_landcover_stats.py --project PROJECT_ID --locations locations.json --resume
        """
    )

    parser.add_argument(
        '--project', type=str, required=True,
        help='Google Cloud project ID for Earth Engine'
    )
    parser.add_argument(
        '--service-account', type=str, default=None,
        help='Path to service account JSON key file'
    )
    parser.add_argument(
        '--locations', type=str, required=True,
        help='Path to locations.json file'
    )
    parser.add_argument(
        '--output', type=str, default='data/landcover_stats.json',
        help='Output JSON file path (default: data/landcover_stats.json)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=BATCH_SIZE,
        help=f'Locations per GEE request (default: {BATCH_SIZE})'
    )
    parser.add_argument(
        '--workers', type=int, default=WORKERS,
        help=f'Parallel workers (default: {WORKERS})'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit number of locations to process'
    )
    parser.add_argument(
        '--test', type=int, metavar='N',
        help='Test mode: process only N locations'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from previous progress'
    )

    args = parser.parse_args()

    # Validate locations file
    if not Path(args.locations).exists():
        print(f"Error: Locations file not found: {args.locations}")
        sys.exit(1)

    # Handle test mode
    limit = args.test if args.test else args.limit

    # Initialize Earth Engine
    print("Initializing Earth Engine...")
    print(f"  Project: {args.project}")

    try:
        if args.service_account:
            print(f"  Auth: Service Account ({args.service_account})")
            credentials = ee.ServiceAccountCredentials(None, args.service_account)
            ee.Initialize(credentials=credentials, project=args.project)
        else:
            print(f"  Auth: OAuth")
            ee.Initialize(project=args.project)
        print("  Status: Connected")
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        sys.exit(1)

    # Run download
    run_download(
        locations_path=args.locations,
        output_path=args.output,
        batch_size=args.batch_size,
        workers=args.workers,
        limit=limit,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
