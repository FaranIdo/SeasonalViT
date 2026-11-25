#!/usr/bin/env python3
"""
Download 35 years of NDVI time series (1990-2024) using GEE High-Volume Endpoint.

Downloads seasonal NDVI data (4 seasons/year x 35 years = 140 bands) for SSL4EO-L locations
as GeoTIFF files. Uses raw Landsat imagery (L5/L7/L8/L9) with QA_PIXEL cloud masking for
better quality than pre-computed composites.

Approach:
- Server-side iteration using ee.List.map() for optimal performance
- QA_PIXEL band cloud masking (cirrus, cloud, cloud shadow bits)
- All 4 Landsat collections queried per season (empty ones optimized away)
- 2 chunks for maximum throughput: fewer requests = more parallel samples

Year chunks:
- Chunk 1: 1990-2006 (17 years, 68 bands)
- Chunk 2: 2007-2024 (18 years, 72 bands)

Usage:
    # Test with 10 locations
    python download_ndvi_35years.py --project PROJECT_ID --locations locations.json --test 10

    # Download first 100 locations
    python download_ndvi_35years.py --project PROJECT_ID --locations locations.json --limit 100

    # Full download (resume-capable)
    python download_ndvi_35years.py --project PROJECT_ID --locations locations.json

    # Forward worker (from start)
    python download_ndvi_35years.py --project PROJECT_ID --locations locations.json --limit 50000

    # Backward worker (from end)
    python download_ndvi_35years.py --project PROJECT_ID --locations locations.json --limit 50000 --reverse

Requirements:
    - earthengine-api
    - requests
    - tqdm
    - rasterio
    - numpy
"""

# CRITICAL: Disable SSL verification for Windows multiprocessing workaround
import os
import ssl
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import atexit
import functools
import json
import os as os_module
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

# Force all prints to flush immediately (critical for Windows subprocess logging)
print = functools.partial(print, flush=True)

import ee
import numpy as np
import requests
import rasterio
from rasterio.transform import from_bounds
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# High-Volume Endpoint
HIGHVOLUME_URL = 'https://earthengine-highvolume.googleapis.com'

# 35-year configuration (1990-2024)
FULL_START_YEAR = 1990
FULL_END_YEAR = 2024

# Year chunks optimized for throughput (2 chunks = fewer requests = more parallel samples)
YEAR_CHUNKS = [
    (1990, 2006),  # 17 years, 68 bands
    (2007, 2024),  # 18 years, 72 bands
]

TILE_SIZE_PIXELS = 264
SCALE = 30
CRS = 'EPSG:4326'
OPTIMAL_WORKERS = 40

# Seasonal DOY targets (±45 day window around each date)
# Season    | Target DOY | Target Date | Window
# ----------|------------|-------------|------------------
# spring    | 79         | Mar 20      | Feb 3  - May 4
# summer    | 172        | Jun 21      | May 7  - Aug 5
# fall      | 265        | Sep 22      | Aug 8  - Nov 6
# winter    | 355        | Dec 21      | Nov 6  - Feb 4
SEASONS = ['spring', 'summer', 'fall', 'winter']
SEASON_DOY = [79, 172, 265, 355]

# QA_PIXEL cloud mask bits: cirrus (2), cloud (3), cloud shadow (4)
CLOUD_BITS = (1 << 2) | (1 << 3) | (1 << 4)


def process_l57(image):
    """Mask clouds and compute NDVI for Landsat 5/7."""
    mask = image.select('QA_PIXEL').bitwiseAnd(CLOUD_BITS).eq(0)
    red = image.select('SR_B3').multiply(0.0000275).add(-0.2)
    nir = image.select('SR_B4').multiply(0.0000275).add(-0.2)
    return nir.subtract(red).divide(nir.add(red)).rename('NDVI').updateMask(mask)


def process_l89(image):
    """Mask clouds and compute NDVI for Landsat 8/9."""
    mask = image.select('QA_PIXEL').bitwiseAnd(CLOUD_BITS).eq(0)
    red = image.select('SR_B4').multiply(0.0000275).add(-0.2)
    nir = image.select('SR_B5').multiply(0.0000275).add(-0.2)
    return nir.subtract(red).divide(nir.add(red)).rename('NDVI').updateMask(mask)


def build_seasonal_stack(geometry: ee.Geometry, start_year: int, end_year: int) -> ee.Image:
    """Build NDVI stack using fully server-side iteration with QA cloud masking."""
    # Pre-filter collections by geometry and cloud cover (done once)
    l5_base = (ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
               .filterBounds(geometry)
               .filter(ee.Filter.lt('CLOUD_COVER', 70)))
    l7_base = (ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
               .filterBounds(geometry)
               .filter(ee.Filter.lt('CLOUD_COVER', 70)))
    l8_base = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
               .filterBounds(geometry)
               .filter(ee.Filter.lt('CLOUD_COVER', 70)))
    l9_base = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
               .filterBounds(geometry)
               .filter(ee.Filter.lt('CLOUD_COVER', 70)))

    years = ee.List.sequence(start_year, end_year)
    doys = ee.List(SEASON_DOY)

    def process_year(year):
        year = ee.Number(year)

        def process_season(doy):
            doy = ee.Number(doy)
            start = ee.Date.fromYMD(year, 1, 1).advance(doy.subtract(45), 'day')
            end = ee.Date.fromYMD(year, 1, 1).advance(doy.add(45), 'day')

            # Get collections for this date range (empty if sensor not active)
            l5 = l5_base.filterDate(start, end).map(process_l57)
            l7 = l7_base.filterDate(start, end).map(process_l57)
            l8 = l8_base.filterDate(start, end).map(process_l89)
            l9 = l9_base.filterDate(start, end).map(process_l89)

            # Merge all - empty collections contribute nothing
            merged = l5.merge(l7).merge(l8).merge(l9)

            # Handle empty collections: add a fully-masked nodata image to ensure
            # collection is never empty. The masked image contributes nothing to
            # median(), but avoids ee.Algorithms.If which causes memory issues.
            nodata_masked = ee.Image.constant(-9999).rename('NDVI').updateMask(ee.Image.constant(0))
            merged_safe = merged.merge(ee.ImageCollection([nodata_masked]))

            return merged_safe.median().unmask(-9999).clip(geometry)

        return doys.map(process_season)

    # Build all bands server-side
    bands = years.map(process_year).flatten()

    return ee.ImageCollection(bands).toBands()


def get_band_names(start_year: int, end_year: int) -> list:
    """Generate ordered list of band names matching toBands() output."""
    names = []
    for year in range(start_year, end_year + 1):
        for season in SEASONS:
            names.append(f'{year}_{season}')
    return names


def create_tile_geometry(lat: float, lon: float) -> ee.Geometry:
    """Create 264x264 pixel geometry centered at lat/lon."""
    half_size_deg = (TILE_SIZE_PIXELS * SCALE) / 111320.0 / 2
    return ee.Geometry.Rectangle([
        lon - half_size_deg,
        lat - half_size_deg,
        lon + half_size_deg,
        lat + half_size_deg
    ])


def format_filename(sample_id: str, lat: float, lon: float) -> str:
    """Create filename with sample_id and coordinates."""
    lat_str = f"{lat:.4f}".replace('.', 'd')
    lon_str = f"{lon:.4f}".replace('.', 'd')
    return f"{sample_id}_{lat_str}_{lon_str}.tif"


def download_chunk(
    geometry: ee.Geometry,
    start_year: int,
    end_year: int,
    max_retries: int = 3
) -> Optional[np.ndarray]:
    """Download a single year chunk and return numpy array."""
    for attempt in range(max_retries):
        try:
            image = build_seasonal_stack(geometry, start_year, end_year)

            url = image.getDownloadURL({
                'scale': SCALE,
                'region': geometry,
                'format': 'GEO_TIFF',
                'crs': CRS,
            })

            response = requests.get(url, timeout=300, verify=False)
            response.raise_for_status()

            # Read GeoTIFF from bytes into numpy array
            with rasterio.open(BytesIO(response.content)) as src:
                data = src.read()  # Shape: (bands, height, width)
                profile = src.profile.copy()

            return data, profile

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise e

    return None, None


def download_single_location_35years(
    sample_id: str,
    lat: float,
    lon: float,
    output_dir: Path,
    band_names: list,
    max_retries: int = 3
) -> dict:
    """Download all 41 years by making 3 API calls and stacking results."""
    filename = format_filename(sample_id, lat, lon)
    output_path = output_dir / filename

    # Skip if already exists
    if output_path.exists():
        return {
            'sample_id': sample_id,
            'status': 'skipped',
            'message': 'Already exists',
            'path': str(output_path)
        }

    try:
        geometry = create_tile_geometry(lat, lon)

        # Download each chunk
        chunks_data = []
        profile = None

        for chunk_start, chunk_end in YEAR_CHUNKS:
            data, chunk_profile = download_chunk(
                geometry, chunk_start, chunk_end, max_retries
            )
            if data is None:
                return {
                    'sample_id': sample_id,
                    'status': 'failed',
                    'error': f'Failed to download chunk {chunk_start}-{chunk_end}'
                }
            chunks_data.append(data)
            if profile is None:
                profile = chunk_profile

        # Stack all chunks: concatenate along band axis
        full_data = np.concatenate(chunks_data, axis=0)  # Shape: (164, height, width)

        # Update profile for full stack
        profile.update(
            count=full_data.shape[0],
            dtype=full_data.dtype
        )

        # Atomic file write: write to temp file, then rename
        temp_path = output_dir / f'.tmp_{filename}'
        try:
            with rasterio.open(temp_path, 'w', **profile) as dst:
                dst.write(full_data)
                # Write band descriptions
                for i, name in enumerate(band_names, start=1):
                    dst.set_band_description(i, name)

            # Ensure data is synced to disk
            os_module.sync() if hasattr(os_module, 'sync') else None

            # Atomic rename
            temp_path.rename(output_path)

        except FileExistsError:
            # Another worker already created this file
            temp_path.unlink(missing_ok=True)
            return {
                'sample_id': sample_id,
                'status': 'skipped',
                'message': 'Already exists (race)',
                'path': str(output_path)
            }
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        return {
            'sample_id': sample_id,
            'status': 'success',
            'size_bytes': output_path.stat().st_size,
            'path': str(output_path)
        }

    except Exception as e:
        return {
            'sample_id': sample_id,
            'status': 'failed',
            'error': str(e)
        }


def load_locations(locations_path: str) -> dict:
    """Load locations from JSON file."""
    with open(locations_path, 'r') as f:
        return json.load(f)


def save_progress(progress_file: Path, completed: set, failed: dict):
    """Save download progress for resume capability using atomic write."""
    progress_dir = progress_file.parent
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=progress_dir,
            prefix='.progress_tmp_',
            suffix='.json',
            delete=False
        ) as f:
            json.dump({
                'completed': list(completed),
                'failed': failed,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
            f.flush()
            os_module.fsync(f.fileno())
            temp_path = f.name
        os_module.replace(temp_path, progress_file)
    except Exception:
        try:
            os_module.unlink(temp_path)
        except Exception:
            pass
        raise


def load_progress(progress_file: Path) -> tuple[set, dict]:
    """Load previous progress if exists."""
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return set(data.get('completed', [])), data.get('failed', {})
    return set(), {}


def run_download(
    locations_path: str,
    output_dir: str,
    workers: int = OPTIMAL_WORKERS,
    limit: Optional[int] = None,
    test_mode: bool = False,
    report_interval: int = 1,
    worker_id: Optional[int] = None,
    num_workers: Optional[int] = None,
    reverse: bool = False,
    skip_first: int = 0
):
    """Run the download process."""
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("Note: Install tqdm for progress bar (pip install tqdm)")

    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use worker-specific progress file if worker_id provided
    if worker_id is not None:
        progress_file = output_path / f'_progress_worker{worker_id}.json'
    else:
        progress_file = output_path / '_progress_35years.json'

    # Load locations
    print(f"\nLoading locations from: {locations_path}")
    locations = load_locations(locations_path)
    total_locations = len(locations)
    print(f"Total locations: {total_locations:,}")

    # Load previous progress
    completed, failed = load_progress(progress_file)
    print(f"Previously completed: {len(completed):,}")
    print(f"Previously failed: {len(failed):,}")

    # Get all location items as a list (preserve order for consistent range slicing)
    all_items = list(locations.items())

    # Apply skip-first BEFORE limit to define the starting position
    # This ensures ranges are calculated from the full list, not filtered by completed
    if skip_first > 0:
        if skip_first >= len(all_items):
            print(f"\nError: --skip-first {skip_first} is >= total locations {len(all_items)}")
            return
        all_items = all_items[skip_first:]
        print(f"Skipping first {skip_first:,} locations, starting from index {skip_first:,}")

    # Apply limit to define the range size (from the skip position)
    if limit:
        all_items = all_items[:limit]
        print(f"Limited to {len(all_items):,} locations (range: {skip_first:,} to {skip_first + len(all_items) - 1:,})")

    # Apply worker partitioning (round-robin across the selected range)
    if worker_id is not None and num_workers is not None:
        all_items = [all_items[i] for i in range(len(all_items)) if i % num_workers == worker_id]
        print(f"Worker {worker_id}/{num_workers}: assigned {len(all_items):,} samples")

    # Filter out completed items (they would be skipped anyway, but no need to queue them)
    to_download = {sid: coords for sid, coords in all_items if sid not in completed}

    if not to_download:
        print("\nAll locations already downloaded!")
        return

    # Calculate stats
    num_years = FULL_END_YEAR - FULL_START_YEAR + 1
    num_bands = num_years * 4
    print(f"\nDownload configuration:")
    print(f"  Year range: {FULL_START_YEAR}-{FULL_END_YEAR} ({num_years} years)")
    print(f"  Bands per file: {num_bands}")
    print(f"  Year chunks: {YEAR_CHUNKS}")
    print(f"  Tile size: {TILE_SIZE_PIXELS}x{TILE_SIZE_PIXELS} pixels")
    print(f"  Workers: {workers}")
    print(f"  To download: {len(to_download):,}")
    print(f"  Output: {output_path}")

    if test_mode:
        print(f"\n[TEST MODE] Downloading {len(to_download)} locations...")
    else:
        print(f"\nStarting download...")

    # Generate band names once (same for all files)
    band_names = get_band_names(FULL_START_YEAR, FULL_END_YEAR)

    # Download
    start_time = time.time()
    success_count = 0
    skip_count = 0
    fail_count = 0

    items = list(to_download.items())
    if reverse:
        items = list(reversed(items))
        print("  Processing in REVERSE order")

    total_items = len(items)

    # Use tqdm only if report_interval is 1
    use_progress_bar = use_tqdm and report_interval == 1

    if use_progress_bar:
        pbar = tqdm(total=total_items, desc="Downloading", unit="loc")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_single_location_35years,
                sample_id, coords[0], coords[1],
                output_path, band_names
            ): sample_id
            for sample_id, coords in items
        }

        for future in as_completed(futures):
            result = future.result()
            sample_id = result['sample_id']

            if result['status'] == 'success':
                success_count += 1
                completed.add(sample_id)
                print(f"[OK] {sample_id}")
            elif result['status'] == 'skipped':
                skip_count += 1
                completed.add(sample_id)
            else:
                fail_count += 1
                error_msg = result.get('error', 'Unknown error')
                failed[sample_id] = error_msg
                # Log failure immediately
                print(f"\n[FAILED] {sample_id}: {error_msg[:100]}")

            processed = success_count + skip_count + fail_count
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta_sec = (total_items - processed) / rate if rate > 0 else 0

            if use_progress_bar:
                pbar.update(1)
                pbar.set_postfix({
                    'ok': success_count,
                    'skip': skip_count,
                    'fail': fail_count,
                    'samples_s': f'{rate:.1f}',
                    'eta': f'{eta_sec/3600:.1f}h' if eta_sec > 3600 else f'{eta_sec/60:.0f}m'
                })
            elif processed % report_interval == 0 or processed == total_items:
                eta_str = f"{eta_sec/3600:.1f}h" if eta_sec > 3600 else f"{eta_sec/60:.1f}m"
                print(f"[{processed:,}/{total_items:,}] ok={success_count:,} skip={skip_count:,} fail={fail_count:,} | {rate:.2f} loc/s | ETA: {eta_str}")

            # Save progress periodically
            if processed % 100 == 0:
                save_progress(progress_file, completed, failed)

    if use_progress_bar:
        pbar.close()

    # Final save
    save_progress(progress_file, completed, failed)

    # Report
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Success: {success_count:,}")
    print(f"  Skipped: {skip_count:,}")
    print(f"  Failed: {fail_count:,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Rate: {(success_count + skip_count) / elapsed:.2f} locations/sec")
    print(f"  Output: {output_path}")

    if fail_count > 0:
        print(f"\nFailed samples saved to: {progress_file}")
        print("Re-run the script to retry failed downloads.")


def main():
    parser = argparse.ArgumentParser(
        description='Download 35 years of NDVI time series (1990-2024) via GEE High-Volume Endpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with 10 locations
    python download_ndvi_35years.py --project PROJECT_ID --locations locations.json --test 10

    # Download first 1000 locations
    python download_ndvi_35years.py --project PROJECT_ID --locations locations.json --limit 1000

    # Forward worker
    python download_ndvi_35years.py --project PROJECT_ID --locations locations.json --limit 50000

    # Backward worker
    python download_ndvi_35years.py --project PROJECT_ID --locations locations.json --limit 50000 --reverse

    # Resume interrupted download
    python download_ndvi_35years.py --project PROJECT_ID --locations locations.json
        """
    )

    parser.add_argument(
        '--project', type=str, required=True,
        help='Google Cloud project ID for Earth Engine'
    )
    parser.add_argument(
        '--service-account', type=str, default=None,
        help='Path to service account JSON key file (optional, uses OAuth if not provided)'
    )
    parser.add_argument(
        '--locations', type=str, required=True,
        help='Path to locations.json file'
    )
    parser.add_argument(
        '--output', type=str, default='ndvi_35years',
        help='Output directory for GeoTIFF files (default: ndvi_35years)'
    )
    parser.add_argument(
        '--workers', type=int, default=OPTIMAL_WORKERS,
        help=f'Number of concurrent workers (default: {OPTIMAL_WORKERS})'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit number of locations to download'
    )
    parser.add_argument(
        '--test', type=int, metavar='N',
        help='Test mode: download only N locations'
    )
    parser.add_argument(
        '--report-interval', type=int, default=1,
        help='Print progress every N downloads (default: 1 = tqdm, use higher for less output)'
    )
    parser.add_argument(
        '--worker-id', type=int, default=None,
        help='Worker ID for parallel execution (0-based index)'
    )
    parser.add_argument(
        '--num-workers', type=int, default=None,
        help='Total number of parallel workers'
    )
    parser.add_argument(
        '--log-file', type=str, default=None,
        help='Path to log file (enables unbuffered output for background execution)'
    )
    parser.add_argument(
        '--reverse', action='store_true',
        help='Process locations in reverse order (useful for running two workers from opposite ends)'
    )
    parser.add_argument(
        '--skip-first', type=int, default=0,
        help='Skip first N locations (after reversing if --reverse is used)'
    )

    args = parser.parse_args()

    # Setup logging to file if specified
    if args.log_file:
        log_file = open(args.log_file, 'w', buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file
        atexit.register(log_file.close)

    # Validate worker parameters
    if (args.worker_id is not None) != (args.num_workers is not None):
        print("Error: --worker-id and --num-workers must be used together")
        sys.exit(1)

    if args.worker_id is not None and args.worker_id >= args.num_workers:
        print(f"Error: --worker-id ({args.worker_id}) must be less than --num-workers ({args.num_workers})")
        sys.exit(1)

    # Validate locations file
    if not Path(args.locations).exists():
        print(f"Error: Locations file not found: {args.locations}")
        sys.exit(1)

    # Handle test mode
    limit = args.test if args.test else args.limit
    test_mode = args.test is not None

    # Initialize Earth Engine
    print("Initializing Earth Engine with High-Volume endpoint...")
    print(f"  Project: {args.project}")

    try:
        if args.service_account:
            print(f"  Auth: Service Account ({args.service_account})")
            credentials = ee.ServiceAccountCredentials(None, args.service_account)
            ee.Initialize(
                credentials=credentials,
                opt_url=HIGHVOLUME_URL,
                project=args.project
            )
        else:
            print(f"  Auth: OAuth")
            ee.Initialize(
                opt_url=HIGHVOLUME_URL,
                project=args.project
            )
        print("  Status: Connected")

        # Set 5-minute timeout for all Earth Engine API calls (QA masking needs more time)
        ee.data.setDeadline(300000)
        print("  Timeout: 300s for API calls")
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        if args.service_account:
            print("Check that service account JSON file is valid and has Earth Engine permissions")
        else:
            print("Run: earthengine authenticate")
        sys.exit(1)

    # Run download
    run_download(
        locations_path=args.locations,
        output_dir=args.output,
        workers=args.workers,
        limit=limit,
        test_mode=test_mode,
        report_interval=args.report_interval,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        reverse=args.reverse,
        skip_first=args.skip_first
    )


if __name__ == '__main__':
    main()
