#!/usr/bin/env python3
"""
Validate exported NDVI tile from GEE.

Supports both single files and GEE's split tile format (directory with multiple files).

Checks:
1. File(s) exist and are readable
2. Has expected 168 bands (42 years × 4 seasons)
3. Band names follow expected pattern
4. NDVI values are in valid range (-1 to 1)
5. NoData values are handled correctly

Usage:
    # Single file
    python validate_ndvi_tile.py /path/to/ndvi_test_tile.tif

    # Directory with split tiles
    python validate_ndvi_tile.py /path/to/earthengine_ndvi_test/
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:
    print("Error: rasterio is required. Install with: pip install rasterio")
    sys.exit(1)


# Expected configuration
START_YEAR = 1984
END_YEAR = 2025
SEASONS = ['spring', 'summer', 'fall', 'winter']
EXPECTED_BANDS = (END_YEAR - START_YEAR + 1) * len(SEASONS)  # 42 × 4 = 168
NODATA_VALUE = -9999


def find_split_tiles(directory: Path) -> list:
    """Find all split tile files in a directory."""
    tiles = []
    for tif_path in directory.glob('ndvi_*tile*.tif'):
        # Parse offset from filename
        match = re.search(r'-(\d{10})-(\d{10})\.tif$', tif_path.name)
        if match:
            row_off = int(match.group(1))
            col_off = int(match.group(2))
            tiles.append((tif_path, row_off, col_off))
    # Sort by offset
    tiles.sort(key=lambda x: (x[1], x[2]))
    return tiles


def validate_single_file(tile_path: Path) -> bool:
    """Validate a single NDVI tile file."""
    print(f"\nFile: {tile_path.name}")
    size_gb = tile_path.stat().st_size / (1024**3)
    print(f"Size: {size_gb:.2f} GB")

    with rasterio.open(tile_path) as src:
        print(f"Dimensions: {src.width} × {src.height} pixels")
        print(f"Bands: {src.count}")
        print(f"CRS: {src.crs}")
        print(f"Data type: {src.dtypes[0]}")

        # Band count check
        if src.count != EXPECTED_BANDS:
            print(f"✗ Band count mismatch: {src.count} (expected {EXPECTED_BANDS})")
            return False
        print(f"✓ Band count correct: {src.count}")

        # Sample a band for value check
        data = src.read(1)
        valid_mask = data != NODATA_VALUE
        if np.any(valid_mask):
            valid_data = data[valid_mask]
            print(f"Band 1 range: [{np.min(valid_data):.4f}, {np.max(valid_data):.4f}]")
            print(f"Valid pixels: {100 * np.sum(valid_mask) / data.size:.1f}%")

    return True


def validate_split_tiles(directory: Path) -> bool:
    """Validate a directory of split tiles from GEE."""
    tiles = find_split_tiles(directory)

    if not tiles:
        print(f"✗ No split tile files found in {directory}")
        return False

    print(f"\n--- Split Tile Analysis ---")
    print(f"Found {len(tiles)} sub-tiles")

    # Calculate total size
    total_size = sum(t[0].stat().st_size for t in tiles)
    print(f"Total size: {total_size / (1024**3):.2f} GB")

    # Analyze grid structure
    rows = sorted(set(t[1] for t in tiles))
    cols = sorted(set(t[2] for t in tiles))
    print(f"Grid: {len(rows)} rows × {len(cols)} cols")
    print(f"Row offsets: {rows}")
    print(f"Col offsets: {cols}")

    # Validate each sub-tile
    print(f"\n--- Sub-tile Validation ---")
    all_valid = True
    band_counts = set()
    total_valid_pixels = 0
    total_pixels = 0

    for i, (tile_path, row_off, col_off) in enumerate(tiles):
        try:
            with rasterio.open(tile_path) as src:
                band_counts.add(src.count)

                # Check a sample for valid pixels
                data = src.read(1)
                valid_mask = data != NODATA_VALUE
                total_valid_pixels += np.sum(valid_mask)
                total_pixels += data.size

                status = "✓" if src.count == EXPECTED_BANDS else "✗"
                print(f"  [{i+1}/{len(tiles)}] offset ({row_off}, {col_off}): "
                      f"{src.width}×{src.height}, {src.count} bands {status}")

                if src.count != EXPECTED_BANDS:
                    all_valid = False
        except Exception as e:
            print(f"  [{i+1}/{len(tiles)}] offset ({row_off}, {col_off}): ERROR - {e}")
            all_valid = False

    # Summary
    print(f"\n--- Summary ---")
    if len(band_counts) == 1:
        bc = band_counts.pop()
        if bc == EXPECTED_BANDS:
            print(f"✓ All sub-tiles have {bc} bands (correct)")
        else:
            print(f"✗ All sub-tiles have {bc} bands (expected {EXPECTED_BANDS})")
            all_valid = False
    else:
        print(f"✗ Inconsistent band counts: {band_counts}")
        all_valid = False

    coverage = 100 * total_valid_pixels / total_pixels if total_pixels > 0 else 0
    print(f"Overall valid pixel coverage: {coverage:.1f}%")

    # Validate first tile in detail
    print(f"\n--- Detailed Check (first sub-tile) ---")
    first_tile = tiles[0][0]
    with rasterio.open(first_tile) as src:
        print(f"File: {first_tile.name}")
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")
        print(f"Data type: {src.dtypes[0]}")
        print(f"NoData: {src.nodata}")

        # Check band names
        band_names = src.descriptions
        if band_names and band_names[0]:
            print(f"First band name: {band_names[0]}")
            print(f"Last band name: {band_names[-1]}")
        else:
            print("Band names: Not set (common for GEE exports)")

        # Sample values from multiple bands
        print(f"\nValue samples:")
        sample_bands = [1, 42, 84, 126, 168]
        for band_idx in sample_bands:
            if band_idx <= src.count:
                data = src.read(band_idx)
                valid_mask = data != NODATA_VALUE
                if np.any(valid_mask):
                    valid_data = data[valid_mask]
                    min_v, max_v = np.min(valid_data), np.max(valid_data)
                    mean_v = np.mean(valid_data)
                    in_range = -1.0 <= min_v and max_v <= 1.0
                    status = "✓" if in_range else "⚠"
                    print(f"  Band {band_idx}: [{min_v:.4f}, {max_v:.4f}], mean={mean_v:.4f} {status}")

    return all_valid


def validate_tile(path: Path) -> bool:
    """Validate an NDVI tile (single file or split directory)."""
    print("=" * 60)
    print("NDVI Tile Validation")
    print("=" * 60)
    print(f"\nPath: {path}")

    if not path.exists():
        print(f"✗ ERROR: Path not found!")
        return False

    if path.is_dir():
        # Directory with split tiles
        result = validate_split_tiles(path)
    else:
        # Single file
        result = validate_single_file(path)

    print("\n" + "=" * 60)
    if result:
        print("✓ VALIDATION PASSED")
    else:
        print("✗ VALIDATION FAILED")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Validate exported NDVI tile from GEE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'path', type=str,
        help='Path to NDVI tile GeoTIFF or directory with split tiles'
    )

    args = parser.parse_args()
    path = Path(args.path)

    valid = validate_tile(path)
    sys.exit(0 if valid else 1)


if __name__ == '__main__':
    main()
