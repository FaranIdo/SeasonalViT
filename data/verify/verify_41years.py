#!/usr/bin/env python3
"""Quick verification script for 41-year NDVI files."""
import sys
from pathlib import Path
import numpy as np
import rasterio

def verify_file(filepath: Path) -> dict:
    """Verify a single 41-year NDVI file."""
    result = {
        'filename': filepath.name,
        'valid': True,
        'issues': []
    }

    try:
        with rasterio.open(filepath) as src:
            # Check band count
            if src.count != 164:
                result['issues'].append(f'Expected 164 bands, got {src.count}')
                result['valid'] = False

            # Check band names
            first_band = src.descriptions[0]
            last_band = src.descriptions[-1]
            if first_band != '1984_spring':
                result['issues'].append(f'First band should be 1984_spring, got {first_band}')
                result['valid'] = False
            if last_band != '2024_winter':
                result['issues'].append(f'Last band should be 2024_winter, got {last_band}')
                result['valid'] = False

            # Check CRS
            if str(src.crs) != 'EPSG:4326':
                result['issues'].append(f'Expected EPSG:4326, got {src.crs}')
                result['valid'] = False

            # Check data coverage across all bands
            bands_with_data = 0
            for i in range(1, src.count + 1):
                band_data = src.read(i)
                if (band_data != -9999).sum() > 0:
                    bands_with_data += 1

            result['bands_with_data'] = bands_with_data
            coverage_pct = bands_with_data / src.count * 100

            # Expect at least 80% of bands to have some data
            if coverage_pct < 80:
                result['issues'].append(f'Low data coverage: {bands_with_data}/{src.count} bands ({coverage_pct:.1f}%)')
                result['valid'] = False

            # Check recent year (2024) has data
            last_band_data = src.read(src.count)  # 2024_winter
            if (last_band_data != -9999).sum() == 0:
                result['issues'].append('No data in most recent band (2024_winter)')

            result['band_count'] = src.count
            result['size_mb'] = filepath.stat().st_size / (1024 * 1024)
            result['shape'] = f'{src.width}x{src.height}'

    except Exception as e:
        result['valid'] = False
        result['issues'].append(f'Error reading file: {str(e)}')

    return result


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('ndvi_41years_test')

    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    tif_files = sorted(data_dir.glob('*.tif'))
    if not tif_files:
        print(f"No .tif files found in {data_dir}")
        sys.exit(1)

    print(f"Verifying {len(tif_files)} files in {data_dir}\n")
    print("=" * 70)

    all_valid = True
    total_size = 0

    for filepath in tif_files:
        result = verify_file(filepath)
        status = "OK" if result['valid'] else "FAILED"
        size_mb = result.get('size_mb', 0)
        total_size += size_mb

        print(f"[{status}] {result['filename']}")
        bands_with_data = result.get('bands_with_data', 0)
        band_count = result.get('band_count', 0)
        coverage = f"{bands_with_data}/{band_count}" if band_count else "N/A"
        print(f"       Bands: {coverage} with data | Size: {size_mb:.1f} MB | Shape: {result.get('shape', 'N/A')}")

        if result['issues']:
            for issue in result['issues']:
                print(f"       Issue: {issue}")

        if not result['valid']:
            all_valid = False
        print()

    print("=" * 70)
    print(f"Total files: {len(tif_files)}")
    print(f"Total size: {total_size:.1f} MB")
    print(f"Status: {'ALL VALID' if all_valid else 'SOME FILES HAVE ISSUES'}")

    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()
