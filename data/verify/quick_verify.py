#!/usr/bin/env python3
"""Quick verification of 41-year NDVI files."""
import sys
from pathlib import Path
import rasterio

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('ndvi_41years_test')
tif_files = sorted(data_dir.glob('*.tif'))

print(f'Verifying {len(tif_files)} files...')
print('=' * 70)

all_ok = True
for filepath in tif_files:
    with rasterio.open(filepath) as src:
        band_count = src.count
        first_band = src.descriptions[0]
        last_band = src.descriptions[-1]

        # Check sample bands: 1, 50, 100, last
        sample_bands = [1, 50, 100, band_count]
        bands_with_data = 0
        for b in sample_bands:
            data = src.read(b)
            if (data != -9999).sum() > 0:
                bands_with_data += 1

        # Accept either 140 bands (1990-2024) or 164 bands (1984-2024)
        ok = band_count in [140, 164] and last_band == '2024_winter'
        status = 'OK' if ok else 'FAIL'
        if not ok:
            all_ok = False

        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f'[{status}] {filepath.name}')
        print(f'       {band_count} bands | {size_mb:.1f} MB | data in {bands_with_data}/{len(sample_bands)} sampled bands')
        print(f'       First: {first_band} | Last: {last_band}')
        print()

print('=' * 70)
print(f'Status: {"ALL FILES VALID" if all_ok else "SOME FILES HAVE ISSUES"}')
