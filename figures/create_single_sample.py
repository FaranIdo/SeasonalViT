#!/usr/bin/env python3
"""
Create a paper-quality visualization of a single NDVI sample.

Shows: World Imagery, Land Cover, Info panel, then selected years with 4 seasons each.

Usage:
    python paper/figures/create_single_sample.py
    python paper/figures/create_single_sample.py --sample 0009710_31d2242_34d6428.tif
"""

import argparse
import json
import math
import os
import re
import warnings
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import rasterio
import requests
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from PIL import Image

warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# ESA WorldCover official color palette
WORLDCOVER_PALETTE = [
    '006400',  # 10: Tree cover (dark green)
    'ffbb22',  # 20: Shrubland (orange)
    'ffff4c',  # 30: Grassland (yellow)
    'f096ff',  # 40: Cropland (pink)
    'fa0000',  # 50: Built-up (red)
    'b4b4b4',  # 60: Bare/sparse vegetation (gray)
    'f0f0f0',  # 70: Snow/ice (white)
    '0064c8',  # 80: Permanent water bodies (blue)
    '0096a0',  # 90: Herbaceous wetland (teal)
    '00cf75',  # 95: Mangroves (green)
    'fae6a0',  # 100: Moss and lichen (beige)
]

WORLDCOVER_NAMES = {
    10: 'Tree cover',
    20: 'Shrubland',
    30: 'Grassland',
    40: 'Cropland',
    50: 'Built-up',
    60: 'Bare/sparse',
    70: 'Snow/ice',
    80: 'Water',
    90: 'Wetland',
    95: 'Mangroves',
    100: 'Moss/lichen',
}

# Selected years to display
SELECTED_YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]

# Season icons with colors
SEASON_CONFIG = [
    ('❀', '#90EE90', '#228B22'),  # Spring: green flower
    ('☀', '#FFEB3B', '#FF8C00'),  # Summer: yellow sun
    ('♣', '#DEB887', '#8B4513'),  # Fall: brown leaf
    ('❄', '#ADD8E6', '#4169E1'),  # Winter: blue snowflake
]
SEASON_POSITIONS = [(0, 0), (0, 1), (1, 0), (1, 1)]

# GEE initialization state
_gee_initialized = False


def init_gee() -> bool:
    """Initialize GEE with service account."""
    global _gee_initialized
    if _gee_initialized:
        return True
    try:
        import ee
        ee.Initialize(project=os.environ.get("EE_PROJECT"))
        _gee_initialized = True
        return True
    except Exception as e:
        print(f"  Warning: GEE initialization failed: {e}")
        return False


def create_ndvi_colormap():
    """Create custom NDVI colormap for -1 to 1 range."""
    colors_with_positions = [
        (0.0,  '#00008B'),
        (0.1,  '#0000CD'),
        (0.2,  '#4169E1'),
        (0.3,  '#6495ED'),
        (0.4,  '#87CEEB'),
        (0.5,  '#8B4513'),
        (0.55, '#D2B48C'),
        (0.6,  '#F4E8C1'),
        (0.65, '#FFFF99'),
        (0.7,  '#CCFF66'),
        (0.75, '#99FF33'),
        (0.8,  '#66CC33'),
        (0.85, '#339900'),
        (0.9,  '#228B22'),
        (1.0,  '#006400'),
    ]
    return LinearSegmentedColormap.from_list(
        'ndvi', [(pos, color) for pos, color in colors_with_positions], N=200
    )


def download_osm_map(bounds, output_size=200):
    """Download OSM map for bounds."""
    lon_span = bounds.right - bounds.left
    zoom = max(1, int(math.log2(output_size * 360 / (lon_span * 256))))

    tile_size = 256
    n = 2 ** zoom

    left_tile = (bounds.left + 180) / 360 * n
    right_tile = (bounds.right + 180) / 360 * n
    top_tile = (1 - math.asinh(math.tan(math.radians(bounds.top))) / math.pi) / 2 * n
    bottom_tile = (1 - math.asinh(math.tan(math.radians(bounds.bottom))) / math.pi) / 2 * n

    start_tile_x = int(left_tile)
    end_tile_x = int(right_tile) + 1
    start_tile_y = int(top_tile)
    end_tile_y = int(bottom_tile) + 1

    tiles_x = end_tile_x - start_tile_x
    tiles_y = end_tile_y - start_tile_y

    canvas = Image.new('RGB', (tiles_x * tile_size, tiles_y * tile_size))

    headers = {'User-Agent': 'AeonProject/1.0'}
    for i in range(tiles_x):
        for j in range(tiles_y):
            tile_x = start_tile_x + i
            tile_y = start_tile_y + j
            try:
                url = f'https://tile.openstreetmap.org/{zoom}/{tile_x}/{tile_y}.png'
                response = requests.get(url, headers=headers, timeout=10, verify=False)
                response.raise_for_status()
                tile = Image.open(BytesIO(response.content))
                canvas.paste(tile, (i * tile_size, j * tile_size))
            except Exception:
                pass

    px_left = (left_tile - start_tile_x) * tile_size
    px_right = (right_tile - start_tile_x) * tile_size
    px_top = (top_tile - start_tile_y) * tile_size
    px_bottom = (bottom_tile - start_tile_y) * tile_size

    cropped = canvas.crop((int(px_left), int(px_top), int(px_right), int(px_bottom)))
    return cropped.resize((output_size, output_size), Image.Resampling.LANCZOS)


def download_worldcover_image(bounds, output_size=200):
    """Download ESA WorldCover land cover map from GEE."""
    if not init_gee():
        return None
    try:
        import ee
        region = ee.Geometry.Rectangle([
            bounds.left, bounds.bottom, bounds.right, bounds.top
        ])
        worldcover = ee.ImageCollection('ESA/WorldCover/v200').first()
        url = worldcover.getThumbURL({
            'region': region,
            'dimensions': output_size,
            'format': 'png',
            'min': 10,
            'max': 100,
            'palette': WORLDCOVER_PALETTE
        })
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"  Warning: Failed to download WorldCover: {e}")
        return None


def reverse_geocode(lat, lon):
    """Get location name from coordinates."""
    url = 'https://nominatim.openstreetmap.org/reverse'
    params = {
        'lat': lat, 'lon': lon, 'format': 'json',
        'zoom': 10, 'addressdetails': 1, 'accept-language': 'en',
    }
    headers = {'User-Agent': 'AeonProject/1.0'}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        address = data.get('address', {})
        city = (address.get('city') or address.get('town') or
                address.get('village') or address.get('county') or
                address.get('state') or '')
        country = address.get('country', '')
        if city and country:
            return f"{city}, {country}"
        elif country:
            return country
        return data.get('display_name', '').split(',')[0]
    except Exception:
        return None


def get_landcover_info(sample_id: str, project_root: Path) -> dict | None:
    """Load land cover stats for a sample."""
    stats_path = project_root / "data" / "landcover_stats.json"
    if not stats_path.exists():
        return None
    with open(stats_path) as f:
        stats = json.load(f)
    return stats.get(sample_id)


def add_season_icons(ax, cell_size: int):
    """Add colored season icons in top-left corner of each quadrant."""
    radius = 8
    offset = 10

    for idx, (row, col) in enumerate(SEASON_POSITIONS):
        icon, bg_color, icon_color = SEASON_CONFIG[idx]
        x = col * cell_size + offset
        y = row * cell_size + offset

        circle = Circle((x, y), radius, facecolor=bg_color, edgecolor='white',
                        linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, icon, fontsize=9, color=icon_color,
                fontweight='bold', ha='center', va='center', zorder=11)


def create_season_grid(data: np.ndarray, year: int, cell_size: int = 100) -> np.ndarray:
    """Create a 2x2 grid of season images for a year."""
    grid = np.full((cell_size * 2, cell_size * 2), np.nan)
    year_idx = year - 1990

    for season_idx, (row, col) in enumerate(SEASON_POSITIONS):
        band_idx = year_idx * 4 + season_idx
        if band_idx < data.shape[0]:
            season_data = data[band_idx]
            h, w = season_data.shape
            row_indices = np.linspace(0, h - 1, cell_size).astype(int)
            col_indices = np.linspace(0, w - 1, cell_size).astype(int)
            resized = season_data[np.ix_(row_indices, col_indices)]
            grid[row * cell_size:(row + 1) * cell_size,
                 col * cell_size:(col + 1) * cell_size] = resized

    return grid


def create_figure(sample_path: Path, output_path: Path, project_root: Path):
    """Create the single sample visualization - compact paper layout."""
    print(f"Loading {sample_path}...")

    with rasterio.open(sample_path) as src:
        data = src.read()
        bounds = src.bounds

    data = data.astype(float)
    data[data == -9999] = np.nan

    # Extract info
    center_lat = (bounds.bottom + bounds.top) / 2
    center_lon = (bounds.left + bounds.right) / 2

    sample_id = re.match(r'^(\d+)_', sample_path.name)
    sample_id = sample_id.group(1) if sample_id else None

    print(f"  Reverse geocoding ({center_lat:.4f}, {center_lon:.4f})...")
    location_name = reverse_geocode(center_lat, center_lon) or "Unknown"
    print(f"  Location: {location_name}")

    landcover_info = get_landcover_info(sample_id, project_root) if sample_id else None

    # Calculate stats
    valid_mask = ~np.isnan(data)
    valid_pct = 100 * valid_mask.sum() / valid_mask.size
    valid_data = data[valid_mask]
    ndvi_mean = np.mean(valid_data) if len(valid_data) > 0 else 0

    # Compact layout: single row with reference images + selected years
    # [World Imagery] [Land Cover] [1990] [2000] [2010] [2020] [2024]
    display_years = [1990, 2000, 2010, 2020, 2024]
    n_cols = 2 + len(display_years)  # 2 reference + years

    cell_size = 64

    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 1.8, 2.5))
    plt.subplots_adjust(wspace=0.08, left=0.02, right=0.88, top=0.85, bottom=0.05)

    # Download reference images
    print("  Downloading OSM map...")
    osm_img = download_osm_map(bounds, output_size=cell_size * 2)

    print("  Downloading WorldCover...")
    wc_img = download_worldcover_image(bounds, output_size=cell_size * 2)

    # Column 0: World Imagery
    ax = axes[0]
    ax.imshow(osm_img)
    ax.set_title('World\nImagery', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    # Column 1: Land Cover
    ax = axes[1]
    if wc_img:
        ax.imshow(wc_img)
    ax.set_title('Land\nCover', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    # Year columns
    cmap = create_ndvi_colormap()
    vmin, vmax = -1, 1

    print("  Generating year panels...")
    for i, year in enumerate(display_years):
        ax = axes[2 + i]
        season_grid = create_season_grid(data, year, cell_size)

        im = ax.imshow(season_grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

        # Grid lines
        ax.axhline(cell_size, color='white', linewidth=0.5)
        ax.axvline(cell_size, color='white', linewidth=0.5)

        # Season icons on first year only
        if i == 0:
            add_season_icons(ax, cell_size)

        ax.set_title(str(year), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Add location info as figure title
    lc_name = landcover_info.get('dominant_name', '') if landcover_info else ''
    title = f"{location_name} ({center_lat:.2f}°, {center_lon:.2f}°)"
    if lc_name:
        title += f" — {lc_name}"
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.98)

    # Colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('NDVI', fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create paper-quality single sample visualization")
    parser.add_argument("--sample", default="0009710_31d2242_34d6428.tif",
                        help="Sample filename (default: 0009710_31d2242_34d6428.tif)")
    parser.add_argument("--output", help="Output path (default: paper/figures/single_sample.png)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    sample_path = project_root / "data" / "samples" / args.sample

    if not sample_path.exists():
        print(f"Error: Sample not found: {sample_path}")
        return

    output_path = Path(args.output) if args.output else Path(__file__).parent / "single_sample.png"

    create_figure(sample_path, output_path, project_root)


if __name__ == "__main__":
    main()
