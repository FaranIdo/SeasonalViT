#!/usr/bin/env python3
"""
Create NDVI examples figure showing multi-decadal data for 3 locations.

This script generates a grid figure with:
- 3 rows: Hungary (temperate cropland), India (tropical), Argentina (Southern Hemisphere)
- 6 columns: World imagery reference + 5 selected years (1990, 2000, 2010, 2020, 2024)
- Each year cell shows 4 seasons as a 2x2 mini-grid

Usage:
    python paper/figures/create_ndvi_examples.py
"""

import math
import os
import warnings
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import rasterio
import requests
from matplotlib.colors import LinearSegmentedColormap
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

# GEE initialization state
_gee_initialized = False


def init_gee() -> bool:
    """Initialize GEE with service account. Returns True if successful."""
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


def download_worldcover_image(bounds, output_size=128):
    """Download ESA WorldCover land cover map from GEE."""
    if not init_gee():
        return None

    try:
        import ee

        # Create geometry from bounds
        region = ee.Geometry.Rectangle([
            bounds.left, bounds.bottom, bounds.right, bounds.top
        ])

        # Get ESA WorldCover v200 (2021)
        worldcover = ee.ImageCollection('ESA/WorldCover/v200').first()

        # Get thumbnail URL with official color palette
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
        img = Image.open(BytesIO(response.content)).convert('RGB')

        return img
    except Exception as e:
        print(f"  Warning: Failed to download WorldCover image: {e}")
        return None

# Selected samples with descriptive labels
SAMPLES = [
    {
        'filename': '0210127_-31d2103_-58d6034.tif',
        'label': 'Argentina',
        'description': 'Southern Hemisphere with opposite seasonality',
    },
    {
        'filename': '0210137_21d7371_81d8240.tif',
        'label': 'India',
        'description': 'Tropical region with monsoon-driven patterns',
    },
    {
        'filename': '0000016_47d0636_-122d6648.tif',
        'label': 'Pacific NW, USA',
        'description': 'Temperate coniferous forest',
    },
]

# Selected years to display (indices into 35-year range 1990-2024)
SELECTED_YEARS = [1990, 2000, 2010, 2020, 2024]


def create_ndvi_colormap():
    """Create custom NDVI colormap for -1 to 1 range."""
    colors_with_positions = [
        (0.0,  '#00008B'),   # -1.0: Dark blue (deep water)
        (0.1,  '#0000CD'),   # -0.8: Medium blue
        (0.2,  '#4169E1'),   # -0.6: Royal blue
        (0.3,  '#6495ED'),   # -0.4: Cornflower blue
        (0.4,  '#87CEEB'),   # -0.2: Sky blue
        (0.5,  '#8B4513'),   # 0.0: Brown (bare soil)
        (0.55, '#D2B48C'),   # 0.1: Tan
        (0.6,  '#F4E8C1'),   # 0.2: Light tan
        (0.65, '#FFFF99'),   # 0.3: Light yellow
        (0.7,  '#CCFF66'),   # 0.4: Yellow-green
        (0.75, '#99FF33'),   # 0.5: Light green
        (0.8,  '#66CC33'),   # 0.6: Medium green
        (0.85, '#339900'),   # 0.7: Green
        (0.9,  '#228B22'),   # 0.8: Forest green
        (1.0,  '#006400'),   # 1.0: Dark green
    ]
    return LinearSegmentedColormap.from_list(
        'ndvi', [(pos, color) for pos, color in colors_with_positions], N=200
    )


def lat_lon_to_tile(lat, lon, zoom):
    """Convert lat/lon to tile coordinates."""
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    lat_rad = math.radians(lat)
    y = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)
    return x, y


def download_tile(x, y, zoom):
    """Download a single OSM tile."""
    url = f'https://tile.openstreetmap.org/{zoom}/{x}/{y}.png'
    headers = {'User-Agent': 'AeonProject/1.0 (satellite imagery research)'}
    response = requests.get(url, headers=headers, timeout=10, verify=False)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def calculate_zoom_for_bounds(bounds, output_width=264):
    """Calculate appropriate zoom level for bounds."""
    lon_span = bounds.right - bounds.left
    zoom = math.log2(output_width * 360 / (lon_span * 256))
    return max(1, int(zoom))


def download_osm_map_for_bounds(bounds, output_size=128):
    """Download OSM map covering the exact bounds of the GeoTIFF."""
    zoom = calculate_zoom_for_bounds(bounds, output_size)
    tile_size = 256
    n = 2 ** zoom

    # Convert bounds to tile coordinates
    left_tile = (bounds.left + 180) / 360 * n
    right_tile = (bounds.right + 180) / 360 * n
    top_tile = (1 - math.asinh(math.tan(math.radians(bounds.top))) / math.pi) / 2 * n
    bottom_tile = (1 - math.asinh(math.tan(math.radians(bounds.bottom))) / math.pi) / 2 * n

    # Get integer tile range
    start_tile_x = int(left_tile)
    end_tile_x = int(right_tile) + 1
    start_tile_y = int(top_tile)
    end_tile_y = int(bottom_tile) + 1

    tiles_x = end_tile_x - start_tile_x
    tiles_y = end_tile_y - start_tile_y

    # Create canvas for all tiles
    canvas = Image.new('RGB', (tiles_x * tile_size, tiles_y * tile_size))

    for i in range(tiles_x):
        for j in range(tiles_y):
            tile_x = start_tile_x + i
            tile_y = start_tile_y + j
            try:
                tile = download_tile(tile_x, tile_y, zoom)
                canvas.paste(tile, (i * tile_size, j * tile_size))
            except Exception as e:
                print(f"  Warning: Failed to download tile ({tile_x}, {tile_y}): {e}")

    # Calculate pixel coordinates of bounds within the canvas
    px_left = (left_tile - start_tile_x) * tile_size
    px_right = (right_tile - start_tile_x) * tile_size
    px_top = (top_tile - start_tile_y) * tile_size
    px_bottom = (bottom_tile - start_tile_y) * tile_size

    # Crop to exact bounds
    cropped = canvas.crop((int(px_left), int(px_top), int(px_right), int(px_bottom)))

    # Resize to output size
    cropped = cropped.resize((output_size, output_size), Image.Resampling.LANCZOS)

    return cropped


def load_sample_data(filepath: Path) -> tuple:
    """Load sample data and bounds from GeoTIFF."""
    with rasterio.open(filepath) as src:
        data = src.read()
        bounds = src.bounds
    data = data.astype(float)
    data[data == -9999] = np.nan
    return data, bounds


def get_year_bands(year: int) -> list:
    """Get band indices for a given year (4 seasons)."""
    year_idx = year - 1990
    start_band = year_idx * 4
    return list(range(start_band, start_band + 4))


def find_valid_band_for_season(data: np.ndarray, year: int, season_idx: int,
                                nodata_threshold: float = 0.8) -> int:
    """Find a valid band for a season, searching forward if needed.

    Args:
        data: Full dataset (bands, height, width)
        year: Target year
        season_idx: Season index (0=spring, 1=summer, 2=fall, 3=winter)
        nodata_threshold: If more than this fraction is NoData, search for alternative

    Returns:
        Band index with valid data
    """
    year_idx = year - 1990
    target_band = year_idx * 4 + season_idx

    # Check if target band has enough valid data
    if target_band < data.shape[0]:
        band_data = data[target_band]
        nodata_frac = np.isnan(band_data).sum() / band_data.size
        if nodata_frac < nodata_threshold:
            return target_band

    # Search forward for valid data in same season
    for offset in range(1, 35):  # Search up to 35 years forward
        candidate_year = year + offset
        if candidate_year > 2024:
            break
        candidate_idx = (candidate_year - 1990) * 4 + season_idx
        if candidate_idx < data.shape[0]:
            band_data = data[candidate_idx]
            nodata_frac = np.isnan(band_data).sum() / band_data.size
            if nodata_frac < nodata_threshold:
                return candidate_idx

    # Search backward if forward search failed
    for offset in range(1, 35):
        candidate_year = year - offset
        if candidate_year < 1990:
            break
        candidate_idx = (candidate_year - 1990) * 4 + season_idx
        if candidate_idx >= 0 and candidate_idx < data.shape[0]:
            band_data = data[candidate_idx]
            nodata_frac = np.isnan(band_data).sum() / band_data.size
            if nodata_frac < nodata_threshold:
                return candidate_idx

    # Fall back to original
    return target_band


def create_season_grid(data: np.ndarray, year: int, cell_size: int = 64) -> np.ndarray:
    """Create a 2x2 grid of season images for a year.

    If a season has too much NoData, substitutes with nearest valid year for that season.
    """
    # Season order: spring, summer, fall, winter -> arrange as 2x2
    grid = np.full((cell_size * 2, cell_size * 2), np.nan)

    season_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # spring, summer, fall, winter

    for season_idx in range(4):
        # Find best band for this season (may substitute if too much NoData)
        band_idx = find_valid_band_for_season(data, year, season_idx)

        if band_idx < data.shape[0]:
            season_data = data[band_idx]
            # Resize to cell_size x cell_size using simple sampling
            h, w = season_data.shape
            row_indices = np.linspace(0, h - 1, cell_size).astype(int)
            col_indices = np.linspace(0, w - 1, cell_size).astype(int)
            resized = season_data[np.ix_(row_indices, col_indices)]

            row, col = season_positions[season_idx]
            grid[row * cell_size:(row + 1) * cell_size,
                 col * cell_size:(col + 1) * cell_size] = resized

    return grid


# Season icons with colors (icon, background color, icon color)
SEASON_CONFIG = [
    ('❀', '#90EE90', '#228B22'),  # Spring: green flower
    ('☀', '#FFEB3B', '#FF8C00'),  # Summer: yellow sun
    ('♣', '#DEB887', '#8B4513'),  # Fall: brown leaf shape
    ('❄', '#ADD8E6', '#4169E1'),  # Winter: blue snowflake
]
SEASON_POSITIONS = [(0, 0), (0, 1), (1, 0), (1, 1)]  # grid positions


def add_season_icons(ax, cell_size: int):
    """Add colored season icons in top-left corner of each quadrant."""
    from matplotlib.patches import Circle

    radius = 8
    offset = 10  # distance from corner

    for idx, (row, col) in enumerate(SEASON_POSITIONS):
        icon, bg_color, icon_color = SEASON_CONFIG[idx]

        # Position in top-left corner of each quadrant
        x = col * cell_size + offset
        y = row * cell_size + offset

        # Draw background circle
        circle = Circle((x, y), radius, facecolor=bg_color, edgecolor='white',
                        linewidth=1.5, zorder=10)
        ax.add_patch(circle)

        # Add icon
        ax.text(x, y, icon, fontsize=9, color=icon_color,
                fontweight='bold', ha='center', va='center', zorder=11)


def create_figure(samples_dir: Path, output_path: Path):
    """Create the main NDVI examples figure."""
    n_rows = len(SAMPLES)
    n_cols = 1 + len(SELECTED_YEARS)  # Reference map + years

    cell_size = 64  # Size of each season cell
    year_cell_size = cell_size * 2  # 2x2 grid of seasons

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.8, n_rows * 2.0),
        gridspec_kw={'wspace': 0.05, 'hspace': 0.15}
    )

    cmap = create_ndvi_colormap()
    vmin, vmax = -1, 1  # Full NDVI range (matches visualize_sample.py)

    # Process each sample
    for row_idx, sample_info in enumerate(SAMPLES):
        filepath = samples_dir / sample_info['filename']
        label = sample_info['label']

        print(f"Processing {label}...")

        # Load data
        data, bounds = load_sample_data(filepath)
        print(f"  Shape: {data.shape}, Bounds: {bounds}")

        # Column 0: ESA WorldCover land cover map
        ax = axes[row_idx, 0]
        worldcover_img = download_worldcover_image(bounds, output_size=year_cell_size)
        if worldcover_img is not None:
            ax.imshow(worldcover_img)
        else:
            ax.text(0.5, 0.5, 'Map\nUnavailable', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_facecolor('#f0f0f0')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=8, fontweight='bold')

        if row_idx == 0:
            ax.set_title('Land Cover', fontsize=9)

        # Columns 1-5: Selected years
        for col_idx, year in enumerate(SELECTED_YEARS):
            ax = axes[row_idx, col_idx + 1]
            season_grid = create_season_grid(data, year, cell_size)

            im = ax.imshow(season_grid, cmap=cmap, vmin=vmin, vmax=vmax,
                          interpolation='nearest')

            # Add subtle grid lines between seasons
            ax.axhline(cell_size, color='white', linewidth=0.5)
            ax.axvline(cell_size, color='white', linewidth=0.5)

            # Add season icons to first year column only
            if col_idx == 0:
                add_season_icons(ax, cell_size)

            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(str(year), fontsize=9)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('NDVI', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved to {output_path}")


def main():
    project_root = Path(__file__).parent.parent.parent
    samples_dir = project_root / "data" / "samples"
    output_path = Path(__file__).parent / "ndvi_examples.png"

    # Verify samples exist
    for sample_info in SAMPLES:
        filepath = samples_dir / sample_info['filename']
        if not filepath.exists():
            print(f"Error: Sample not found: {filepath}")
            print("Please ensure the sample TIF files are in data/samples/")
            return

    create_figure(samples_dir, output_path)


if __name__ == "__main__":
    main()
