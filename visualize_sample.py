#!/usr/bin/env python3
"""
Visualize all NDVI images for a single sample (35 years × 4 seasons) with OSM reference map.

Usage:
    python visualize_sample.py --input data/test_qa_masked/sample.tif
    python visualize_sample.py --input data/test_qa_masked/sample.tif --output output.png
"""

import argparse
import json
import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import requests
from PIL import Image, ImageDraw
from io import BytesIO

warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# GEE initialization state
_gee_initialized = False

# ESA WorldCover official color palette (class values: 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
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

# WorldCover legend entries: (color_hex, label)
WORLDCOVER_LEGEND = [
    ('#006400', 'Tree cover'),
    ('#ffbb22', 'Shrubland'),
    ('#ffff4c', 'Grassland'),
    ('#f096ff', 'Cropland'),
    ('#fa0000', 'Built-up'),
    ('#b4b4b4', 'Bare/sparse'),
    ('#f0f0f0', 'Snow/ice'),
    ('#0064c8', 'Water'),
    ('#0096a0', 'Wetland'),
    ('#00cf75', 'Mangroves'),
    ('#fae6a0', 'Moss/lichen'),
]


def extract_sample_id(filename: str) -> str | None:
    """Extract sample_id from filename format: {sample_id}_{lat}_{lon}.tif"""
    match = re.match(r'^(\d+)_', filename)
    if match:
        return match.group(1)
    return None


def get_landcover_info(sample_id: str) -> dict | None:
    """Load land cover stats for a sample from landcover_stats.json."""
    stats_path = Path(__file__).parent / "data" / "landcover_stats.json"
    if not stats_path.exists():
        return None
    with open(stats_path) as f:
        stats = json.load(f)
    return stats.get(sample_id)


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


def draw_worldcover_legend(img):
    """Draw WorldCover legend overlay on the image."""
    draw = ImageDraw.Draw(img)

    # Legend position and sizing
    box_w, box_h = 12, 10
    padding = 4
    x_start = 5
    y_start = 5

    # Draw semi-transparent background
    legend_height = len(WORLDCOVER_LEGEND) * (box_h + 2) + padding * 2
    legend_width = 75

    # Draw white background with transparency effect (draw filled rectangle)
    for i in range(legend_height):
        for j in range(legend_width):
            try:
                px = img.getpixel((x_start + j, y_start + i))
                # Blend with white (alpha ~0.7)
                blended = tuple(int(0.3 * c + 0.7 * 255) for c in px[:3])
                img.putpixel((x_start + j, y_start + i), blended)
            except IndexError:
                pass

    # Draw legend entries
    for i, (color, label) in enumerate(WORLDCOVER_LEGEND):
        y = y_start + padding + i * (box_h + 2)
        # Color box
        rgb = tuple(int(color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
        draw.rectangle([x_start + padding, y, x_start + padding + box_w, y + box_h],
                       fill=rgb, outline='black')
        # Label text
        draw.text((x_start + padding + box_w + 3, y - 1), label, fill='black')

    return img


def download_worldcover_image(bounds, output_size=264):
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

        # Add legend overlay
        img = draw_worldcover_legend(img)

        return img
    except Exception as e:
        print(f"  Warning: Failed to download WorldCover image: {e}")
        return None


def reverse_geocode(lat, lon):
    """Get location name from coordinates using OpenStreetMap Nominatim API."""
    url = 'https://nominatim.openstreetmap.org/reverse'
    params = {
        'lat': lat,
        'lon': lon,
        'format': 'json',
        'zoom': 10,  # City/town level
        'addressdetails': 1,
        'accept-language': 'en',  # Request English names
    }
    headers = {'User-Agent': 'AeonProject/1.0 (satellite imagery research)'}
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
    except Exception as e:
        print(f"  Warning: Reverse geocoding failed: {e}")
        return None


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
    """Calculate the appropriate zoom level to fit the bounds in output_width pixels."""
    lon_span = bounds.right - bounds.left
    center_lat = (bounds.top + bounds.bottom) / 2

    # At zoom z, the world is 256 * 2^z pixels wide
    # Longitude span in pixels = (lon_span / 360) * 256 * 2^z
    # We want this to equal output_width, so:
    # 2^z = output_width * 360 / (lon_span * 256)
    # z = log2(output_width * 360 / (lon_span * 256))

    zoom = math.log2(output_width * 360 / (lon_span * 256))
    # Round down to avoid cutting off edges
    return max(1, int(zoom))


def download_osm_map_for_bounds(bounds, output_size=264):
    """Download OSM map covering the exact bounds of the GeoTIFF."""
    zoom = calculate_zoom_for_bounds(bounds, output_size)
    print(f"  Using zoom level {zoom} to match ~{(bounds.right - bounds.left) * 111:.1f}km extent")

    tile_size = 256
    n = 2 ** zoom

    # Convert bounds to tile coordinates (floating point for precision)
    left_tile = (bounds.left + 180) / 360 * n
    right_tile = (bounds.right + 180) / 360 * n
    top_tile = (1 - math.asinh(math.tan(math.radians(bounds.top))) / math.pi) / 2 * n
    bottom_tile = (1 - math.asinh(math.tan(math.radians(bounds.bottom))) / math.pi) / 2 * n

    # Get integer tile range (expand to cover full bounds)
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

    # Draw center crosshair
    draw = ImageDraw.Draw(cropped)
    cx, cy = output_size // 2, output_size // 2
    draw.line((cx - 10, cy, cx + 10, cy), fill='red', width=3)
    draw.line((cx, cy - 10, cx, cy + 10), fill='red', width=3)

    return cropped


def create_ndvi_colormap():
    """Create custom NDVI colormap for -1 to 1 range."""
    colors_with_positions = [
        (0.0,  '#00008B'),  # -1.0: Dark blue (deep water)
        (0.1,  '#0000CD'),  # -0.8: Medium blue
        (0.2,  '#4169E1'),  # -0.6: Royal blue
        (0.3,  '#6495ED'),  # -0.4: Cornflower blue
        (0.4,  '#87CEEB'),  # -0.2: Sky blue
        (0.5,  '#8B4513'),  # 0.0: Brown (bare soil)
        (0.55, '#D2B48C'),  # 0.1: Tan
        (0.6,  '#F4E8C1'),  # 0.2: Light tan
        (0.65, '#FFFF99'),  # 0.3: Light yellow
        (0.7,  '#CCFF66'),  # 0.4: Yellow-green
        (0.75, '#99FF33'),  # 0.5: Light green
        (0.8,  '#66CC33'),  # 0.6: Medium green
        (0.85, '#339900'),  # 0.7: Green
        (0.9,  '#228B22'),  # 0.8: Forest green
        (1.0,  '#006400'),  # 1.0: Dark green
    ]
    return LinearSegmentedColormap.from_list(
        'ndvi', [(pos, color) for pos, color in colors_with_positions], N=200
    )


def visualize(input_path, output_path):
    """Generate NDVI time series visualization."""
    print(f"Loading NDVI data from {input_path}...")
    with rasterio.open(input_path) as src:
        data = src.read()
        bounds = src.bounds

    print(f"Data shape: {data.shape}")
    print(f"Number of bands: {data.shape[0]}")
    print(f"Bounds: {bounds}")

    data = data.astype(float)
    data[data == -9999] = np.nan

    valid_mask = ~np.isnan(data)
    valid_pct = 100 * valid_mask.sum() / valid_mask.size
    valid_data = data[valid_mask]
    ndvi_mean = np.mean(valid_data)
    ndvi_median = np.median(valid_data)

    years = list(range(1990, 2025))
    seasons = ['spring', 'summer', 'fall', 'winter']

    center_lat = (bounds.bottom + bounds.top) / 2
    center_lon = (bounds.left + bounds.right) / 2

    # Extract sample_id and get land cover info
    sample_id = extract_sample_id(input_path.name)
    landcover_info = None
    if sample_id:
        print(f"\nLooking up land cover for sample {sample_id}...")
        landcover_info = get_landcover_info(sample_id)
        if landcover_info:
            print(f"  Land cover: {landcover_info.get('dominant_name', 'Unknown')}")
            print(f"  Vegetation fraction: {landcover_info.get('vegetation_fraction', 0):.1%}")
        else:
            print("  No land cover data found")

    print(f"\nReverse geocoding ({center_lat:.4f}, {center_lon:.4f})...")
    location_name = reverse_geocode(center_lat, center_lon)
    if location_name:
        print(f"  Location: {location_name}")
    else:
        location_name = "Unknown"

    print(f"\nDownloading OSM map for bounds ({bounds.left:.4f} to {bounds.right:.4f}°E)...")
    try:
        osm_map = download_osm_map_for_bounds(bounds, output_size=264)
        print("  OSM map downloaded successfully")
    except Exception as e:
        print(f"  Warning: Failed to download OSM map: {e}")
        osm_map = None

    print("\nDownloading ESA WorldCover map from GEE...")
    worldcover_map = download_worldcover_image(bounds, output_size=264)
    if worldcover_map:
        print("  WorldCover map downloaded successfully")

    fig, axes = plt.subplots(36, 4, figsize=(16, 144))
    title = f'NDVI Time Series - {location_name}\n({center_lat:.4f}°N, {center_lon:.4f}°E)'
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.998)
    plt.subplots_adjust(left=0.05, right=0.92, top=0.99, bottom=0.01, hspace=0.3, wspace=0.1)

    # Build land cover text for info panel
    landcover_text = ""
    if landcover_info:
        dominant_name = landcover_info.get('dominant_name')
        veg_frac = landcover_info.get('vegetation_fraction', 0)
        if dominant_name:
            landcover_text = f"Land Cover: {dominant_name}\nVegetation: {veg_frac:.0%}\n\n"
        else:
            landcover_text = "Land Cover: No data\n\n"

    for col in range(4):
        ax = axes[0, col]
        if col == 0 and osm_map is not None:
            ax.imshow(osm_map)
            ax.set_title('OpenStreetMap (Current)', fontsize=10, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
        elif col == 1 and worldcover_map is not None:
            ax.imshow(worldcover_map)
            ax.set_title('ESA WorldCover (2021)', fontsize=10, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(2)
        elif col == 2:
            info_text = (
                f'{location_name}\n'
                f'{center_lat:.4f}°N, {center_lon:.4f}°E\n\n'
                f'{landcover_text}'
                f'Tile: ~8km × 8km\n'
                f'Pixels: {data.shape[1]}×{data.shape[2]}\n'
                f'Years: {len(years)} ({years[0]}-{years[-1]})\n'
                f'Bands: {data.shape[0]} ({len(years)}×{len(seasons)})\n\n'
                f'Valid pixels: {valid_pct:.1f}%\n'
                f'NDVI mean: {ndvi_mean:.3f}\n'
                f'NDVI median: {ndvi_median:.3f}'
            )
            ax.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=9,
                    transform=ax.transAxes, family='monospace')
            ax.set_title('Info', fontsize=10, fontweight='bold')
        else:
            ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

    cmap = create_ndvi_colormap()
    vmin, vmax = -1, 1

    print("\nGenerating visualization...")
    for year_idx, year in enumerate(years):
        row = year_idx + 1
        for season_idx, season in enumerate(seasons):
            band_idx = year_idx * 4 + season_idx
            ax = axes[row, season_idx]
            im = ax.imshow(data[band_idx], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_title(f'{year} {season.capitalize()}', fontsize=8, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.5)

    cbar_ax = fig.add_axes([0.97, 0.1, 0.01, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('NDVI', fontsize=12, fontweight='bold')
    cbar.set_ticks([-1.0, -0.5, 0, 0.5, 1.0])
    cbar.ax.set_yticklabels(['-1.0\n(Water)', '-0.5', '0.0\n(Bare)', '0.5', '1.0\n(Dense)'])

    print(f"\nSaving visualization to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Done!")

    print(f"\nNDVI Statistics:")
    print(f"  Min: {np.min(valid_data):.3f}")
    print(f"  Max: {np.max(valid_data):.3f}")
    print(f"  Mean: {np.mean(valid_data):.3f}")
    print(f"  Median: {np.median(valid_data):.3f}")
    print(f"  NoData pixels: {np.sum(np.isnan(data))} / {data.size} ({100*np.sum(np.isnan(data))/data.size:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 35-year NDVI time series")
    parser.add_argument("--input", required=True, help="Path to input GeoTIFF file")
    parser.add_argument("--output", help="Path to output PNG file (default: derived from input)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.png')

    visualize(input_path, output_path)
