"""Download OpenStreetMap image for a specific location using direct tile fetching."""

import math
import warnings
import requests
from PIL import Image
from io import BytesIO

# Suppress SSL warnings for this script (network proxy intercepts SSL)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

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
    headers = {
        'User-Agent': 'AeonProject/1.0 (satellite imagery research; contact@example.com)'
    }
    response = requests.get(url, headers=headers, timeout=10, verify=False)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def download_osm_map(lat, lon, width, height, zoom):
    """Download OSM map centered on lat/lon with given dimensions."""
    tile_size = 256

    center_x, center_y = lat_lon_to_tile(lat, lon, zoom)

    n = 2 ** zoom
    x_float = (lon + 180) / 360 * n
    y_float = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n

    pixel_x = (x_float - center_x) * tile_size
    pixel_y = (y_float - center_y) * tile_size

    tiles_needed_x = math.ceil((width / 2 + pixel_x) / tile_size) + math.ceil((width / 2 - pixel_x) / tile_size)
    tiles_needed_y = math.ceil((height / 2 + pixel_y) / tile_size) + math.ceil((height / 2 - pixel_y) / tile_size)

    start_tile_x = center_x - math.ceil((width / 2 - pixel_x) / tile_size)
    start_tile_y = center_y - math.ceil((height / 2 - pixel_y) / tile_size)

    canvas_width = tiles_needed_x * tile_size
    canvas_height = tiles_needed_y * tile_size
    canvas = Image.new('RGB', (canvas_width, canvas_height))

    for i in range(tiles_needed_x):
        for j in range(tiles_needed_y):
            tile_x = start_tile_x + i
            tile_y = start_tile_y + j
            try:
                tile = download_tile(tile_x, tile_y, zoom)
                canvas.paste(tile, (i * tile_size, j * tile_size))
                print(f"Downloaded tile ({tile_x}, {tile_y})")
            except Exception as e:
                print(f"Failed to download tile ({tile_x}, {tile_y}): {e}")

    center_canvas_x = (center_x - start_tile_x) * tile_size + pixel_x
    center_canvas_y = (center_y - start_tile_y) * tile_size + pixel_y

    left = int(center_canvas_x - width / 2)
    top = int(center_canvas_y - height / 2)
    right = left + width
    bottom = top + height

    cropped = canvas.crop((left, top, right, bottom))

    # Draw red cross at center
    from PIL import ImageDraw
    draw = ImageDraw.Draw(cropped)
    cx, cy = width // 2, height // 2
    draw.line((cx - 10, cy, cx + 10, cy), fill='red', width=3)
    draw.line((cx, cy - 10, cx, cy + 10), fill='red', width=3)

    return cropped


if __name__ == '__main__':
    # 31°33'54"N 34°49'54"E converted to decimal
    lat = 31 + 33/60 + 54/3600  # 31.565
    lon = 34 + 49/60 + 54/3600  # 34.8317

    image = download_osm_map(lat, lon, width=264, height=264, zoom=14)

    output_path = 'data/osm_map_31d565_34d8317.png'
    image.save(output_path)
    print(f"Saved map for location ({lat}, {lon}) to {output_path}")
