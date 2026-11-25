#!/usr/bin/env python3
"""
Create global map showing dominant land cover class per grid cell.

Divides the world into a grid and colors each cell by the most frequent
land cover class among samples in that cell.

Usage:
    python paper/figures/create_global_map_dominant.py
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not installed, using basic matplotlib")


# ESA WorldCover color palette
LANDCOVER_COLORS = {
    10: '#006400',   # Tree cover (dark green)
    20: '#ffbb22',   # Shrubland (orange)
    30: '#ffff4c',   # Grassland (yellow)
    40: '#f096ff',   # Cropland (pink)
    50: '#fa0000',   # Built-up (red)
    60: '#b4b4b4',   # Bare/sparse vegetation (gray)
    70: '#f0f0f0',   # Snow/ice (white)
    80: '#0064c8',   # Permanent water bodies (blue)
    90: '#0096a0',   # Herbaceous wetland (teal)
    95: '#00cf75',   # Mangroves (green)
    100: '#fae6a0',  # Moss and lichen (beige)
    None: '#888888', # Unknown/no data (dark gray)
}

LANDCOVER_NAMES = {
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
    None: 'No data',
}


def load_landcover_stats(stats_path: Path) -> dict:
    """Load land cover statistics from JSON file."""
    print(f"Loading land cover stats from {stats_path}...")
    with open(stats_path) as f:
        stats = json.load(f)
    print(f"  Loaded {len(stats)} samples")
    return stats


def extract_coordinates_and_classes(stats: dict) -> tuple:
    """Extract lat/lon coordinates and dominant classes from stats."""
    lats = []
    lons = []
    classes = []

    for sample_id, info in stats.items():
        lat = info.get('lat')
        lon = info.get('lon')
        dominant_class = info.get('dominant_class')

        if lat is not None and lon is not None:
            lats.append(lat)
            lons.append(lon)
            classes.append(dominant_class)

    return np.array(lats), np.array(lons), classes


def compute_dominant_grid(lats, lons, classes, grid_size=2.0):
    """
    Compute dominant class for each grid cell.

    Args:
        lats, lons: coordinates
        classes: land cover classes
        grid_size: size of grid cells in degrees

    Returns:
        grid_lons, grid_lats: center coordinates of cells
        grid_colors: color for each cell
        grid_counts: number of samples in each cell
    """
    # Create grid
    lon_bins = np.arange(-180, 180 + grid_size, grid_size)
    lat_bins = np.arange(-90, 90 + grid_size, grid_size)

    # Assign each point to a grid cell
    lon_idx = np.digitize(lons, lon_bins) - 1
    lat_idx = np.digitize(lats, lat_bins) - 1

    # Count classes per cell
    cell_classes = {}
    for i in range(len(lats)):
        cell_key = (lon_idx[i], lat_idx[i])
        if cell_key not in cell_classes:
            cell_classes[cell_key] = []
        cell_classes[cell_key].append(classes[i])

    # Find dominant class per cell
    grid_lons = []
    grid_lats = []
    grid_colors = []
    grid_counts = []

    for (li, la), cell_list in cell_classes.items():
        if 0 <= li < len(lon_bins) - 1 and 0 <= la < len(lat_bins) - 1:
            # Get most common class
            counter = Counter(cell_list)
            dominant = counter.most_common(1)[0][0]

            # Cell center
            cell_lon = lon_bins[li] + grid_size / 2
            cell_lat = lat_bins[la] + grid_size / 2

            grid_lons.append(cell_lon)
            grid_lats.append(cell_lat)
            grid_colors.append(LANDCOVER_COLORS.get(dominant, LANDCOVER_COLORS[None]))
            grid_counts.append(len(cell_list))

    return grid_lons, grid_lats, grid_colors, grid_counts


def create_map_with_cartopy(lats, lons, classes, output_path: Path, grid_size=2.0):
    """Create gridded dominant class map using cartopy."""
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    # Add map features
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#666666')
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='#999999')
    ax.set_global()

    # Compute grid
    print(f"Computing dominant class per {grid_size}° grid cell...")
    grid_lons, grid_lats, grid_colors, grid_counts = compute_dominant_grid(
        lats, lons, classes, grid_size
    )
    print(f"  {len(grid_lons)} cells with samples")

    # Plot as rectangles
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    patches = []
    colors = []
    for glon, glat, color, count in zip(grid_lons, grid_lats, grid_colors, grid_counts):
        # Rectangle from cell corner
        rect = plt.Rectangle(
            (glon - grid_size/2, glat - grid_size/2),
            grid_size, grid_size,
            transform=ccrs.PlateCarree()
        )
        patches.append(rect)
        colors.append(color)

    # Add patches one by one (for proper transform handling)
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('none')
        patch.set_alpha(0.8)
        ax.add_patch(patch)

    # Create legend
    class_counts = Counter(classes)
    legend_elements = []
    for cls in [10, 20, 30, 40, 60, 80, 90]:
        if cls in class_counts and class_counts[cls] > 100:
            color = LANDCOVER_COLORS[cls]
            name = LANDCOVER_NAMES[cls]
            count = class_counts[cls]
            pct = 100 * count / len(lats)
            legend_elements.append(
                plt.scatter([], [], c=color, s=20, marker='s', label=f'{name} ({pct:.1f}%)')
            )

    ax.legend(
        handles=legend_elements,
        loc='lower left',
        fontsize=8,
        framealpha=0.9,
        markerscale=1.5,
    )

    plt.title(f'Dominant Land Cover per {grid_size}° Grid Cell', fontsize=12)
    plt.tight_layout()
    print(f"Saving to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done!")


def create_map_basic(lats, lons, classes, output_path: Path, grid_size=2.0):
    """Create basic gridded map without cartopy (fallback)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    print(f"Computing dominant class per {grid_size}° grid cell...")
    grid_lons, grid_lats, grid_colors, grid_counts = compute_dominant_grid(
        lats, lons, classes, grid_size
    )
    print(f"  {len(grid_lons)} cells with samples")

    # Plot rectangles
    for glon, glat, color in zip(grid_lons, grid_lats, grid_colors):
        rect = plt.Rectangle(
            (glon - grid_size/2, glat - grid_size/2),
            grid_size, grid_size,
            facecolor=color,
            edgecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    # Legend
    class_counts = Counter(classes)
    legend_elements = []
    for cls in [10, 20, 30, 40, 60, 80]:
        if cls in class_counts and class_counts[cls] > 100:
            color = LANDCOVER_COLORS[cls]
            name = LANDCOVER_NAMES[cls]
            pct = 100 * class_counts[cls] / len(lats)
            legend_elements.append(
                plt.scatter([], [], c=color, s=20, marker='s', label=f'{name} ({pct:.1f}%)')
            )

    ax.legend(handles=legend_elements, loc='lower left', fontsize=8)
    plt.title(f'Dominant Land Cover per {grid_size}° Grid Cell', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    stats_path = project_root / "data" / "landcover_stats.json"
    output_path = Path(__file__).parent / "global_distribution_dominant.png"

    # Load data
    stats = load_landcover_stats(stats_path)
    lats, lons, classes = extract_coordinates_and_classes(stats)
    print(f"  {len(lats)} locations with valid coordinates")

    # Create map with 2-degree grid
    grid_size = 2.0
    if HAS_CARTOPY:
        create_map_with_cartopy(lats, lons, classes, output_path, grid_size)
    else:
        create_map_basic(lats, lons, classes, output_path, grid_size)


if __name__ == "__main__":
    main()
