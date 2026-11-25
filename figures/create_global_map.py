#!/usr/bin/env python3
"""
Create global sampling distribution map with land cover coloring.

This script generates a publication-quality world map showing all 250k
sampling locations colored by their dominant ESA WorldCover land cover class.

Usage:
    python paper/figures/create_global_map.py
"""

import json
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


# ESA WorldCover color palette (matching visualize_sample.py)
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


def create_map_with_cartopy(lats, lons, classes, output_path: Path):
    """Create map using cartopy for publication quality."""
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#666666')
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='#999999')
    ax.set_global()

    # Group points by land cover class for efficient plotting
    class_groups = {}
    for lat, lon, cls in zip(lats, lons, classes):
        if cls not in class_groups:
            class_groups[cls] = {'lats': [], 'lons': []}
        class_groups[cls]['lats'].append(lat)
        class_groups[cls]['lons'].append(lon)

    # Plot order: less common classes on top for visibility
    class_counts = {cls: len(data['lats']) for cls, data in class_groups.items()}
    sorted_classes = sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True)

    # Plot each class
    for cls in sorted_classes:
        data = class_groups[cls]
        color = LANDCOVER_COLORS.get(cls, LANDCOVER_COLORS[None])
        ax.scatter(
            data['lons'], data['lats'],
            c=color,
            s=0.5,
            alpha=0.7,
            transform=ccrs.PlateCarree(),
            rasterized=True,  # Improves PDF rendering
        )

    # Create legend (only for classes with significant counts)
    legend_elements = []
    legend_threshold = 100  # Only show classes with >100 samples
    for cls in [10, 20, 30, 40, 60, 80, 90]:  # Main classes
        if cls in class_counts and class_counts[cls] > legend_threshold:
            color = LANDCOVER_COLORS[cls]
            name = LANDCOVER_NAMES[cls]
            count = class_counts[cls]
            pct = 100 * count / len(lats)
            legend_elements.append(
                plt.scatter([], [], c=color, s=20, label=f'{name} ({pct:.1f}%)')
            )

    ax.legend(
        handles=legend_elements,
        loc='lower left',
        fontsize=8,
        framealpha=0.9,
        markerscale=1.5,
    )

    plt.tight_layout()
    print(f"Saving to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done!")


def create_map_basic(lats, lons, classes, output_path: Path):
    """Create basic map without cartopy (fallback)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by class
    class_groups = {}
    for lat, lon, cls in zip(lats, lons, classes):
        if cls not in class_groups:
            class_groups[cls] = {'lats': [], 'lons': []}
        class_groups[cls]['lats'].append(lat)
        class_groups[cls]['lons'].append(lon)

    # Plot order
    class_counts = {cls: len(data['lats']) for cls, data in class_groups.items()}
    sorted_classes = sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True)

    for cls in sorted_classes:
        data = class_groups[cls]
        color = LANDCOVER_COLORS.get(cls, LANDCOVER_COLORS[None])
        ax.scatter(data['lons'], data['lats'], c=color, s=0.5, alpha=0.7)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    # Legend
    legend_elements = []
    for cls in [10, 20, 30, 40, 60, 80]:
        if cls in class_counts and class_counts[cls] > 100:
            color = LANDCOVER_COLORS[cls]
            name = LANDCOVER_NAMES[cls]
            pct = 100 * class_counts[cls] / len(lats)
            legend_elements.append(
                plt.scatter([], [], c=color, s=20, label=f'{name} ({pct:.1f}%)')
            )

    ax.legend(handles=legend_elements, loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    stats_path = project_root / "data" / "landcover_stats.json"
    output_path = Path(__file__).parent / "global_distribution.png"

    # Load data
    stats = load_landcover_stats(stats_path)
    lats, lons, classes = extract_coordinates_and_classes(stats)
    print(f"  {len(lats)} locations with valid coordinates")

    # Count classes
    from collections import Counter
    class_counts = Counter(classes)
    print("\nLand cover distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        name = LANDCOVER_NAMES.get(cls, 'Unknown')
        print(f"  {name}: {count} ({100*count/len(classes):.1f}%)")

    # Create map
    if HAS_CARTOPY:
        create_map_with_cartopy(lats, lons, classes, output_path)
    else:
        create_map_basic(lats, lons, classes, output_path)


if __name__ == "__main__":
    main()
