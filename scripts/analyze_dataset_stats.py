#!/usr/bin/env python3
"""Analyze GloSeaN dataset statistics and generate reports.

This script analyzes:
1. Land cover distribution from ESA WorldCover
2. Geographic distribution of samples
3. Split sizes and configurations
4. Vegetation fraction statistics

Outputs:
- data/dataset_statistics.json: Raw statistics for programmatic use
- docs/dataset_statistics.md: Human-readable analysis report

Usage:
    python scripts/analyze_dataset_stats.py

    # Custom paths
    python scripts/analyze_dataset_stats.py \
        --landcover data/landcover_stats.json \
        --splits data/splits_200k.json \
        --output-json data/dataset_statistics.json \
        --output-md docs/dataset_statistics.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


# ESA WorldCover class definitions
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow/ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# Vegetation classes (for computing vegetation_fraction)
VEGETATION_CLASSES = {10, 20, 30, 40, 90, 95, 100}


def analyze_landcover(landcover_path: Path) -> dict:
    """Analyze land cover statistics."""
    with open(landcover_path) as f:
        stats = json.load(f)

    total = len(stats)

    # Count dominant classes
    class_counts = {}
    veg_fractions = []
    lats = []
    lons = []

    for s in stats.values():
        dom = s.get("dominant_class")
        if dom is not None:
            class_counts[dom] = class_counts.get(dom, 0) + 1
        veg_fractions.append(s.get("vegetation_fraction", 0))
        lats.append(s["lat"])
        lons.append(s["lon"])

    vf = np.array(veg_fractions)
    lats = np.array(lats)
    lons = np.array(lons)

    # Land cover distribution
    landcover_dist = {}
    for code in sorted(class_counts.keys()):
        count = class_counts[code]
        landcover_dist[code] = {
            "name": WORLDCOVER_CLASSES.get(code, "Unknown"),
            "count": count,
            "percentage": round(100 * count / total, 2),
        }

    # Vegetation fraction stats
    veg_stats = {
        "mean": round(float(vf.mean()), 4),
        "median": round(float(np.median(vf)), 4),
        "std": round(float(vf.std()), 4),
        "min": round(float(vf.min()), 4),
        "max": round(float(vf.max()), 4),
    }

    # Vegetation fraction distribution
    veg_distribution = {}
    for low, high, label in [
        (0, 0.2, "0-20%"),
        (0.2, 0.4, "20-40%"),
        (0.4, 0.6, "40-60%"),
        (0.6, 0.8, "60-80%"),
        (0.8, 1.0, "80-<100%"),
        (1.0, 1.01, "100%"),
    ]:
        count = int(((vf >= low) & (vf < high)).sum())
        veg_distribution[label] = {
            "count": count,
            "percentage": round(100 * count / total, 2),
        }

    # Geographic distribution
    geo_stats = {
        "latitude": {
            "min": round(float(lats.min()), 2),
            "max": round(float(lats.max()), 2),
            "mean": round(float(lats.mean()), 2),
        },
        "longitude": {
            "min": round(float(lons.min()), 2),
            "max": round(float(lons.max()), 2),
            "mean": round(float(lons.mean()), 2),
        },
    }

    # Latitude bands
    latitude_bands = {}
    for low, high, name in [
        (-90, -60, "Antarctic"),
        (-60, -30, "S Temperate"),
        (-30, 0, "S Tropical"),
        (0, 30, "N Tropical"),
        (30, 60, "N Temperate"),
        (60, 90, "Arctic"),
    ]:
        count = int(((lats >= low) & (lats < high)).sum())
        latitude_bands[name] = {
            "range": f"{low} to {high}",
            "count": count,
            "percentage": round(100 * count / total, 2),
        }

    # Hemisphere distribution
    north = int((lats >= 0).sum())
    south = int((lats < 0).sum())
    hemisphere = {
        "northern": {"count": north, "percentage": round(100 * north / total, 2)},
        "southern": {"count": south, "percentage": round(100 * south / total, 2)},
    }

    return {
        "total_locations": total,
        "landcover_distribution": landcover_dist,
        "vegetation_fraction": veg_stats,
        "vegetation_fraction_distribution": veg_distribution,
        "geographic": geo_stats,
        "latitude_bands": latitude_bands,
        "hemisphere": hemisphere,
    }


def analyze_splits(splits_path: Path) -> dict:
    """Analyze split configuration."""
    with open(splits_path) as f:
        manifest = json.load(f)

    splits_info = {}
    for split_name, split_info in manifest["splits"].items():
        splits_info[split_name] = {
            "count": split_info["count"],
            "input_years": split_info["input_years"],
            "predict_years": split_info["predict_years"],
        }

    return {
        "total_unique_samples": manifest["total_unique_samples"],
        "n_blocks": manifest["n_blocks"],
        "random_state": manifest["random_state"],
        "created": manifest.get("created"),
        "splits": splits_info,
    }


def generate_markdown(stats: dict, output_path: Path) -> None:
    """Generate markdown report from statistics."""
    lines = [
        "# GloSeaN Dataset Statistics",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"- **Total locations**: {stats['landcover']['total_locations']:,}",
        f"- **Unique samples in splits**: {stats['splits']['total_unique_samples']:,}",
        f"- **K-means blocks**: {stats['splits']['n_blocks']}",
        "",
        "## Train/Validation/Test Splits",
        "",
        "| Split | Samples | Input Years | Predict Years | Purpose |",
        "|-------|---------|-------------|---------------|---------|",
    ]

    split_purposes = {
        "train": "Training",
        "val_iid": "In-distribution validation",
        "ood_t": "Temporal OOD evaluation",
        "ood_s": "Spatial OOD evaluation",
        "ood_st": "Spatial + Temporal OOD",
    }

    for name, info in stats["splits"]["splits"].items():
        input_yrs = f"{info['input_years'][0]}-{info['input_years'][1]}"
        pred_yrs = f"{info['predict_years'][0]}-{info['predict_years'][1]}"
        purpose = split_purposes.get(name, "")
        lines.append(
            f"| {name} | {info['count']:,} | {input_yrs} | {pred_yrs} | {purpose} |"
        )

    lines.extend([
        "",
        "## Land Cover Distribution",
        "",
        "Based on ESA WorldCover v200 (2021) dominant class per tile.",
        "",
        "| Code | Class | Count | Percentage |",
        "|------|-------|------:|----------:|",
    ])

    for code, info in stats["landcover"]["landcover_distribution"].items():
        lines.append(
            f"| {code} | {info['name']} | {info['count']:,} | {info['percentage']:.1f}% |"
        )

    veg = stats["landcover"]["vegetation_fraction"]
    lines.extend([
        "",
        "## Vegetation Fraction",
        "",
        "Fraction of each tile covered by vegetation classes "
        "(tree cover, shrubland, grassland, cropland, wetland, mangroves, moss/lichen).",
        "",
        f"- **Mean**: {veg['mean']:.3f}",
        f"- **Median**: {veg['median']:.3f}",
        f"- **Std Dev**: {veg['std']:.3f}",
        f"- **Range**: {veg['min']:.3f} - {veg['max']:.3f}",
        "",
        "### Distribution",
        "",
        "| Range | Count | Percentage |",
        "|-------|------:|----------:|",
    ])

    for label, info in stats["landcover"]["vegetation_fraction_distribution"].items():
        lines.append(f"| {label} | {info['count']:,} | {info['percentage']:.1f}% |")

    geo = stats["landcover"]["geographic"]
    lines.extend([
        "",
        "## Geographic Distribution",
        "",
        f"- **Latitude**: {geo['latitude']['min']}° to {geo['latitude']['max']}° "
        f"(mean: {geo['latitude']['mean']}°)",
        f"- **Longitude**: {geo['longitude']['min']}° to {geo['longitude']['max']}° "
        f"(mean: {geo['longitude']['mean']}°)",
        "",
        "### Hemisphere",
        "",
        f"- **Northern**: {stats['landcover']['hemisphere']['northern']['count']:,} "
        f"({stats['landcover']['hemisphere']['northern']['percentage']:.1f}%)",
        f"- **Southern**: {stats['landcover']['hemisphere']['southern']['count']:,} "
        f"({stats['landcover']['hemisphere']['southern']['percentage']:.1f}%)",
        "",
        "### Latitude Bands",
        "",
        "| Band | Range | Count | Percentage |",
        "|------|-------|------:|----------:|",
    ])

    for name, info in stats["landcover"]["latitude_bands"].items():
        lines.append(
            f"| {name} | {info['range']}° | {info['count']:,} | {info['percentage']:.1f}% |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- Land cover data from ESA WorldCover v200 (2021) at 10m resolution",
        "- NDVI tiles are 264×264 pixels at 30m Landsat resolution (~7.9km × 7.9km)",
        "- Sampling locations from SSL4EO-L dataset",
        "- Geographic splits use K-means clustering to prevent spatial data leakage",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GloSeaN dataset statistics"
    )
    parser.add_argument(
        "--landcover",
        type=Path,
        default=Path("data/landcover_stats.json"),
        help="Path to landcover statistics JSON",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/splits_200k.json"),
        help="Path to splits manifest JSON",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/dataset_statistics.json"),
        help="Output path for JSON statistics",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/dataset_statistics.md"),
        help="Output path for markdown report",
    )
    args = parser.parse_args()

    print(f"Analyzing landcover stats from: {args.landcover}")
    landcover_stats = analyze_landcover(args.landcover)

    print(f"Analyzing splits from: {args.splits}")
    splits_stats = analyze_splits(args.splits)

    # Combine stats
    all_stats = {
        "generated": datetime.now().isoformat(),
        "landcover": landcover_stats,
        "splits": splits_stats,
    }

    # Save JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Saved JSON statistics to: {args.output_json}")

    # Generate markdown
    generate_markdown(all_stats, args.output_md)
    print(f"Saved markdown report to: {args.output_md}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total locations: {landcover_stats['total_locations']:,}")
    print(f"Unique samples in splits: {splits_stats['total_unique_samples']:,}")
    print(f"\nTop land cover classes:")
    sorted_lc = sorted(
        landcover_stats["landcover_distribution"].items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    )
    for code, info in sorted_lc[:5]:
        print(f"  {info['name']:25s}: {info['count']:6,} ({info['percentage']:.1f}%)")


if __name__ == "__main__":
    main()
