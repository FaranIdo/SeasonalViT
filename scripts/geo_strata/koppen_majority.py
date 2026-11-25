"""Compute Koppen tags via majority lookup over tile bounds (not center pixel).

Outputs (under data/geo_strata/):
    koppen_majority.csv          sample_id -> koppen_majority (A/B/C/D/E/Unknown)
    nodata_by_koppen_season.csv  5x4 grid (mean NoData fraction by Koppen x season)
    split_balance_koppen_v2.csv  Pool A vs Pool B by majority-Koppen (replaces v1)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from _strata_utils import (  # noqa: E402
    KoppenLookup,
    SEASONS,
    sample_id_from_filename,
)

KOPPEN_ORDER = ["A", "B", "C", "D", "E"]
TILE_HALF_KM = 4.0  # tiles are ~8 km wide; sample +/- 4 km
LAT_DEG_PER_KM = 1.0 / 111.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--landcover-stats", default="data/landcover_stats.json")
    p.add_argument("--splits-json", default="data/splits_200k.json")
    p.add_argument("--nodata-parquet", default="data/geo_strata/nodata_per_tile.parquet")
    p.add_argument("--koppen-raster", default="data/external/Beck_KG_V1/Beck_KG_V1_present_0p0083.tif")
    p.add_argument("--output-dir", default="data/geo_strata")
    p.add_argument("--grid-half", type=int, default=4,
                   help="Half-size of pixel grid sampled around the tile center (px).")
    return p.parse_args()


def lookup_majority(lookup: KoppenLookup, lat: float, lon: float, half: int) -> str:
    """Sample a (2*half+1) grid of pixels around (lat,lon) and return the majority Koppen major class."""
    if not lookup.available or lookup._array is None or lookup._transform is None:
        return "Unknown"
    inv = ~lookup._transform
    h, w = lookup._array.shape
    counter: Counter = Counter()
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            x, y = inv * (lon, lat)
            r, c = int(round(y)) + dr, int(round(x)) + dc
            if 0 <= r < h and 0 <= c < w:
                code = int(lookup._array[r, c])
                if code > 0:
                    label = lookup.MAJOR_BY_CODE.get(code, None)
                    if label is not None:
                        counter[label] += 1
    if not counter:
        return "Unknown"
    return counter.most_common(1)[0][0]


def tag_all_tiles(stats: dict, lookup: KoppenLookup, half: int) -> pd.DataFrame:
    rows = []
    for sid, rec in stats.items():
        lat = rec.get("lat")
        lon = rec.get("lon")
        if lat is None or lon is None:
            rows.append((sid, "Unknown"))
            continue
        rows.append((sid, lookup_majority(lookup, float(lat), float(lon), half)))
    return pd.DataFrame(rows, columns=["sample_id", "koppen_majority"])


def split_balance(tags: pd.DataFrame, splits: dict) -> pd.DataFrame:
    pool_a_files = (
        set(splits["splits"]["train"]["files"])
        | set(splits["splits"]["val_iid"]["files"])
        | set(splits["splits"]["ood_t"]["files"])
    )
    pool_b_files = (
        set(splits["splits"]["ood_s"]["files"])
        | set(splits["splits"]["ood_st"]["files"])
    )
    pool_a_ids = {sample_id_from_filename(f) for f in pool_a_files}
    pool_b_ids = {sample_id_from_filename(f) for f in pool_b_files}

    tags = tags.set_index("sample_id")
    a_counts = tags.loc[tags.index.intersection(pool_a_ids), "koppen_majority"].value_counts()
    b_counts = tags.loc[tags.index.intersection(pool_b_ids), "koppen_majority"].value_counts()

    rows = []
    classes = KOPPEN_ORDER + ["Unknown"]
    for c in classes:
        a_pct = 100.0 * a_counts.get(c, 0) / max(1, a_counts.sum())
        b_pct = 100.0 * b_counts.get(c, 0) / max(1, b_counts.sum())
        rows.append((c, a_pct, b_pct, b_pct - a_pct))
    return pd.DataFrame(rows, columns=["zone", "pool_a_pct", "pool_b_pct", "delta_pp"])


def nodata_by_koppen_season(tags: pd.DataFrame, parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    merged = df.merge(tags, on="sample_id", how="inner")
    pivot = merged.pivot_table(
        index="koppen_majority", columns="season",
        values="nodata_frac", aggfunc="mean"
    )
    pivot = pivot.reindex(index=KOPPEN_ORDER + ["Unknown"], columns=SEASONS)
    return (pivot * 100.0).round(2)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    stats = json.loads(Path(args.landcover_stats).read_text())
    splits = json.loads(Path(args.splits_json).read_text())
    lookup = KoppenLookup(args.koppen_raster)
    if not lookup.available:
        raise SystemExit(f"Koppen raster not found at {args.koppen_raster}")

    print(f"Tagging {len(stats)} tiles via {2*args.grid_half+1}x{2*args.grid_half+1} pixel majority...", flush=True)
    tags = tag_all_tiles(stats, lookup, args.grid_half)
    tags.to_csv(out / "koppen_majority.csv", index=False)
    print(f"  wrote {out / 'koppen_majority.csv'} ({len(tags)} rows)")
    print("Distribution:", tags["koppen_majority"].value_counts().to_dict())

    sb = split_balance(tags, splits)
    sb.to_csv(out / "split_balance_koppen_v2.csv", index=False)
    print(f"\nSplit balance v2:\n{sb.to_string(index=False)}\n")

    nd = nodata_by_koppen_season(tags, args.nodata_parquet)
    nd.to_csv(out / "nodata_by_koppen_season.csv")
    print(f"NoData by Koppen x Season (%):\n{nd.to_string()}\n")


if __name__ == "__main__":
    main()
