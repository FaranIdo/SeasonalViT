"""A1 — Tag the OOD-s 2022 manifest with stratification columns.

Adds: latitude_band, koppen_class, nodata_tier, nodata_mean, dominant_landcover,
      lat, lon. One row per OOD-s 2022 tile.

Inputs:
    --splits-json          data/splits_200k.json
    --landcover-stats      data/landcover_stats.json
    --nodata-summary       output of A2: nodata_per_tile_summary.csv (optional)
    --koppen-raster        Beck et al. 2018 .tif (optional, falls back to "unknown")

Output:
    --output-csv           e.g. data/geo_strata/ood_s_2022_manifest.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from _strata_utils import (  # noqa: E402
    KoppenLookup,
    latitude_band,
    load_landcover_stats,
    load_split_files,
    nodata_tier,
    sample_id_from_filename,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits-json", default="data/splits_200k.json")
    p.add_argument("--landcover-stats", default="data/landcover_stats.json")
    p.add_argument("--nodata-summary", default=None,
                   help="CSV from A2 with sample_id, nodata_mean, nodata_max, n_bands.")
    p.add_argument("--koppen-raster", default=None)
    p.add_argument("--split-name", default="ood_s")
    p.add_argument("--output-csv", required=True)
    return p.parse_args()


def build_base_frame(files: list[str], stats: dict) -> pd.DataFrame:
    rows = []
    for fname in files:
        sid = sample_id_from_filename(fname)
        s = stats.get(sid, {})
        rows.append({
            "sample_id": sid,
            "filename": fname,
            "lat": s.get("lat"),
            "lon": s.get("lon"),
            "dominant_landcover": s.get("dominant_class"),
            "dominant_landcover_name": s.get("dominant_name"),
            "vegetation_fraction": s.get("vegetation_fraction"),
        })
    return pd.DataFrame(rows)


def add_lat_band(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["latitude_band"] = df["lat"].map(lambda v: latitude_band(float(v)) if pd.notna(v) else "unknown")
    df["hemisphere"] = df["lat"].map(lambda v: "N" if pd.notna(v) and v >= 0 else "S")
    return df


def add_koppen(df: pd.DataFrame, raster_path: str | None) -> pd.DataFrame:
    lookup = KoppenLookup(raster_path)
    if not lookup.available:
        print(f"[koppen] raster not available at {raster_path!r}; setting koppen_class='unknown'.")
        df = df.assign(koppen_class="unknown")
        return df
    df = df.copy()
    lats = df["lat"].astype(float).to_numpy()
    lons = df["lon"].astype(float).to_numpy()
    df["koppen_class"] = lookup.lookup_many(lats, lons)
    return df


def add_nodata_tier(df: pd.DataFrame, summary_csv: str | None) -> pd.DataFrame:
    if summary_csv is None:
        print("[nodata] --nodata-summary not provided; nodata_tier='unknown'.")
        df = df.copy()
        df["nodata_mean"] = np.nan
        df["nodata_tier"] = "unknown"
        return df
    summary = pd.read_csv(summary_csv, dtype={"sample_id": str})
    df = df.merge(summary[["sample_id", "nodata_mean", "nodata_max"]], on="sample_id", how="left")
    df["nodata_tier"] = df["nodata_mean"].map(
        lambda v: nodata_tier(float(v)) if pd.notna(v) else "unknown"
    )
    return df


def main() -> None:
    args = parse_args()
    stats = load_landcover_stats(args.landcover_stats)
    splits = load_split_files(args.splits_json)
    if args.split_name not in splits:
        raise SystemExit(f"split '{args.split_name}' not in splits_200k.json (have: {list(splits)})")

    files = splits[args.split_name]
    df = build_base_frame(files, stats)
    df = add_lat_band(df)
    df = add_koppen(df, args.koppen_raster)
    df = add_nodata_tier(df, args.nodata_summary)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows).")
    print(df[["latitude_band", "koppen_class", "nodata_tier"]].apply(pd.Series.value_counts).fillna(0))


if __name__ == "__main__":
    main()
