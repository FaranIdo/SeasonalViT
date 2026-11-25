"""A3 — Split-balance audit between Group A and Group B.

Group A = train ∪ val_iid ∪ ood_t (tiles seen during training/temporal-OOD).
Group B = ood_s ∪ ood_st          (tiles held out for spatial-OOD evaluation).
ood_s and ood_st share the same tile list (different prediction years), so
Group B's unique tile count == |ood_s|.

Outputs (under --output-dir):
    split_balance_landcover.csv   per-class share for each group + abs gap
    split_balance_geography.csv   mean ± std lat, hemisphere %, latitude-band %
    split_balance_koppen.csv      Köppen-zone share per group + abs gap
    split_balance_summary.csv     headline numbers (max gaps)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from _strata_utils import (  # noqa: E402
    LAT_BANDS_PLAN,
    KoppenLookup,
    latitude_band,
    load_landcover_stats,
    load_split_files,
    sample_id_from_filename,
)

LANDCOVER_NAMES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare/sparse vegetation", 70: "Snow and ice",
    80: "Permanent water bodies", 90: "Herbaceous wetland", 95: "Mangroves",
    100: "Moss and lichen",
}
LANDCOVER_CODES = sorted(LANDCOVER_NAMES.keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits-json", default="data/splits_200k.json")
    p.add_argument("--landcover-stats", default="data/landcover_stats.json")
    p.add_argument("--koppen-raster", default=None)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def collect_group_ids(splits: dict[str, list[str]]) -> tuple[set[str], set[str]]:
    a = set()
    for name in ("train", "val_iid", "ood_t"):
        if name in splits:
            a.update(sample_id_from_filename(f) for f in splits[name])
    b = set()
    for name in ("ood_s", "ood_st"):
        if name in splits:
            b.update(sample_id_from_filename(f) for f in splits[name])
    return a, b


def landcover_shares(ids: set[str], stats: dict) -> pd.Series:
    counts = Counter()
    n = 0
    for sid in ids:
        c = stats.get(sid, {}).get("dominant_class")
        if c is None:
            continue
        counts[int(c)] += 1
        n += 1
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series({code: 100.0 * counts.get(code, 0) / n for code in LANDCOVER_CODES})


def geography_summary(ids: set[str], stats: dict) -> dict:
    lats = np.array([stats[sid]["lat"] for sid in ids if sid in stats and stats[sid].get("lat") is not None])
    if lats.size == 0:
        return {"n": 0}
    bands = pd.Series([latitude_band(float(v)) for v in lats]).value_counts(normalize=True) * 100.0
    band_pct = {f"latband_{name}_pct": float(bands.get(name, 0.0)) for name, _, _ in LAT_BANDS_PLAN}
    return {
        "n": int(lats.size),
        "lat_mean": float(lats.mean()),
        "lat_std": float(lats.std()),
        "pct_north": float(100.0 * (lats >= 0).sum() / lats.size),
        "pct_south": float(100.0 * (lats < 0).sum() / lats.size),
        **band_pct,
    }


def koppen_shares(ids: set[str], stats: dict, raster_path: str | None) -> pd.Series:
    lookup = KoppenLookup(raster_path)
    classes = ["A", "B", "C", "D", "E", "unknown"]
    if not lookup.available:
        print(f"[koppen] raster not available at {raster_path!r}; reporting all 'unknown'.")
        return pd.Series({c: (100.0 if c == "unknown" else 0.0) for c in classes})
    lats, lons = [], []
    for sid in ids:
        s = stats.get(sid)
        if s and s.get("lat") is not None and s.get("lon") is not None:
            lats.append(float(s["lat"]))
            lons.append(float(s["lon"]))
    arr_lats = np.array(lats)
    arr_lons = np.array(lons)
    labels = lookup.lookup_many(arr_lats, arr_lons)
    counts = Counter(labels.tolist())
    n = sum(counts.values()) or 1
    return pd.Series({c: 100.0 * counts.get(c, 0) / n for c in classes})


def landcover_table(a: set[str], b: set[str], stats: dict) -> pd.DataFrame:
    sa = landcover_shares(a, stats)
    sb = landcover_shares(b, stats)
    df = pd.DataFrame({
        "code": LANDCOVER_CODES,
        "name": [LANDCOVER_NAMES[c] for c in LANDCOVER_CODES],
        "group_a_pct": [sa.get(c, 0.0) for c in LANDCOVER_CODES],
        "group_b_pct": [sb.get(c, 0.0) for c in LANDCOVER_CODES],
    })
    df["abs_gap_pp"] = (df["group_a_pct"] - df["group_b_pct"]).abs()
    return df.sort_values("abs_gap_pp", ascending=False)


def koppen_table(a: set[str], b: set[str], stats: dict, raster_path: str | None) -> pd.DataFrame:
    sa = koppen_shares(a, stats, raster_path)
    sb = koppen_shares(b, stats, raster_path)
    classes = list(sa.index)
    df = pd.DataFrame({
        "koppen_major": classes,
        "group_a_pct": [sa[c] for c in classes],
        "group_b_pct": [sb[c] for c in classes],
    })
    df["abs_gap_pp"] = (df["group_a_pct"] - df["group_b_pct"]).abs()
    return df


def headline_summary(lc_df: pd.DataFrame, kg_df: pd.DataFrame,
                     a_geo: dict, b_geo: dict,
                     a: set[str], b: set[str]) -> pd.DataFrame:
    rows = [
        ("group_a_n_unique_tiles", float(len(a))),
        ("group_b_n_unique_tiles", float(len(b))),
        ("max_landcover_gap_pp", float(lc_df["abs_gap_pp"].to_numpy().max())),
        ("max_landcover_gap_class", str(lc_df.iloc[0]["name"])),
        ("max_koppen_gap_pp", float(kg_df["abs_gap_pp"].to_numpy().max())),
        ("group_a_lat_mean", float(a_geo.get("lat_mean", 0.0))),
        ("group_b_lat_mean", float(b_geo.get("lat_mean", 0.0))),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats = load_landcover_stats(args.landcover_stats)
    splits = load_split_files(args.splits_json)
    a_ids, b_ids = collect_group_ids(splits)
    print(f"Group A unique tiles: {len(a_ids)}; Group B: {len(b_ids)}")

    lc_df = landcover_table(a_ids, b_ids, stats)
    a_geo = geography_summary(a_ids, stats)
    b_geo = geography_summary(b_ids, stats)
    geo_df = pd.DataFrame({"group_a": a_geo, "group_b": b_geo})
    kg_df = koppen_table(a_ids, b_ids, stats, args.koppen_raster)

    lc_df.to_csv(out / "split_balance_landcover.csv", index=False)
    geo_df.to_csv(out / "split_balance_geography.csv")
    kg_df.to_csv(out / "split_balance_koppen.csv", index=False)

    summary = headline_summary(lc_df, kg_df, a_geo, b_geo, a_ids, b_ids)
    summary.to_csv(out / "split_balance_summary.csv", index=False)

    print(f"\nWrote outputs to {out}.\n")
    print("Headline:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
