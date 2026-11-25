"""A2 — Missing-data inventory across the full 200k-tile clean dataset.

Outputs (under --output-dir):
    nodata_per_tile.parquet      one row per tile × band: sample_id, year, season, nodata_frac
    nodata_per_tile_summary.csv  one row per tile: sample_id, mean nodata, max nodata
    nodata_by_lat_season.csv     5 latitude bands × 4 seasons grid (mean nodata fraction)
    nodata_by_era.csv            6 sensor eras (mean nodata fraction + tile coverage)
    nodata_heatmap.png           heatmap rendering of nodata_by_lat_season
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from _strata_utils import (  # noqa: E402
    SEASONS,
    LAT_BANDS_PLAN,
    era_for_year,
    latitude_band,
    load_landcover_stats,
    parse_band_label,
    sample_id_from_filename,
)

NODATA_VALUE = -9999
DEFAULT_LANDCOVER_STATS = "data/landcover_stats.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles-dir", required=True, help="Directory containing the .tif tiles.")
    p.add_argument("--output-dir", required=True, help="Where to write CSVs and heatmap.")
    p.add_argument("--landcover-stats", default=DEFAULT_LANDCOVER_STATS,
                   help="JSON with sample_id -> {lat, lon, ...}.")
    p.add_argument("--workers", type=int, default=2,
                   help="Worker processes (keep low; disk is the bottleneck).")
    p.add_argument("--limit", type=int, default=None, help="Hard cap (debug).")
    p.add_argument("--sample-size", type=int, default=None,
                   help="Random sample size (after applying --include-files).")
    p.add_argument("--include-files", default=None,
                   help="Path to a text file with one tile filename per line (filter).")
    p.add_argument("--seed", type=int, default=17, help="Random seed for sampling.")
    p.add_argument("--chunk-size", type=int, default=8, help="Tiles per worker chunk.")
    p.add_argument("--resume", action="store_true",
                   help="Skip tiles already represented in nodata_per_tile.parquet.")
    return p.parse_args()


def list_tiles(tiles_dir: Path, limit: int | None, include_file: str | None,
               sample_size: int | None, seed: int) -> list[Path]:
    files = sorted(tiles_dir.glob("*.tif"))
    if include_file:
        wanted = {ln.strip() for ln in Path(include_file).read_text().splitlines() if ln.strip()}
        files = [p for p in files if p.name in wanted]
        print(f"[filter] {len(files)} tiles match --include-files.")
    if sample_size is not None and sample_size < len(files):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(files), size=sample_size, replace=False)
        files = [files[i] for i in sorted(idx.tolist())]
        print(f"[sample] randomly selected {len(files)} tiles (seed={seed}).")
    if limit is not None:
        files = files[:limit]
    return files


def _scan_tile(path_str: str) -> list[tuple] | None:
    """Worker: load all bands once, compute NoData fraction per band."""
    import rasterio

    path = Path(path_str)
    sid = sample_id_from_filename(path.name)
    try:
        with rasterio.open(path) as src:
            descs = list(src.descriptions)
            stack = src.read()  # shape (bands, H, W)
        per_band_total = stack.shape[1] * stack.shape[2]
        bad_per_band = (stack == NODATA_VALUE).reshape(stack.shape[0], -1).sum(axis=1)
        rows: list[tuple] = []
        for b, bad in enumerate(bad_per_band.tolist()):
            label = descs[b] or f"band{b + 1}"
            year, season = _safe_parse_label(label)
            frac = bad / per_band_total if per_band_total else 1.0
            rows.append((sid, year, season, frac))
        return rows
    except Exception as exc:  # pragma: no cover - logged by main
        print(f"[scan-error] {path.name}: {exc}", file=sys.stderr)
        return None


def _safe_parse_label(label: str) -> tuple[int | None, str | None]:
    try:
        return parse_band_label(label)
    except Exception:
        return None, None


def scan_tiles_parallel(paths: list[Path], workers: int, chunk_size: int) -> pd.DataFrame:
    print(f"Scanning {len(paths)} tiles with {workers} workers...", flush=True)
    rows: list[tuple] = []
    with mp.Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_scan_tile, [str(p) for p in paths], chunksize=chunk_size)):
            if result is not None:
                rows.extend(result)
            if (i + 1) % 1000 == 0:
                print(f"  ...{i + 1}/{len(paths)} tiles done", flush=True)
    print(f"Done. {len(rows)} band rows collected.", flush=True)
    return pd.DataFrame(rows, columns=["sample_id", "year", "season", "nodata_frac"])


def load_existing_parquet(parquet_path: Path) -> set[str]:
    if not parquet_path.exists():
        return set()
    print(f"[resume] loading existing {parquet_path.name}...", flush=True)
    df = pd.read_parquet(parquet_path, columns=["sample_id"])
    return set(df["sample_id"].unique().tolist())


def attach_lat(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    lats = df["sample_id"].map(lambda s: stats.get(s, {}).get("lat"))
    df = df.assign(lat=lats.astype(float))
    df = df.dropna(subset=["lat", "year", "season"])
    df["lat_band"] = df["lat"].map(latitude_band)
    df["era"] = df["year"].astype(int).map(era_for_year)
    return df


def aggregate_by_lat_season(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="lat_band", columns="season", values="nodata_frac", aggfunc="mean"
    )
    pivot = pivot.reindex(index=[b[0] for b in LAT_BANDS_PLAN], columns=SEASONS)
    return pivot


def aggregate_by_era(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("era", observed=True)
    out = grouped["nodata_frac"].agg(["mean", "median", "count"]).reset_index()
    out = out.rename(columns={"mean": "nodata_mean", "median": "nodata_median", "count": "n_band_obs"})
    return out


def render_heatmap(pivot: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    arr = pivot.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="magma_r", vmin=0.0, vmax=max(0.5, np.nanmax(arr)))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Season")
    ax.set_ylabel("Latitude band")
    ax.set_title("Mean NoData fraction per band")
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            v = arr[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{v*100:.1f}%", ha="center", va="center",
                        color="white" if v > 0.25 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def write_per_tile_summary(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    g = df.groupby("sample_id")["nodata_frac"]
    summary = g.agg(["mean", "max", "count"]).reset_index()
    summary = summary.rename(columns={"mean": "nodata_mean", "max": "nodata_max", "count": "n_bands"})
    summary.to_csv(out_path, index=False)
    return summary


def main() -> None:
    args = parse_args()
    tiles_dir = Path(args.tiles_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = list_tiles(tiles_dir, args.limit, args.include_files, args.sample_size, args.seed)
    parquet_path = out / "nodata_per_tile.parquet"
    already = load_existing_parquet(parquet_path) if args.resume else set()
    paths = [p for p in paths if sample_id_from_filename(p.name) not in already]
    if not paths:
        print("Nothing to scan (resume hit everything).")
        df_new = pd.DataFrame(columns=["sample_id", "year", "season", "nodata_frac"])
    else:
        df_new = scan_tiles_parallel(paths, args.workers, args.chunk_size)

    if args.resume and parquet_path.exists() and not df_new.empty:
        df_old = pd.read_parquet(parquet_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    elif not df_new.empty:
        df = df_new
    else:
        df = pd.read_parquet(parquet_path)

    df.to_parquet(parquet_path, index=False)
    print(f"Wrote {parquet_path} ({len(df)} rows).")

    write_per_tile_summary(df, out / "nodata_per_tile_summary.csv")

    stats = load_landcover_stats(args.landcover_stats)
    df_tagged = attach_lat(df, stats)

    pivot = aggregate_by_lat_season(df_tagged)
    pivot.to_csv(out / "nodata_by_lat_season.csv")
    render_heatmap(pivot, out / "nodata_heatmap.png")

    era_df = aggregate_by_era(df_tagged)
    era_df.to_csv(out / "nodata_by_era.csv", index=False)

    print(f"Done. Outputs in {out}:")
    for name in ("nodata_per_tile.parquet", "nodata_per_tile_summary.csv",
                 "nodata_by_lat_season.csv", "nodata_by_era.csv", "nodata_heatmap.png"):
        print(f"  {name}")


if __name__ == "__main__":
    main()
