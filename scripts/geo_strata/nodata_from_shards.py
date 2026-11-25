"""A2 — Missing-data inventory from precomputed shard masks.

Reads `nodata_masks` (bool, [N, 140, 256, 256]) from one or more shard
directories under `data/ndvi_shards/<split>/<split>/shard_*.{safetensors,pt}`.
This is much faster than re-reading the original GeoTIFFs because:
  (a) the conversion step already computed the mask once, and
  (b) we read big contiguous tensor blocks instead of tile-by-tile decompression.

Outputs (under --output-dir):
    nodata_per_tile.parquet     sample_id × band → nodata_frac (used by A1)
    nodata_per_tile_summary.csv per-tile mean / max / count
    nodata_by_lat_season.csv    5×4 grid (mean nodata fraction)
    nodata_by_era.csv           6 sensor eras
    nodata_heatmap.png          heatmap of the lat × season grid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from _strata_utils import (  # noqa: E402
    LAT_BANDS_PLAN,
    SEASONS,
    era_for_year,
    latitude_band,
    parse_band_label,
    sample_id_from_filename,
)

PIXELS_PER_BAND = 256 * 256
BAND_LABELS_140: list[str] = [
    f"{year}_{season}"
    for year in range(1990, 2025)
    for season in SEASONS
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard-dirs", nargs="+", required=True,
                   help="One or more directories containing shard_*.safetensors files.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit-shards", type=int, default=None,
                   help="Process only the first N shards (debug).")
    p.add_argument("--prefer-pt", action="store_true",
                   help="Use shard_*.pt files instead of shard_*.safetensors.")
    return p.parse_args()


def find_shards(dirs: list[str], prefer_pt: bool) -> list[Path]:
    out: list[Path] = []
    pattern = "shard_*.pt" if prefer_pt else "shard_*.safetensors"
    for d in dirs:
        path = Path(d)
        out.extend(sorted(path.glob(pattern)))
    return out


def load_files_for_shard(shard_path: Path) -> list[str] | None:
    sibling = shard_path.with_suffix("").parent / f"{shard_path.stem}.files.json"
    if not sibling.exists():
        return None
    return json.loads(sibling.read_text())


def process_shard_safetensors(path: Path) -> tuple[list[str], np.ndarray]:
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        keys = f.keys()
        masks = f.get_tensor("nodata_masks")  # bool [N, 140, 256, 256]
        files = list(f.get_tensor("files")) if "files" in keys else load_files_for_shard(path) or []
    fracs = masks.reshape(masks.shape[0], masks.shape[1], -1).sum(axis=2) / PIXELS_PER_BAND
    return list(map(_decode_filename, files)), fracs


def process_shard_torch(path: Path) -> tuple[list[str], np.ndarray]:
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    masks = blob["nodata_masks"].numpy().astype(bool)  # [N, 140, 256, 256]
    files = blob.get("files") or load_files_for_shard(path) or []
    fracs = masks.reshape(masks.shape[0], masks.shape[1], -1).sum(axis=2) / PIXELS_PER_BAND
    return list(map(_decode_filename, files)), fracs


def _decode_filename(item) -> str:
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def shard_to_long_df(files: list[str], fracs: np.ndarray) -> pd.DataFrame:
    n_samples, n_bands = fracs.shape
    if not files or len(files) != n_samples:
        sample_ids = [f"unknown_{i}" for i in range(n_samples)]
    else:
        sample_ids = [sample_id_from_filename(f) for f in files]
    if n_bands != 140:
        labels = [f"band{b + 1}" for b in range(n_bands)]
    else:
        labels = BAND_LABELS_140
    sid_arr = np.repeat(sample_ids, n_bands)
    label_arr = np.tile(labels, n_samples)
    years = np.array([_safe_year(lbl) for lbl in label_arr], dtype=object)
    seasons = np.array([_safe_season(lbl) for lbl in label_arr], dtype=object)
    return pd.DataFrame({
        "sample_id": sid_arr,
        "year": years,
        "season": seasons,
        "nodata_frac": fracs.reshape(-1),
    })


def _safe_year(label: str) -> int | None:
    try:
        return parse_band_label(label)[0]
    except Exception:
        return None


def _safe_season(label: str) -> str | None:
    try:
        return parse_band_label(label)[1]
    except Exception:
        return None


def attach_lat_band(df: pd.DataFrame, files_to_lat: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["lat"] = df["sample_id"].map(files_to_lat).astype(float)
    df = df.dropna(subset=["lat", "year", "season"])
    df["lat_band"] = df["lat"].map(latitude_band)
    df["era"] = df["year"].astype(int).map(era_for_year)
    return df


def aggregate_by_lat_season(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(index="lat_band", columns="season",
                           values="nodata_frac", aggfunc="mean")
    pivot = pivot.reindex(index=[b[0] for b in LAT_BANDS_PLAN], columns=SEASONS)
    return pivot


def aggregate_by_era(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("era", observed=True)["nodata_frac"]
    out = g.agg(["mean", "median", "count"]).reset_index()
    return out.rename(columns={"mean": "nodata_mean", "median": "nodata_median", "count": "n_band_obs"})


def render_heatmap(pivot: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    arr = pivot.to_numpy(dtype=float)
    vmax = max(0.5, float(np.nanmax(arr)) if np.isfinite(arr).any() else 0.5)
    im = ax.imshow(arr, aspect="auto", cmap="magma_r", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Season"); ax.set_ylabel("Latitude band")
    ax.set_title("Mean NoData fraction per band")
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            v = arr[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{v*100:.1f}%", ha="center", va="center",
                        color="white" if v > 0.25 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
    fig.savefig(out_path, dpi=160); fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def write_per_tile_summary(df: pd.DataFrame, out_path: Path) -> None:
    g = df.groupby("sample_id")["nodata_frac"]
    summary = g.agg(["mean", "max", "count"]).reset_index()
    summary = summary.rename(columns={"mean": "nodata_mean", "max": "nodata_max", "count": "n_bands"})
    summary.to_csv(out_path, index=False)


def load_landcover_lat_map(path: str) -> dict[str, float]:
    import json as _json

    raw = _json.loads(Path(path).read_text())
    return {sid: float(rec["lat"]) for sid, rec in raw.items() if rec.get("lat") is not None}


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    shards = find_shards(args.shard_dirs, args.prefer_pt)
    if args.limit_shards is not None:
        shards = shards[: args.limit_shards]
    print(f"Found {len(shards)} shards.", flush=True)

    rows = []
    for i, shard in enumerate(shards):
        print(f"  [{i + 1}/{len(shards)}] {shard.name}", flush=True)
        if args.prefer_pt or shard.suffix == ".pt":
            files, fracs = process_shard_torch(shard)
        else:
            files, fracs = process_shard_safetensors(shard)
        rows.append(shard_to_long_df(files, fracs))
        print(f"    samples: {fracs.shape[0]}, mean nodata: {fracs.mean():.4f}", flush=True)

    df = pd.concat(rows, ignore_index=True)
    parquet = out / "nodata_per_tile.parquet"
    df.to_parquet(parquet, index=False)
    print(f"Wrote {parquet} ({len(df)} rows).")

    write_per_tile_summary(df, out / "nodata_per_tile_summary.csv")

    lat_map = load_landcover_lat_map("data/landcover_stats.json")
    df_tagged = attach_lat_band(df, lat_map)

    pivot = aggregate_by_lat_season(df_tagged)
    pivot.to_csv(out / "nodata_by_lat_season.csv")
    render_heatmap(pivot, out / "nodata_heatmap.png")

    era_df = aggregate_by_era(df_tagged)
    era_df.to_csv(out / "nodata_by_era.csv", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
