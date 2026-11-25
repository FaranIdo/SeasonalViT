"""Global residual map on OOD-s 2022, faceted by land cover.

Three panels, one per dominant land cover (Tree cover, Shrubland,
Grassland). Each tile is plotted at its centroid (lon, lat) and colored
by signed-bias-over-tile-variability:

    z = bias_annual / sigma_tile

clipped at +/- 2 for visualization. A diverging colormap (red = under-
prediction, white = on-target, green = over-prediction) lets the reader
see *where* the model is biased and *how strongly* relative to each
tile's own natural NDVI variability. The three-LC split decorrelates
biome-driven climate from geography: tree-cover panel is dominated by
boreal-margin and tropical-forest tiles, shrubland by arid belts,
grassland by steppe/savanna.

Coastlines fetched once from Natural Earth (110m) and cached locally.

Usage:
    python scripts/geo_strata/residual_map.py \\
        --per-tile-tsv results/geo_strata/geo_strata_per_tile.tsv \\
        --manifest data/geo_strata/ood_s_2022_manifest.csv \\
        --output figures/geo_strata_map.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import json

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

LAND_COVER_PANELS = [
    "Tree cover", "Shrubland", "Grassland", "Cropland", "Bare/sparse vegetation",
]
SEASONS = ["spring", "summer", "fall", "winter"]
COASTLINE_URL = (
    "https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip"
)
DEFAULT_COASTLINE_CACHE = Path("data/geo_strata/ne_110m_coastline.geojson")
DEFAULT_LANDCOVER_STATS = Path("data/landcover_stats.json")
DEFAULT_SPLITS = Path("data/splits_200k.json")
DEFAULT_Z_CLIP = 0.5  # 90th percentile of |z| across all 3 LCs is ~0.5
DEFAULT_RAW_BIAS_CLIP = 0.10  # NDVI units; ~99th percentile of |bias_annual|
DEFAULT_POOL_A_SUBSAMPLE = 8000  # subsampled gray-context dots

# Custom diverging cmap: zero -> white (visually quiet, "all good"), green
# -> negative bias (reality greener than predicted), red -> positive bias
# (reality browner than predicted). This matches NDVI change-detection
# convention (greening green, browning red): bias = pred - gt, so bias < 0
# means reality > prediction = greening, bias > 0 means reality < prediction
# = browning. Endpoints from ColorBrewer RdYlGn 11-class with the yellow
# midpoint replaced by white. Registered into matplotlib's colormap registry
# the first time the module loads so callers can pass --cmap GnWRd from the CLI.
GN_W_RD = LinearSegmentedColormap.from_list(
    "GnWRd", ["#1a9850", "#f7f7f7", "#b2182b"], N=256,
)
if "GnWRd" not in mpl.colormaps:
    mpl.colormaps.register(GN_W_RD, name="GnWRd")
DEFAULT_CMAP = "GnWRd"

# ESA WorldCover palette (matches Figure 5: global_distribution.png)
LANDCOVER_PALETTE = {
    "Tree cover": "#006400",
    "Shrubland": "#FFBB22",
    "Grassland": "#FFFF4C",
    "Cropland": "#F096FF",
    "Bare/sparse vegetation": "#B4B4B4",
    "Permanent water bodies": "#0064C8",
    "Herbaceous wetland": "#0096A0",
    "Mangroves": "#00CF75",
    "Built-up": "#FA0000",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-tile-tsv", type=Path, nargs="+", required=True,
                   help="One or more TSVs from geo_strata_gather.py. "
                        "Multiple are concatenated (use to combine ood_s + "
                        "val_iid + ood_t for a densified residual map).")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--coastline-cache", type=Path, default=DEFAULT_COASTLINE_CACHE)
    p.add_argument("--dot-size", type=float, default=8.0)
    p.add_argument("--cmap", type=str, default=DEFAULT_CMAP,
                   help="Diverging colormap. Default GnWRd (custom) follows "
                        "NDVI change-detection convention: green = reality "
                        "greener than expected (bias < 0), red = reality "
                        "browner (bias > 0), white midpoint at zero.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Alpha for OOD-s dots. Lower = density visible via overlap.")
    p.add_argument("--z-clip", type=float, default=DEFAULT_Z_CLIP,
                   help="Saturate colormap at +/- z-clip standard deviations. "
                        "Default 0.5 matches the ~90th percentile of |z| and "
                        "produces strongest visual contrast.")
    p.add_argument("--raw-bias-clip", type=float, default=DEFAULT_RAW_BIAS_CLIP,
                   help="Saturate raw-bias map colormap at +/- this value in "
                        "NDVI units. Default 0.10 covers ~99th percentile of "
                        "|bias_annual|.")
    p.add_argument("--landcover-stats", type=Path, default=DEFAULT_LANDCOVER_STATS)
    p.add_argument("--splits-json", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--pool-a-subsample", type=int, default=DEFAULT_POOL_A_SUBSAMPLE,
                   help="How many Pool-A tiles to render as gray context.")
    return p.parse_args()


def load_and_join(
    per_tile_tsvs: list[Path],
    manifest: Path,
    landcover_stats_path: Path,
) -> pd.DataFrame:
    """Load one or more per-tile TSVs, concatenate, join landcover from stats.

    The OOD-s manifest at `manifest` only carries Pool B tiles; landcover for
    val_iid and ood_t comes from `landcover_stats_path` (all 250k tiles).
    Manifest is preserved as the optional source for any future strata
    columns (Koppen, lat band, nodata_tier) but not used here.
    """
    _ = manifest  # reserved for future use
    frames = [pd.read_csv(p, sep="\t", dtype={"sample_id": str}) for p in per_tile_tsvs]
    per_tile = pd.concat(frames, ignore_index=True)
    dup = per_tile["sample_id"].duplicated().sum()
    if dup:
        print(f"WARN: {dup} duplicate sample_ids across splits, keeping first.", flush=True)
        per_tile = per_tile.drop_duplicates(subset=["sample_id"], keep="first")
    with open(landcover_stats_path) as f:
        stats = json.load(f)
    lc = pd.DataFrame([
        {"sample_id": sid, "dominant_landcover_name": v["dominant_name"]}
        for sid, v in stats.items()
    ])
    return per_tile.merge(lc, on="sample_id", how="left", validate="many_to_one")


def add_z_score(df: pd.DataFrame, z_clip: float) -> pd.DataFrame:
    """Compute z = bias_annual / sigma_tile, clipped at +/- z_clip for color."""
    df = df.copy()
    bias_cols = [f"bias_{s}" for s in SEASONS]
    df["bias_annual"] = df[bias_cols].mean(axis=1)
    df["z"] = df["bias_annual"] / df["sigma_tile"].replace(0, np.nan)
    df["z_clipped"] = df["z"].clip(-z_clip, z_clip)
    return df


def add_raw_bias_clipped(df: pd.DataFrame, clip: float) -> pd.DataFrame:
    """Add a clipped copy of `bias_annual` for the raw-bias map encoding."""
    df = df.copy()
    if "bias_annual" not in df.columns:
        bias_cols = [f"bias_{s}" for s in SEASONS]
        df["bias_annual"] = df[bias_cols].mean(axis=1)
    df["bias_annual_clipped"] = df["bias_annual"].clip(-clip, clip)
    return df


def _filename_to_sample_id(filename: str) -> str:
    """Recover sample_id from 'NNNNNNN_*.tif'."""
    return filename.split("_", 1)[0]


def load_pool_a_context(
    landcover_stats_path: Path,
    splits_path: Path,
    subsample: int,
) -> pd.DataFrame:
    """Return a (lat, lon) DataFrame of Pool A locations, optionally subsampled.

    Pool A = train + val_iid + ood_t locations (everything that isn't OOD-s/-st).
    Used to draw a faint gray geographic context behind the OOD-s color dots.
    """
    with open(landcover_stats_path) as f:
        stats = json.load(f)
    with open(splits_path) as f:
        splits = json.load(f)["splits"]
    pool_a_files = (
        splits["train"]["files"] + splits["val_iid"]["files"] + splits["ood_t"]["files"]
    )
    pool_a_ids = {_filename_to_sample_id(name) for name in pool_a_files}
    rows = [
        {"lat": stats[sid]["lat"], "lon": stats[sid]["lon"]}
        for sid in pool_a_ids if sid in stats
    ]
    df = pd.DataFrame(rows)
    if subsample and len(df) > subsample:
        df = df.sample(n=subsample, random_state=17).reset_index(drop=True)
    return df


def load_coastlines(cache_path: Path) -> gpd.GeoDataFrame:
    if cache_path.exists():
        return gpd.read_file(cache_path)
    print(f"Fetching coastlines from Natural Earth -> {cache_path}", flush=True)
    gdf = gpd.read_file(COASTLINE_URL)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache_path, driver="GeoJSON")
    return gdf


def setup_panel_axes(
    ax: plt.Axes,
    coastlines: gpd.GeoDataFrame,
    pool_a: pd.DataFrame | None = None,
) -> None:
    """Draw coastlines, faint Pool-A context, and a graticule on the panel."""
    if pool_a is not None and not pool_a.empty:
        ax.scatter(
            pool_a["lon"], pool_a["lat"],
            c="0.85", s=1.0, linewidths=0, zorder=1,
        )
    coastlines.plot(ax=ax, color="0.4", linewidth=0.4, zorder=2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)  # GloSeaN cap is 55 N, leave headroom for visual balance
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 76, 30))
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, linestyle=":", linewidth=0.4, color="0.7")
    ax.set_aspect("equal")


def plot_one_panel(
    ax: plt.Axes,
    df_lc: pd.DataFrame,
    coastlines: gpd.GeoDataFrame,
    title: str,
    dot_size: float,
    cmap: str,
    clip: float,
    alpha: float,
    pool_a: pd.DataFrame | None,
    value_col: str = "z_clipped",
) -> mpl.collections.PathCollection:
    setup_panel_axes(ax, coastlines, pool_a=pool_a)
    sc = ax.scatter(
        df_lc["lon"],
        df_lc["lat"],
        c=df_lc[value_col],
        cmap=cmap,
        vmin=-clip,
        vmax=clip,
        s=dot_size,
        linewidths=0,
        alpha=alpha,
        zorder=3,
    )
    ax.set_title(f"{title}  (n={len(df_lc):,})", fontsize=9)
    return sc


def plot_global_map(
    df: pd.DataFrame, output: Path, coastline_cache: Path,
    dot_size: float, cmap: str, clip: float, alpha: float,
    value_col: str = "z_clipped",
    cbar_label: str | None = None,
) -> None:
    coastlines = load_coastlines(coastline_cache)
    n_panels = len(LAND_COVER_PANELS)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(7.5, 1.7 * n_panels + 1.5),
        constrained_layout=True,
    )
    last_sc = None
    for ax, lc in zip(axes, LAND_COVER_PANELS):
        df_lc = df[df["dominant_landcover_name"] == lc]
        if df_lc.empty:
            ax.set_title(f"{lc}: no tiles", fontsize=9)
            continue
        last_sc = plot_one_panel(
            ax, df_lc, coastlines, lc, dot_size, cmap, clip, alpha, None,
            value_col=value_col,
        )
        ax.set_xlabel("")
    axes[-1].set_xlabel("Longitude (degrees)", fontsize=8)
    for ax in axes:
        ax.set_ylabel("Lat (deg)", fontsize=8)
    if last_sc is not None:
        cbar = fig.colorbar(
            last_sc, ax=axes, orientation="horizontal",
            pad=0.04, aspect=40, shrink=0.85,
        )
        cbar.set_label(
            cbar_label or (
                r"Tile-mean signed bias / $\sigma_{\mathrm{tile}}$  "
                rf"(saturated at $\pm${clip:g}; "
                r"red: model under-predicts; green: over-predicts)"
            ),
            fontsize=8,
        )
        cbar.ax.tick_params(labelsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_residuals(
    df: pd.DataFrame,
    output: Path,
    coastline_cache: Path,
    dot_size: float,
    cmap: str,
    clip: float,
    alpha: float,
    value_col: str = "z_clipped",
    cbar_label: str | None = None,
    title_suffix: str = "",
) -> None:
    """Single-panel residual map across all tiles, no LC faceting."""
    coastlines = load_coastlines(coastline_cache)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    setup_panel_axes(ax, coastlines, pool_a=None)
    sc = ax.scatter(
        df["lon"], df["lat"],
        c=df[value_col], cmap=cmap,
        vmin=-clip, vmax=clip,
        s=dot_size, linewidths=0, alpha=alpha, zorder=3,
    )
    ax.set_xlabel("Longitude (degrees)", fontsize=8)
    ax.set_ylabel("Latitude (degrees)", fontsize=8)
    splits_present = (
        sorted(df["split"].dropna().unique().tolist())
        if "split" in df.columns else ["unknown"]
    )
    ax.set_title(
        f"Held-out 2022 residuals across all land covers{title_suffix} "
        f"(n={len(df):,}; splits: {', '.join(splits_present)})",
        fontsize=10,
    )
    cbar = fig.colorbar(
        sc, ax=ax, orientation="horizontal",
        pad=0.06, aspect=40, shrink=0.85,
    )
    cbar.set_label(
        cbar_label or (
            r"Tile-mean signed bias / $\sigma_{\mathrm{tile}}$  "
            rf"(saturated at $\pm${clip:g}; "
            r"green: reality greener than predicted (bias $<$ 0); "
            r"red: reality browner than predicted (bias $>$ 0))"
        ),
        fontsize=8,
    )
    cbar.ax.tick_params(labelsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_split_context(
    pool_a: pd.DataFrame,
    pool_b: pd.DataFrame,
    output: Path,
    coastline_cache: Path,
    dot_size: float,
) -> None:
    """One-panel map showing Pool A vs Pool B locations to illustrate the split."""
    coastlines = load_coastlines(coastline_cache)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    coastlines.plot(ax=ax, color="0.4", linewidth=0.4, zorder=2)
    ax.scatter(
        pool_a["lon"], pool_a["lat"],
        c="0.75", s=dot_size, linewidths=0, alpha=0.6, zorder=1,
        label=f"Pool A (train + val_iid + ood_t, subsampled n={len(pool_a):,})",
    )
    ax.scatter(
        pool_b["lon"], pool_b["lat"],
        c="#d62728", s=dot_size + 2, linewidths=0, alpha=0.7, zorder=3,
        label=f"Pool B (OOD-s + ood_st, n={len(pool_b):,})",
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 76, 30))
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, linestyle=":", linewidth=0.4, color="0.7")
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (degrees)", fontsize=8)
    ax.set_ylabel("Latitude (degrees)", fontsize=8)
    ax.set_title(
        "GloSeaN spatial split: Pool A (training pool) vs Pool B (OOD-s holdout)",
        fontsize=10,
    )
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pool_b_landcover(
    df: pd.DataFrame, output: Path, coastline_cache: Path, dot_size: float,
) -> None:
    """One-panel map of all OOD-s tiles colored by dominant land cover."""
    coastlines = load_coastlines(coastline_cache)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    coastlines.plot(ax=ax, color="0.4", linewidth=0.4, zorder=1)
    classes_present = [c for c in LANDCOVER_PALETTE if (df["dominant_landcover_name"] == c).any()]
    for lc in classes_present:
        sub = df[df["dominant_landcover_name"] == lc]
        ax.scatter(
            sub["lon"], sub["lat"],
            c=LANDCOVER_PALETTE[lc], s=dot_size, linewidths=0, alpha=0.7,
            label=f"{lc} ({len(sub):,})", zorder=2,
        )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 76, 30))
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, linestyle=":", linewidth=0.4, color="0.7")
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (degrees)", fontsize=8)
    ax.set_ylabel("Latitude (degrees)", fontsize=8)
    ax.set_title(
        f"Pool B (OOD-s) tiles by dominant ESA WorldCover class (n={len(df):,})",
        fontsize=10,
    )
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def report_panel_stats(df: pd.DataFrame) -> None:
    """Console summary: per-LC count, median |z|, fraction with |z| > 1."""
    print("\n=== Per-LC stats (full TSV) ===")
    rows = []
    for lc in LAND_COVER_PANELS:
        sub = df[df["dominant_landcover_name"] == lc]
        if sub.empty:
            continue
        z = sub["z"].dropna()
        rows.append({
            "landcover": lc,
            "n": len(sub),
            "median_|z|": z.abs().median(),
            "frac_|z|>1": (z.abs() > 1).mean(),
            "frac_|z|>2": (z.abs() > 2).mean(),
        })
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def _derive_sibling(output: Path, suffix: str) -> Path:
    """Append a suffix before the extension: foo.png -> foo<suffix>.png."""
    return output.with_name(f"{output.stem}{suffix}{output.suffix}")


RAW_BIAS_CBAR = (
    "Tile-mean signed bias (NDVI units)  "
    "(green: reality greener than predicted, bias $<$ 0; "
    "red: reality browner than predicted, bias $>$ 0)"
)


def main() -> None:
    args = parse_args()
    df = load_and_join(args.per_tile_tsv, args.manifest, args.landcover_stats)
    df = add_z_score(df, args.z_clip)
    df = add_raw_bias_clipped(df, args.raw_bias_clip)
    print(f"Loaded {len(df)} tiles after manifest join.", flush=True)
    report_panel_stats(df)

    # Residual maps use ALL evaluated tiles (ood_s + val_iid + ood_t).
    # z-score variant: model-quality / anomaly view.
    plot_global_map(
        df, args.output, args.coastline_cache,
        args.dot_size, args.cmap, args.z_clip, args.alpha,
    )
    print(f"Wrote {args.output}", flush=True)

    all_path = _derive_sibling(args.output, "_all")
    plot_all_residuals(
        df, all_path, args.coastline_cache,
        args.dot_size, args.cmap, args.z_clip, args.alpha,
    )
    print(f"Wrote {all_path}", flush=True)

    # Raw-bias variant: change-magnitude view (no normalization by sigma_tile).
    raw_path = _derive_sibling(args.output, "_raw_bias")
    plot_global_map(
        df, raw_path, args.coastline_cache,
        args.dot_size, args.cmap, args.raw_bias_clip, args.alpha,
        value_col="bias_annual_clipped",
        cbar_label=(
            f"{RAW_BIAS_CBAR}; saturated at $\\pm${args.raw_bias_clip:g}"
        ),
    )
    print(f"Wrote {raw_path}", flush=True)

    raw_all_path = _derive_sibling(args.output, "_raw_bias_all")
    plot_all_residuals(
        df, raw_all_path, args.coastline_cache,
        args.dot_size, args.cmap, args.raw_bias_clip, args.alpha,
        value_col="bias_annual_clipped",
        cbar_label=(
            f"{RAW_BIAS_CBAR}; saturated at $\\pm${args.raw_bias_clip:g}"
        ),
        title_suffix=" - raw bias",
    )
    print(f"Wrote {raw_all_path}", flush=True)

    # split_context and pool_b_landcover describe the SPLIT STRUCTURE itself
    # (Pool A vs Pool B, OOD-s land-cover composition), so they stay focused on
    # ood_s tiles even when the residual maps densify with val_iid + ood_t.
    df_ood_s = (
        df[df["split"] == "ood_s"] if "split" in df.columns else df
    )

    pool_a = load_pool_a_context(
        args.landcover_stats, args.splits_json, args.pool_a_subsample,
    )
    pool_b = df_ood_s[["lat", "lon"]].copy()
    split_path = _derive_sibling(args.output, "_split_context")
    plot_split_context(
        pool_a, pool_b, split_path, args.coastline_cache, args.dot_size,
    )
    print(f"Wrote {split_path}", flush=True)

    landcover_path = _derive_sibling(args.output, "_pool_b_landcover")
    plot_pool_b_landcover(
        df_ood_s, landcover_path, args.coastline_cache, args.dot_size,
    )
    print(f"Wrote {landcover_path}", flush=True)


if __name__ == "__main__":
    main()
