"""Paper-ready single-column figure for the geographic-strata analysis.

Emits a 4-row x 1-column figure sized for a JSTARS single-column slot
(~3.4 in wide). Panels in order:

  (a) Tree cover           - Pool B + val_iid + ood_t tiles where
                             dominant_landcover_name == 'Tree cover'
  (b) Shrubland            - same, 'Shrubland'
  (c) Grassland            - same, 'Grassland'
  (d) All land covers      - every held-out tile regardless of class

Each tile is colored by the standardized NDVI residual

    z(x) = ( NDVI_hat(x, 2022) - NDVI_obs(x, 2022) ) / sigma_NDVI(x)

clipped at +/- 0.5 standardized units. The denominator is the per-tile
temporal standard deviation of seasonal NDVI over the 1990 spring -
2021 winter input window. Sign convention follows the NDVI greening /
browning convention used in remote-sensing change-detection literature:

    green: z < 0  ->  reality greener than predicted  (greening)
    red:   z > 0  ->  reality browner than predicted  (browning)

This script is deliberately separated from residual_map.py: that
module is the exploratory tool, this one is the camera-ready
deliverable. Don't add CLI knobs unless they affect paper output.

Usage:
    python scripts/geo_strata/paper_figure_geo_strata.py \\
        --output figures/geo_strata.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GEO_STRATA_DIR = Path(__file__).resolve().parent
if str(GEO_STRATA_DIR) not in sys.path:
    sys.path.insert(0, str(GEO_STRATA_DIR))

from residual_map import (  # type: ignore[import-not-found]
    DEFAULT_COASTLINE_CACHE,
    load_coastlines,
)

DEFAULT_TSVS = [
    Path("results/geo_strata/geo_strata_per_tile.tsv"),
    Path("results/geo_strata/geo_strata_per_tile_val_iid.tsv"),
    Path("results/geo_strata/geo_strata_per_tile_ood_t.tsv"),
]
DEFAULT_LANDCOVER_STATS = Path("data/landcover_stats.json")
DEFAULT_OUTPUT = Path("results/geo_strata/geo_strata.pdf")

SEASONS = ["spring", "summer", "fall", "winter"]
PANELS: list[tuple[str, str, str | None]] = [
    ("(a)", "Tree cover", "Tree cover"),
    ("(b)", "Shrubland", "Shrubland"),
    ("(c)", "Grassland", "Grassland"),
    ("(d)", "All land covers", None),
]
Z_CLIP = 0.5
DOT_SIZE = 4.0
ALPHA = 0.6
COLUMN_WIDTH_IN = 3.4  # JSTARS single-column nominal width
PANEL_HEIGHT_IN = 1.30  # 360:135 lon:lat aspect at this width
TITLE_PAD = 0.10  # extra inches per panel for the title row
COLORBAR_HEIGHT = 0.35  # inches reserved for the bottom colorbar

GN_W_RD = LinearSegmentedColormap.from_list(
    "GnWRd", ["#1a9850", "#f7f7f7", "#b2182b"], N=256,
)
if "GnWRd" not in mpl.colormaps:
    mpl.colormaps.register(GN_W_RD, name="GnWRd")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-tile-tsv", type=Path, nargs="+", default=DEFAULT_TSVS)
    p.add_argument("--landcover-stats", type=Path, default=DEFAULT_LANDCOVER_STATS)
    p.add_argument("--coastline-cache", type=Path, default=DEFAULT_COASTLINE_CACHE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def load_data(tsvs: list[Path], landcover_stats: Path) -> pd.DataFrame:
    """Concatenate per-split TSVs and join landcover names from stats JSON."""
    import json
    frames = [pd.read_csv(p, sep="\t", dtype={"sample_id": str}) for p in tsvs]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["sample_id"], keep="first")
    with open(landcover_stats) as f:
        stats = json.load(f)
    lc = pd.DataFrame([
        {"sample_id": sid, "dominant_landcover_name": v["dominant_name"]}
        for sid, v in stats.items()
    ])
    return df.merge(lc, on="sample_id", how="left", validate="many_to_one")


def add_z_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bias_annual"] = df[[f"bias_{s}" for s in SEASONS]].mean(axis=1)
    df["z"] = df["bias_annual"] / df["sigma_tile"].replace(0, np.nan)
    df["z_clipped"] = df["z"].clip(-Z_CLIP, Z_CLIP)
    return df


def configure_axes(ax: plt.Axes, coastlines: gpd.GeoDataFrame) -> None:
    """Equirectangular world frame: coastlines + faint graticule, no tick labels."""
    coastlines.plot(ax=ax, color="0.45", linewidth=0.35, zorder=2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 76, 30))
    ax.tick_params(
        which="both", left=False, right=False, top=False, bottom=False,
        labelleft=False, labelbottom=False,
    )
    ax.grid(True, linestyle=":", linewidth=0.3, color="0.75", zorder=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("0.4")
    ax.set_aspect("equal")


def panel_dataframe(df: pd.DataFrame, lc_filter: str | None) -> pd.DataFrame:
    if lc_filter is None:
        return df
    return df[df["dominant_landcover_name"] == lc_filter]


def draw_panel(
    ax: plt.Axes,
    df_panel: pd.DataFrame,
    coastlines: gpd.GeoDataFrame,
    label: str,
    title: str,
) -> mpl.collections.PathCollection | None:
    configure_axes(ax, coastlines)
    sc = None
    if not df_panel.empty:
        sc = ax.scatter(
            df_panel["lon"], df_panel["lat"],
            c=df_panel["z_clipped"], cmap="GnWRd",
            vmin=-Z_CLIP, vmax=Z_CLIP,
            s=DOT_SIZE, linewidths=0, alpha=ALPHA, zorder=3,
        )
    ax.set_title(f"{label} {title}", fontsize=8, loc="center", pad=2)
    return sc


def make_figure(
    df: pd.DataFrame, coastlines: gpd.GeoDataFrame, output: Path,
) -> None:
    n_panels = len(PANELS)
    fig_height = (
        n_panels * (PANEL_HEIGHT_IN + TITLE_PAD)
        + COLORBAR_HEIGHT
        + 0.20  # bottom + top margins
    )
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(COLUMN_WIDTH_IN, fig_height),
        constrained_layout=True,
    )
    last_sc = None
    for ax, (label, title, lc_filter) in zip(axes, PANELS):
        sc = draw_panel(
            ax, panel_dataframe(df, lc_filter), coastlines, label, title,
        )
        if sc is not None:
            last_sc = sc
    if last_sc is None:
        raise RuntimeError("No data drawn on any panel")
    cbar = fig.colorbar(
        last_sc, ax=axes, orientation="horizontal",
        pad=0.02, aspect=30, shrink=0.85,
    )
    cbar.set_label("Standardized NDVI residual", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_ticks([-Z_CLIP, -Z_CLIP / 2, 0.0, Z_CLIP / 2, Z_CLIP])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_data(args.per_tile_tsv, args.landcover_stats)
    df = add_z_score(df)
    coastlines = load_coastlines(args.coastline_cache)
    print(f"Loaded {len(df):,} tiles. Drawing 4-panel paper figure ...", flush=True)
    make_figure(df, coastlines, args.output)
    print(f"Wrote {args.output} (and .png companion).", flush=True)


if __name__ == "__main__":
    main()
