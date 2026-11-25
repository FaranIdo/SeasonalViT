"""Stratified performance tables on OOD-s 2022.

Reads the per-tile TSV produced by `geo_strata_gather.py` and the
strata manifest at `data/geo_strata/ood_s_2022_manifest.csv`, joins them, and
emits three LaTeX tables, one per stratification axis:

  - Table C.1 — by latitude band
  - Table C.2 — by Koppen-Geiger major class
  - Table C.3 — by cloud-contamination tercile (input-window NoData%)

Each table reports n, MAE, RMSE, signed bias, and median sigma_tile
across tiles in the stratum, aggregated over all four 2022 target
seasons. r2 is intentionally omitted: it is unstable on small
per-stratum samples.

Usage:
    python scripts/geo_strata/strata_tables.py \\
        --per-tile-tsv results/geo_strata/geo_strata_per_tile.tsv \\
        --manifest data/geo_strata/ood_s_2022_manifest.csv \\
        --output-dir results/geo_strata
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

SEASONS = ["spring", "summer", "fall", "winter"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-tile-tsv", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def load_and_join(per_tile_tsv: Path, manifest_csv: Path) -> pd.DataFrame:
    per_tile = pd.read_csv(per_tile_tsv, sep="\t", dtype={"sample_id": str})
    manifest = pd.read_csv(manifest_csv, dtype={"sample_id": str})
    keep = ["sample_id", "dominant_landcover_name", "latitude_band",
            "koppen_class", "nodata_mean", "nodata_tier"]
    df = per_tile.merge(manifest[keep], on="sample_id", how="left", validate="one_to_one")
    return df


def collapse_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-tile annual MAE / RMSE / bias as the mean across the 4 seasons.

    Per-tile MAE annual = mean(MAE_spring..MAE_winter). Same for RMSE and
    bias. This matches the per-tile aggregation already used in the
    headline OOD-s table (Section IV-B / Table 5).
    """
    df = df.copy()
    df["mae_annual"] = df[[f"mae_{s}" for s in SEASONS]].mean(axis=1)
    df["rmse_annual"] = df[[f"rmse_{s}" for s in SEASONS]].mean(axis=1)
    df["bias_annual"] = df[[f"bias_{s}" for s in SEASONS]].mean(axis=1)
    return df


def stratum_summary(
    df: pd.DataFrame, group_col: str, group_order: Iterable[str] | None = None,
) -> pd.DataFrame:
    """One row per stratum with n, MAE, RMSE, bias, sigma_tile median."""
    subset = df[df[group_col].notna()].copy()
    grouped = subset.groupby(group_col, sort=False)
    rows = []
    for name, g in grouped:
        rows.append({
            group_col: name,
            "n": len(g),
            "MAE": g["mae_annual"].mean(),
            "RMSE": g["rmse_annual"].mean(),
            "bias": g["bias_annual"].mean(),
            "sigma_tile_median": g["sigma_tile"].median(),
        })
    summary = pd.DataFrame(rows)
    if group_order is not None:
        summary[group_col] = pd.Categorical(
            summary[group_col], categories=list(group_order), ordered=True,
        )
        summary = summary.sort_values(group_col).reset_index(drop=True)
    return summary


def signed_lat_band(lat: float) -> str:
    """Six-band signed latitude binning aligned with the GloSeaN coverage cap."""
    if lat <= -30:
        return "(-55,-30]"
    if lat <= -15:
        return "(-30,-15]"
    if lat <= 0:
        return "(-15,0]"
    if lat <= 15:
        return "(0,15]"
    if lat <= 30:
        return "(15,30]"
    return "(30,55]"


def add_signed_lat_band(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lat_band_signed"] = df["lat"].apply(signed_lat_band)
    return df


def add_cloud_regime_tercile(df: pd.DataFrame) -> pd.DataFrame:
    """Bin tiles into low / medium / high input-window NoData% terciles."""
    df = df.copy()
    df["cloud_regime"] = pd.qcut(
        df["nodata_mean"],
        q=[0, 1 / 3, 2 / 3, 1.0],
        labels=["low", "medium", "high"],
    )
    return df


def fmt_float(v: float, digits: int = 4) -> str:
    if pd.isna(v):
        return "--"
    return f"{v:.{digits}f}"


def fmt_signed(v: float, digits: int = 4) -> str:
    if pd.isna(v):
        return "--"
    return f"{v:+.{digits}f}"


def build_latex_rows(
    summary: pd.DataFrame, group_col: str, group_label: str,
) -> List[str]:
    lines = []
    for _, row in summary.iterrows():
        cells = [
            str(row[group_col]),
            f"{int(row['n']):,}",
            fmt_float(row["MAE"]),
            fmt_float(row["RMSE"]),
            fmt_signed(row["bias"]),
            fmt_float(row["sigma_tile_median"]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    return lines


def write_latex_table(
    summary: pd.DataFrame, group_col: str, group_label: str,
    caption: str, label: str, output: Path,
) -> None:
    body = build_latex_rows(summary, group_col, group_label)
    header = (
        f"{group_label} & $n$ & MAE $\\downarrow$ & RMSE $\\downarrow$ "
        f"& bias & $\\sigma_{{\\mathrm{{tile}}}}$ med. \\\\"
    )
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{l rrrrr}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def write_tsv(summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, sep="\t", index=False, float_format="%.6f")


def emit_lat_band(df: pd.DataFrame, output_dir: Path) -> None:
    df = add_signed_lat_band(df)
    order = ["(-55,-30]", "(-30,-15]", "(-15,0]", "(0,15]", "(15,30]", "(30,55]"]
    summary = stratum_summary(df, "lat_band_signed", group_order=order)
    write_tsv(summary, output_dir / "stage2_table_latitude.tsv")
    write_latex_table(
        summary, "lat_band_signed", r"Latitude band ($^{\circ}$)",
        caption=(r"\rev{Per-tile performance on OOD-s 2022 stratified by signed "
                 r"latitude band ($n=22{,}476$). Metrics are tile means over the "
                 r"four 2022 target seasons.}"),
        label="tab:strata_latitude",
        output=output_dir / "stage2_table_latitude.tex",
    )


def emit_koppen(df: pd.DataFrame, output_dir: Path) -> None:
    order = ["A", "B", "C", "D", "E"]
    keep = df[df["koppen_class"].isin(order)]
    summary = stratum_summary(keep, "koppen_class", group_order=order)
    write_tsv(summary, output_dir / "stage2_table_koppen.tsv")
    write_latex_table(
        summary, "koppen_class", r"K\"oppen class",
        caption=(r"\rev{Per-tile performance on OOD-s 2022 stratified by "
                 r"K\"oppen-Geiger major class. Tiles whose center falls on "
                 r"water and tiles outside the land-only Beck raster are "
                 r"excluded. Metrics are tile means over the four 2022 "
                 r"target seasons.}"),
        label="tab:strata_koppen",
        output=output_dir / "stage2_table_koppen.tex",
    )


def emit_cloud_regime(df: pd.DataFrame, output_dir: Path) -> None:
    df = add_cloud_regime_tercile(df)
    order = ["low", "medium", "high"]
    summary = stratum_summary(df, "cloud_regime", group_order=order)
    summary[["cloud_regime"]] = summary[["cloud_regime"]].astype(str)
    write_tsv(summary, output_dir / "stage2_table_cloud.tsv")
    write_latex_table(
        summary, "cloud_regime", r"Cloud regime",
        caption=(r"\rev{Per-tile performance on OOD-s 2022 stratified by "
                 r"input-window NoData fraction terciles (low/medium/high). "
                 r"Reflects historical cloud contamination across the "
                 r"1990--2021 input window. Metrics are tile means over the "
                 r"four 2022 target seasons.}"),
        label="tab:strata_cloud",
        output=output_dir / "stage2_table_cloud.tex",
    )


def main() -> None:
    args = parse_args()
    df = load_and_join(args.per_tile_tsv, args.manifest)
    df = collapse_seasons(df)
    print(f"Loaded {len(df)} tiles (after manifest join).", flush=True)

    emit_lat_band(df, args.output_dir)
    emit_koppen(df, args.output_dir)
    emit_cloud_regime(df, args.output_dir)
    print(f"Wrote 3 tables (TSV + LaTeX) to {args.output_dir}.", flush=True)


if __name__ == "__main__":
    main()
