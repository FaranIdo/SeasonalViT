"""Plot val-MAE learning curves for all SeasonalViT ablations.

Parses each ablation's `training.log` for per-epoch val MAE and draws
one overlaid learning-curve figure for the paper. Also includes two
reference horizontal lines: the full model's best val MAE (ep 17, 0.0719) and naive
same-season persistence (val_anchor_mae = 0.0839).

Usage (from the repo root):
    python scripts/plot_ablation_curves.py \
        --runs-dir runs \
        --output docs/ablation_curves.png

--runs-dir expects a directory that contains the ablation_* subdirs,
each holding a training.log. Works on local or remote-mounted paths.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# (label, subdir name under runs-dir, matplotlib color, marker)
ABLATIONS: List[Tuple[str, str, str, str]] = [
    ("SeasonalViT (full)", "seasonalvit_full", "#1f77b4", "o"),
    ("cross-season only", "ablation_cross_season", "#2ca02c", "s"),
    ("cross-year only", "ablation_cross_year", "#ff7f0e", "^"),
    ("no anchor", "ablation_no_anchor", "#d62728", "D"),
    ("last-step anchor", "ablation_last_step_anchor", "#8c564b", "P"),
    ("no pre-training", "ablation_no_pretrain", "#9467bd", "v"),
]

FULL_MODEL_BEST = 0.0719  # epoch 17
ANCHOR_MAE = 0.0839
VAL_MAE_RE = re.compile(r"Val: loss=[\d.]+ \| MAE=(?P<mae>[\d.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/ablation_curves.png"))
    parser.add_argument("--max-epochs", type=int, default=10)
    return parser.parse_args()


def parse_val_mae_per_epoch(log_path: Path, max_epochs: int) -> List[float]:
    """Return a list of val MAE per epoch, in order, from the training log."""
    if not log_path.exists():
        print(f"  (skip) missing log: {log_path}")
        return []
    maes = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = VAL_MAE_RE.search(line)
            if m:
                maes.append(float(m.group("mae")))
                if len(maes) >= max_epochs:
                    break
    return maes


def build_curves(runs_dir: Path, max_epochs: int) -> Dict[str, List[float]]:
    """Parse every ablation log and return {label: [val_mae per epoch]}."""
    curves: Dict[str, List[float]] = {}
    for label, subdir, _, _ in ABLATIONS:
        log_path = runs_dir / subdir / "training.log"
        maes = parse_val_mae_per_epoch(log_path, max_epochs)
        curves[label] = maes
        print(f"  {label:22s} {len(maes):2d} epochs  best={min(maes):.4f}" if maes else f"  {label}: no data")
    return curves


def _plot_curve(ax, epochs, maes, color, marker, label):
    ax.plot(epochs, maes, color=color, marker=marker, linewidth=1.5,
            markersize=5, label=label)


def _plot_reference_lines(ax) -> None:
    ax.axhline(y=FULL_MODEL_BEST, color="#1f77b4", linestyle="--", linewidth=1,
               alpha=0.5, label=f"full model ep17 best ({FULL_MODEL_BEST})")
    ax.axhline(y=ANCHOR_MAE, color="gray", linestyle=":", linewidth=1,
               alpha=0.7, label=f"naive persistence ({ANCHOR_MAE})")


def render_plot(curves: Dict[str, List[float]], output: Path) -> None:
    """Render overlaid learning curves to a PNG."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=140)
    for label, _subdir, color, marker in ABLATIONS:
        maes = curves.get(label, [])
        if not maes:
            continue
        epochs = list(range(1, len(maes) + 1))
        _plot_curve(ax, epochs, maes, color, marker, label)
    _plot_reference_lines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE (NDVI)")
    ax.set_title("SeasonalViT ablation study — val MAE vs epoch (50 shards, 10 ep budget)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    ax.set_xticks(range(1, 11))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    print(f"Wrote {output}")


def main() -> None:
    args = parse_args()
    print(f"Parsing ablation logs under: {args.runs_dir}")
    curves = build_curves(args.runs_dir, args.max_epochs)
    render_plot(curves, args.output)


if __name__ == "__main__":
    main()
