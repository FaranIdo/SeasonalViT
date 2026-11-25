"""Plot training convergence for the SeasonalViT pretraining and final fine-tuning runs.

Reads the three training.log files from the training runs and produces a
two-panel figure: pretraining (epochs 1-39) and fine-tuning (epochs 1-20).

Outputs:
  results/training_logs/convergence_curves.png
  results/training_logs/convergence_curves.tsv
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "results" / "training_logs"

PRETRAIN_PHASE1_LOG = LOG_DIR / "pretrain_stage1.log"
PRETRAIN_PHASE2_LOG = LOG_DIR / "pretrain_stage2.log"
FINETUNE_LOG = LOG_DIR / "finetune.log"

OUT_PNG = LOG_DIR / "convergence_curves.png"
OUT_PRETRAIN_PNG = LOG_DIR / "convergence_pretrain.png"
OUT_PRETRAIN_PDF = LOG_DIR / "convergence_pretrain.pdf"
OUT_FINETUNE_PNG = LOG_DIR / "convergence_finetune.png"
OUT_FINETUNE_PDF = LOG_DIR / "convergence_finetune.pdf"
OUT_TSV = LOG_DIR / "convergence_curves.tsv"

EPOCH_LINE = re.compile(
    r"Epoch (?P<epoch>\d+) complete \| train_loss=(?P<train>[0-9.]+) \| "
    r"(?:val_loss=(?P<val>[0-9.]+)|val_mae=(?P<val_mae>[0-9.]+))"
)
VAL_DETAIL = re.compile(
    r"Val: loss=(?P<val>[0-9.]+)(?: \| MAE=(?P<mae>[0-9.]+))?"
    r"(?: \| R.=(?P<r2>-?[0-9.]+))?"
)


@dataclass
class EpochRow:
    epoch: int
    train_loss: float
    val_loss: float | None
    val_mae: float | None
    val_r2: float | None


def parse_pretrain(*log_paths: Path) -> list[EpochRow]:
    rows: dict[int, EpochRow] = {}
    pending_val: float | None = None
    for log_path in log_paths:
        for line in log_path.read_text(errors="replace").splitlines():
            if "Val: loss=" in line:
                m = VAL_DETAIL.search(line)
                if m:
                    pending_val = float(m.group("val"))
            m = EPOCH_LINE.search(line)
            if m:
                epoch = int(m.group("epoch"))
                train = float(m.group("train"))
                val = float(m.group("val")) if m.group("val") else pending_val
                rows[epoch] = EpochRow(epoch, train, val, None, None)
                pending_val = None
    return [rows[k] for k in sorted(rows)]


def parse_finetune(log_path: Path) -> list[EpochRow]:
    rows: dict[int, EpochRow] = {}
    pending_val: float | None = None
    pending_mae: float | None = None
    pending_r2: float | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        if "Val: loss=" in line:
            m = VAL_DETAIL.search(line)
            if m:
                pending_val = float(m.group("val"))
                pending_mae = float(m.group("mae")) if m.group("mae") else None
                pending_r2 = float(m.group("r2")) if m.group("r2") else None
        m = EPOCH_LINE.search(line)
        if m:
            epoch = int(m.group("epoch"))
            train = float(m.group("train"))
            val_mae = float(m.group("val_mae")) if m.group("val_mae") else pending_mae
            rows[epoch] = EpochRow(epoch, train, pending_val, val_mae, pending_r2)
            pending_val = pending_mae = pending_r2 = None
    return [rows[k] for k in sorted(rows)]


def write_tsv(pre: list[EpochRow], fine: list[EpochRow]) -> None:
    lines = ["stage\tepoch\ttrain_loss\tval_loss\tval_mae\tval_r2"]
    for row in pre:
        lines.append(
            f"pretrain\t{row.epoch}\t{row.train_loss:.4f}\t"
            f"{row.val_loss:.4f}\t\t"
        )
    for row in fine:
        val = f"{row.val_loss:.4f}" if row.val_loss is not None else ""
        mae = f"{row.val_mae:.4f}" if row.val_mae is not None else ""
        r2 = f"{row.val_r2:.4f}" if row.val_r2 is not None else ""
        lines.append(
            f"finetune\t{row.epoch}\t{row.train_loss:.4f}\t{val}\t{mae}\t{r2}"
        )
    OUT_TSV.write_text("\n".join(lines) + "\n")


PAPER_RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Single column ~= 3.5 in; we use a slightly larger size that scales well.
PAPER_FIGSIZE = (3.6, 2.6)


def plot(pre: list[EpochRow], fine: list[EpochRow]) -> None:
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        plot_pretrain(axes[0], pre)
        plot_finetune(axes[1], fine)
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=200)
        plt.close(fig)
    print(f"Wrote {OUT_PNG}")
    save_single_panel(plot_pretrain, pre, OUT_PRETRAIN_PNG, OUT_PRETRAIN_PDF)
    save_single_panel(plot_finetune, fine, OUT_FINETUNE_PNG, OUT_FINETUNE_PDF)


def save_single_panel(plot_fn, rows, png_path: Path, pdf_path: Path) -> None:
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
        plot_fn(ax, rows)
        fig.tight_layout()
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
        plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def plot_pretrain(ax, rows: list[EpochRow]) -> None:
    epochs = [r.epoch for r in rows]
    train = [r.train_loss for r in rows]
    val = [r.val_loss for r in rows]

    ax.plot(epochs, train, marker="o", ms=3, label="Train", color="#1f77b4")
    ax.plot(epochs, val, marker="s", ms=3, label="Validation", color="#d62728")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction loss")
    ax.set_xlim(0, max(epochs) + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", frameon=False, handlelength=1.5)


def plot_finetune(ax, rows: list[EpochRow]) -> None:
    epochs = [r.epoch for r in rows]
    train = [r.train_loss for r in rows]
    val = [r.val_loss for r in rows]

    ax.plot(epochs, train, marker="o", ms=3, label="Train", color="#1f77b4")
    ax.plot(epochs, val, marker="s", ms=3, label="Validation", color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Forecast loss")
    ax.grid(alpha=0.25, lw=0.5)

    ax.set_xlim(0, max(epochs) + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right", frameon=False, handlelength=1.5)


def main() -> None:
    pre = parse_pretrain(PRETRAIN_PHASE1_LOG, PRETRAIN_PHASE2_LOG)
    fine = parse_finetune(FINETUNE_LOG)
    print(f"Pretrain epochs parsed: {len(pre)} (range {pre[0].epoch}..{pre[-1].epoch})")
    print(f"Finetune epochs parsed: {len(fine)} (range {fine[0].epoch}..{fine[-1].epoch})")
    write_tsv(pre, fine)
    plot(pre, fine)


if __name__ == "__main__":
    main()
