"""Run every ablation's best.pt through FinetuningTrainer.validate() on a
chosen evaluation split. Prints per-run progress and a final TSV of
metrics suitable for direct copy-paste into the paper's ablation table.

Each ablation is loaded once and evaluated in-process (no subprocess
launch), reusing the val_dataset across runs since the eval split is the
same. The model architecture (anchor_mode, temporal_pathway) and
fine-tuning config are read from each checkpoint, so each ablation is
evaluated under the exact recipe it was trained with.

NOTE: val_anchor_mae / val_delta_abs_mean are auto-disabled by the
trainer when anchor_mode='none' and will appear as NaN in the TSV for
the no_anchor ablation by design (not an eval failure).

Usage:
    python scripts/eval_ablations_on_split.py \\
        --runs-dir runs \\
        --eval-dir data/shards/ood_spatial/ood_spatial \\
        --output docs/ablation_results_ood_s.tsv \\
        --predict-years 2022
"""

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from datasets.sharded_dataset import InMemoryNDVIDataset, create_dataloader
from model.config import FinetuningConfig, SeasonalViTConfig
from model.seasonal_vit import SeasonalViT
from train.finetune import FinetuningTrainer

logging.basicConfig(level=logging.WARNING, format="%(message)s")  # quiet child loggers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# (display_label, subdir under runs-dir)
ABLATIONS: List[Tuple[str, str]] = [
    ("SeasonalViT (full)", "seasonalvit_full"),
    ("cross-season only", "ablation_cross_season"),
    ("cross-year only", "ablation_cross_year"),
    ("no anchor", "ablation_no_anchor"),
    ("last-step anchor", "ablation_last_step_anchor"),
    ("no pre-training", "ablation_no_pretrain"),
]

# Fallback model-arch overrides for checkpoints saved BEFORE
# train/finetune.py started persisting `model_config`. New checkpoints
# embed their own model_config; this map is only used when the ckpt
# doesn't have one (`model_config` key missing).
ARCH_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "ablation_cross_season":     {"temporal_pathway": "cross_season"},
    "ablation_cross_year":       {"temporal_pathway": "cross_year"},
    "ablation_no_anchor":        {"anchor_mode": "none"},
    "ablation_last_step_anchor": {"anchor_mode": "last_step"},
}

REPORT_KEYS = [
    "val_mae", "val_rmse", "val_r2", "val_R2_det",
    "val_bias", "val_anchor_mae", "val_delta_abs_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/ablation_results.tsv"))
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument(
        "--predict-years", type=int, nargs="+", default=[2022],
        help="Target year(s) to predict. Override per OOD split: "
             "[2022] for OOD-s/val_id, [2023] or [2024] for OOD-t/OOD-st.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Smoke-test: cap eval samples loaded from --eval-dir.",
    )
    parser.add_argument(
        "--only", type=str, nargs="+", default=None,
        help="Evaluate only the listed subdir names (e.g. ablation_no_anchor). "
             "Default: every entry in ABLATIONS.",
    )
    return parser.parse_args()


def _resolve_model_config(
    ckpt: Dict, ckpt_subdir_name: str,
) -> SeasonalViTConfig:
    """Resolve SeasonalViTConfig for an ablation checkpoint.

    Priority:
      1. `ckpt["model_config"]` if present (new checkpoints — preferred,
         since it's the exact config the model was trained with).
      2. ARCH_OVERRIDES[subdir_name] for older checkpoints saved before
         model_config was persisted. Looked up by the ablation directory
         name, which is the only stable identifier we have for these.
      3. SeasonalViTConfig() defaults (seasonalvit_full and ablation_no_pretrain both use
         defaults).
    """
    if "model_config" in ckpt:
        return ckpt["model_config"]
    overrides = ARCH_OVERRIDES.get(ckpt_subdir_name, {})
    return SeasonalViTConfig(**overrides)


def load_model_and_train_config(
    checkpoint_path: Path,
    predict_years: List[int],
) -> Tuple[SeasonalViT, FinetuningConfig]:
    """Load weights with the model arch the ckpt was trained with.

    Prefers the ckpt's saved `model_config` (new format); falls back to
    ARCH_OVERRIDES keyed on the parent dir name for older checkpoints.
    Forces strict state_dict loading; mismatched shapes (wrong arch)
    raise instead of silently re-initializing layers.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_train_config = ckpt.get("config", FinetuningConfig())

    model_config = _resolve_model_config(ckpt, checkpoint_path.parent.name)
    model = SeasonalViT(model_config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # Force the predict-years for this eval run; keep other training
    # hyperparameters as-saved so anchor_mode / loss weights match.
    # predict_target_indices takes priority over predict_years, so clear it.
    train_config = replace(
        saved_train_config,
        predict_years=list(predict_years),
        predict_target_indices=None,
        temporal_subset_years=0,  # validation always uses all timesteps
    )
    return model, train_config


def build_trainer(
    model: SeasonalViT,
    train_config: FinetuningConfig,
    output_dir: Path,
) -> FinetuningTrainer:
    """Single-GPU trainer using the checkpoint's saved training config."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return FinetuningTrainer(
        model=model, config=train_config, device=device,
        output_dir=output_dir, distributed=False,
    )


def evaluate_one(
    ckpt: Path, val_loader, scratch_dir: Path, predict_years: List[int],
) -> Dict:
    """Load checkpoint with its saved arch, run validation, return metrics."""
    print(f"  Loading {ckpt} ...", flush=True)
    t0 = time.perf_counter()
    model, train_config = load_model_and_train_config(ckpt, predict_years)
    print(
        f"  Arch: anchor_mode={getattr(model.config, 'anchor_mode', 'same_season')} "
        f"temporal_pathway={getattr(model.config, 'temporal_pathway', 'both')} "
        f"predict_years={train_config.predict_years}",
        flush=True,
    )
    trainer = build_trainer(model, train_config, scratch_dir)
    print(f"  Validating on {len(val_loader.dataset)} samples ...", flush=True)
    metrics = trainer.validate(val_loader)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.0f}s. MAE={metrics['val_mae']:.4f}", flush=True)
    return metrics


def collect_results(
    runs_dir: Path,
    val_loader,
    scratch_dir: Path,
    predict_years: List[int],
    only: Optional[List[str]] = None,
) -> List[Tuple[str, Dict]]:
    """Loop over every ablation, evaluate, collect (label, metrics) pairs.

    Re-raises load failures (KeyError, RuntimeError) so silent
    state_dict-shape mismatches don't drop rows from the table; only
    catches CUDA OOM to allow the sweep to continue.
    """
    ablations = ABLATIONS if only is None else [
        (label, subdir) for label, subdir in ABLATIONS if subdir in only
    ]
    results = []
    n = len(ablations)
    for i, (label, subdir) in enumerate(ablations, 1):
        ckpt = runs_dir / subdir / "best.pt"
        print(f"\n[{i}/{n}] {label}", flush=True)
        if not ckpt.exists():
            print(f"  SKIP: {ckpt} not found", flush=True)
            continue
        try:
            metrics = evaluate_one(ckpt, val_loader, scratch_dir, predict_years)
            results.append((label, metrics))
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  OOM (skipping): {exc}", flush=True)
            torch.cuda.empty_cache()
    return results


def write_tsv(results: List[Tuple[str, Dict]], output: Path) -> None:
    """Write per-ablation metrics as a TSV file for paper copy-paste."""
    output.parent.mkdir(parents=True, exist_ok=True)
    header = ["ablation"] + REPORT_KEYS
    lines = ["\t".join(header)]
    for label, metrics in results:
        row = [label] + [f"{metrics.get(k, float('nan')):.6f}" for k in REPORT_KEYS]
        lines.append("\t".join(row))
    output.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {output}", flush=True)


def print_summary_table(results: List[Tuple[str, Dict]]) -> None:
    """Print a summary table that's nicer than raw TSV."""
    print("\n=== Summary ===")
    header = f"{'Ablation':<22s}  {'MAE':>7s}  {'RMSE':>7s}  {'r²':>6s}  {'R²_det':>7s}  {'bias':>7s}"
    print(header)
    print("-" * len(header))
    for label, m in results:
        print(
            f"{label:<22s}  {m['val_mae']:>7.4f}  {m['val_rmse']:>7.4f}  "
            f"{m['val_r2']:>6.3f}  {m['val_R2_det']:>+7.3f}  {m['val_bias']:>+7.4f}"
        )


def main() -> None:
    args = parse_args()
    scratch_dir = Path("/tmp/eval_ablations_scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading val data from {args.eval_dir} ...", flush=True)
    # num_workers=2 because our training machine had SATA SSDs (no NVMe).
    # 8 parallel readers thrash the disk queue; 2 keeps it saturated
    # without contention, which is faster for sequential-bandwidth disks.
    val_dataset = InMemoryNDVIDataset(
        args.eval_dir, max_samples=args.limit, num_workers=2,
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, prefetch_factor=2,
    )
    print(f"Val samples: {len(val_dataset)}  predict_years={args.predict_years}", flush=True)

    results = collect_results(
        args.runs_dir, val_loader, scratch_dir, args.predict_years, args.only,
    )
    print_summary_table(results)
    write_tsv(results, args.output)


if __name__ == "__main__":
    main()
