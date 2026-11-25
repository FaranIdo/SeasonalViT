"""Run a single SeasonalViT checkpoint at multiple forecast horizons on a
chosen evaluation split. For each horizon h in {1, 2, 4, 8} (seasons after
the last input observation), pins the target to a specific timestep and
runs FinetuningTrainer.validate(). Writes a TSV of metrics per horizon
and prints a summary table.

Fine-grained
horizon analysis to show how performance degrades with lead time. Default
split is OOD-st (predict 2023-2024 at unseen locations), which is the
only split with 8 future seasons available from one anchor.

Horizon -> absolute target index (T=140 sequence, 1990 spring at idx 0):
  h=1 -> 132 (2023 spring),  h=2 -> 133 (2023 summer),
  h=4 -> 135 (2023 winter),  h=8 -> 139 (2024 winter).

The model is loaded once per horizon with the same checkpoint; only the
training-config target indices change. Validation uses all available
input timesteps before the target.

Usage:
    python scripts/eval_horizons.py \\
        --checkpoint checkpoints/seasonalvit_finetune.pt \\
        --eval-dir data/shards/ood_spatiotemporal/ood_spatiotemporal \\
        --output docs/horizon_results_ood_st.tsv
"""

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from datasets.sharded_dataset import InMemoryNDVIDataset, create_dataloader
from model.config import FinetuningConfig
from model.seasonal_vit import SeasonalViT
from train.finetune import FinetuningTrainer
from scripts.eval_ablations_on_split import (
    _resolve_model_config,
    build_trainer,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Surface shard-loader progress (file count, parallel-load timings,
# allocation sizes, copy throughput) so a long warm-up isn't a silent hang.
logging.getLogger("datasets.sharded_dataset").setLevel(logging.INFO)

LAST_INPUT_IDX = 131  # 2022 winter (the anchor for OOD-st / OOD-t horizons)
DEFAULT_HORIZONS = [1, 2, 4, 8]
SEASON_NAMES = ["spring", "summer", "fall", "winter"]

REPORT_KEYS = [
    "val_mae", "val_rmse", "val_r2", "val_R2_det",
    "val_bias", "val_anchor_mae", "val_delta_abs_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to the fine-tuned best.pt (or any single checkpoint).")
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="Shard dir for the eval split (default expects OOD-st).")
    parser.add_argument("--output", type=Path,
                        default=Path("docs/horizon_results.tsv"))
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=DEFAULT_HORIZONS,
                        help="Seasons after last input. Default: 1 2 4 8.")
    parser.add_argument("--last-input-idx", type=int, default=LAST_INPUT_IDX,
                        help="Absolute timestep index of the last input "
                             "observation. 131=2022 winter (OOD-t/OOD-st), "
                             "127=2021 winter (OOD-s).")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke-test: cap eval samples loaded.")
    return parser.parse_args()


def horizon_to_index(h: int, last_input_idx: int) -> int:
    """Convert horizon-in-seasons to absolute timestep index."""
    return last_input_idx + h


def index_to_label(idx: int) -> str:
    """Pretty label like '2023 spring' for an absolute timestep index."""
    year = 1990 + idx // 4
    season = SEASON_NAMES[idx % 4]
    return f"{year} {season}"


def load_model_and_trainer_once(
    checkpoint_path: Path,
    scratch_dir: Path,
) -> FinetuningTrainer:
    """Load the checkpoint once and build a trainer we'll reuse across horizons.

    The same model serves every horizon; only `config.predict_target_indices`
    changes between calls. Reloading the model per horizon (as the ablation
    script does, because each ablation has a different checkpoint) wastes
    ~5 GB of VRAM allocations and creates fragmentation that cascades into
    OOMs on later horizons.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_train_config = ckpt.get("config", FinetuningConfig())

    model_config = _resolve_model_config(ckpt, checkpoint_path.parent.name)
    model = SeasonalViT(model_config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # Start with predict_years cleared and temporal subsampling off; the
    # per-horizon loop sets predict_target_indices on this same config.
    train_config = replace(
        saved_train_config,
        predict_target_indices=None,
        predict_years=None,
        temporal_subset_years=0,
    )
    print(
        f"  Arch: anchor_mode={getattr(model.config, 'anchor_mode', 'same_season')} "
        f"temporal_pathway={getattr(model.config, 'temporal_pathway', 'both')}",
        flush=True,
    )
    return build_trainer(model, train_config, scratch_dir)


def evaluate_one_horizon(
    h: int, last_input_idx: int, trainer: FinetuningTrainer, val_loader,
) -> Dict:
    """Pin target to horizon h on the existing trainer, run validate(), return metrics.

    Mutates `trainer.config.predict_target_indices` in place. The trainer's
    `_sample_target_indices` reads this at runtime, so no model reload is
    needed between horizons.
    """
    target_idx = horizon_to_index(h, last_input_idx)
    label = index_to_label(target_idx)
    print(f"\n[h={h}] target_idx={target_idx} ({label})", flush=True)
    t0 = time.perf_counter()
    trainer.config.predict_target_indices = [target_idx]
    print(f"  Validating on {len(val_loader.dataset)} samples ...", flush=True)
    metrics = trainer.validate(val_loader)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.0f}s. MAE={metrics['val_mae']:.4f}", flush=True)
    return metrics


def collect_results(
    horizons: List[int], last_input_idx: int, trainer: FinetuningTrainer,
    val_loader, output: Path,
) -> List[Tuple[int, Dict]]:
    """Loop horizons, evaluate each, collect (h, metrics) pairs.

    Rewrites `output` after every completed horizon so a mid-sweep kill
    leaves you with the rows you've already paid for. Each rewrite is
    idempotent and tiny (the TSV is at most 8 rows + header).
    """
    results = []
    for h in horizons:
        try:
            metrics = evaluate_one_horizon(h, last_input_idx, trainer, val_loader)
            results.append((h, metrics))
            write_tsv(results, output, last_input_idx)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  OOM (skipping h={h}): {exc}", flush=True)
            torch.cuda.empty_cache()
    return results


def write_tsv(
    results: List[Tuple[int, Dict]], output: Path, last_input_idx: int,
) -> None:
    """Write per-horizon metrics as a TSV file for paper copy-paste."""
    output.parent.mkdir(parents=True, exist_ok=True)
    header = ["horizon_seasons", "target_idx", "target_label"] + REPORT_KEYS
    lines = ["\t".join(header)]
    for h, metrics in results:
        idx = horizon_to_index(h, last_input_idx)
        row = [str(h), str(idx), index_to_label(idx)] + [
            f"{metrics.get(k, float('nan')):.6f}" for k in REPORT_KEYS
        ]
        lines.append("\t".join(row))
    output.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {output}", flush=True)


def print_summary_table(
    results: List[Tuple[int, Dict]], last_input_idx: int,
) -> None:
    """Print a summary table that's nicer than raw TSV."""
    print("\n=== Summary ===")
    header = (
        f"{'h':>3s}  {'target':<14s}  {'MAE':>7s}  {'RMSE':>7s}  "
        f"{'r²':>6s}  {'R²_det':>7s}  {'bias':>7s}"
    )
    print(header)
    print("-" * len(header))
    for h, m in results:
        idx = horizon_to_index(h, last_input_idx)
        print(
            f"{h:>3d}  {index_to_label(idx):<14s}  "
            f"{m['val_mae']:>7.4f}  {m['val_rmse']:>7.4f}  "
            f"{m['val_r2']:>6.3f}  {m['val_R2_det']:>+7.3f}  "
            f"{m['val_bias']:>+7.4f}"
        )


def main() -> None:
    args = parse_args()
    scratch_dir = Path("/tmp/eval_horizons_scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading val data from {args.eval_dir} ...", flush=True)
    val_dataset = InMemoryNDVIDataset(
        args.eval_dir, max_samples=args.limit, num_workers=2,
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, prefetch_factor=2,
    )
    print(
        f"Val samples: {len(val_dataset)}  "
        f"horizons={args.horizons}  last_input_idx={args.last_input_idx} "
        f"({index_to_label(args.last_input_idx)})",
        flush=True,
    )

    print(f"Loading checkpoint {args.checkpoint} ...", flush=True)
    trainer = load_model_and_trainer_once(args.checkpoint, scratch_dir)

    results = collect_results(
        args.horizons, args.last_input_idx, trainer,
        val_loader, args.output,
    )
    print_summary_table(results, args.last_input_idx)


if __name__ == "__main__":
    main()
