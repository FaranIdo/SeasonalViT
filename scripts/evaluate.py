#!/usr/bin/env python3
"""Evaluate a fine-tuned SeasonalViT checkpoint on one shard split.

Scores the checkpoint with FinetuningTrainer.validate(), the same protocol
used for validation during training and by scripts/eval_ablations_on_split.py:
per-sample MAE / RMSE / r^2 / bias on the 256x256 crop with nodata pixels
excluded, then averaged over samples. Each sample is scored on one target
season drawn with a fixed seed from --predict-years. With --per-season, every
target season of those years is also scored on its own.

The model architecture and loss settings are read from the checkpoint
(`model_config` / `config`), so the model is evaluated as it was trained.

Usage:
    # OOD-s / val_iid: 2022 targets
    python scripts/evaluate.py --checkpoint runs/finetune/best.pt \\
        --eval-dir data/shards/ood_s --predict-years 2022

    # OOD-st: one future year, with per-season scores and a prediction grid
    python scripts/evaluate.py --checkpoint runs/finetune/best.pt \\
        --eval-dir data/shards/ood_st --predict-years 2023 --per-season --visualize
"""

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from datasets.sharded_dataset import InMemoryNDVIDataset, create_dataloader
from eval.visualize import create_prediction_grid
from model.config import FinetuningConfig
from scripts.eval_ablations_on_split import build_trainer, load_model_and_train_config
from train.finetune import FinetuningTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FIRST_YEAR = 1990
SEASON_NAMES = ["spring", "summer", "fall", "winter"]
REPORT_KEYS = ["val_mae", "val_rmse", "val_r2", "val_bias", "val_anchor_mae"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned SeasonalViT checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Fine-tuning checkpoint (best.pt)")
    parser.add_argument(
        "--eval-dir", type=Path, required=True,
        help="Directory containing evaluation shards (shard_*.safetensors files)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument(
        "--predict-years", type=int, nargs="+", default=[2022],
        help="Target year(s): 2022 for val_iid / OOD-s, 2023 or 2024 for OOD-t / OOD-st",
    )
    parser.add_argument("--batch-size", type=int, default=3, help="Evaluation batch size")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of samples loaded")
    parser.add_argument("--per-season", action="store_true", help="Also score each target season separately")
    parser.add_argument("--visualize", action="store_true", help="Save a grid of example predictions")
    return parser.parse_args()


def create_eval_dataloader(eval_dir: Path, batch_size: int, limit: Optional[int] = None):
    """Load the evaluation shards into memory and wrap them in a DataLoader."""
    logger.info(f"Loading evaluation data from: {eval_dir}")
    dataset = InMemoryNDVIDataset(eval_dir, max_samples=limit, num_workers=2)
    logger.info(f"Evaluation samples: {len(dataset)}")
    return create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def target_label(index: int) -> str:
    """Human-readable name of a timestep index, e.g. 128 -> '2022 spring'."""
    return f"{FIRST_YEAR + index // 4} {SEASON_NAMES[index % 4]}"


def evaluate_per_season(
    trainer: FinetuningTrainer,
    base_config: FinetuningConfig,
    dataloader,
) -> Dict[str, Dict[str, float]]:
    """Score every target timestep of the predict years on its own."""
    results = {}
    for index in base_config.get_target_indices():
        trainer.config = replace(base_config, predict_target_indices=[index])
        logger.info(f"Scoring target {target_label(index)} (index {index})")
        results[target_label(index)] = trainer.validate(dataloader)
    trainer.config = base_config
    return results


@torch.no_grad()
def collect_examples(
    trainer: FinetuningTrainer,
    dataloader,
    n_samples: int,
) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor]]:
    """Run the validation forward pass on the first batches and keep n_samples outputs."""
    trainer._base_model.eval()
    generator = torch.Generator(device=trainer.device)
    generator.manual_seed(42)
    outputs, targets = [], []
    for batch in dataloader:
        _, output, target, _, batch_size = trainer._validate_batch(batch, generator)
        for j in range(batch_size):
            outputs.append({key: output[key][j:j + 1].float().cpu() for key in ("prediction", "delta", "anchor")})
            targets.append(target[j:j + 1].float().cpu())
        if len(outputs) >= n_samples:
            break
    return outputs[:n_samples], targets[:n_samples]


def save_prediction_grid(trainer: FinetuningTrainer, dataloader, output_dir: Path, n_samples: int = 8) -> None:
    """Save example predictions next to their targets."""
    outputs, targets = collect_examples(trainer, dataloader, n_samples)
    fig = create_prediction_grid(outputs, targets, n_samples=n_samples)
    path = output_dir / "predictions_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved prediction grid to {path}")


def format_row(label: str, metrics: Dict[str, float]) -> str:
    return (
        f"{label:<16s}  {metrics['val_mae']:>7.4f}  {metrics['val_rmse']:>7.4f}  "
        f"{metrics['val_r2']:>6.3f}  {metrics['val_bias']:>+8.4f}  {metrics['val_anchor_mae']:>10.4f}"
    )


def print_results(overall: Dict[str, float], per_season: Dict[str, Dict[str, float]]) -> None:
    """Print overall (and per-season) metrics as a table."""
    header = f"{'Target':<16s}  {'MAE':>7s}  {'RMSE':>7s}  {'r^2':>6s}  {'bias':>8s}  {'anchor MAE':>10s}"
    print("\n" + header)
    print("-" * len(header))
    print(format_row("all (sampled)", overall))
    for label, metrics in per_season.items():
        print(format_row(label, metrics))
    print()


def save_results(
    path: Path,
    args: argparse.Namespace,
    overall: Dict[str, float],
    per_season: Dict[str, Dict[str, float]],
) -> None:
    """Write metrics and the evaluation settings to JSON."""
    results = {
        "checkpoint": str(args.checkpoint),
        "eval_dir": str(args.eval_dir),
        "predict_years": args.predict_years,
        "overall": overall,
        "per_season": per_season,
    }
    path.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved metrics to {path}")


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = create_eval_dataloader(args.eval_dir, args.batch_size, args.limit)
    model, train_config = load_model_and_train_config(args.checkpoint, args.predict_years)
    trainer = build_trainer(model, train_config, args.output_dir)
    logger.info(f"Evaluating {args.checkpoint} on {trainer.device}, predict_years={args.predict_years}")

    overall = trainer.validate(dataloader)
    per_season = evaluate_per_season(trainer, train_config, dataloader) if args.per_season else {}
    print_results(overall, per_season)
    save_results(args.output_dir / "metrics.json", args, overall, per_season)

    if args.visualize:
        save_prediction_grid(trainer, dataloader, args.output_dir)


if __name__ == "__main__":
    main()
