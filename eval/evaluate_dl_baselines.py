#!/usr/bin/env python3
"""Evaluate deep learning baseline models for NDVI prediction.

Shared evaluation script for ConvLSTM, PredRNN, and Prithvi baselines.
Loads a trained checkpoint, runs inference on evaluation shards, and
produces the same JSON output format as evaluate_baselines.py.

Usage:
    python eval/evaluate_dl_baselines.py \
        --model convlstm \
        --checkpoint outputs/convlstm/best.pt \
        --split ood_t \
        --shard-root /path/to/shards \
        --output-dir outputs/convlstm/eval

    python eval/evaluate_dl_baselines.py \
        --model predrnn \
        --checkpoint outputs/predrnn/best.pt \
        --split val_iid \
        --shard-root /path/to/shards

    python eval/evaluate_dl_baselines.py \
        --model prithvi \
        --checkpoint outputs/prithvi/best.pt \
        --split ood_s \
        --shard-root /path/to/shards
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.shard_io import load_shard
from eval.evaluate_baselines import (
    SEASON_NAMES,
    SPLIT_CONFIGS,
    aggregate_results,
    compute_batch_stats,
    extract_sample_id,
    log_method_results,
    print_comparison_table,
    create_comparison_plot,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate DL baselines")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["convlstm", "predrnn", "prithvi"],
        help="Model type to evaluate",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split", type=str, required=True,
        choices=["val_iid", "ood_t", "ood_s", "ood_st"],
    )
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--predict-years", type=int, nargs="+", default=None)
    parser.add_argument("--context-gap-years", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[32, 64],
        help="Hidden dims for ConvLSTM/PredRNN (must match training)",
    )
    return parser.parse_args()


def load_model(
    model_type: str,
    checkpoint_path: Path,
    device: torch.device,
    hidden_dims: List[int],
) -> torch.nn.Module:
    """Load a trained DL baseline model.

    Args:
        model_type: One of "convlstm", "predrnn", "prithvi".
        checkpoint_path: Path to saved checkpoint.
        device: Target device.
        hidden_dims: Hidden dimensions (for ConvLSTM/PredRNN).

    Returns:
        Loaded model in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    model = _create_model(model_type, hidden_dims)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded {model_type} model ({n_params:,} params) from {checkpoint_path}")
    return model


def _create_model(model_type: str, hidden_dims: List[int]) -> torch.nn.Module:
    """Instantiate model by type."""
    if model_type == "convlstm":
        from baselines.convlstm.model import ConvLSTMModel
        return ConvLSTMModel(hidden_dims=hidden_dims)
    elif model_type == "predrnn":
        from baselines.predrnn.model import PredRNNModel
        return PredRNNModel(hidden_dims=hidden_dims)
    elif model_type == "prithvi":
        from baselines.prithvi.model import PrithviForecaster
        return PrithviForecaster()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_target_indices(predict_years: List[int]) -> List[int]:
    """Convert years to timestep indices."""
    indices = []
    for year in predict_years:
        start = (year - 1990) * 4
        indices.extend(range(start, start + 4))
    return indices


@torch.no_grad()
def evaluate_on_shard(
    model: torch.nn.Module,
    shard_path: Path,
    target_indices: List[int],
    context_end_idx: int,
    batch_size: int,
    device: torch.device,
    model_type: str,
) -> List[Dict]:
    """Evaluate model on a single shard.

    Args:
        model: Trained model in eval mode.
        shard_path: Path to shard_*.safetensors file.
        target_indices: Target timestep indices.
        context_end_idx: Last context index.
        batch_size: Inference batch size.
        device: Compute device.
        model_type: Model type string.

    Returns:
        List of per-sample statistics dictionaries.
    """
    data = load_shard(shard_path)
    images = data["images"]
    nodata_masks = data["nodata_masks"]
    files_list = data.get("files", [])
    n_samples = images.shape[0]

    sample_ids = [
        extract_sample_id(files_list[i]) if i < len(files_list) else f"sample_{i}"
        for i in range(n_samples)
    ]

    all_stats = []

    for target_idx in target_indices:
        # Process in batches
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_images = images[start:end].to(device)
            batch_nodata = nodata_masks[start:end].to(device)
            batch_ids = sample_ids[start:end]

            # Prepare input: all timesteps up to context_end_idx
            input_seq = batch_images[:, :context_end_idx + 1]
            if input_seq.dim() == 4:
                input_seq = input_seq.unsqueeze(2)  # Add channel dim

            # Forward pass
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                pred = model(input_seq)  # (B, 1, H, W)

            pred_squeezed = pred.squeeze(1).cpu()  # (B, H, W)
            target = batch_images[:, target_idx].cpu()  # (B, H, W)
            target_valid = ~batch_nodata[:, target_idx].cpu()
            pred_valid = torch.ones_like(target_valid)

            stats = compute_batch_stats(
                pred_squeezed, target, pred_valid, target_valid,
                batch_ids, target_idx,
            )
            all_stats.extend(stats)

    del data, images, nodata_masks
    return all_stats


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Get split config
    config = SPLIT_CONFIGS[args.split]
    predict_years = args.predict_years or config["predict_years"]
    target_indices = get_target_indices(predict_years)
    context_end_idx = min(target_indices) - 1 - (args.context_gap_years - 1) * 4

    logger.info(f"Split: {args.split}, predict years: {predict_years}")
    logger.info(f"Target indices: {target_indices}, context_end: {context_end_idx}")

    # Output dir
    if args.output_dir is None:
        year_suffix = "_".join(str(y) for y in sorted(predict_years))
        args.output_dir = Path(f"outputs/{args.model}/eval/{args.split}_{year_suffix}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model, args.checkpoint, device, args.hidden_dims)

    # Find shard files
    all_shard_files = []
    for split_dir in config["dirs"]:
        shard_path = args.shard_root / split_dir
        if not shard_path.exists():
            logger.error(f"Split directory not found: {shard_path}")
            sys.exit(1)
        shard_files = sorted(shard_path.glob("shard_*.safetensors"))
        all_shard_files.extend(shard_files)
        logger.info(f"Found {len(shard_files)} shards in {shard_path}")

    # Evaluate
    all_stats = []
    start_time = time.time()

    for shard_path in tqdm(all_shard_files, desc="Evaluating"):
        stats = evaluate_on_shard(
            model, shard_path, target_indices, context_end_idx,
            args.batch_size, device, args.model,
        )
        all_stats.extend(stats)

    elapsed = time.time() - start_time
    logger.info(f"Evaluation complete: {len(all_stats)} stats in {elapsed:.1f}s")

    # Aggregate results
    method_name = args.model
    aggregated = aggregate_results(all_stats)
    all_results = {method_name: aggregated}

    log_method_results(method_name, aggregated)

    # Save results
    output_json = args.output_dir / "metrics.json"
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved metrics to {output_json}")

    per_sample_json = args.output_dir / "per_sample_stats.json"
    with open(per_sample_json, "w") as f:
        json.dump({method_name: all_stats}, f, indent=2)
    logger.info(f"Saved per-sample stats to {per_sample_json}")

    create_comparison_plot(all_results, args.output_dir / "comparison.png")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
