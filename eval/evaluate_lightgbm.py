#!/usr/bin/env python3
"""Evaluate trained LightGBM baseline on NDVI prediction splits.

Uses the same evaluation pipeline (metrics, aggregation, plots) as other
baselines, but runs LightGBM per-pixel inference via LightGBMPredictor.

Usage:
    python eval/evaluate_lightgbm.py \
        --model-path outputs/lightgbm/model.txt \
        --split val_iid \
        --shard-root /path/to/shards

    python eval/evaluate_lightgbm.py \
        --model-path outputs/lightgbm/model.txt \
        --split ood_t \
        --shard-root /path/to/shards \
        --output-dir outputs/lightgbm/eval/ood_t

    # Parallel processing (4 workers):
    python eval/evaluate_lightgbm.py \
        --model-path outputs/lightgbm/model.txt \
        --split val_iid \
        --shard-root /path/to/shards \
        --num-workers 4
"""

import argparse
import gc
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.shard_io import load_shard
from eval.evaluate_baselines import (
    SPLIT_CONFIGS,
    aggregate_results,
    compute_batch_stats,
    create_comparison_plot,
    extract_sample_id,
    log_method_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LightGBM baseline")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to trained model file (model.txt)")
    parser.add_argument("--split", type=str, required=True,
                        choices=list(SPLIT_CONFIGS.keys()))
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--predict-years", type=int, nargs="+", default=None)
    parser.add_argument("--context-gap-years", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None,
                        help="Device for feature extraction (e.g., cuda, cpu)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Parallel workers for shard processing (default: 1)")
    return parser.parse_args()


def get_target_indices(predict_years: List[int]) -> List[int]:
    """Convert years to timestep indices."""
    indices = []
    for year in predict_years:
        start = (year - 1990) * 4
        indices.extend(range(start, start + 4))
    return indices


def evaluate_shard_worker(
    args: Tuple[Path, Path, List[int], int, int],
) -> List[Dict]:
    """Worker function for parallel shard evaluation.

    Args:
        args: Tuple of (model_path, shard_path, target_indices,
              context_end_idx, batch_size).

    Returns:
        List of per-sample statistics dictionaries.
    """
    model_path, shard_path, target_indices, context_end_idx, batch_size = args

    from baselines.lightgbm.model import LightGBMPredictor
    predictor = LightGBMPredictor(model_path)

    stats = evaluate_on_shard(
        predictor, shard_path, target_indices, context_end_idx, batch_size,
    )
    gc.collect()
    return stats


def evaluate_on_shard(
    predictor,
    shard_path: Path,
    target_indices: List[int],
    context_end_idx: int,
    batch_size: int,
) -> List[Dict]:
    """Evaluate LightGBM on a single shard.

    Args:
        predictor: Trained LightGBM predictor.
        shard_path: Path to shard_*.safetensors file.
        target_indices: Target timestep indices.
        context_end_idx: Last context index.
        batch_size: Samples per batch for feature extraction.

    Returns:
        List of per-sample statistics dictionaries.
    """
    t0 = time.time()
    data = load_shard(shard_path)
    images = data["images"]
    nodata_masks = data["nodata_masks"]
    files_list = data.get("files", [])
    n_samples = images.shape[0]
    t_load = time.time() - t0
    logger.info(f"  {shard_path.name}: loaded {n_samples} samples in {t_load:.1f}s")

    sample_ids = [
        extract_sample_id(files_list[i]) if i < len(files_list) else f"sample_{i}"
        for i in range(n_samples)
    ]

    all_stats = []
    n_batches = (n_samples + batch_size - 1) // batch_size

    for ti, target_idx in enumerate(target_indices):
        t_target = time.time()
        for batch_i, start in enumerate(range(0, n_samples, batch_size)):
            end = min(start + batch_size, n_samples)
            batch_images = images[start:end]
            batch_nodata = nodata_masks[start:end]
            batch_ids = sample_ids[start:end]

            pred, pred_valid = predictor.predict_batch(
                batch_images, batch_nodata, target_idx, context_end_idx,
            )

            target = batch_images[:, target_idx].float()
            target_valid = ~batch_nodata[:, target_idx]

            stats = compute_batch_stats(
                pred, target, pred_valid, target_valid,
                batch_ids, target_idx,
            )
            all_stats.extend(stats)

        elapsed = time.time() - t_target
        logger.info(
            f"  {shard_path.name}: target {ti+1}/{len(target_indices)} "
            f"(idx={target_idx}) done in {elapsed:.1f}s"
        )

    del data, images, nodata_masks
    total = time.time() - t0
    logger.info(f"  {shard_path.name}: complete in {total:.1f}s")
    return all_stats


def run_sequential(
    predictor,
    shard_files: List[Path],
    target_indices: List[int],
    context_end_idx: int,
    batch_size: int,
) -> List[Dict]:
    """Evaluate shards sequentially."""
    all_stats = []
    for i, shard_path in enumerate(shard_files):
        logger.info(f"Shard {i+1}/{len(shard_files)}: {shard_path.name}")
        stats = evaluate_on_shard(
            predictor, shard_path, target_indices, context_end_idx, batch_size,
        )
        all_stats.extend(stats)
    return all_stats


def run_parallel(
    model_path: Path,
    shard_files: List[Path],
    target_indices: List[int],
    context_end_idx: int,
    batch_size: int,
    num_workers: int,
) -> List[Dict]:
    """Evaluate shards in parallel using ProcessPoolExecutor."""
    tasks = [
        (model_path, sp, target_indices, context_end_idx, batch_size)
        for sp in shard_files
    ]

    all_stats = []
    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(evaluate_shard_worker, t): t[1] for t in tasks}
        for future in as_completed(futures):
            shard_path = futures[future]
            stats = future.result()
            all_stats.extend(stats)
            completed += 1
            logger.info(
                f"Completed {completed}/{len(shard_files)}: "
                f"{shard_path.name} ({len(stats)} stats)"
            )

    return all_stats


def main():
    """Main evaluation entry point."""
    args = parse_args()

    config = SPLIT_CONFIGS[args.split]
    predict_years = args.predict_years or config["predict_years"]
    target_indices = get_target_indices(predict_years)
    context_end_idx = min(target_indices) - 1 - (args.context_gap_years - 1) * 4

    logger.info(f"Split: {args.split}, predict years: {predict_years}")
    logger.info(f"Target indices: {target_indices}, context_end: {context_end_idx}")

    if args.output_dir is None:
        year_suffix = "_".join(str(y) for y in sorted(predict_years))
        args.output_dir = Path(f"outputs/lightgbm/eval/{args.split}_{year_suffix}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = None
    logger.info(f"Feature extraction device: {device}")

    start_time = time.time()

    if args.num_workers > 1:
        logger.info(f"Parallel evaluation with {args.num_workers} workers")
        all_stats = run_parallel(
            args.model_path, all_shard_files, target_indices,
            context_end_idx, args.batch_size, args.num_workers,
        )
    else:
        from baselines.lightgbm.model import LightGBMPredictor
        predictor = LightGBMPredictor(args.model_path, device=device)
        all_stats = run_sequential(
            predictor, all_shard_files, target_indices,
            context_end_idx, args.batch_size,
        )

    elapsed = time.time() - start_time
    logger.info(f"Evaluation complete: {len(all_stats)} stats in {elapsed:.1f}s")

    # Aggregate and save
    method_name = "lightgbm"
    aggregated = aggregate_results(all_stats)
    all_results = {method_name: aggregated}

    log_method_results(method_name, aggregated)

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
