#!/usr/bin/env python3
"""Train a LightGBM per-pixel model for NDVI prediction.

Samples pixels from training shards, extracts per-pixel features, and trains
a single global LightGBM model. The model can then be used for evaluation
via the LightGBMPredictor wrapper.

Usage:
    python baselines/lightgbm/train.py \
        --train-dir /path/to/train/shards \
        --output-dir outputs/lightgbm \
        --max-pixels 1000000 \
        --predict-years 2022

    # Quick test with fewer pixels and shards
    python baselines/lightgbm/train.py \
        --train-dir /path/to/train/shards \
        --output-dir outputs/lightgbm \
        --max-pixels 100000 \
        --shard-range 0:5
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets.shard_io import load_shard
from baselines.lightgbm.model import (
    FEATURE_NAMES,
    extract_pixel_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LightGBM per-pixel NDVI prediction model"
    )
    parser.add_argument(
        "--train-dir", type=Path, required=True,
        help="Directory containing training shard_*.safetensors files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/lightgbm"),
        help="Output directory for model and logs",
    )
    parser.add_argument(
        "--max-pixels", type=int, default=1_000_000,
        help="Maximum number of training pixels to sample (default: 1M)",
    )
    parser.add_argument(
        "--predict-years", type=int, nargs="+", default=[2022],
        help="Years to predict (used for target indices during training)",
    )
    parser.add_argument(
        "--context-gap-years", type=int, default=1,
        help="Years between context end and prediction start (default: 1)",
    )
    parser.add_argument(
        "--shard-range", type=str, default=None,
        help="Range of shards to use, format: START:END (e.g., 0:50)",
    )
    parser.add_argument(
        "--val-dir", type=Path, default=None,
        help="Optional validation shard directory for early stopping",
    )
    parser.add_argument(
        "--val-max-pixels", type=int, default=200_000,
        help="Max validation pixels (default: 200K)",
    )
    parser.add_argument(
        "--workers", type=int, default=min(4, os.cpu_count() or 1),
        help="Number of parallel workers for shard loading (default: 4)",
    )
    return parser.parse_args()


def compute_target_indices(predict_years: List[int]) -> List[int]:
    """Convert years to timestep indices."""
    indices = []
    for year in predict_years:
        start = (year - 1990) * 4
        indices.extend(range(start, start + 4))
    return indices


def sample_pixels_from_shards(
    shard_dir: Path,
    target_indices: List[int],
    context_end_idx: int,
    max_pixels: int,
    shard_range: Optional[Tuple[int, int]] = None,
    workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample training pixels from shards in parallel.

    Args:
        shard_dir: Directory with shard_*.safetensors files.
        target_indices: Target timestep indices.
        context_end_idx: Last context index.
        max_pixels: Maximum pixels to collect.
        shard_range: Optional (start, end) shard filter.
        workers: Number of parallel workers.

    Returns:
        Tuple of (features, targets) numpy arrays.
    """
    shard_files = sorted(shard_dir.glob("shard_*.safetensors"))
    if shard_range is not None:
        shard_files = shard_files[shard_range[0]:shard_range[1]]

    if not shard_files:
        logger.error(f"No shard files found in {shard_dir}")
        sys.exit(1)

    # Fixed per-shard budget: 100 images × 324 pixels/image = 32,400 pixels
    # Load only as many shards as needed to reach max_pixels
    pixels_per_shard = 100 * 324
    n_available = len(shard_files)
    n_shards_needed = min(n_available, max(1, -(-max_pixels // pixels_per_shard)))
    rng = np.random.default_rng(42)
    rng.shuffle(shard_files)
    shard_files = shard_files[:n_shards_needed]
    logger.info(
        f"Sampling from {n_shards_needed} of {n_available} shards with {workers} workers, "
        f"~{pixels_per_shard:,} pixels/shard, target_indices={target_indices}"
    )

    all_features = []
    all_targets = []
    total_pixels = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_shard, shard_path, target_indices,
                context_end_idx, pixels_per_shard,
            ): shard_path
            for shard_path in shard_files
        }
        for future in as_completed(futures):
            shard_path = futures[future]
            features, targets = future.result()
            if features is not None:
                all_features.append(features)
                all_targets.append(targets)
                total_pixels += len(targets)
                logger.info(f"  {shard_path.name}: {len(targets):,} pixels (total: {total_pixels:,})")

    if not all_features:
        logger.error("No valid pixels found!")
        sys.exit(1)

    features = np.concatenate(all_features, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    if len(targets) > max_pixels:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(targets), size=max_pixels, replace=False)
        features = features[idx]
        targets = targets[idx]

    logger.info(f"Final dataset: {len(targets):,} pixels, {features.shape[1]} features")
    return features, targets


def _process_shard(
    shard_path: Path,
    target_indices: List[int],
    context_end_idx: int,
    max_pixels: int = 0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract features from a single shard.

    Args:
        shard_path: Path to shard file.
        target_indices: Target timestep indices.
        context_end_idx: Last context index.
        max_pixels: Max pixels to extract. 0 = no limit.

    Returns:
        Tuple of (features, targets) or (None, None) if no valid pixels.
    """
    t0 = time.time()
    data = load_shard(shard_path)
    images = data["images"]
    nodata_masks = data["nodata_masks"]
    t_load = time.time() - t0
    logger.info(
        f"  {shard_path.name}: loaded {images.shape[0]} samples "
        f"({images.shape}) in {t_load:.1f}s"
    )

    # Cap images per shard: balance diversity vs feature extraction speed
    # 100 images × 256×256 × 4 targets = ~26M pixels before subsampling
    max_images = 100
    if images.shape[0] > max_images:
        rng = np.random.default_rng(42)
        idx = rng.choice(images.shape[0], size=max_images, replace=False)
        idx.sort()
        images = images[idx]
        nodata_masks = nodata_masks[idx]
        logger.info(
            f"  {shard_path.name}: subsampled {max_images} of "
            f"{data['images'].shape[0]} images"
        )

    del data

    t0 = time.time()
    shard_features = []
    shard_targets = []

    # Budget pixels evenly across target indices
    n_images = images.shape[0]
    pixels_per_target = max_pixels // len(target_indices) if max_pixels > 0 else 0

    for target_idx in target_indices:
        features, targets, valid = extract_pixel_features(
            images, nodata_masks, target_idx, context_end_idx
        )
        valid_features = features[valid]
        valid_targets = targets[valid]

        # Subsample pixels to keep total within budget
        if pixels_per_target > 0 and len(valid_targets) > pixels_per_target:
            rng = np.random.default_rng(42 + target_idx)
            idx = rng.choice(len(valid_targets), size=pixels_per_target, replace=False)
            valid_features = valid_features[idx]
            valid_targets = valid_targets[idx]

        shard_features.append(valid_features)
        shard_targets.append(valid_targets)

    del images, nodata_masks
    t_feat = time.time() - t0
    logger.info(f"  {shard_path.name}: feature extraction in {t_feat:.1f}s")

    if not shard_features:
        return None, None

    return np.concatenate(shard_features), np.concatenate(shard_targets)


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> "lgb.Booster":
    """Train a LightGBM model.

    Args:
        X_train: Training features (N, F).
        y_train: Training targets (N,).
        X_val: Optional validation features.
        y_val: Optional validation targets.

    Returns:
        Trained LightGBM Booster.
    """
    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "num_threads": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)

    callbacks = [lgb.log_evaluation(period=10)]
    valid_sets = []
    valid_names = []

    if X_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES)
        valid_sets.append(val_data)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=20))

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    logger.info(f"Training complete: {model.current_iteration()} iterations")
    return model


def save_model_and_metadata(
    model: "lgb.Booster",
    output_dir: Path,
    train_info: dict,
) -> None:
    """Save trained model and training metadata.

    Args:
        model: Trained LightGBM model.
        output_dir: Output directory.
        train_info: Training metadata dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.txt"
    model.save_model(str(model_path))
    logger.info(f"Saved model to {model_path}")

    # Feature importance
    importance = dict(zip(FEATURE_NAMES, model.feature_importance("gain").tolist()))
    train_info["feature_importance"] = importance
    train_info["num_iterations"] = model.current_iteration()

    meta_path = output_dir / "train_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(train_info, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")


def main():
    """Main training entry point."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_indices = compute_target_indices(args.predict_years)
    context_end_idx = min(target_indices) - 1 - (args.context_gap_years - 1) * 4

    logger.info(f"Target indices: {target_indices}")
    logger.info(f"Context end idx: {context_end_idx}")

    # Parse shard range
    shard_range = None
    if args.shard_range:
        start, end = map(int, args.shard_range.split(":"))
        shard_range = (start, end)

    # Sample training pixels
    t0 = time.time()
    X_train, y_train = sample_pixels_from_shards(
        args.train_dir, target_indices, context_end_idx,
        args.max_pixels, shard_range, args.workers,
    )
    logger.info(f"Pixel sampling: {time.time() - t0:.1f}s")

    # Sample validation pixels
    X_val, y_val = None, None
    if args.val_dir:
        X_val, y_val = sample_pixels_from_shards(
            args.val_dir, target_indices, context_end_idx,
            args.val_max_pixels, workers=args.workers,
        )

    # Train
    t0 = time.time()
    model = train_lightgbm(X_train, y_train, X_val, y_val)
    train_time = time.time() - t0
    logger.info(f"Training time: {train_time:.1f}s")

    # Save
    train_info = {
        "predict_years": args.predict_years,
        "target_indices": target_indices,
        "context_end_idx": context_end_idx,
        "n_train_pixels": len(y_train),
        "n_val_pixels": len(y_val) if y_val is not None else 0,
        "train_time_s": train_time,
        "train_mae": float(np.abs(model.predict(X_train) - y_train).mean()),
    }
    if y_val is not None:
        train_info["val_mae"] = float(np.abs(model.predict(X_val) - y_val).mean())

    save_model_and_metadata(model, args.output_dir, train_info)
    logger.info("Done!")


if __name__ == "__main__":
    main()
