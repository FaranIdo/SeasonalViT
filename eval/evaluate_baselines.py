"""Evaluate naive baseline methods for NDVI prediction.

Runs persistence, previous_year, and climatology baselines on evaluation data
and reports MAE, R², and NSE metrics with per-season breakdown.

Basic Usage:
    # Evaluate on OOD splits (recommended):
    python eval/evaluate_baselines.py \
        --split ood_t \
        --shard-root /path/to/ndvi_shards

    # Available splits: val_iid, ood_t, ood_s, ood_st
    # Each split auto-configures directories and predict years

    # Override predict years for a split:
    python eval/evaluate_baselines.py \
        --split ood_s \
        --shard-root /path/to/ndvi_shards \
        --predict-years 2021 2022

    # Process shards directly (legacy mode):
    python eval/evaluate_baselines.py \
        --eval-dir /path/to/shards \
        --output-dir outputs/baselines \
        --methods persistence previous_year climatology \
        --forecast-horizon 8

Context Gap (--context-gap-years):
    Controls how many years before the prediction target the context data ends.
    This creates a gap between available input data and the prediction target.

    Default is 1, meaning when predicting 2024, baselines use data up to 2023.
    Set to 2 to use data only up to 2022 when predicting 2024.

    # Default (gap=1): predict 2024 using data up to 2023
    python eval/evaluate_baselines.py \
        --split ood_t \
        --shard-root /path/to/ndvi_shards

    # Gap=2: predict 2024 using data only up to 2022
    python eval/evaluate_baselines.py \
        --split ood_t \
        --shard-root /path/to/ndvi_shards \
        --context-gap-years 2

Parallel Processing (--num-workers):
    Process multiple shards in parallel using separate processes.
    Each worker loads one shard at a time, guaranteeing memory cleanup.

    Default is 3 workers. Set to 1 for sequential processing.

    # Parallel with 6 workers:
    python eval/evaluate_baselines.py \
        --split ood_t \
        --shard-root /path/to/ndvi_shards \
        --num-workers 6

    # Sequential processing (single worker):
    python eval/evaluate_baselines.py \
        --split ood_t \
        --shard-root /path/to/ndvi_shards \
        --num-workers 1

Re-aggregate / Regenerate Plots (--load-stats):
    Skip shard processing and regenerate metrics/plots from existing per-sample stats.
    Useful for tweaking visualizations or adding land-cover breakdown without reprocessing.

    # Regenerate plots from existing stats:
    python eval/evaluate_baselines.py \
        --load-stats outputs/baselines/ood_t/nonml_per_sample_stats.json \
        --output-dir outputs/baselines/ood_t

    # Re-aggregate with land-cover breakdown:
    python eval/evaluate_baselines.py \
        --load-stats outputs/baselines/ood_t/nonml_per_sample_stats.json \
        --landcover-json data/landcover_stats.json \
        --output-dir outputs/baselines/ood_t

Other Options:
    # With per-land-cover analysis:
    python eval/evaluate_baselines.py \
        --eval-dir /path/to/shards \
        --landcover-json data/landcover_stats.json

Progress reporting shows:
    - Samples processed and remaining
    - Estimated time remaining (ETA)
    - Processing speed (samples/sec)
"""

import argparse
import gc
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines import ALL_BASELINES, get_baseline
from datasets.shard_io import load_shard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
SEASON_NAMES = ["spring", "summer", "fall", "winter"]
N_TIMESTEPS = 140  # 35 years × 4 seasons

# Land cover class names (ESA WorldCover)
LANDCOVER_NAMES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse",
    70: "Snow/ice",
    80: "Water",
    90: "Wetland",
    95: "Mangroves",
    100: "Moss/lichen",
}

# Split configurations for OOD evaluation
SPLIT_CONFIGS = {
    "val_iid": {
        "dirs": ["val_iid/val_iid"],
        "predict_years": [2022],
    },
    "ood_t": {
        "dirs": ["val_iid/val_iid", "ood_t_extra/ood_t_extra"],
        "predict_years": [2023, 2024],
    },
    "ood_s": {
        "dirs": ["ood_spatial/ood_spatial"],
        "predict_years": [2022],
    },
    "ood_st": {
        "dirs": ["ood_spatial/ood_spatial"],
        "predict_years": [2023, 2024],
    },
}


# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate naive baselines for NDVI prediction"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Directory containing shard_*.safetensors files, or path to single shard",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/baselines"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=ALL_BASELINES,
        choices=ALL_BASELINES,
        help="Baseline methods to evaluate",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=None,
        help="Sample targets from last N timesteps (default: 4). Ignored if --predict-year is set.",
    )
    parser.add_argument(
        "--predict-year",
        type=int,
        default=None,
        help="Specific year to predict (e.g., 2023). Overrides --forecast-horizon.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)",
    )
    parser.add_argument(
        "--landcover-json",
        type=Path,
        default=None,
        help="Path to landcover_stats.json for per-land-cover analysis",
    )
    parser.add_argument(
        "--load-stats",
        type=Path,
        default=None,
        help="Load existing per-sample stats JSON and re-aggregate (skips shard processing)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val_iid", "ood_t", "ood_s", "ood_st"],
        help="Evaluation split (auto-configures paths and years)",
    )
    parser.add_argument(
        "--shard-root",
        type=Path,
        default=None,
        help="Root directory containing split folders (used with --split)",
    )
    parser.add_argument(
        "--predict-years",
        type=int,
        nargs="+",
        default=None,
        help="Years to predict (e.g., --predict-years 2023 2024). Overrides --predict-year.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help="Number of parallel workers for shard processing (default: 3)",
    )
    parser.add_argument(
        "--context-gap-years",
        type=int,
        default=1,
        help="Years between context end and prediction start (default: 1). "
        "Set to 2 to use data until 2022 when predicting 2024.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for prediction computation: 'cpu', 'cuda', or 'cuda:0' (default: cpu)",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------


def resolve_shard_files(eval_dir: Path) -> List[Path]:
    """Find shard files to process.

    Args:
        eval_dir: Directory containing shard_*.safetensors files, or path to single shard.

    Returns:
        List of shard file paths.

    Raises:
        SystemExit: If no shard files found.
    """
    if eval_dir.is_file():
        return [eval_dir]

    shard_files = sorted(eval_dir.glob("shard_*.safetensors"))
    if not shard_files:
        logger.error(f"No shard_*.safetensors files found in {eval_dir}")
        sys.exit(1)

    return shard_files


def extract_sample_id(filename: str) -> str:
    """Extract sample_id from filename like '0005487_35d408_140d754.tif'."""
    return filename.split("_")[0]


def load_landcover_data(landcover_path: Path) -> Dict[str, Dict]:
    """Load land cover statistics from JSON file.

    Args:
        landcover_path: Path to landcover_stats.json.

    Returns:
        Dictionary mapping sample_id to land cover info.
    """
    logger.info(f"Loading land cover data: {landcover_path}")
    with open(landcover_path) as f:
        data = json.load(f)
    logger.info(f"Loaded land cover data for {len(data)} samples")
    return data



# -----------------------------------------------------------------------------
# Target Index Computation
# -----------------------------------------------------------------------------


def compute_target_indices(
    predict_years: Optional[List[int]],
    predict_year: Optional[int],
    forecast_horizon: Optional[int],
) -> List[int]:
    """Compute target timestep indices based on prediction settings.

    Args:
        predict_years: List of years to predict (1990-2024), or None.
        predict_year: Specific year to predict (1990-2024), or None.
        forecast_horizon: Number of timesteps from end, or None (default: 4).

    Returns:
        List of target timestep indices.

    Raises:
        SystemExit: If predict_year(s) is out of valid range.
    """
    # Handle multiple years (new behavior)
    if predict_years is not None:
        target_indices = []
        for year in predict_years:
            start_idx = (year - 1990) * 4
            end_idx = start_idx + 4
            if start_idx < 0 or end_idx > N_TIMESTEPS:
                logger.error(f"Invalid year {year}. Must be 1990-2024.")
                sys.exit(1)
            target_indices.extend(range(start_idx, end_idx))
        logger.info(f"Target indices: {target_indices} (years {predict_years})")
        return target_indices

    # Handle single year (existing behavior)
    if predict_year is not None:
        start_idx = (predict_year - 1990) * 4
        end_idx = start_idx + 4
        if start_idx < 0 or end_idx > N_TIMESTEPS:
            logger.error(f"Invalid predict_year {predict_year}. Must be 1990-2024.")
            sys.exit(1)
        target_indices = list(range(start_idx, end_idx))
        logger.info(f"Target indices: {target_indices} (year {predict_year}, 4 seasons)")
        return target_indices

    # Fall back to forecast horizon
    horizon = forecast_horizon if forecast_horizon is not None else 4
    target_indices = list(range(N_TIMESTEPS - horizon, N_TIMESTEPS))
    logger.info(f"Target indices: {target_indices} (last {horizon} timesteps)")
    return target_indices


# -----------------------------------------------------------------------------
# Core Evaluation
# -----------------------------------------------------------------------------


def compute_sample_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_id: str,
    target_idx: int,
) -> Optional[Dict]:
    """Compute sufficient statistics and metrics for a single prediction.

    Args:
        pred: Predicted values (flattened valid pixels).
        target: Target values (flattened valid pixels).
        sample_id: Sample identifier.
        target_idx: Target timestep index.

    Returns:
        Dictionary with sufficient statistics and computed metrics, or None if no valid pixels.
    """
    valid_mask = torch.isfinite(pred) & torch.isfinite(target)
    p = pred[valid_mask].float()
    t = target[valid_mask].float()

    n = p.numel()
    if n == 0:
        return None

    # Sufficient statistics
    sum_p = p.sum().item()
    sum_t = t.sum().item()
    sum_p2 = (p**2).sum().item()
    sum_t2 = (t**2).sum().item()
    sum_pt = (p * t).sum().item()
    sum_ae = (p - t).abs().sum().item()
    sum_se = ((p - t) ** 2).sum().item()

    # Compute metrics directly
    mean_p = sum_p / n
    mean_t = sum_t / n
    var_p = sum_p2 / n - mean_p**2
    var_t = sum_t2 / n - mean_t**2
    cov = sum_pt / n - mean_p * mean_t

    mae = sum_ae / n
    mse = sum_se / n
    nse = 1 - mse / var_t if var_t > 0 else None

    if var_p > 0 and var_t > 0:
        corr = cov / (var_p**0.5 * var_t**0.5)
        r2 = corr**2
    else:
        r2 = None

    return {
        "sample_id": sample_id,
        "target_idx": target_idx,
        "target_season": target_idx % 4,
        "target_year": 1990 + target_idx // 4,
        "n_valid": n,
        # Computed metrics
        "mae": mae,
        "r2": r2,
        "nse": nse,
        # Sufficient statistics (for re-aggregation)
        "sum_pred": sum_p,
        "sum_target": sum_t,
        "sum_pred_sq": sum_p2,
        "sum_target_sq": sum_t2,
        "sum_pred_target": sum_pt,
        "sum_abs_error": sum_ae,
        "sum_sq_error": sum_se,
    }


def compute_batch_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_valid: torch.Tensor,
    target_valid: torch.Tensor,
    sample_ids: List[str],
    target_idx: int,
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """Compute sufficient statistics for all samples in a batch using vectorized ops.

    Args:
        pred: Predicted values of shape (B, H, W).
        target: Target values of shape (B, H, W).
        pred_valid: Prediction valid mask of shape (B, H, W), True = valid.
        target_valid: Target valid mask of shape (B, H, W), True = valid.
        sample_ids: List of sample identifiers, length B.
        target_idx: Target timestep index.
        device: Device to use for computation (None = use input device).

    Returns:
        List of per-sample statistics dictionaries.
    """
    B, H, W = pred.shape

    # Move to specified device if provided
    if device is not None:
        pred = pred.to(device)
        target = target.to(device)
        pred_valid = pred_valid.to(device)
        target_valid = target_valid.to(device)

    # Combined valid mask: both prediction and target must be valid
    valid = pred_valid & target_valid  # (B, H, W)

    # Flatten spatial dimensions for vectorized computation
    pred_flat = pred.view(B, -1).float()  # (B, H*W)
    target_flat = target.view(B, -1).float()  # (B, H*W)
    valid_flat = valid.view(B, -1)  # (B, H*W)

    # Mask invalid values to zero for summing
    pred_masked = pred_flat.masked_fill(~valid_flat, 0.0)
    target_masked = target_flat.masked_fill(~valid_flat, 0.0)

    # Per-sample sufficient statistics (all shape (B,))
    n_valid = valid_flat.sum(dim=1)  # Count of valid pixels
    sum_pred = pred_masked.sum(dim=1)
    sum_target = target_masked.sum(dim=1)
    sum_pred_sq = (pred_masked**2).sum(dim=1)
    sum_target_sq = (target_masked**2).sum(dim=1)
    sum_pred_target = (pred_masked * target_masked).sum(dim=1)

    # Compute errors where valid
    diff = pred_flat - target_flat
    diff_masked = diff.masked_fill(~valid_flat, 0.0)
    sum_abs_error = diff_masked.abs().sum(dim=1)
    sum_sq_error = (diff_masked**2).sum(dim=1)

    # Vectorized metric computation (all shape (B,))
    n_float = n_valid.float().clamp(min=1)  # Avoid division by zero
    mean_p = sum_pred / n_float
    mean_t = sum_target / n_float
    var_p = sum_pred_sq / n_float - mean_p**2
    var_t = sum_target_sq / n_float - mean_t**2
    cov = sum_pred_target / n_float - mean_p * mean_t

    mae = sum_abs_error / n_float
    mse = sum_sq_error / n_float
    nse = 1 - mse / var_t.clamp(min=1e-10)  # Clamp to avoid division by zero

    # R² computation
    std_p = var_p.clamp(min=0).sqrt()
    std_t = var_t.clamp(min=0).sqrt()
    corr = cov / (std_p * std_t).clamp(min=1e-10)
    r2 = corr**2

    # Mark invalid metrics where var_t or var_p is zero
    invalid_nse = var_t <= 0
    invalid_r2 = (var_p <= 0) | (var_t <= 0)

    # Filter to valid samples and convert to lists in one go
    has_valid = n_valid > 0
    valid_indices = has_valid.nonzero(as_tuple=True)[0].cpu().tolist()

    # Move to CPU and convert to lists (single conversion per tensor)
    n_valid_list = n_valid.cpu().tolist()
    sum_pred_list = sum_pred.cpu().tolist()
    sum_target_list = sum_target.cpu().tolist()
    sum_pred_sq_list = sum_pred_sq.cpu().tolist()
    sum_target_sq_list = sum_target_sq.cpu().tolist()
    sum_pred_target_list = sum_pred_target.cpu().tolist()
    sum_abs_error_list = sum_abs_error.cpu().tolist()
    sum_sq_error_list = sum_sq_error.cpu().tolist()
    mae_list = mae.cpu().tolist()
    r2_list = r2.cpu().tolist()
    nse_list = nse.cpu().tolist()
    invalid_nse_list = invalid_nse.cpu().tolist()
    invalid_r2_list = invalid_r2.cpu().tolist()

    # Build result list only for valid samples
    target_season = target_idx % 4
    target_year = 1990 + target_idx // 4

    results = [
        {
            "sample_id": sample_ids[i],
            "target_idx": target_idx,
            "target_season": target_season,
            "target_year": target_year,
            "n_valid": int(n_valid_list[i]),
            "mae": mae_list[i],
            "r2": None if invalid_r2_list[i] else r2_list[i],
            "nse": None if invalid_nse_list[i] else nse_list[i],
            "sum_pred": sum_pred_list[i],
            "sum_target": sum_target_list[i],
            "sum_pred_sq": sum_pred_sq_list[i],
            "sum_target_sq": sum_target_sq_list[i],
            "sum_pred_target": sum_pred_target_list[i],
            "sum_abs_error": sum_abs_error_list[i],
            "sum_sq_error": sum_sq_error_list[i],
        }
        for i in valid_indices
    ]

    return results


def _compute_metrics_from_sums(
    n: int,
    sum_p: float,
    sum_t: float,
    sum_p2: float,
    sum_t2: float,
    sum_pt: float,
    sum_ae: float,
    sum_se: float,
) -> Dict:
    """Compute metrics from sufficient statistics."""
    if n == 0:
        return {"mae": None, "r2": None, "nse": None, "bias": None}

    mean_p = sum_p / n
    mean_t = sum_t / n
    var_p = sum_p2 / n - mean_p**2
    var_t = sum_t2 / n - mean_t**2
    cov = sum_pt / n - mean_p * mean_t

    mae = sum_ae / n
    mse = sum_se / n
    bias = abs(mean_p - mean_t)
    nse = 1 - mse / var_t if var_t > 0 else None

    if var_p > 0 and var_t > 0:
        corr = cov / (var_p**0.5 * var_t**0.5)
        r2 = corr**2
    else:
        r2 = None

    return {"mae": mae, "r2": r2, "nse": nse, "bias": bias}


def aggregate_from_stats(stats_list: List[Dict]) -> Dict:
    """Compute MAE, R², NSE, bias with per-sample averaging.

    Computes metrics per sample first, then averages across samples.
    This gives meaningful R²/NSE values (not dominated by between-sample variance).

    Args:
        stats_list: List of per-(sample, target_idx) statistics dictionaries.

    Returns:
        Dictionary with aggregated metrics: mae, r2, nse, bias, n_valid, n_samples.
    """
    if not stats_list:
        return {"mae": 0.0, "r2": 0.0, "nse": 0.0, "bias": 0.0, "n_valid": 0, "n_samples": 0}

    # Group stats by sample_id
    from collections import defaultdict
    by_sample: Dict[str, List[Dict]] = defaultdict(list)
    for s in stats_list:
        by_sample[s["sample_id"]].append(s)

    # Compute metrics per sample, then average
    sample_metrics = []
    total_pixels = 0

    for sample_id, sample_stats in by_sample.items():
        n = sum(s["n_valid"] for s in sample_stats)
        if n == 0:
            continue

        total_pixels += n
        metrics = _compute_metrics_from_sums(
            n=n,
            sum_p=sum(s["sum_pred"] for s in sample_stats),
            sum_t=sum(s["sum_target"] for s in sample_stats),
            sum_p2=sum(s["sum_pred_sq"] for s in sample_stats),
            sum_t2=sum(s["sum_target_sq"] for s in sample_stats),
            sum_pt=sum(s["sum_pred_target"] for s in sample_stats),
            sum_ae=sum(s["sum_abs_error"] for s in sample_stats),
            sum_se=sum(s["sum_sq_error"] for s in sample_stats),
        )
        sample_metrics.append(metrics)

    if not sample_metrics:
        return {"mae": 0.0, "r2": 0.0, "nse": 0.0, "bias": 0.0, "n_valid": 0, "n_samples": 0}

    # Average across samples (excluding None and NaN values)
    import math
    def avg(key):
        vals = [m[key] for m in sample_metrics
                if m[key] is not None and math.isfinite(m[key])]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "mae": avg("mae"),
        "r2": avg("r2"),
        "nse": avg("nse"),
        "bias": avg("bias"),
        "n_valid": total_pixels,
        "n_samples": len(sample_metrics),
    }


def evaluate_single_target(
    baseline,
    images_batch: torch.Tensor,
    nodata_batch: torch.Tensor,
    target_idx: int,
    sample_id: str,
    context_end_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Evaluate a baseline on a single target timestep.

    Args:
        baseline: Baseline method instance.
        images_batch: Images with batch dim (1, T, H, W).
        nodata_batch: Nodata mask with batch dim (1, T, H, W).
        target_idx: Target timestep index.
        sample_id: Sample identifier.
        context_end_idx: Last index that can be used as context (inclusive).

    Returns:
        Statistics dictionary, or None if no valid pixels.
    """
    pred, valid_mask = baseline.predict(
        images_batch, target_idx, nodata_batch, context_end_idx
    )

    target = images_batch[0, target_idx]
    target_valid = ~nodata_batch[0, target_idx]
    combined_valid = valid_mask[0] & target_valid

    if combined_valid.sum() == 0:
        return None

    pred_valid = pred[0][combined_valid]
    target_valid_vals = target[combined_valid]

    return compute_sample_stats(pred_valid, target_valid_vals, sample_id, target_idx)


def evaluate_baseline_on_sample(
    baseline,
    images: torch.Tensor,
    nodata_mask: torch.Tensor,
    target_indices: List[int],
    sample_id: str,
    context_end_idx: Optional[int] = None,
) -> List[Dict]:
    """Evaluate a baseline on a single sample across multiple target indices.

    Args:
        baseline: Baseline method instance.
        images: NDVI images of shape (T, H, W).
        nodata_mask: Nodata mask of shape (T, H, W).
        target_indices: List of timestep indices to predict.
        sample_id: Sample identifier.
        context_end_idx: Last index that can be used as context (inclusive).

    Returns:
        List of per-target_idx stats dictionaries.
    """
    images_batch = images.unsqueeze(0)
    nodata_batch = nodata_mask.unsqueeze(0)

    stats_list = []
    for target_idx in target_indices:
        stats = evaluate_single_target(
            baseline, images_batch, nodata_batch, target_idx, sample_id, context_end_idx
        )
        if stats is not None:
            stats_list.append(stats)

    return stats_list


# -----------------------------------------------------------------------------
# Parallel Processing
# -----------------------------------------------------------------------------


def _sub_batch_ranges(n: int, sub_batch: int | None) -> list:
    """Return list of (start, end) ranges for sub-batching."""
    if sub_batch is None or sub_batch >= n:
        return [(0, n)]
    return [(i, min(i + sub_batch, n)) for i in range(0, n, sub_batch)]


def process_single_shard(
    shard_path: Path,
    method_names: List[str],
    target_indices: List[int],
    context_end_idx: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, List[Dict]]:
    """Worker function to process a single shard in a separate process.

    This function uses batch processing for efficiency: instead of processing
    one sample at a time, it passes all samples (B=1024) through each baseline
    at once and uses vectorized operations to compute statistics.

    Args:
        shard_path: Path to the shard file.
        method_names: List of baseline method names to evaluate.
        target_indices: List of target timestep indices.
        context_end_idx: Last index that can be used as context (inclusive).

    Returns:
        Dictionary mapping method names to lists of per-target stats.
    """
    shard_name = shard_path.name
    shard_start = time.time()

    compute_device = torch.device(device)
    use_gpu = compute_device.type == "cuda"
    sub_batch = 256 if use_gpu else None
    print(f"[{shard_name}] Starting (device={compute_device}, sub_batch={sub_batch})...",
          flush=True)

    # Create baselines in worker process (avoids pickling)
    baselines = {name: get_baseline(name) for name in method_names}

    # Load shard (keep on CPU, sub-batch to GPU)
    load_start = time.time()
    print(f"[{shard_name}] Loading...", flush=True)
    shard_data = load_shard(shard_path)
    images = shard_data["images"]  # (B, T, H, W) stays on CPU
    nodata_masks = shard_data["nodata_masks"]  # (B, T, H, W)
    shard_files_list = shard_data.get("files", [])
    del shard_data
    n_samples = images.shape[0]
    load_time = time.time() - load_start
    print(f"[{shard_name}] Loaded {n_samples} samples in {load_time:.1f}s", flush=True)

    # Build sample_ids list
    sample_ids = [
        extract_sample_id(shard_files_list[i])
        if i < len(shard_files_list)
        else f"sample_{i}"
        for i in range(n_samples)
    ]

    method_results = {name: [] for name in method_names}

    for method_name, baseline in baselines.items():
        method_start = time.time()
        predict_time = 0.0
        stats_time = 0.0

        for t_idx, target_idx in enumerate(target_indices):
            # Process sub-batches: predict + stats on GPU, collect lightweight dicts
            batch_ranges = _sub_batch_ranges(n_samples, sub_batch)

            for start, end in batch_ranges:
                t0 = time.time()
                imgs = images[start:end].to(compute_device)
                masks = nodata_masks[start:end].to(compute_device)
                pred, pred_valid = baseline.predict(
                    imgs, target_idx, masks, context_end_idx
                )
                target_vals = imgs[:, target_idx]
                target_valid = ~masks[:, target_idx]
                predict_time += time.time() - t0

                t0 = time.time()
                stats_list = compute_batch_stats(
                    pred, target_vals, pred_valid, target_valid,
                    sample_ids[start:end], target_idx,
                )
                stats_time += time.time() - t0
                method_results[method_name].extend(stats_list)

                del imgs, masks, pred, pred_valid, target_vals, target_valid

        method_time = time.time() - method_start
        print(
            f"[{shard_name}] {method_name}: {method_time:.1f}s "
            f"(predict={predict_time:.1f}s, stats={stats_time:.1f}s)",
            flush=True,
        )

    total_time = time.time() - shard_start
    print(f"[{shard_name}] Done in {total_time:.1f}s", flush=True)
    return method_results


def process_shards_parallel(
    shard_files: List[Path],
    method_names: List[str],
    target_indices: List[int],
    num_workers: int,
    context_end_idx: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, List[Dict]]:
    """Process shards in parallel using multiple processes.

    Each worker process handles one shard at a time, creating its own baseline
    instances to avoid pickling issues. This guarantees memory cleanup as each
    process terminates after completing its shard.

    Args:
        shard_files: List of shard file paths.
        method_names: List of baseline method names to evaluate.
        target_indices: List of target timestep indices.
        num_workers: Number of parallel worker processes.
        context_end_idx: Last index that can be used as context (inclusive).

    Returns:
        Dictionary mapping method names to combined lists of per-target stats.
    """
    combined_results = {name: [] for name in method_names}
    num_gpus = torch.cuda.device_count() if device.startswith("cuda") else 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit shards, assigning GPUs round-robin
        futures = {}
        for i, path in enumerate(shard_files):
            shard_device = f"cuda:{i % num_gpus}" if num_gpus > 0 else device
            futures[executor.submit(
                process_single_shard, path, method_names, target_indices,
                context_end_idx, shard_device
            )] = path

        # Collect results with progress bar
        with tqdm(total=len(futures), desc="Processing shards", unit="shard") as pbar:
            for future in as_completed(futures):
                shard_path = futures[future]
                try:
                    shard_results = future.result()
                    for name in method_names:
                        combined_results[name].extend(shard_results[name])
                except Exception as e:
                    logger.error(f"Failed to process {shard_path}: {e}")
                pbar.update(1)

    return combined_results


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def create_empty_metrics() -> Dict:
    """Create empty metrics dictionary with zero values."""
    return {"mae": 0.0, "r2": 0.0, "nse": 0.0, "bias": 0.0, "n_valid_pixels": 0, "n_samples": 0}


def _stats_to_metrics(agg: Dict) -> Dict:
    """Convert aggregate_from_stats output to metrics format."""
    return {
        "mae": agg["mae"],
        "r2": agg["r2"],
        "nse": agg["nse"],
        "bias": agg["bias"],
        "n_valid_pixels": agg["n_valid"],
        "n_samples": agg.get("n_samples", 0),
    }


def aggregate_results(all_stats: List[Dict]) -> Dict:
    """Aggregate results across all samples using sufficient statistics.

    Args:
        all_stats: Flat list of per-target stats dictionaries.

    Returns:
        Aggregated metrics with overall and per-season breakdown.
    """
    empty_result = {
        "overall": create_empty_metrics(),
        "per_season": {s: create_empty_metrics() for s in SEASON_NAMES},
    }

    if not all_stats:
        return empty_result

    overall = aggregate_from_stats(all_stats)
    aggregated = {"overall": _stats_to_metrics(overall), "per_season": {}}

    for season_idx, season_name in enumerate(SEASON_NAMES):
        season_stats = [s for s in all_stats if s["target_season"] == season_idx]
        if season_stats:
            aggregated["per_season"][season_name] = _stats_to_metrics(
                aggregate_from_stats(season_stats)
            )
        else:
            aggregated["per_season"][season_name] = create_empty_metrics()

    return aggregated


def aggregate_results_by_landcover(
    all_stats: List[Dict], landcover_data: Dict[str, Dict]
) -> Dict[str, Dict]:
    """Aggregate results by land cover class.

    Args:
        all_stats: Flat list of per-target stats dictionaries.
        landcover_data: Dictionary mapping sample_id to land cover info.

    Returns:
        Dictionary mapping land cover class name to aggregated metrics.
    """
    # Group stats by dominant land cover class
    stats_by_class: Dict[str, List[Dict]] = {}

    for stat in all_stats:
        sample_id = stat["sample_id"]
        lc_info = landcover_data.get(sample_id)

        if lc_info is None or lc_info.get("dominant_class") is None:
            class_name = "Unknown"
        else:
            class_code = lc_info["dominant_class"]
            class_name = LANDCOVER_NAMES.get(class_code, f"Class {class_code}")

        if class_name not in stats_by_class:
            stats_by_class[class_name] = []
        stats_by_class[class_name].append(stat)

    # Aggregate each class
    result = {}
    for class_name, class_stats in stats_by_class.items():
        agg = aggregate_from_stats(class_stats)
        result[class_name] = {
            **_stats_to_metrics(agg),
            "n_samples": len(set(s["sample_id"] for s in class_stats)),
        }

    return result


def process_all_shards(
    shard_files: List[Path],
    method_names: List[str],
    target_indices: List[int],
    max_samples: Optional[int],
    num_workers: int = 1,
    context_end_idx: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, List[Dict]]:
    """Process all shards and collect statistics for each baseline.

    Args:
        shard_files: List of shard file paths.
        method_names: List of baseline method names to evaluate.
        target_indices: List of target timestep indices.
        max_samples: Maximum samples to process, or None for unlimited.
        num_workers: Number of parallel worker processes (default: 1 for sequential).
        context_end_idx: Last index that can be used as context (inclusive).
        device: Device for computation ('cpu', 'cuda', 'cuda:0', etc.).

    Returns:
        Dictionary mapping method names to lists of per-sample stats.
    """
    shards_to_process = shard_files
    if max_samples is not None:
        max_shards = max(1, (max_samples + 999) // 1000)
        shards_to_process = shard_files[:max_shards]
        logger.info(f"Limiting to {len(shards_to_process)} shards for ~{max_samples} samples")

    if num_workers > 1:
        logger.info(f"Processing {len(shards_to_process)} shards with {num_workers} workers")
        return process_shards_parallel(
            shards_to_process, method_names, target_indices, num_workers,
            context_end_idx, device
        )

    combined_results = {name: [] for name in method_names}
    for shard_path in shards_to_process:
        shard_results = process_single_shard(
            shard_path, method_names, target_indices, context_end_idx, device
        )
        for name in method_names:
            combined_results[name].extend(shard_results[name])

    return combined_results


def process_split_directories(
    shard_root: Path,
    split_dirs: List[str],
    method_names: List[str],
    target_indices: List[int],
    max_samples: Optional[int],
    num_workers: int = 1,
    context_end_idx: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, List[Dict]]:
    """Process multiple shard directories and combine results.

    Args:
        shard_root: Root directory containing split folders.
        split_dirs: List of subdirectory paths relative to shard_root.
        method_names: List of baseline method names to evaluate.
        target_indices: List of target timestep indices.
        max_samples: Maximum samples to process, or None for unlimited.
        num_workers: Number of parallel worker processes (default: 1 for sequential).
        context_end_idx: Last index that can be used as context (inclusive).
        device: Device for computation ('cpu', 'cuda', 'cuda:0', etc.).

    Returns:
        Dictionary mapping method names to lists of per-sample stats.
    """
    combined_results = {name: [] for name in method_names}
    samples_remaining = max_samples

    for split_dir in split_dirs:
        shard_path = shard_root / split_dir
        if not shard_path.exists():
            logger.error(f"Split directory not found: {shard_path}")
            sys.exit(1)

        logger.info(f"\nProcessing split directory: {shard_path}")
        shard_files = resolve_shard_files(shard_path)
        logger.info(f"Found {len(shard_files)} shard file(s)")

        method_results = process_all_shards(
            shard_files, method_names, target_indices, samples_remaining,
            num_workers=num_workers, context_end_idx=context_end_idx,
            device=device
        )

        for name in method_names:
            combined_results[name].extend(method_results[name])

        # Update samples remaining
        if samples_remaining is not None:
            unique_samples = len(
                set(s["sample_id"] for s in method_results[method_names[0]])
            )
            samples_remaining -= unique_samples
            if samples_remaining <= 0:
                break

    return combined_results


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def log_method_results(method_name: str, aggregated: Dict) -> None:
    """Log results for a single method."""
    overall = aggregated["overall"]
    logger.info(f"\n{method_name}:")
    logger.info(f"  Overall MAE:  {overall['mae']:.4f}")
    logger.info(f"  Overall R²:   {overall['r2']:.4f}")
    logger.info(f"  Overall NSE:  {overall['nse']:.4f}")
    logger.info(f"  Overall bias: {overall['bias']:.4f}")
    logger.info(f"  Samples: {overall.get('n_samples', 'N/A'):,}  |  Valid pixels: {overall['n_valid_pixels']:,}")

    logger.info("  Per-season:")
    for season in SEASON_NAMES:
        s = aggregated["per_season"][season]
        logger.info(
            f"    {season:8s}: MAE={s['mae']:.4f}, R²={s['r2']:.4f}, "
            f"NSE={s['nse']:.4f}, bias={s['bias']:.4f}"
        )


def save_results_json(
    all_results: Dict, method_results: Dict[str, List[Dict]], output_dir: Path
) -> None:
    """Save aggregated and per-sample results to JSON files."""
    output_json = output_dir / "nonml_metrics.json"
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved metrics to: {output_json}")

    per_sample_json = output_dir / "nonml_per_sample_stats.json"
    with open(per_sample_json, "w") as f:
        json.dump(method_results, f, indent=2)
    logger.info(f"Saved per-sample stats to: {per_sample_json}")


def print_comparison_table(all_results: Dict, methods: List[str]) -> None:
    """Print a summary comparison table."""
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Method':<20} {'MAE':>10} {'R²':>10} {'NSE':>10} {'Bias':>10}")
    logger.info("-" * 80)
    for method_name in methods:
        m = all_results[method_name]["overall"]
        logger.info(
            f"{method_name:<20} {m['mae']:>10.4f} {m['r2']:>10.4f} "
            f"{m['nse']:>10.4f} {m['bias']:>10.4f}"
        )
    logger.info("=" * 80)


def log_landcover_results(method_name: str, per_landcover: Dict[str, Dict]) -> None:
    """Log per-land-cover results for a method."""
    if not per_landcover:
        return

    logger.info(f"  Per-land-cover:")
    # Sort by number of samples descending
    sorted_classes = sorted(
        per_landcover.items(), key=lambda x: x[1]["n_samples"], reverse=True
    )
    for class_name, metrics in sorted_classes:
        logger.info(
            f"    {class_name:<18}: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}, "
            f"NSE={metrics['nse']:.4f}, n_samples={metrics['n_samples']:,}"
        )


def print_landcover_comparison_table(
    all_results: Dict, methods: List[str], landcover_data: Dict
) -> None:
    """Print comparison table broken down by land cover class."""
    # Get all unique land cover classes across all methods
    all_classes = set()
    for method_name in methods:
        if "per_landcover" in all_results[method_name]:
            all_classes.update(all_results[method_name]["per_landcover"].keys())

    if not all_classes:
        return

    logger.info("\n" + "=" * 100)
    logger.info("LAND COVER BREAKDOWN (MAE by method)")
    logger.info("=" * 100)

    # Header
    header = f"{'Land Cover':<20}"
    for method in methods:
        header += f" {method:>15}"
    header += f" {'Samples':>10}"
    logger.info(header)
    logger.info("-" * 100)

    # Sort classes by total samples
    class_samples = {}
    for class_name in all_classes:
        for method_name in methods:
            per_lc = all_results[method_name].get("per_landcover", {})
            if class_name in per_lc:
                class_samples[class_name] = per_lc[class_name]["n_samples"]
                break

    sorted_classes = sorted(
        all_classes, key=lambda x: class_samples.get(x, 0), reverse=True
    )

    for class_name in sorted_classes:
        row = f"{class_name:<20}"
        n_samples = 0
        for method_name in methods:
            per_lc = all_results[method_name].get("per_landcover", {})
            if class_name in per_lc:
                row += f" {per_lc[class_name]['mae']:>15.4f}"
                n_samples = per_lc[class_name]["n_samples"]
            else:
                row += f" {'N/A':>15}"
        row += f" {n_samples:>10,}"
        logger.info(row)

    logger.info("=" * 100)


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def create_comparison_plot(
    results: Dict[str, Dict],
    output_path: Path,
) -> None:
    """Create bar chart comparing baselines on each metric.

    Args:
        results: Dictionary mapping baseline name to aggregated results.
        output_path: Path to save the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
        return

    methods = list(results.keys())
    metrics = ["mae", "r2", "nse", "bias"]
    # Arrows: ↓ = lower is better, ↑ = higher is better
    metric_titles = ["MAE ↓", "R² ↑", "NSE ↑", "Bias ↓"]
    metric_ylabels = ["MAE", "R²", "NSE", "Bias"]
    # Colors for up to 4 methods
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, metric, title, ylabel in zip(axes, metrics, metric_titles, metric_ylabels):
        values = [results[m]["overall"][metric] for m in methods]
        bars = ax.bar(methods, values, color=colors[: len(methods)])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight="bold")
        # Rotate x-axis labels to prevent overlap
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison plot: {output_path}")


def create_predictions_grid(
    images: torch.Tensor,
    nodata_mask: torch.Tensor,
    target_idx: int,
    baselines: Dict,
    output_path: Path,
) -> None:
    """Create side-by-side visualization of predictions.

    Args:
        images: Single sample images (T, H, W).
        nodata_mask: Nodata mask (T, H, W).
        target_idx: Target timestep index.
        baselines: Dictionary of baseline instances.
        output_path: Path to save the plot.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return

    # NDVI colormap
    colors = ["#2166ac", "#92c5de", "#f7f7f7", "#fddbc7", "#b2182b", "#1a9641", "#006400"]
    positions = [0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    cmap = LinearSegmentedColormap.from_list("ndvi", list(zip(positions, colors)))

    # Add batch dimension for baseline prediction
    images_batch = images.unsqueeze(0)
    mask_batch = nodata_mask.unsqueeze(0)

    # Get target
    target = images[target_idx].numpy()
    target_mask = nodata_mask[target_idx].numpy()

    # Get predictions from each baseline
    predictions = {}
    for name, baseline in baselines.items():
        pred, valid = baseline.predict(images_batch, target_idx, mask_batch)
        predictions[name] = (pred[0].numpy(), valid[0].numpy())

    # Create plot
    n_cols = 2 + len(baselines)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))

    # Plot target
    ax = axes[0]
    masked_target = target.copy()
    masked_target[target_mask] = float("nan")
    im = ax.imshow(masked_target, cmap=cmap, vmin=-0.2, vmax=1.0)
    ax.set_title(f"Target (t={target_idx})")
    ax.axis("off")

    # Plot previous observation (t-4)
    ax = axes[1]
    if target_idx >= 4:
        prev_obs = images[target_idx - 4].numpy()
        prev_mask = nodata_mask[target_idx - 4].numpy()
        masked_prev = prev_obs.copy()
        masked_prev[prev_mask] = float("nan")
        ax.imshow(masked_prev, cmap=cmap, vmin=-0.2, vmax=1.0)
    ax.set_title(f"Prev Year (t-4)")
    ax.axis("off")

    # Plot each baseline prediction
    for i, (name, (pred, valid)) in enumerate(predictions.items()):
        ax = axes[2 + i]
        masked_pred = pred.copy()
        masked_pred[~valid] = float("nan")
        ax.imshow(masked_pred, cmap=cmap, vmin=-0.2, vmax=1.0)
        ax.set_title(name.replace("_", " ").title())
        ax.axis("off")

    # Add colorbar
    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02, label="NDVI")

    season = SEASON_NAMES[target_idx % 4]
    year = 1990 + target_idx // 4
    fig.suptitle(f"Baseline Predictions - {season.title()} {year}", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved predictions grid: {output_path}")


def main():
    """Main entry point for baseline evaluation."""
    args = parse_args()

    # Validate arguments
    if args.load_stats is None and args.eval_dir is None and args.split is None:
        logger.error("One of --eval-dir, --load-stats, or --split is required")
        sys.exit(1)

    if args.split and not args.shard_root:
        logger.error("--shard-root is required when using --split")
        sys.exit(1)

    # Note: Output dir with year suffix is set later, after predict_years is determined
    # For load_stats mode, we need to create output_dir now (no year suffix needed)
    if args.load_stats:
        if args.split and args.output_dir == Path("outputs/baselines"):
            args.output_dir = Path("outputs/baselines") / args.split
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load land cover data if provided
    landcover_data = None
    if args.landcover_json:
        landcover_data = load_landcover_data(args.landcover_json)

    # Mode 1: Load existing stats and re-aggregate
    if args.load_stats:
        logger.info(f"Loading existing stats from: {args.load_stats}")
        with open(args.load_stats) as f:
            method_results = json.load(f)

        methods = list(method_results.keys())
        logger.info(f"Found methods: {methods}")

        # Aggregate and report results
        all_results = {}
        for method_name in methods:
            aggregated = aggregate_results(method_results[method_name])

            if landcover_data:
                aggregated["per_landcover"] = aggregate_results_by_landcover(
                    method_results[method_name], landcover_data
                )

            all_results[method_name] = aggregated
            log_method_results(method_name, aggregated)

            if landcover_data:
                log_landcover_results(method_name, aggregated.get("per_landcover", {}))

        print_comparison_table(all_results, methods)

        if landcover_data:
            print_landcover_comparison_table(all_results, methods, landcover_data)

        # Save updated metrics (with land cover breakdown)
        output_json = args.output_dir / "nonml_metrics.json"
        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nSaved metrics to: {output_json}")

        # Generate comparison plot
        create_comparison_plot(all_results, args.output_dir / "nonml_comparison.png")
        return

    # Mode 2: Process shards (either via --split or --eval-dir)
    logger.info(f"Methods: {args.methods}, device: {args.device}")

    # Determine predict_years from args or split config
    predict_years = args.predict_years
    if args.split:
        config = SPLIT_CONFIGS[args.split]
        if predict_years is None:
            predict_years = config["predict_years"]
        logger.info(f"Split: {args.split}")

    # Set output dir with year suffix (only if using default path and split mode)
    if args.split and args.output_dir == Path("outputs/baselines"):
        year_suffix = "_".join(str(y) for y in sorted(predict_years))
        args.output_dir = Path("outputs/baselines") / f"{args.split}_{year_suffix}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_indices = compute_target_indices(
        predict_years, args.predict_year, args.forecast_horizon
    )

    # Compute context_end_idx based on context_gap_years
    # gap=1 means context ends 1 year before prediction (e.g., predict 2024, use up to 2023)
    # gap=2 means context ends 2 years before prediction (e.g., predict 2024, use up to 2022)
    context_end_idx = min(target_indices) - 1 - (args.context_gap_years - 1) * 4
    context_end_year = 1990 + context_end_idx // 4
    logger.info(
        f"Context gap: {args.context_gap_years} year(s) "
        f"(using data up to {context_end_year}, context_end_idx={context_end_idx})"
    )

    # Process shards based on mode
    method_names = args.methods
    if args.split:
        config = SPLIT_CONFIGS[args.split]
        method_results = process_split_directories(
            args.shard_root, config["dirs"], method_names, target_indices,
            args.max_samples, args.num_workers, context_end_idx, args.device,
        )
    else:
        shard_files = resolve_shard_files(args.eval_dir)
        logger.info(f"Found {len(shard_files)} shard file(s)")
        method_results = process_all_shards(
            shard_files, method_names, target_indices,
            args.max_samples, args.num_workers, context_end_idx, args.device,
        )

    # Aggregate and report results
    all_results = {}
    for method_name in method_names:
        aggregated = aggregate_results(method_results[method_name])
        if landcover_data:
            aggregated["per_landcover"] = aggregate_results_by_landcover(
                method_results[method_name], landcover_data
            )
        all_results[method_name] = aggregated
        log_method_results(method_name, aggregated)
        if landcover_data:
            log_landcover_results(method_name, aggregated.get("per_landcover", {}))

    # Save results and generate visualizations
    save_results_json(all_results, method_results, args.output_dir)
    print_comparison_table(all_results, method_names)
    if landcover_data:
        print_landcover_comparison_table(all_results, method_names, landcover_data)
    create_comparison_plot(all_results, args.output_dir / "nonml_comparison.png")

    logger.info(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
