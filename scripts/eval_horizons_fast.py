"""Fast horizon evaluation: encode each batch ONCE, decode 8x.

Same analysis as scripts/eval_horizons.py, but ~6-8x faster by
sharing the encoder pipeline across horizons. The original script ran
the full forward (encoder + decoder) per (batch, horizon) pair = 8x
encoder cost. Here we run the encoder once per batch and only the
decoder per horizon.

This is correct only because the input window is FIXED at [0:132]
(1990 spring through 2022 winter, the OOD-st input range). The slow
script silently shifted the window per horizon (see "Horizon Evaluation" in
docs/specific/forecasting_horizon.md), letting the model peek at
intermediate seasons. We use a fixed window here.

Per-horizon, what changes:
  - target_season (one of {0,1,2,3})
  - target_year (33 for 2023, 34 for 2024)
  - anchor (recomputed from images + target_season — cheap)

Per-horizon, what stays the same:
  - spatial_features, temporal_enc, location_enc, temporal_features
    (output of the spatial encoder + temporal transformer pipeline)

Output schema is identical to eval_horizons.py: same TSV, same columns,
incremental write per horizon (so we can pivot any subset for the paper).

Usage:
    python scripts/eval_horizons_fast.py \\
        --checkpoint checkpoints/seasonalvit_finetune.pt \\
        --eval-dir data/shards/ood_spatial/ood_spatial \\
        --output docs/horizon_results_ood_st.tsv
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List

import torch

from datasets.sharded_dataset import InMemoryNDVIDataset, create_dataloader
from model.seasonal_vit import SeasonalViT
from train.metrics import PerSampleMetricsTracker
from scripts.eval_ablations_on_split import _resolve_model_config

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("datasets.sharded_dataset").setLevel(logging.INFO)


LAST_INPUT_IDX = 131  # 2022 winter — last input timestep on OOD-st
INPUT_LEN = LAST_INPUT_IDX + 1  # 132 timesteps fed to the encoder
DEFAULT_HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8]
SEASON_NAMES = ["spring", "summer", "fall", "winter"]
REPORT_KEYS = [
    "val_mae", "val_rmse", "val_r2", "val_R2_det",
    "val_bias", "val_anchor_mae", "val_delta_abs_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("docs/horizon_results.tsv"))
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=DEFAULT_HORIZONS)
    parser.add_argument("--last-input-idx", type=int, default=LAST_INPUT_IDX)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def horizon_to_index(h: int, last_input_idx: int) -> int:
    return last_input_idx + h


def index_to_label(idx: int) -> str:
    return f"{1990 + idx // 4} {SEASON_NAMES[idx % 4]}"


def load_model(checkpoint_path: Path, device: torch.device) -> SeasonalViT:
    """Load the SeasonalViT checkpoint with its saved arch."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = _resolve_model_config(ckpt, checkpoint_path.parent.name)
    model = SeasonalViT(model_config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device).eval()
    print(
        f"  Arch: anchor_mode={getattr(model.config, 'anchor_mode', 'same_season')} "
        f"temporal_pathway={getattr(model.config, 'temporal_pathway', 'both')}",
        flush=True,
    )
    return model


def encode_batch_once(
    model: SeasonalViT,
    input_images: torch.Tensor,
    season_ids: torch.Tensor,
    year_indices: torch.Tensor,
    lat: torch.Tensor,
    lon: torch.Tensor,
) -> torch.Tensor:
    """Run spatial encoder + temporal encoder + location + temporal transformer.

    Returns temporal_features of shape (B, T, N, D), shareable across horizons.
    """
    B, T, H, W = input_images.shape
    if season_ids.dim() == 1:
        season_ids = season_ids.unsqueeze(0).expand(B, -1)
    if year_indices.dim() == 1:
        year_indices = year_indices.unsqueeze(0).expand(B, -1)

    images_for_encoding = input_images.clone()
    images_for_encoding[images_for_encoding == model.config.nodata_value] = (
        model.config.ndvi_mean
    )
    images_flat = images_for_encoding.reshape(B * T, 1, H, W)
    flat_features = model._encode_spatial_chunked(images_flat)  # (B*T, N, D)
    N = model.config.num_spatial_patches
    D = flat_features.shape[-1]
    spatial_features = flat_features.view(B, T, N, D)

    temporal_enc = model.temporal_encoding(season_ids, year_indices)
    location_enc, _ = model.location_encoder(lat, lon)
    spatial_features = spatial_features + location_enc.unsqueeze(1).unsqueeze(2)
    return model.temporal_transformer(spatial_features, temporal_enc)


def decode_for_horizon(
    model: SeasonalViT,
    temporal_features: torch.Tensor,
    input_images: torch.Tensor,
    season_ids: torch.Tensor,
    target_idx: int,
) -> Dict[str, torch.Tensor]:
    """Compute anchor + run decoder for a single target timestep.

    Builds (B,) target_season and target_year tensors from `target_idx`,
    computes the same-season anchor from the input window, then calls the
    decoder. Returns the model's standard output dict.
    """
    B = input_images.shape[0]
    device = input_images.device

    target_season = torch.full(
        (B,), target_idx % 4, dtype=torch.long, device=device,
    )
    target_year = torch.full(
        (B,), target_idx // 4, dtype=torch.long, device=device,
    )

    if season_ids.dim() == 1:
        season_ids_b = season_ids.unsqueeze(0).expand(B, -1)
    else:
        season_ids_b = season_ids

    anchor_mode = getattr(model.config, "anchor_mode", "same_season")
    if anchor_mode == "none":
        anchor = None
    else:
        anchor = model._get_climatology_anchor(
            input_images, season_ids_b, target_season,
        )
    return model.decoder(temporal_features, anchor, target_season, target_year)


@torch.no_grad()
def evaluate_all_horizons(
    model: SeasonalViT, val_loader, horizons: List[int],
    last_input_idx: int, device: torch.device,
) -> Dict[int, Dict[str, float]]:
    """Single-pass eval: encode each batch once, decode every horizon.

    The @torch.no_grad() decorator is load-bearing: without it, the
    encoder's per-timestep activations stay live for an unrun backward
    pass, ballooning memory ~10x at T=132. The original trainer.validate()
    has the same decorator -- this script bypasses that path so we have
    to add it ourselves.
    """
    input_len = last_input_idx + 1

    trackers = {h: PerSampleMetricsTracker() for h in horizons}
    anchor_trackers = {h: PerSampleMetricsTracker() for h in horizons}
    delta_sums = {h: 0.0 for h in horizons}
    num_batches = 0

    iterator = val_loader
    try:
        from tqdm import tqdm
        iterator = tqdm(val_loader, desc="Encode+decode (all horizons)")
    except ImportError:
        pass

    for batch in iterator:
        full_images = batch["images"].to(device, non_blocking=True)
        full_season_ids = batch["season_ids"].to(device, non_blocking=True)
        full_year_indices = batch["year_indices"].to(device, non_blocking=True)
        lat = batch["lat"].to(device, non_blocking=True)
        lon = batch["lon"].to(device, non_blocking=True)
        nodata_mask = batch.get("nodata_mask")
        if nodata_mask is not None:
            nodata_mask = nodata_mask.to(device, non_blocking=True)

        input_images = full_images[:, :input_len]
        input_season_ids = full_season_ids[..., :input_len]
        input_year_indices = full_year_indices[..., :input_len]

        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            temporal_features = encode_batch_once(
                model, input_images, input_season_ids, input_year_indices, lat, lon,
            )

            for h in horizons:
                target_idx = horizon_to_index(h, last_input_idx)
                output = decode_for_horizon(
                    model, temporal_features, input_images,
                    input_season_ids, target_idx,
                )
                target = full_images[:, target_idx:target_idx + 1]
                if nodata_mask is not None:
                    target_nodata = nodata_mask[:, target_idx:target_idx + 1]
                    valid_mask = ~target_nodata
                else:
                    valid_mask = None

                trackers[h].update(output["prediction"], target, valid_mask)
                if output.get("anchor") is not None:
                    anchor_trackers[h].update(output["anchor"], target, valid_mask)
                delta_sums[h] += output["delta"].abs().mean().item()

        num_batches += 1

    return _finalize_metrics(trackers, anchor_trackers, delta_sums, num_batches)


def _finalize_metrics(
    trackers: Dict[int, PerSampleMetricsTracker],
    anchor_trackers: Dict[int, PerSampleMetricsTracker],
    delta_sums: Dict[int, float],
    num_batches: int,
) -> Dict[int, Dict[str, float]]:
    """Convert per-horizon trackers into the same metric schema as validate()."""
    results = {}
    for h in trackers:
        pred = trackers[h].compute()
        anc = (
            anchor_trackers[h].compute() if anchor_trackers[h].n_samples > 0
            else {"mae": float("nan")}
        )
        results[h] = {
            "val_mae": pred["mae"],
            "val_rmse": pred["rmse"],
            "val_r2": pred["r2"],
            "val_R2_det": pred["R2_det"],
            "val_bias": pred["bias"],
            "val_anchor_mae": anc["mae"],
            "val_delta_abs_mean": delta_sums[h] / max(num_batches, 1),
        }
    return results


def write_tsv(
    results: Dict[int, Dict[str, float]], horizons: List[int],
    output: Path, last_input_idx: int,
) -> None:
    """Write per-horizon metrics as a TSV file (matches eval_horizons.py schema)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    header = ["horizon_seasons", "target_idx", "target_label"] + REPORT_KEYS
    lines = ["\t".join(header)]
    for h in horizons:
        if h not in results:
            continue
        idx = horizon_to_index(h, last_input_idx)
        m = results[h]
        row = [str(h), str(idx), index_to_label(idx)] + [
            f"{m.get(k, float('nan')):.6f}" for k in REPORT_KEYS
        ]
        lines.append("\t".join(row))
    output.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {output}", flush=True)


def print_summary_table(
    results: Dict[int, Dict[str, float]], horizons: List[int],
    last_input_idx: int,
) -> None:
    print("\n=== Summary ===")
    header = (
        f"{'h':>3s}  {'target':<14s}  {'MAE':>7s}  {'RMSE':>7s}  "
        f"{'r²':>6s}  {'R²_det':>7s}  {'bias':>7s}"
    )
    print(header)
    print("-" * len(header))
    for h in horizons:
        if h not in results:
            continue
        idx = horizon_to_index(h, last_input_idx)
        m = results[h]
        print(
            f"{h:>3d}  {index_to_label(idx):<14s}  "
            f"{m['val_mae']:>7.4f}  {m['val_rmse']:>7.4f}  "
            f"{m['val_r2']:>6.3f}  {m['val_R2_det']:>+7.3f}  "
            f"{m['val_bias']:>+7.4f}"
        )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        f"({index_to_label(args.last_input_idx)})  "
        f"input_window=[0:{args.last_input_idx + 1}] (fixed across all horizons)",
        flush=True,
    )

    print(f"Loading checkpoint {args.checkpoint} ...", flush=True)
    model = load_model(args.checkpoint, device)

    t0 = time.perf_counter()
    results = evaluate_all_horizons(
        model, val_loader, args.horizons, args.last_input_idx, device,
    )
    print(f"\nElapsed: {(time.perf_counter() - t0) / 60:.1f} min")

    print_summary_table(results, args.horizons, args.last_input_idx)
    write_tsv(results, args.horizons, args.output, args.last_input_idx)


if __name__ == "__main__":
    main()
