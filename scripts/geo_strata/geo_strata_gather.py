"""Per-tile metrics on OOD-s 2022 for stratified analysis.

Runs the fine-tuned SeasonalViT checkpoint over the OOD-s 2022 shards in a
single pass and writes one row per tile to a TSV. The map and table scripts
read this TSV to stratify by latitude, Koppen, cloud regime, and land cover
without re-running the model.

Per tile we record:
  - sample_id, filename, lat, lon
  - sigma_tile : temporal std of seasonal NDVI over the 1990-spring through
                 2021-winter input window (NoData-aware, fp32)
  - {mae, rmse, bias, pred_mean, gt_mean, n_valid}_{spring,summer,fall,winter}

Cost: ~30-60 min on remote L40S for the full ~22.5k OOD-s tiles. Use
--smoke for a 500-tile sanity pass (~3-5 min) before the full run.

Usage:
    python scripts/geo_strata/geo_strata_gather.py \\
        --checkpoint checkpoints/seasonalvit_finetune.pt \\
        --eval-dir data/shards/ood_spatial/ood_spatial \\
        --output-tsv results/geo_strata/per_tile_ood_s_2022.tsv \\
        --smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.sharded_dataset import InMemoryNDVIDataset, create_dataloader  # noqa: E402
from scripts.eval_horizons_fast import encode_batch_once, load_model  # noqa: E402

GEO_STRATA_DIR = Path(__file__).resolve().parent
if str(GEO_STRATA_DIR) not in sys.path:
    sys.path.insert(0, str(GEO_STRATA_DIR))

from _geo_strata_helpers import (  # noqa: E402
    SEASON_NAMES,
    format_tsv_value,
    per_tile_stats_one_season,
    sigma_tile_temporal,
)
from _strata_utils import sample_id_from_filename  # noqa: E402


LAST_INPUT_IDX_OOD_S = 127  # 2021 winter
INPUT_LEN = LAST_INPUT_IDX_OOD_S + 1  # 128 timesteps fed to the encoder
TARGET_INDICES_2022 = [128, 129, 130, 131]  # 2022 spring..winter
SMOKE_TILE_COUNT = 500


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--output-tsv", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap total tiles. --smoke overrides to 500.")
    p.add_argument("--smoke", action="store_true",
                   help=f"Run on the first {SMOKE_TILE_COUNT} tiles only.")
    p.add_argument("--split-name", type=str, required=True,
                   help="Tag every emitted row with this split label (e.g. "
                        "'ood_s', 'val_iid', 'ood_t'). Stored in the 'split' "
                        "column so downstream scripts can filter or color by it.")
    return p.parse_args()


def build_dataset(eval_dir: Path, max_samples: int | None) -> InMemoryNDVIDataset:
    return InMemoryNDVIDataset(eval_dir, max_samples=max_samples, num_workers=2)


def build_loader(dataset: InMemoryNDVIDataset, batch_size: int, num_workers: int):
    return create_dataloader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2,
    )


def decode_target_season(
    model,
    temporal_features: torch.Tensor,
    input_images: torch.Tensor,
    input_season_ids: torch.Tensor,
    target_idx: int,
) -> torch.Tensor:
    """Decode one target season; return predictions of shape (B, 1, H, W)."""
    B = input_images.shape[0]
    device = input_images.device
    target_season = torch.full((B,), target_idx % 4, dtype=torch.long, device=device)
    target_year = torch.full((B,), target_idx // 4, dtype=torch.long, device=device)
    if input_season_ids.dim() == 1:
        season_ids_b = input_season_ids.unsqueeze(0).expand(B, -1)
    else:
        season_ids_b = input_season_ids
    anchor_mode = getattr(model.config, "anchor_mode", "same_season")
    anchor = (
        None if anchor_mode == "none"
        else model._get_climatology_anchor(input_images, season_ids_b, target_season)
    )
    out = model.decoder(temporal_features, anchor, target_season, target_year)
    return out["prediction"]


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "images": batch["images"].to(device, non_blocking=True),
        "season_ids": batch["season_ids"].to(device, non_blocking=True),
        "year_indices": batch["year_indices"].to(device, non_blocking=True),
        "lat": batch["lat"].to(device, non_blocking=True),
        "lon": batch["lon"].to(device, non_blocking=True),
        "nodata_mask": batch["nodata_mask"].to(device, non_blocking=True),
    }


def _score_seasons(
    model,
    temporal_features: torch.Tensor,
    full_images: torch.Tensor,
    nodata_mask: torch.Tensor,
    input_images: torch.Tensor,
    input_seasons: torch.Tensor,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Decode each 2022 target season and compute per-tile stats."""
    per_season: Dict[str, Dict[str, torch.Tensor]] = {}
    for season_name, target_idx in zip(SEASON_NAMES, TARGET_INDICES_2022):
        pred = decode_target_season(
            model, temporal_features, input_images, input_seasons, target_idx,
        )
        gt = full_images[:, target_idx]
        valid = ~nodata_mask[:, target_idx]
        per_season[season_name] = per_tile_stats_one_season(
            pred.squeeze(1).float(), gt.float(), valid,
        )
    return per_season


@torch.inference_mode()
def process_batch(batch, model, device) -> Dict[str, Any]:
    """Run encode + 4-season decode + sigma; return per-tile aggregates as tensors.

    Uses `torch.inference_mode()` rather than `torch.no_grad()`: same correctness
    guarantees, but additionally disables autograd version-counter bookkeeping,
    so the encoder + decoder pass is strictly faster on the gather workload.
    Requires `model.eval()`, which `load_model()` already sets.
    """
    b = _move_batch(batch, device)
    input_images = b["images"][:, :INPUT_LEN]
    input_seasons = b["season_ids"][..., :INPUT_LEN]
    input_years = b["year_indices"][..., :INPUT_LEN]
    input_nodata = b["nodata_mask"][:, :INPUT_LEN]

    sigma = sigma_tile_temporal(input_images, input_nodata)

    with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):  # type: ignore[attr-defined]
        temporal_features = encode_batch_once(
            model, input_images, input_seasons, input_years, b["lat"], b["lon"],
        )
        per_season = _score_seasons(
            model, temporal_features, b["images"], b["nodata_mask"],
            input_images, input_seasons,
        )

    return {
        "lat": b["lat"].detach().cpu(),
        "lon": b["lon"].detach().cpu(),
        "sigma": sigma.detach().cpu(),
        "per_season": {s: {k: v.detach().cpu() for k, v in m.items()}
                       for s, m in per_season.items()},
    }


def _row_for_tile(
    b: int,
    sample_id: str,
    filename: str,
    lat: float,
    lon: float,
    sigma: float,
    per_season: Dict[str, Dict[str, torch.Tensor]],
    split_name: str,
) -> Dict[str, float | int | str]:
    row: Dict[str, float | int | str] = {
        "sample_id": sample_id,
        "filename": filename,
        "split": split_name,
        "lat": lat,
        "lon": lon,
        "sigma_tile": sigma,
    }
    for season in SEASON_NAMES:
        stats = per_season[season]
        row[f"mae_{season}"] = float(stats["mae"][b].item())
        row[f"rmse_{season}"] = float(stats["rmse"][b].item())
        row[f"bias_{season}"] = float(stats["bias"][b].item())
        row[f"pred_mean_{season}"] = float(stats["pred_mean"][b].item())
        row[f"gt_mean_{season}"] = float(stats["gt_mean"][b].item())
        row[f"n_valid_{season}"] = int(stats["n_valid"][b].item())
    return row


def emit_rows(
    batch_out, dataset, sample_offset: int, split_name: str,
) -> List[dict]:
    rows = []
    n = batch_out["lat"].shape[0]
    for b in range(n):
        idx = sample_offset + b
        filename = dataset.files[idx]
        rows.append(_row_for_tile(
            b=b,
            sample_id=sample_id_from_filename(filename),
            filename=filename,
            lat=float(batch_out["lat"][b].item()),
            lon=float(batch_out["lon"][b].item()),
            sigma=float(batch_out["sigma"][b].item()),
            per_season=batch_out["per_season"],
            split_name=split_name,
        ))
    return rows


def write_tsv(rows: List[dict], output_tsv: Path) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_tsv.write_text("")
        return
    columns = list(rows[0].keys())
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(format_tsv_value(row[c]) for c in columns))
    output_tsv.write_text("\n".join(lines) + "\n")


def _resolve_max_samples(args: argparse.Namespace) -> int | None:
    if args.smoke:
        return SMOKE_TILE_COUNT
    return args.max_samples


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_samples = _resolve_max_samples(args)

    print(f"Loading data from {args.eval_dir} (max_samples={max_samples})...", flush=True)
    dataset = build_dataset(args.eval_dir, max_samples=max_samples)
    loader = build_loader(dataset, args.batch_size, args.num_workers)
    print(f"Tiles: {len(dataset)}", flush=True)

    print(f"Loading checkpoint {args.checkpoint}...", flush=True)
    model = load_model(args.checkpoint, device)

    t0 = time.perf_counter()
    rows: List[dict] = []
    sample_offset = 0
    n_total = len(dataset)
    pbar = tqdm(total=n_total, desc="Gather (encode+decode)", unit="tile")
    for batch in loader:
        batch_out = process_batch(batch, model, device)
        rows.extend(emit_rows(batch_out, dataset, sample_offset, args.split_name))
        b = batch_out["lat"].shape[0]
        sample_offset += b
        pbar.update(b)
    pbar.close()

    elapsed_min = (time.perf_counter() - t0) / 60
    print(f"\nElapsed: {elapsed_min:.1f} min", flush=True)
    write_tsv(rows, args.output_tsv)
    print(f"Wrote {args.output_tsv} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
