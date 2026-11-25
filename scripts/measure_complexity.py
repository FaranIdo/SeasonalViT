"""Measure SeasonalViT computational complexity.

Reports per-component parameter counts and GFLOPs for a single forward pass,
plus the total counts for both the pretraining model and the fine-tuning model.

Run from repo root with the venv activated:
    python scripts/measure_complexity.py
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from fvcore.nn import FlopCountAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.config import SeasonalViTConfig
from model.pretrain_model import PretrainingModel
from model.seasonal_vit import SeasonalViT


logging.basicConfig(level=logging.WARNING)


# Inputs used when measuring fine-tune model FLOPs: an input sequence of
# T_in = 136 timesteps, with the target set to 2022 spring.
T_IN_FINETUNE = 136
T_TOTAL_PRETRAIN = 140  # full 35 years × 4 seasons
H = W = 256


def _human(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return str(n)


def _gflops(n: int) -> str:
    return f"{n / 1e9:.2f} G"


def count_params_per_submodule(model: torch.nn.Module) -> Dict[str, int]:
    """Return a dict of {top-level submodule name: parameter count}."""
    counts = {}
    for name, module in model.named_children():
        counts[name] = sum(p.numel() for p in module.parameters())
    return counts


def build_finetune_inputs(_config: SeasonalViTConfig) -> Tuple:
    """Synthetic inputs matching the SeasonalViT.forward signature."""
    B = 1
    T = T_IN_FINETUNE
    images = torch.randn(B, T, H, W) * 0.2 + 0.4
    season_ids = torch.arange(T) % 4
    year_indices = torch.arange(T) // 4
    season_ids = season_ids.unsqueeze(0).expand(B, -1)
    year_indices = year_indices.unsqueeze(0).expand(B, -1)
    lat = torch.tensor([35.0])
    lon = torch.tensor([-100.0])
    target_season = torch.tensor([0])
    target_year = torch.tensor([32])  # 2022 = year index 32
    return (
        images,
        season_ids,
        year_indices,
        lat,
        lon,
        target_season,
        target_year,
    )


def build_pretrain_inputs() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic inputs for PretrainingModel.forward."""
    B = 1
    T = T_TOTAL_PRETRAIN
    images = torch.randn(B, T, H, W) * 0.2 + 0.4
    season_ids = (torch.arange(T) % 4).unsqueeze(0).expand(B, -1)
    year_indices = (torch.arange(T) // 4).unsqueeze(0).expand(B, -1)
    return images, season_ids, year_indices


def measure_finetune(config: SeasonalViTConfig) -> Tuple[Dict[str, int], int, int]:
    """Return (per-submodule params, total params, FLOPs) for SeasonalViT."""
    model = SeasonalViT(config)
    model.eval()
    params_by_sub = count_params_per_submodule(model)
    total = sum(params_by_sub.values())
    inputs = build_finetune_inputs(config)
    flops = FlopCountAnalysis(model, inputs)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return params_by_sub, total, flops.total()


def measure_pretrain(config: SeasonalViTConfig) -> Tuple[Dict[str, int], int, int]:
    """Return (per-submodule params, total params, FLOPs) for PretrainingModel."""
    model = PretrainingModel(config)
    model.eval()
    params_by_sub = count_params_per_submodule(model)
    total = sum(params_by_sub.values())
    inputs = build_pretrain_inputs()
    flops = FlopCountAnalysis(model, inputs)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return params_by_sub, total, flops.total()


def print_table(title: str, params_by_sub: Dict[str, int], total_params: int, total_flops: int) -> None:
    print(f"\n## {title}")
    print(f"{'Submodule':<28} {'Params':>14}")
    for name, n in params_by_sub.items():
        print(f"{name:<28} {_human(n):>14}")
    print(f"{'TOTAL':<28} {_human(total_params):>14}")
    print(f"FLOPs (single forward): {_gflops(total_flops)}")
    print(f"FLOPs (raw):             {total_flops:,}")


def main() -> None:
    config = SeasonalViTConfig(swin_pretrained=False, use_gradient_checkpointing=False)

    print(f"Inputs (finetune):  B=1, T={T_IN_FINETUNE}, H=W={H}")
    print(f"Inputs (pretrain):  B=1, T={T_TOTAL_PRETRAIN}, H=W={H}")

    ft_params, ft_total, ft_flops = measure_finetune(config)
    print_table("SeasonalViT (forecasting / fine-tuning)", ft_params, ft_total, ft_flops)

    pt_params, pt_total, pt_flops = measure_pretrain(config)
    print_table("PretrainingModel", pt_params, pt_total, pt_flops)


if __name__ == "__main__":
    main()
