"""End-to-end tests for scripts/evaluate.py on a tiny CPU model.

InMemoryNDVIDataset is fixed to 256x256 tiles, which makes a CPU run of the
Swin encoder slow, so these tests swap in a small in-memory dataset with the
same item fields. Shard loading itself is covered by tests/test_shard_io.py
and tests/test_dataset.py.
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from model.config import FinetuningConfig, SeasonalViTConfig
from model.seasonal_vit import SeasonalViT
from scripts import evaluate as evaluate_script
from scripts.eval_ablations_on_split import load_model_and_train_config

NUM_TIMESTEPS = 140
IMAGE_SIZE = 64


def _tiny_model_config() -> SeasonalViTConfig:
    return SeasonalViTConfig(
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        spatial_grid_size=2,
        num_spatial_patches=4,
        temporal_depth=2,
        temporal_heads=4,
        swin_pretrained=False,
    )


def _write_checkpoint(path: Path, train_config: FinetuningConfig) -> None:
    """Save a checkpoint with the same keys FinetuningTrainer._save_checkpoint writes."""
    torch.manual_seed(0)
    model_config = _tiny_model_config()
    model = SeasonalViT(model_config)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": train_config, "model_config": model_config},
        path,
    )


class _TinyNDVIDataset(Dataset):
    """Same item fields and image dtype (float16) as InMemoryNDVIDataset, at 64x64."""

    def __init__(self, num_samples: int = 2) -> None:
        torch.manual_seed(1)
        shape = (num_samples, NUM_TIMESTEPS, IMAGE_SIZE, IMAGE_SIZE)
        self.images = (torch.rand(shape) * 0.8).to(torch.float16)
        self.nodata_masks = torch.zeros(shape, dtype=torch.bool)
        self.lats = torch.tensor([45.0, -10.0][:num_samples])
        self.lons = torch.tensor([10.0, 120.0][:num_samples])
        self.season_ids = torch.tensor([t % 4 for t in range(NUM_TIMESTEPS)])
        self.year_indices = torch.tensor([t // 4 for t in range(NUM_TIMESTEPS)])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        return {
            "images": self.images[idx], "season_ids": self.season_ids,
            "year_indices": self.year_indices, "lat": self.lats[idx],
            "lon": self.lons[idx], "nodata_mask": self.nodata_masks[idx],
        }


def _run_evaluate(monkeypatch, argv: list) -> None:
    def tiny_dataloader(eval_dir, batch_size, limit=None):
        return DataLoader(_TinyNDVIDataset(), batch_size=batch_size, shuffle=False)

    monkeypatch.setattr(evaluate_script, "create_eval_dataloader", tiny_dataloader)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", *argv])
    evaluate_script.main()


def test_evaluate_writes_overall_and_per_season_metrics(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint, FinetuningConfig())
    output_dir = tmp_path / "eval"

    _run_evaluate(monkeypatch, [
        "--checkpoint", str(checkpoint), "--eval-dir", str(tmp_path / "unused"),
        "--output-dir", str(output_dir), "--predict-years", "2022",
        "--batch-size", "2", "--per-season", "--visualize",
    ])

    results = json.loads((output_dir / "metrics.json").read_text())
    assert results["predict_years"] == [2022]
    assert results["overall"]["num_samples"] == 2
    assert sorted(results["per_season"]) == [
        "2022 fall", "2022 spring", "2022 summer", "2022 winter",
    ]
    for metrics in [results["overall"], *results["per_season"].values()]:
        assert 0.0 <= metrics["val_mae"] < 2.0
    assert (output_dir / "predictions_grid.png").exists()


def test_target_label_maps_index_to_year_and_season():
    assert evaluate_script.target_label(0) == "1990 spring"
    assert evaluate_script.target_label(131) == "2022 winter"
    assert evaluate_script.target_label(139) == "2024 winter"


def test_predict_years_override_clears_saved_target_indices(tmp_path):
    """A saved predict_target_indices would otherwise win over --predict-years."""
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint, FinetuningConfig(predict_target_indices=[128]))

    _, train_config = load_model_and_train_config(checkpoint, [2023])

    assert train_config.predict_target_indices is None
    assert train_config.get_target_indices() == [132, 133, 134, 135]


def test_prediction_grid_handles_a_single_sample():
    """plt.subplots returns 1-D axes for one row unless squeeze=False."""
    from eval.visualize import create_prediction_grid

    image = torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    outputs = [{"prediction": image, "delta": image * 0.1, "anchor": image}]

    fig = create_prediction_grid(outputs, [image], n_samples=1)

    assert len(fig.axes) == 4
