"""Pytest fixtures for SeasonalViT tests.

Provides reusable fixtures for model testing, including:
- Sample data paths from data/samples/
- Dummy tensor batches for shape testing
- Model configurations
"""

from pathlib import Path
from typing import List

import pytest
import torch

from model.config import SeasonalViTConfig, PretrainingConfig, FinetuningConfig


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def samples_dir() -> Path:
    """Return path to sample data directory."""
    return Path("data/samples")


@pytest.fixture
def sample_tif_paths(samples_dir: Path) -> List[Path]:
    """Return list of sample GeoTIFF paths."""
    if not samples_dir.exists():
        pytest.skip("Sample data not available")
    paths = sorted(samples_dir.glob("*.tif"))
    if not paths:
        pytest.skip("No sample TIF files found")
    return paths


@pytest.fixture
def sample_tif_path(sample_tif_paths: List[Path]) -> Path:
    """Return a single sample GeoTIFF path."""
    return sample_tif_paths[0]


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def model_config() -> SeasonalViTConfig:
    """Return default model configuration for testing.

    Note: Pretrained weights disabled by default for faster CI testing.
    """
    return SeasonalViTConfig(swin_pretrained=False)


@pytest.fixture
def small_model_config() -> SeasonalViTConfig:
    """Return smaller model configuration for faster testing."""
    return SeasonalViTConfig(
        image_size=(64, 64),
        spatial_grid_size=2,
        num_spatial_patches=4,
        temporal_depth=2,
        temporal_heads=4,
        swin_pretrained=False,
    )


@pytest.fixture
def pretrain_config() -> PretrainingConfig:
    """Return default pre-training configuration."""
    return PretrainingConfig()


@pytest.fixture
def finetune_config() -> FinetuningConfig:
    """Return default fine-tuning configuration."""
    return FinetuningConfig()


# =============================================================================
# Tensor Fixtures
# =============================================================================


@pytest.fixture
def dummy_ndvi_batch() -> torch.Tensor:
    """Return dummy NDVI batch tensor.

    Shape: (B, T, H, W) = (2, 140, 256, 256)
    Values: Random in [-1, 1] range (valid NDVI range)
    """
    batch_size = 2
    num_timesteps = 140
    height = 256
    width = 256
    return torch.randn(batch_size, num_timesteps, height, width).clamp(-1, 1)


@pytest.fixture
def dummy_ndvi_single() -> torch.Tensor:
    """Return single NDVI image tensor.

    Shape: (1, H, W) = (1, 256, 256)
    Values: Random in [-1, 1] range
    """
    return torch.randn(1, 256, 256).clamp(-1, 1)


@pytest.fixture
def dummy_spatial_input() -> torch.Tensor:
    """Return input tensor for spatial encoder.

    Shape: (B*T, 1, H, W) = (280, 1, 256, 256)
    This represents 2 batches × 140 timesteps flattened.
    """
    batch_size = 2
    num_timesteps = 140
    return torch.randn(batch_size * num_timesteps, 1, 256, 256).clamp(-1, 1)


@pytest.fixture
def dummy_temporal_input(model_config: SeasonalViTConfig) -> torch.Tensor:
    """Return input tensor for temporal transformer.

    Shape: (B, T, N, D) = (2, 140, 64, 768)
    Where N = num_spatial_patches, D = embed_dim
    """
    batch_size = 2
    return torch.randn(
        batch_size,
        model_config.num_timesteps,
        model_config.num_spatial_patches,
        model_config.embed_dim,
    )


@pytest.fixture
def dummy_location_coords() -> torch.Tensor:
    """Return dummy latitude/longitude coordinates.

    Shape: (B, 2) where columns are [lat, lon]
    Values: lat in [-90, 90], lon in [-180, 180]
    """
    batch_size = 2
    lat = torch.rand(batch_size) * 180 - 90  # [-90, 90]
    lon = torch.rand(batch_size) * 360 - 180  # [-180, 180]
    return torch.stack([lat, lon], dim=1)


@pytest.fixture
def dummy_season_ids() -> torch.Tensor:
    """Return dummy season IDs for all timesteps.

    Shape: (T,) = (140,)
    Values: [0, 1, 2, 3, 0, 1, 2, 3, ...] repeating
    """
    num_years = 35
    seasons_per_year = 4
    return torch.tensor([s for _ in range(num_years) for s in range(seasons_per_year)])


@pytest.fixture
def dummy_year_indices() -> torch.Tensor:
    """Return dummy year indices for all timesteps.

    Shape: (T,) = (140,)
    Values: [0, 0, 0, 0, 1, 1, 1, 1, ...] incrementing each 4 timesteps
    """
    num_years = 35
    seasons_per_year = 4
    return torch.tensor([y for y in range(num_years) for _ in range(seasons_per_year)])


# =============================================================================
# Device Fixtures
# =============================================================================


@pytest.fixture
def device() -> torch.device:
    """Return available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
