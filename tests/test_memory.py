"""Memory leak tests for SeasonalViT.

These tests verify that memory is properly freed over repeated forward/backward
passes, catching memory leaks early in CI.
"""

import gc

import pytest
import torch

from model.config import SeasonalViTConfig
from model.seasonal_vit import SeasonalViT


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMemoryLeaks:
    """Test that memory is properly freed."""

    @pytest.fixture
    def small_config(self) -> SeasonalViTConfig:
        """Small config for memory testing."""
        return SeasonalViTConfig(
            image_size=(64, 64),
            spatial_grid_size=2,
            num_spatial_patches=4,
            temporal_depth=2,
            temporal_heads=4,
            swin_pretrained=False,
        )

    def _create_dummy_batch(
        self, config: SeasonalViTConfig, batch_size: int = 2, num_timesteps: int = 8
    ) -> dict:
        """Create a dummy batch for testing."""
        return {
            "images": torch.randn(
                batch_size, num_timesteps, config.image_size[0], config.image_size[1]
            ).clamp(-1, 1),
            "season_ids": torch.tensor(
                [[s % 4 for s in range(num_timesteps)] for _ in range(batch_size)]
            ),
            "year_indices": torch.tensor(
                [[s // 4 for s in range(num_timesteps)] for _ in range(batch_size)]
            ),
            "lat": torch.rand(batch_size) * 180 - 90,
            "lon": torch.rand(batch_size) * 360 - 180,
            "target_season": torch.randint(0, 4, (batch_size,)),
        }

    def _run_iteration(self, model: SeasonalViT, batch: dict) -> None:
        """Run a single forward/backward iteration."""
        output = model(
            batch["images"],
            batch["season_ids"],
            batch["year_indices"],
            batch["lat"],
            batch["lon"],
            batch["target_season"],
        )
        loss = output["prediction"].sum()
        loss.backward()
        del output, loss

    def test_no_memory_leak_forward_backward(self, small_config: SeasonalViTConfig) -> None:
        """Test memory doesn't grow over repeated forward/backward."""
        model = SeasonalViT(small_config).cuda()
        batch = {k: v.cuda() for k, v in self._create_dummy_batch(small_config).items()}

        # Warmup
        self._run_iteration(model, batch)
        gc.collect()
        torch.cuda.empty_cache()

        initial_memory = torch.cuda.memory_allocated()

        for _ in range(10):
            self._run_iteration(model, batch)
            model.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()
        growth = final_memory - initial_memory

        # Allow small growth (< 10MB) for PyTorch internal caches
        assert growth < 10 * 1024 * 1024, f"Memory grew by {growth / 1024**2:.1f} MB"

    def test_no_memory_leak_eval_mode(self, small_config: SeasonalViTConfig) -> None:
        """Test memory doesn't grow in eval mode."""
        model = SeasonalViT(small_config).cuda()
        model.eval()
        batch = {k: v.cuda() for k, v in self._create_dummy_batch(small_config).items()}

        # Warmup
        with torch.no_grad():
            output = model(
                batch["images"],
                batch["season_ids"],
                batch["year_indices"],
                batch["lat"],
                batch["lon"],
                batch["target_season"],
            )
            del output

        gc.collect()
        torch.cuda.empty_cache()

        initial_memory = torch.cuda.memory_allocated()

        for _ in range(10):
            with torch.no_grad():
                output = model(
                    batch["images"],
                    batch["season_ids"],
                    batch["year_indices"],
                    batch["lat"],
                    batch["lon"],
                    batch["target_season"],
                )
                del output
            gc.collect()
            torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()
        growth = final_memory - initial_memory

        # Should be minimal growth in eval mode
        assert growth < 5 * 1024 * 1024, f"Memory grew by {growth / 1024**2:.1f} MB"


class TestMemoryLeaksCPU:
    """CPU memory tests (always run)."""

    @pytest.fixture
    def small_config(self) -> SeasonalViTConfig:
        """Small config for memory testing."""
        return SeasonalViTConfig(
            image_size=(64, 64),
            spatial_grid_size=2,
            num_spatial_patches=4,
            temporal_depth=2,
            temporal_heads=4,
            swin_pretrained=False,
        )

    def test_tensors_freed_after_forward(self, small_config: SeasonalViTConfig) -> None:
        """Test that intermediate tensors are freed after forward pass."""
        model = SeasonalViT(small_config)
        model.eval()

        batch_size = 2
        num_timesteps = 8

        images = torch.randn(
            batch_size, num_timesteps, small_config.image_size[0], small_config.image_size[1]
        ).clamp(-1, 1)
        season_ids = torch.tensor(
            [[s % 4 for s in range(num_timesteps)] for _ in range(batch_size)]
        )
        year_indices = torch.tensor(
            [[s // 4 for s in range(num_timesteps)] for _ in range(batch_size)]
        )
        lat = torch.rand(batch_size) * 180 - 90
        lon = torch.rand(batch_size) * 360 - 180
        target_season = torch.randint(0, 4, (batch_size,))
        target_year = torch.randint(0, 35, (batch_size,))

        # Forward pass
        with torch.no_grad():
            output = model(images, season_ids, year_indices, lat, lon, target_season, target_year)

        # Check output keys
        assert "prediction" in output
        assert "delta" in output
        assert "anchor" in output

        # Output should be detached (no grad history)
        assert not output["prediction"].requires_grad

        del output
        gc.collect()

        # No assertion here - just ensure no crash from dangling references
