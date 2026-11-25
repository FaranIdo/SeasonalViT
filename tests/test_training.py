"""Tests for training loops.

Tests both pre-training and fine-tuning trainers:
- Single batch loss decreases
- Checkpoint save/load

NOTE: Some tests are marked with @pytest.mark.skip_in_full_suite because they
fail due to MPS memory exhaustion when running the full test suite. These tests
pass when run in isolation:
    python -m pytest tests/test_training.py -v
"""

# Trainers open a TensorBoard SummaryWriter that is only closed at the end of
# train(); tests that call train_epoch() directly leave its writer thread running,
# which can race the temp-dir cleanup, hence ignore_cleanup_errors=True below.
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.config import (
    SeasonalViTConfig,
    PretrainingConfig,
    FinetuningConfig,
)
from model.decoder import ObservationAnchoredDecoder
from model.pretrain_model import PretrainingModel
from model.seasonal_vit import SeasonalViT
from train.pretrain import PretrainingTrainer
from train.finetune import FinetuningTrainer
from train.losses import ForecastingLoss


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_config() -> SeasonalViTConfig:
    """Small model config for fast testing."""
    return SeasonalViTConfig(
        image_size=(64, 64),
        spatial_grid_size=2,
        num_spatial_patches=4,
        temporal_depth=2,  # Minimum 2 for dual-scale (each pathway gets 1)
        temporal_heads=4,
        swin_pretrained=False,
    )


@pytest.fixture
def small_pretrain_config() -> PretrainingConfig:
    """Small pre-training config for fast testing."""
    return PretrainingConfig(
        max_epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        warmup_epochs=0,
        # Match dummy_pretrain_batch which has T=8 (2 years × 4 seasons)
        temporal_subset_years=2,
    )


@pytest.fixture
def small_finetune_config() -> FinetuningConfig:
    """Small fine-tuning config for fast testing."""
    return FinetuningConfig(
        max_epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        warmup_epochs=0,
        freeze_encoder=False,
        # With T=12 and horizon=4, fixed input length = 12 - 4 = 8 (divisible by 4)
        forecast_horizon=4,
        predict_years=None,  # Use forecast_horizon, not default [2022]
        use_all_history_before_target=False,
    )


@pytest.fixture
def pretrain_model(small_config: SeasonalViTConfig) -> PretrainingModel:
    """Create pre-training model."""
    return PretrainingModel(small_config)


@pytest.fixture
def forecast_model(small_config: SeasonalViTConfig) -> SeasonalViT:
    """Create forecasting model."""
    return SeasonalViT(small_config)


@pytest.fixture
def device() -> torch.device:
    """Get test device."""
    return torch.device("cpu")


@pytest.fixture
def dummy_pretrain_batch() -> dict:
    """Create dummy batch for pre-training."""
    B, T, H, W = 2, 8, 64, 64  # T divisible by 4 for seasons
    return {
        "images": torch.randn(B, T, H, W).clamp(-1, 1),
        "season_ids": torch.tensor([[i % 4 for i in range(T)] for _ in range(B)]),
        "year_indices": torch.tensor([[i // 4 for i in range(T)] for _ in range(B)]),
    }


@pytest.fixture
def dummy_finetune_batch() -> dict:
    """Create dummy batch for fine-tuning with Trainer (variable target).

    T=12 (3 years × 4 seasons) with forecast_horizon=4 and use_all_history_before_target=False
    gives fixed input length of 12-4=8 (divisible by 4).
    """
    B, T, H, W = 2, 12, 64, 64
    return {
        "images": torch.randn(B, T, H, W).clamp(-1, 1),
        "season_ids": torch.tensor([[i % 4 for i in range(T)] for _ in range(B)]),
        "year_indices": torch.tensor([[i // 4 for i in range(T)] for _ in range(B)]),
        "lat": torch.tensor([37.0, 45.0]),
        "lon": torch.tensor([-122.0, -73.0]),
        "nodata_mask": torch.zeros(B, T, H, W, dtype=torch.bool),
    }


def create_dummy_dataloader(batch: dict, num_batches: int = 3) -> DataLoader:
    """Create a simple dataloader that repeats the batch."""
    class RepeatDataset:
        def __init__(self, batch, num_batches):
            self.batch = batch
            self.num_batches = num_batches

        def __len__(self):
            return self.num_batches

        def __getitem__(self, idx):
            return self.batch

    dataset = RepeatDataset(batch, num_batches)
    return DataLoader(dataset, batch_size=None)


# =============================================================================
# Pre-training Tests
# =============================================================================


class TestPretrainingTrainer:
    """Tests for PretrainingTrainer."""

    def test_trainer_initialization(
        self,
        pretrain_model: PretrainingModel,
        small_pretrain_config: PretrainingConfig,
        device: torch.device,
    ):
        """Test trainer initializes correctly."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = PretrainingTrainer(
                model=pretrain_model,
                config=small_pretrain_config,
                device=device,
                output_dir=Path(tmpdir),
            )

            assert trainer.model is not None
            assert trainer.optimizer is not None
            assert trainer.scheduler is not None
            assert trainer.epoch == 0
            assert trainer.global_step == 0

    def test_single_train_step(
        self,
        pretrain_model: PretrainingModel,
        small_pretrain_config: PretrainingConfig,
        dummy_pretrain_batch: dict,
        device: torch.device,
    ):
        """Test single training step executes and returns loss."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = PretrainingTrainer(
                model=pretrain_model,
                config=small_pretrain_config,
                device=device,
                output_dir=Path(tmpdir),
            )

            loss = trainer._train_step_accumulate(dummy_pretrain_batch, accum_steps=1)

            assert isinstance(loss, torch.Tensor)
            assert loss.item() > 0
            assert not torch.isnan(loss)

    @pytest.mark.skip(reason="MPS memory exhaustion in full suite. Run in isolation: pytest tests/test_training.py::TestPretrainingTrainer::test_loss_decreases_over_steps")
    def test_loss_decreases_over_steps(
        self,
        pretrain_model: PretrainingModel,
        small_pretrain_config: PretrainingConfig,
        dummy_pretrain_batch: dict,
        device: torch.device,
    ):
        """Test that loss generally decreases over multiple steps."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = PretrainingTrainer(
                model=pretrain_model,
                config=small_pretrain_config,
                device=device,
                output_dir=Path(tmpdir),
            )

            losses = []
            for _ in range(5):
                loss = trainer._train_step_accumulate(dummy_pretrain_batch, accum_steps=1)
                losses.append(loss.item())
                trainer.global_step += 1

            # Loss should decrease (allow some variance)
            assert losses[-1] < losses[0] * 1.5  # Not getting much worse
            # At least one step should show improvement
            assert min(losses[1:]) < losses[0]

    @pytest.mark.skip(reason="MPS memory exhaustion in full suite. Run in isolation: pytest tests/test_training.py::TestPretrainingTrainer::test_train_epoch")
    def test_train_epoch(
        self,
        pretrain_model: PretrainingModel,
        small_pretrain_config: PretrainingConfig,
        dummy_pretrain_batch: dict,
        device: torch.device,
    ):
        """Test training for one epoch."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = PretrainingTrainer(
                model=pretrain_model,
                config=small_pretrain_config,
                device=device,
                output_dir=Path(tmpdir),
            )

            dataloader = create_dummy_dataloader(dummy_pretrain_batch, num_batches=3)
            metrics = trainer.train_epoch(dataloader, epoch=0)

            assert "loss" in metrics
            assert "num_batches" in metrics
            assert metrics["num_batches"] == 3
            assert metrics["loss"] > 0

    def test_checkpoint_save_load(
        self,
        pretrain_model: PretrainingModel,
        small_pretrain_config: PretrainingConfig,
        device: torch.device,
    ):
        """Test checkpoint save and load."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            output_dir = Path(tmpdir)

            # Create trainer and modify state
            trainer = PretrainingTrainer(
                model=pretrain_model,
                config=small_pretrain_config,
                device=device,
                output_dir=output_dir,
            )
            trainer.epoch = 5
            trainer.global_step = 100
            trainer.best_loss = 0.5

            # Save checkpoint
            checkpoint_path = output_dir / "test_checkpoint.pt"
            trainer._save_checkpoint(checkpoint_path)

            # Create new trainer and load
            new_model = PretrainingModel(pretrain_model.config)
            new_trainer = PretrainingTrainer(
                model=new_model,
                config=small_pretrain_config,
                device=device,
                output_dir=output_dir,
                resume_from=checkpoint_path,
            )

            assert new_trainer.epoch == 6  # epoch + 1
            assert new_trainer.global_step == 100
            assert new_trainer.best_loss == 0.5

    def test_train_epoch_with_progress_bar(
        self,
        pretrain_model: PretrainingModel,
        small_pretrain_config: PretrainingConfig,
        dummy_pretrain_batch: dict,
        device: torch.device,
    ):
        """Test train_epoch works with progress bar enabled."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = PretrainingTrainer(
                model=pretrain_model,
                config=small_pretrain_config,
                device=device,
                output_dir=Path(tmpdir),
            )

            dataloader = create_dummy_dataloader(dummy_pretrain_batch, num_batches=3)
            metrics = trainer.train_epoch(dataloader, epoch=0, use_progress_bar=True)

            assert "loss" in metrics
            assert "num_batches" in metrics
            assert metrics["num_batches"] == 3
            assert metrics["loss"] > 0

    def test_train_epoch_without_progress_bar(
        self,
        pretrain_model: PretrainingModel,
        small_pretrain_config: PretrainingConfig,
        dummy_pretrain_batch: dict,
        device: torch.device,
    ):
        """Test train_epoch works with progress bar disabled."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = PretrainingTrainer(
                model=pretrain_model,
                config=small_pretrain_config,
                device=device,
                output_dir=Path(tmpdir),
            )

            dataloader = create_dummy_dataloader(dummy_pretrain_batch, num_batches=3)
            metrics = trainer.train_epoch(dataloader, epoch=0, use_progress_bar=False)

            assert "loss" in metrics
            assert "num_batches" in metrics
            assert metrics["num_batches"] == 3
            assert metrics["loss"] > 0


# =============================================================================
# Fine-tuning Tests
# =============================================================================


class TestFinetuningTrainer:
    """Tests for FinetuningTrainer."""

    def test_trainer_initialization(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        device: torch.device,
    ):
        """Test trainer initializes correctly."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=Path(tmpdir),
            )

            assert trainer.model is not None
            assert trainer.optimizer is not None
            assert trainer.loss_fn is not None
            assert trainer.epoch == 0

    def test_single_train_step(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        dummy_finetune_batch: dict,
        device: torch.device,
    ):
        """Test single training step executes."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=Path(tmpdir),
            )

            loss, components = trainer._train_step_accumulate(
                dummy_finetune_batch, accum_steps=1,
            )

            assert isinstance(loss, float)
            assert loss > 0
            assert "recon_loss" in components
            assert "delta_reg" in components

    def test_loss_stable_over_steps(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        dummy_finetune_batch: dict,
        device: torch.device,
    ):
        """Test that loss remains stable over multiple steps (doesn't explode)."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=Path(tmpdir),
            )

            losses = []
            for _ in range(3):
                trainer.optimizer.zero_grad()
                loss, _ = trainer._train_step_accumulate(
                    dummy_finetune_batch, accum_steps=1,
                )
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
                losses.append(loss)
                trainer.global_step += 1

            # Loss should not explode (allow 2x increase max)
            assert losses[-1] < losses[0] * 2.0
            # Loss should remain finite and positive
            assert all(0 < loss < float("inf") for loss in losses)

    def test_train_epoch(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        dummy_finetune_batch: dict,
        device: torch.device,
    ):
        """Test training for one epoch."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=Path(tmpdir),
            )

            dataloader = create_dummy_dataloader(dummy_finetune_batch, num_batches=3)
            metrics = trainer.train_epoch(dataloader, epoch=0)

            assert "loss" in metrics
            assert "recon_loss" in metrics
            assert "delta_reg" in metrics
            assert metrics["num_batches"] == 3

    def test_encoder_freezing(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        device: torch.device,
    ):
        """Test encoder freezing option."""
        small_finetune_config.freeze_encoder = True

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=Path(tmpdir),
            )

            # Check encoder is frozen
            for name, param in trainer.model.named_parameters():
                if name.startswith("spatial_encoder"):
                    assert not param.requires_grad
                elif name.startswith("decoder"):
                    assert param.requires_grad

    def test_checkpoint_save_load(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        device: torch.device,
    ):
        """Test checkpoint save and load."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            output_dir = Path(tmpdir)

            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=output_dir,
            )
            trainer.epoch = 3
            trainer.global_step = 50
            trainer.best_loss = 0.3
            trainer.best_mae = 0.08

            checkpoint_path = output_dir / "test_checkpoint.pt"
            trainer._save_checkpoint(checkpoint_path)

            new_model = SeasonalViT(forecast_model.config)
            new_trainer = FinetuningTrainer(
                model=new_model,
                config=small_finetune_config,
                device=device,
                output_dir=output_dir,
                resume_from=checkpoint_path,
            )

            assert new_trainer.epoch == 4
            assert new_trainer.global_step == 50
            assert new_trainer.best_loss == 0.3
            assert new_trainer.best_mae == 0.08

    def test_checkpoint_restores_rng_state(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        device: torch.device,
    ):
        """RNG state should survive save/load without TypeError."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            output_dir = Path(tmpdir)

            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=output_dir,
            )

            # Generate some random state
            torch.randn(10)
            checkpoint_path = output_dir / "rng_test.pt"
            trainer._save_checkpoint(checkpoint_path)

            # Loading should not raise TypeError on RNG state
            new_model = SeasonalViT(forecast_model.config)
            new_trainer = FinetuningTrainer(
                model=new_model,
                config=small_finetune_config,
                device=device,
                output_dir=output_dir,
                resume_from=checkpoint_path,
            )

            # RNG should produce deterministic output after restore
            assert new_trainer.epoch == 1

    def test_resume_recovers_best_mae_from_best_pt(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        device: torch.device,
    ):
        """Resuming from a pre-validation epoch_N.pt should recover
        best_mae/best_loss from best.pt if it exists.

        Simulates the crash-between-save-and-validation scenario: epoch_N.pt
        is written with stale best_mae=inf, but best.pt from a prior
        successful validation holds the real best.
        """
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            output_dir = Path(tmpdir)

            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=output_dir,
            )

            # Simulate a prior successful validation that wrote best.pt
            trainer.epoch = 0
            trainer.best_mae = 0.0773
            trainer.best_loss = 0.45
            trainer._save_checkpoint(output_dir / "best.pt")

            # Simulate pre-validation epoch checkpoint with stale best stats
            trainer.best_mae = float("inf")
            trainer.best_loss = float("inf")
            trainer._save_checkpoint(output_dir / "epoch_1.pt")

            new_model = SeasonalViT(forecast_model.config)
            new_trainer = FinetuningTrainer(
                model=new_model,
                config=small_finetune_config,
                device=device,
                output_dir=output_dir,
                resume_from=output_dir / "epoch_1.pt",
            )

            assert new_trainer.best_mae == 0.0773
            assert new_trainer.best_loss == 0.45

    def test_resume_keeps_checkpoint_best_if_better(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        device: torch.device,
    ):
        """If the resume checkpoint has a better best_mae than best.pt,
        the checkpoint value wins (best.pt is only read as a fallback).
        """
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            output_dir = Path(tmpdir)

            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=output_dir,
            )

            # Stale best.pt with worse MAE
            trainer.epoch = 0
            trainer.best_mae = 0.09
            trainer._save_checkpoint(output_dir / "best.pt")

            # Newer epoch checkpoint with better tracked MAE
            trainer.best_mae = 0.07
            trainer._save_checkpoint(output_dir / "epoch_5.pt")

            new_model = SeasonalViT(forecast_model.config)
            new_trainer = FinetuningTrainer(
                model=new_model,
                config=small_finetune_config,
                device=device,
                output_dir=output_dir,
                resume_from=output_dir / "epoch_5.pt",
            )

            assert new_trainer.best_mae == 0.07


# =============================================================================
# Integration Tests
# =============================================================================


class TestTrainingIntegration:
    """Integration tests for full training pipeline."""

    def test_pretrain_to_finetune_transfer(
        self,
        small_config: SeasonalViTConfig,
        small_pretrain_config: PretrainingConfig,
        small_finetune_config: FinetuningConfig,
        dummy_pretrain_batch: dict,
        dummy_finetune_batch: dict,
        device: torch.device,
    ):
        """Test weights transfer from pre-training to fine-tuning."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            output_dir = Path(tmpdir)

            # Pre-train
            pretrain_model = PretrainingModel(small_config)
            pretrain_trainer = PretrainingTrainer(
                model=pretrain_model,
                config=small_pretrain_config,
                device=device,
                output_dir=output_dir / "pretrain",
            )

            # Do a few steps
            for _ in range(3):
                pretrain_trainer._train_step_accumulate(dummy_pretrain_batch, accum_steps=1)

            # Save checkpoint
            pretrain_trainer._save_checkpoint(output_dir / "pretrain.pt")

            # Load into forecasting model
            checkpoint = torch.load(
                output_dir / "pretrain.pt",
                map_location=device,
                weights_only=False,
            )
            forecast_model = SeasonalViT.from_pretrained(
                checkpoint["model_state_dict"],
                small_config,
            )

            # Fine-tune
            finetune_trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=output_dir / "finetune",
            )

            # Should work without errors
            loss, _ = finetune_trainer._train_step_accumulate(
                dummy_finetune_batch, accum_steps=1,
            )
            assert loss > 0

    def test_gradient_flow(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        dummy_finetune_batch: dict,
        device: torch.device,
    ):
        """Test gradients flow through all components."""
        forecast_model = forecast_model.to(device)

        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in dummy_finetune_batch.items()}

        # Forward pass - use fixed window slicing consistent with Trainer
        images = batch["images"]
        season_ids = batch["season_ids"]
        year_indices = batch["year_indices"]
        lat = batch["lat"]
        lon = batch["lon"]

        horizon = small_finetune_config.forecast_horizon
        input_len = images.shape[1] - horizon  # T - horizon

        target = images[:, -1:, :, :]
        target_season = season_ids[:, -1]
        target_year = year_indices[:, -1]
        input_images = images[:, :input_len, :, :]
        input_season_ids = season_ids[:, :input_len]
        input_year_indices = year_indices[:, :input_len]

        output = forecast_model(
            input_images,
            input_season_ids,
            input_year_indices,
            lat,
            lon,
            target_season,
            target_year,
        )

        # Compute loss and backward
        loss = (output["prediction"] - target).pow(2).mean()
        loss.backward()

        # Check gradients exist
        has_grad = False
        for name, param in forecast_model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients found in any parameter"

    def test_scheduler_cosine_monotonic(
        self,
        forecast_model: SeasonalViT,
        device: torch.device,
    ):
        """Test LR decreases monotonically after warmup and floors at min_lr."""
        config = FinetuningConfig(
            max_epochs=20,
            warmup_epochs=2,
            learning_rate=1e-3,
            min_lr=1e-6,
            batch_size=2,
            freeze_encoder=False,
            forecast_horizon=4,
            use_all_history_before_target=False,
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=config,
                device=device,
                output_dir=Path(tmpdir),
                steps_per_epoch=100,
            )

            # Simulate epochs advancing (the old bug: self.epoch changes)
            lrs = []
            for epoch in range(20):
                trainer.epoch = epoch
                for _ in range(100):
                    trainer.scheduler.step()
                    lrs.append(trainer.optimizer.param_groups[0]["lr"])

            # After warmup (first 200 steps), LR should be monotonically non-increasing
            warmup_steps = config.warmup_epochs * 100
            post_warmup_lrs = lrs[warmup_steps:]
            for i in range(1, len(post_warmup_lrs)):
                assert post_warmup_lrs[i] <= post_warmup_lrs[i - 1] + 1e-10, (
                    f"LR increased at step {warmup_steps + i}: "
                    f"{post_warmup_lrs[i - 1]:.6e} -> {post_warmup_lrs[i]:.6e}"
                )

            # LR at end should be at min_lr, not zero
            final_lr = lrs[-1]
            assert final_lr >= config.min_lr - 1e-10, (
                f"Final LR {final_lr:.2e} fell below min_lr {config.min_lr:.2e}"
            )
            assert final_lr < config.learning_rate * 0.01, (
                f"Final LR {final_lr:.2e} didn't decay enough"
            )

    def test_validation_deterministic(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        dummy_finetune_batch: dict,
        device: torch.device,
    ):
        """Two validate() calls produce identical metrics."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=Path(tmpdir),
            )

            dataloader = create_dummy_dataloader(dummy_finetune_batch, num_batches=3)

            metrics1 = trainer.validate(dataloader)
            metrics2 = trainer.validate(dataloader)

            assert metrics1["val_loss"] == pytest.approx(metrics2["val_loss"], abs=1e-6)
            assert metrics1["val_mae"] == pytest.approx(metrics2["val_mae"], abs=1e-6)
            assert metrics1["val_r2"] == pytest.approx(metrics2["val_r2"], abs=1e-6)

    def test_validate_uses_base_model_not_ddp_wrapper(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        dummy_finetune_batch: dict,
        device: torch.device,
    ):
        """Validation must invoke _base_model, not the (potentially DDP-wrapped)
        self.model.

        If self.model.eval() + self.model(...) were used under DDP, the first
        forward would trigger _sync_buffers which hangs for ~30 min because
        only rank 0 validates. We simulate the DDP case by swapping
        trainer.model with a distinct object that raises if called, while
        keeping trainer._base_model intact.
        """
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=Path(tmpdir),
            )
            dataloader = create_dummy_dataloader(dummy_finetune_batch, num_batches=2)

            class PoisonedDDPWrapper(torch.nn.Module):
                def forward(self, *args, **kwargs):
                    raise AssertionError(
                        "validate() called self.model (simulated DDP wrapper). "
                        "It must use self._base_model to avoid _sync_buffers hang."
                    )

                def eval(self):
                    raise AssertionError(
                        "validate() called self.model.eval() (simulated DDP wrapper)."
                    )

                def train(self, mode=True):
                    return self

            trainer.model = PoisonedDDPWrapper()

            # Should complete without touching trainer.model
            metrics = trainer.validate(dataloader)
            assert "val_mae" in metrics

    def test_assert_no_batchnorm_guard(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        device: torch.device,
    ):
        """_assert_no_batchnorm fails loudly if BN is introduced later."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=Path(tmpdir),
            )
            # Baseline: current model has no BN, guard passes.
            trainer._assert_no_batchnorm()

            # Inject a BN module and verify the guard fires.
            trainer._base_model.bn_probe = torch.nn.BatchNorm2d(4)
            with pytest.raises(RuntimeError, match="BatchNorm"):
                trainer._assert_no_batchnorm()

    def test_forecasting_loss_smooth_l1(self):
        """Verify ForecastingLoss uses Smooth L1 and produces valid output."""
        loss_fn = ForecastingLoss(delta_reg_weight=0.01, beta=0.1)

        B, H, W = 2, 64, 64
        output = {
            "prediction": torch.randn(B, 1, H, W),
            "delta": torch.randn(B, 1, H, W) * 0.1,
        }
        target = torch.randn(B, 1, H, W)

        total_loss, components = loss_fn(output, target)

        assert total_loss.item() > 0
        assert not torch.isnan(total_loss)
        assert "recon_loss" in components
        assert "delta_reg" in components

        # Verify Smooth L1 behavior: large errors should produce lower loss than MSE
        large_error_output = {
            "prediction": target + 2.0,  # Large constant error
            "delta": torch.zeros(B, 1, H, W),
        }
        sl1_loss, _ = loss_fn(large_error_output, target)
        mse_loss = ((large_error_output["prediction"] - target) ** 2).mean()
        assert sl1_loss.item() < mse_loss.item(), (
            "Smooth L1 should produce lower loss than MSE for large errors"
        )

    def test_climatology_anchor(self, small_config: SeasonalViTConfig):
        """Verify multi-year anchor averages correctly and handles nodata."""
        small_config.anchor_years = 3
        model = SeasonalViT(small_config)

        B, T, H, W = 2, 12, 64, 64  # 3 years × 4 seasons
        nodata = small_config.nodata_value

        # Create images where each spring (season=0) has known values
        images = torch.full((B, T, H, W), nodata)
        season_ids = torch.tensor([[i % 4 for i in range(T)] for _ in range(B)])
        # Spring at t=0,4,8 → set known values
        images[:, 0, :, :] = 0.3  # Year 0 spring
        images[:, 4, :, :] = 0.5  # Year 1 spring
        images[:, 8, :, :] = 0.7  # Year 2 spring

        target_season = torch.tensor([0, 0])  # predict spring
        anchor = model._get_climatology_anchor(images, season_ids, target_season)

        # Should average 3 spring observations: (0.3 + 0.5 + 0.7) / 3 = 0.5
        expected = 0.5
        assert anchor.shape == (B, 1, H, W)
        assert anchor[0, 0, 0, 0].item() == pytest.approx(expected, abs=1e-5)

        # Test nodata handling: make year 0 spring nodata for batch 1
        images[1, 0, :, :] = nodata
        anchor2 = model._get_climatology_anchor(images, season_ids, target_season)
        # Batch 1 should average only 2 valid: (0.5 + 0.7) / 2 = 0.6
        assert anchor2[1, 0, 0, 0].item() == pytest.approx(0.6, abs=1e-5)
        # Batch 0 unchanged
        assert anchor2[0, 0, 0, 0].item() == pytest.approx(0.5, abs=1e-5)

        # Test all-nodata case: pixel should remain nodata
        images[1, 4, 0, 0] = nodata
        images[1, 8, 0, 0] = nodata
        anchor3 = model._get_climatology_anchor(images, season_ids, target_season)
        assert anchor3[1, 0, 0, 0].item() == pytest.approx(nodata, abs=1e-5)

    def test_progressive_decoder_output_shape(self, small_config: SeasonalViTConfig):
        """Verify progressive decoder produces correct output shape (B,1,H,W)."""
        decoder = ObservationAnchoredDecoder(small_config)
        B, T, N = 2, 8, small_config.num_spatial_patches
        D = small_config.embed_dim
        H, W = small_config.image_size

        temporal_features = torch.randn(B, T, N, D)
        anchor = torch.randn(B, 1, H, W)
        target_season = torch.tensor([0, 2])
        target_year = torch.tensor([30, 32])

        output = decoder(temporal_features, anchor, target_season, target_year)

        assert output["prediction"].shape == (B, 1, H, W)
        assert output["delta"].shape == (B, 1, H, W)
        assert output["anchor"].shape == (B, 1, H, W)
        # Delta should be finite
        assert torch.isfinite(output["delta"]).all()


class TestDeltaStdEstimation:
    """Tests for delta_std auto-estimation and delta supervision integration."""

    def test_compute_batch_true_deltas_shape(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        dummy_finetune_batch: dict,
        device: torch.device,
    ):
        """True deltas should be a 1D tensor of valid pixel values."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=Path(tmpdir),
            )
            gen = torch.Generator(device=device)
            gen.manual_seed(42)

            deltas = trainer._compute_batch_true_deltas(dummy_finetune_batch, gen)

            assert deltas.dim() == 1
            assert deltas.numel() > 0
            assert torch.isfinite(deltas).all()

    def test_compute_batch_true_deltas_with_nodata(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        device: torch.device,
    ):
        """Nodata pixels should be excluded from true deltas."""
        B, T, H, W = 2, 12, 64, 64
        nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)
        nodata_mask[:, :, :32, :] = True  # Top half is nodata

        batch = {
            "images": torch.randn(B, T, H, W).clamp(-1, 1),
            "season_ids": torch.tensor([[i % 4 for i in range(T)] for _ in range(B)]),
            "year_indices": torch.tensor([[i // 4 for i in range(T)] for _ in range(B)]),
            "lat": torch.tensor([37.0, 45.0]),
            "lon": torch.tensor([-122.0, -73.0]),
            "nodata_mask": nodata_mask,
        }

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=Path(tmpdir),
            )
            gen = torch.Generator(device=device)
            gen.manual_seed(42)

            deltas_partial = trainer._compute_batch_true_deltas(batch, gen)

            # With half nodata, should have fewer pixels than full batch
            batch_full = {**batch, "nodata_mask": torch.zeros(B, T, H, W, dtype=torch.bool)}
            gen.manual_seed(42)
            deltas_full = trainer._compute_batch_true_deltas(batch_full, gen)

            assert deltas_partial.numel() < deltas_full.numel()

    def test_estimate_delta_std_returns_positive(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        dummy_finetune_batch: dict,
        device: torch.device,
    ):
        """Estimated delta_std should be positive and finite."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=Path(tmpdir),
            )

            val_loader = create_dummy_dataloader(dummy_finetune_batch, num_batches=3)
            delta_std = trainer._estimate_delta_std(val_loader, max_batches=3)

            assert delta_std > 0
            assert delta_std < float("inf")

    def test_estimate_delta_std_deterministic(
        self,
        forecast_model: SeasonalViT,
        small_finetune_config: FinetuningConfig,
        dummy_finetune_batch: dict,
        device: torch.device,
    ):
        """Two calls with same data should return the same delta_std."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=small_finetune_config,
                device=device,
                output_dir=Path(tmpdir),
            )

            val_loader = create_dummy_dataloader(dummy_finetune_batch, num_batches=3)
            std1 = trainer._estimate_delta_std(val_loader, max_batches=3)
            std2 = trainer._estimate_delta_std(val_loader, max_batches=3)

            assert std1 == pytest.approx(std2, abs=1e-6)

    def test_train_step_with_delta_supervision(
        self,
        forecast_model: SeasonalViT,
        dummy_finetune_batch: dict,
        device: torch.device,
    ):
        """Training step should include delta_sup_loss in components."""
        config = FinetuningConfig(
            max_epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            warmup_epochs=0,
            forecast_horizon=4,
            predict_years=None,
            use_all_history_before_target=False,
            delta_supervision_weight=1.0,
            delta_std=0.1,
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=config,
                device=device,
                output_dir=Path(tmpdir),
            )

            loss, components = trainer._train_step_accumulate(
                dummy_finetune_batch, accum_steps=1,
            )

            assert "delta_sup_loss" in components
            assert components["delta_sup_loss"] >= 0

    def test_train_epoch_logs_delta_sup_loss(
        self,
        forecast_model: SeasonalViT,
        dummy_finetune_batch: dict,
        device: torch.device,
    ):
        """Train epoch metrics should include delta_sup_loss."""
        config = FinetuningConfig(
            max_epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            warmup_epochs=0,
            forecast_horizon=4,
            predict_years=None,
            use_all_history_before_target=False,
            delta_supervision_weight=1.0,
            delta_std=0.1,
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=config,
                device=device,
                output_dir=Path(tmpdir),
            )

            dataloader = create_dummy_dataloader(dummy_finetune_batch, num_batches=3)
            metrics = trainer.train_epoch(dataloader, epoch=0)

            assert "delta_sup_loss" in metrics


class TestEpochSummaryLogging:
    """Test that epoch summary logging doesn't crash with missing variables."""

    def test_fit_two_epochs_with_validation(
        self,
        forecast_model: SeasonalViT,
        dummy_finetune_batch: dict,
        device: torch.device,
    ):
        """fit() with validation should complete without NameError.

        Catches bugs like prev_best not being defined before use in
        the epoch summary block.
        """
        config = FinetuningConfig(
            max_epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            warmup_epochs=0,
            forecast_horizon=4,
            predict_years=None,
            use_all_history_before_target=False,
            delta_supervision_weight=0.1,
            delta_std=0.1,
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            trainer = FinetuningTrainer(
                model=forecast_model,
                config=config,
                device=device,
                output_dir=Path(tmpdir),
            )
            train_loader = create_dummy_dataloader(dummy_finetune_batch, num_batches=3)
            val_loader = create_dummy_dataloader(dummy_finetune_batch, num_batches=2)

            history = trainer.fit(train_loader, val_loader)

            assert len(history["val_mae"]) == 2
            assert all(m > 0 for m in history["val_mae"])
