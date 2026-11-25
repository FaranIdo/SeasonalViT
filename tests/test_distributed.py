"""Tests for distributed training utilities.

Tests the helper functions in train/distributed.py using mocking
since local development doesn't have multi-GPU setup.
"""

import pytest
from unittest.mock import patch, MagicMock
import torch


class TestDistributedUtilitiesNonDistributed:
    """Test distributed helpers when torch.distributed is NOT initialized."""

    def test_is_main_process_non_distributed(self):
        """Returns True when distributed is not initialized."""
        from train.distributed import is_main_process
        # When not in distributed mode, should always return True
        with patch('torch.distributed.is_initialized', return_value=False):
            assert is_main_process() is True

    def test_get_rank_non_distributed(self):
        """Returns 0 when distributed is not initialized."""
        from train.distributed import get_rank
        with patch('torch.distributed.is_initialized', return_value=False):
            assert get_rank() == 0

    def test_get_world_size_non_distributed(self):
        """Returns 1 when distributed is not initialized."""
        from train.distributed import get_world_size
        with patch('torch.distributed.is_initialized', return_value=False):
            assert get_world_size() == 1

    def test_get_local_rank_no_env_var(self):
        """Returns 0 when LOCAL_RANK env var is not set."""
        from train.distributed import get_local_rank
        with patch.dict('os.environ', {}, clear=True):
            assert get_local_rank() == 0

    def test_get_local_rank_with_env_var(self):
        """Returns the value from LOCAL_RANK env var."""
        from train.distributed import get_local_rank
        with patch.dict('os.environ', {'LOCAL_RANK': '1'}):
            assert get_local_rank() == 1

    def test_is_distributed_non_distributed(self):
        """Returns False when distributed is not initialized."""
        from train.distributed import is_distributed
        with patch('torch.distributed.is_initialized', return_value=False):
            assert is_distributed() is False

    def test_barrier_non_distributed(self):
        """barrier() is a no-op when not distributed."""
        from train.distributed import barrier
        # Should not raise and not call dist.barrier
        with patch('torch.distributed.is_initialized', return_value=False):
            with patch('torch.distributed.barrier') as mock_barrier:
                barrier()
                mock_barrier.assert_not_called()


class TestDistributedUtilitiesWithMocking:
    """Test distributed behavior with mocked torch.distributed."""

    def test_is_main_process_rank0(self):
        """Returns True when rank is 0."""
        from train.distributed import is_main_process
        with patch('torch.distributed.is_initialized', return_value=True):
            with patch('torch.distributed.get_rank', return_value=0):
                assert is_main_process() is True

    def test_is_main_process_rank1(self):
        """Returns False when rank is not 0."""
        from train.distributed import is_main_process
        with patch('torch.distributed.is_initialized', return_value=True):
            with patch('torch.distributed.get_rank', return_value=1):
                assert is_main_process() is False

    def test_get_rank_distributed(self):
        """Returns the actual rank when distributed."""
        from train.distributed import get_rank
        with patch('torch.distributed.is_initialized', return_value=True):
            with patch('torch.distributed.get_rank', return_value=2):
                assert get_rank() == 2

    def test_get_world_size_distributed(self):
        """Returns the actual world size when distributed."""
        from train.distributed import get_world_size
        with patch('torch.distributed.is_initialized', return_value=True):
            with patch('torch.distributed.get_world_size', return_value=4):
                assert get_world_size() == 4

    def test_is_distributed_when_initialized(self):
        """Returns True when distributed is initialized."""
        from train.distributed import is_distributed
        with patch('torch.distributed.is_initialized', return_value=True):
            assert is_distributed() is True

    def test_barrier_calls_dist_barrier(self):
        """barrier() calls dist.barrier when distributed."""
        from train.distributed import barrier
        with patch('torch.distributed.is_initialized', return_value=True):
            with patch('torch.distributed.barrier') as mock_barrier:
                barrier()
                mock_barrier.assert_called_once()


class TestDataLoaderWithSampler:
    """Test DistributedSampler integration with dataloaders."""

    def test_create_dataloader_with_sampler(self):
        """Verify sampler parameter disables shuffle."""
        from datasets.sharded_dataset import create_dataloader

        # Create a mock dataset
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=100)

        # Create a mock sampler
        sampler = MagicMock()

        # Create dataloader with sampler
        loader = create_dataloader(
            dataset,
            batch_size=8,
            shuffle=True,  # This should be ignored when sampler is provided
            sampler=sampler,
        )

        # Verify sampler is used
        assert loader.sampler is sampler
        # When sampler is provided, shuffle should be disabled
        # (DataLoader sets shuffle=False internally when sampler is provided)

    def test_create_dataloader_without_sampler(self):
        """Verify shuffle works when no sampler is provided."""
        from datasets.sharded_dataset import create_dataloader

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=100)

        loader = create_dataloader(
            dataset,
            batch_size=8,
            shuffle=True,
            sampler=None,
        )

        # Without sampler, shuffle should work normally
        assert loader.sampler is not None  # RandomSampler or similar


class TestPretrainingTrainerDistributed:
    """Test PretrainingTrainer DDP integration."""

    def test_trainer_init_non_distributed(self):
        """Trainer initializes correctly in non-distributed mode."""
        from train.pretrain import PretrainingTrainer
        from model.pretrain_model import PretrainingModel
        from model.config import PretrainingConfig, SeasonalViTConfig
        import tempfile

        # Create minimal config
        config = PretrainingConfig(max_epochs=1)
        model_config = SeasonalViTConfig(swin_pretrained=False)

        # Create a minimal model (will be slow but works)
        model = PretrainingModel(model_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trainer in non-distributed mode
            trainer = PretrainingTrainer(
                model=model,
                config=config,
                device=torch.device("cpu"),
                output_dir=tmpdir,
                distributed=False,
                rank=0,
            )

            # Verify model is not wrapped in DDP
            assert not isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel)
            # Verify writer is created (main process)
            assert trainer.writer is not None

    def test_save_checkpoint_skipped_on_non_main(self):
        """Verify checkpoint saving is skipped on rank != 0.

        Tests _save_checkpoint directly with a non-distributed trainer
        but mocked is_main_process to simulate non-main rank.
        """
        from train.pretrain import PretrainingTrainer
        from model.pretrain_model import PretrainingModel
        from model.config import PretrainingConfig, SeasonalViTConfig
        import tempfile
        import os

        config = PretrainingConfig(max_epochs=1)
        model_config = SeasonalViTConfig(swin_pretrained=False)
        model = PretrainingModel(model_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create non-distributed trainer first
            trainer = PretrainingTrainer(
                model=model,
                config=config,
                device=torch.device("cpu"),
                output_dir=tmpdir,
                distributed=False,
                rank=0,
            )

            # Now mock is_main_process to return False and test save
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
            with patch('train.pretrain.is_main_process', return_value=False):
                trainer._save_checkpoint(checkpoint_path)

            # File should NOT be created since we're not main process
            assert not os.path.exists(checkpoint_path)

    def test_save_checkpoint_works_on_main(self):
        """Verify checkpoint saved on rank 0."""
        from train.pretrain import PretrainingTrainer
        from model.pretrain_model import PretrainingModel
        from model.config import PretrainingConfig, SeasonalViTConfig
        import tempfile
        import os

        config = PretrainingConfig(max_epochs=1)
        model_config = SeasonalViTConfig(swin_pretrained=False)
        model = PretrainingModel(model_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trainer in non-distributed mode (simulates rank=0 behavior)
            trainer = PretrainingTrainer(
                model=model,
                config=config,
                device=torch.device("cpu"),
                output_dir=tmpdir,
                distributed=False,
                rank=0,
            )

            # Try to save checkpoint (is_main_process returns True by default)
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
            trainer._save_checkpoint(checkpoint_path)

            # File SHOULD be created since we're main process
            assert os.path.exists(checkpoint_path)

            # Verify checkpoint contents
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "epoch" in checkpoint


class TestFinetuningTrainerDistributed:
    """Test FinetuningTrainer DDP integration."""

    def test_writer_not_created_on_non_main(self):
        """Verify TensorBoard writer is not created on non-main process.

        Tests by creating trainer with mocked is_main_process.
        """
        from train.finetune import FinetuningTrainer
        from model.seasonal_vit import SeasonalViT
        from model.config import FinetuningConfig, SeasonalViTConfig
        import tempfile

        config = FinetuningConfig(max_epochs=1)
        model_config = SeasonalViTConfig(swin_pretrained=False)
        model = SeasonalViT(model_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock is_main_process during trainer creation
            with patch('train.finetune.is_main_process', return_value=False):
                trainer = FinetuningTrainer(
                    model=model,
                    config=config,
                    device=torch.device("cpu"),
                    output_dir=tmpdir,
                    distributed=False,  # Use non-distributed to avoid DDP init
                    rank=1,
                )

            # Writer should be None on non-main process
            assert trainer.writer is None

    def test_writer_created_on_main(self):
        """Verify TensorBoard writer is created on main process."""
        from train.finetune import FinetuningTrainer
        from model.seasonal_vit import SeasonalViT
        from model.config import FinetuningConfig, SeasonalViTConfig
        import tempfile

        config = FinetuningConfig(max_epochs=1)
        model_config = SeasonalViTConfig(swin_pretrained=False)
        model = SeasonalViT(model_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FinetuningTrainer(
                model=model,
                config=config,
                device=torch.device("cpu"),
                output_dir=tmpdir,
                distributed=False,
                rank=0,
            )

            # Writer should exist on main process
            assert trainer.writer is not None


class TestReduceTensor:
    """Test tensor reduction utilities."""

    def test_reduce_tensor_non_distributed(self):
        """reduce_tensor returns unchanged tensor when not distributed."""
        from train.distributed import reduce_tensor

        tensor = torch.tensor([1.0, 2.0, 3.0])
        with patch('torch.distributed.is_initialized', return_value=False):
            result = reduce_tensor(tensor)
            assert torch.equal(result, tensor)

    def test_all_gather_tensors_non_distributed(self):
        """all_gather_tensors returns single-element list when not distributed."""
        from train.distributed import all_gather_tensors

        tensor = torch.tensor([1.0, 2.0, 3.0])
        with patch('torch.distributed.is_initialized', return_value=False):
            result = all_gather_tensors(tensor)
            assert len(result) == 1
            assert torch.equal(result[0], tensor)
