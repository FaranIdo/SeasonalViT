"""Tests for InMemoryNDVIDataset and related utilities."""

import json

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from safetensors.torch import save_file


def _save_test_shard(shard_data: dict, path: Path) -> None:
    """Save test shard data in safetensors format with JSON sidecar."""
    tensors = {
        "images": shard_data["images"],
        "nodata_masks": shard_data["nodata_masks"].to(torch.uint8),
        "lats": shard_data["lats"],
        "lons": shard_data["lons"],
    }
    save_file(tensors, str(path))
    json_path = path.with_suffix("").with_suffix(".files.json")
    with open(json_path, "w") as f:
        json.dump(shard_data["files"], f)


class TestInMemoryNDVIDataset:
    """Tests for InMemoryNDVIDataset."""

    @pytest.fixture(scope="class")
    def shard_dir(self, tmp_path_factory):
        """Create temporary shard directory with test shards."""
        shard_path = tmp_path_factory.mktemp("shards_inmemory")

        # Create 3 small test shards (10 samples each)
        for i in range(3):
            shard_data = {
                "images": torch.randn(10, 140, 256, 256, dtype=torch.float16),
                "nodata_masks": torch.zeros(10, 140, 256, 256, dtype=torch.bool),
                "lats": torch.randn(10),
                "lons": torch.randn(10),
                "files": [f"sample_{i}_{j}.tif" for j in range(10)],
            }
            _save_test_shard(shard_data, shard_path / f"shard_{i:03d}.safetensors")

        return shard_path

    def test_init_and_properties(self, shard_dir):
        """Test dataset initialization and basic properties."""
        from datasets.sharded_dataset import InMemoryNDVIDataset

        dataset = InMemoryNDVIDataset(shard_dir)

        assert len(dataset) == 30  # 3 shards * 10 samples
        assert dataset.images.shape == (30, 140, 256, 256)
        assert dataset.nodata_masks.shape == (30, 140, 256, 256)
        assert dataset.lats.shape == (30,)
        assert dataset.lons.shape == (30,)

    def test_getitem_returns_correct_structure(self, shard_dir):
        """Test that __getitem__ returns correct sample structure."""
        from datasets.sharded_dataset import InMemoryNDVIDataset

        dataset = InMemoryNDVIDataset(shard_dir)
        sample = dataset[0]

        assert "images" in sample
        assert "season_ids" in sample
        assert "year_indices" in sample
        assert "lat" in sample
        assert "lon" in sample
        assert "nodata_mask" in sample

        assert sample["images"].shape == (140, 256, 256)
        assert sample["season_ids"].shape == (140,)
        assert sample["year_indices"].shape == (140,)

    def test_season_ids_correct(self, shard_dir):
        """Test that season IDs cycle correctly (0, 1, 2, 3, 0, 1, ...)."""
        from datasets.sharded_dataset import InMemoryNDVIDataset

        dataset = InMemoryNDVIDataset(shard_dir)
        sample = dataset[0]

        # First 8 seasons
        expected = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
        assert torch.equal(sample["season_ids"][:8], expected)

        # All values should be 0-3
        assert sample["season_ids"].min() == 0
        assert sample["season_ids"].max() == 3

    def test_year_indices_correct(self, shard_dir):
        """Test that year indices increase correctly."""
        from datasets.sharded_dataset import InMemoryNDVIDataset

        dataset = InMemoryNDVIDataset(shard_dir)
        sample = dataset[0]

        # First 8 timesteps should be years 0, 0, 0, 0, 1, 1, 1, 1
        expected = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        assert torch.equal(sample["year_indices"][:8], expected)

        # Year range
        assert sample["year_indices"].min() == 0
        assert sample["year_indices"].max() == 34

    def test_max_samples_limits_loading(self, shard_dir):
        """Test that max_samples limits the number of samples loaded."""
        from datasets.sharded_dataset import InMemoryNDVIDataset

        dataset = InMemoryNDVIDataset(shard_dir, max_samples=15)

        assert len(dataset) == 15
        assert dataset.images.shape[0] == 15

    def test_shard_range_limits_loading(self, shard_dir):
        """Test that shard_range limits which shards are loaded."""
        from datasets.sharded_dataset import InMemoryNDVIDataset

        # Load only first 2 shards (0, 1)
        dataset = InMemoryNDVIDataset(shard_dir, shard_range=(0, 2))

        assert len(dataset) == 20  # 2 shards * 10 samples
        assert dataset.images.shape[0] == 20

    def test_share_memory(self, shard_dir):
        """Test share_memory_() method."""
        from datasets.sharded_dataset import InMemoryNDVIDataset

        dataset = InMemoryNDVIDataset(shard_dir)
        result = dataset.share_memory_()

        # Should return self for chaining
        assert result is dataset

        # Tensors should be in shared memory
        assert dataset.images.is_shared()
        assert dataset.nodata_masks.is_shared()
        assert dataset.lats.is_shared()
        assert dataset.lons.is_shared()

    def test_no_shards_raises_error(self, tmp_path):
        """Test that empty directory raises ValueError."""
        from datasets.sharded_dataset import InMemoryNDVIDataset

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No shard_.*files found"):
            InMemoryNDVIDataset(empty_dir)


class TestCreateDataloader:
    """Tests for create_dataloader function."""

    @pytest.fixture(scope="class")
    def shard_dir(self, tmp_path_factory):
        """Create temporary shard directory with test shards."""
        shard_path = tmp_path_factory.mktemp("shards_dataloader")

        # Create 2 small test shards
        for i in range(2):
            shard_data = {
                "images": torch.randn(5, 140, 256, 256, dtype=torch.float16),
                "nodata_masks": torch.zeros(5, 140, 256, 256, dtype=torch.bool),
                "lats": torch.randn(5),
                "lons": torch.randn(5),
                "files": [f"sample_{i}_{j}.tif" for j in range(5)],
            }
            _save_test_shard(shard_data, shard_path / f"shard_{i:03d}.safetensors")

        return shard_path

    def test_create_dataloader(self, shard_dir):
        """Test creating a DataLoader."""
        from datasets.sharded_dataset import InMemoryNDVIDataset, create_dataloader

        dataset = InMemoryNDVIDataset(shard_dir)
        dataloader = create_dataloader(dataset, batch_size=4, num_workers=0)

        batch = next(iter(dataloader))

        assert batch["images"].shape == (4, 140, 256, 256)
        assert batch["season_ids"].shape == (4, 140)
        assert batch["year_indices"].shape == (4, 140)
        assert batch["lat"].shape == (4,)
        assert batch["lon"].shape == (4,)

    def test_sampler_disables_shuffle(self, shard_dir):
        """Test that providing a sampler disables shuffle."""
        from datasets.sharded_dataset import InMemoryNDVIDataset, create_dataloader
        from torch.utils.data import SequentialSampler

        dataset = InMemoryNDVIDataset(shard_dir)
        sampler = SequentialSampler(dataset)

        # This should not raise even with shuffle=True
        dataloader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=True,
            sampler=sampler,
            num_workers=0,
        )

        # Verify we can iterate
        batch = next(iter(dataloader))
        assert batch["images"].shape[0] == 4


class TestTransforms:
    """Tests for data transforms."""

    def test_compose(self):
        """Test Compose transform."""
        from datasets.transforms import Compose

        def add_one(sample):
            sample["images"] = sample["images"] + 1
            return sample

        def multiply_two(sample):
            sample["images"] = sample["images"] * 2
            return sample

        transform = Compose([add_one, multiply_two])
        sample = {"images": torch.zeros(10, 32, 32)}
        result = transform(sample)

        # (0 + 1) * 2 = 2
        assert torch.allclose(result["images"], torch.ones(10, 32, 32) * 2)

    def test_temporal_dropout(self):
        """Test TemporalDropout transform."""
        from datasets.transforms import TemporalDropout

        transform = TemporalDropout(drop_prob=0.5)
        sample = {"images": torch.ones(100, 32, 32)}

        # Run multiple times to check randomness
        torch.manual_seed(42)
        result = transform(sample)

        # Some timesteps should be zeroed
        zero_timesteps = (result["images"].sum(dim=[1, 2]) == 0).sum()
        assert zero_timesteps > 0
        assert zero_timesteps < 100

    def test_temporal_noise(self):
        """Test TemporalNoise transform."""
        from datasets.transforms import TemporalNoise

        transform = TemporalNoise(noise_std=0.1)
        original = torch.zeros(10, 32, 32)
        sample = {"images": original.clone()}

        result = transform(sample)

        # Values should have changed
        assert not torch.allclose(result["images"], original)

        # Should still be in valid range
        assert result["images"].min() >= -1
        assert result["images"].max() <= 1

    def test_seasonal_jitter(self):
        """Test SeasonalJitter transform."""
        from datasets.transforms import SeasonalJitter

        transform = SeasonalJitter(swap_prob=1.0)

        # Create images with distinct values per timestep
        images = torch.arange(8).float().view(8, 1, 1).expand(8, 32, 32)
        sample = {"images": images.clone()}

        torch.manual_seed(42)
        result = transform(sample)

        # Some values should have been swapped
        # The exact swaps depend on random state
        assert result["images"].shape == images.shape

    def test_random_horizontal_flip(self):
        """Test RandomHorizontalFlip transform."""
        from datasets.transforms import RandomHorizontalFlip

        transform = RandomHorizontalFlip(p=1.0)

        # Create asymmetric image
        images = torch.zeros(2, 4, 4)
        images[:, :, 0] = 1.0

        sample = {"images": images.clone()}
        result = transform(sample)

        # After horizontal flip, column 0 should become column 3
        assert torch.allclose(result["images"][:, :, 3], torch.ones(2, 4))
        assert torch.allclose(result["images"][:, :, 0], torch.zeros(2, 4))

    def test_random_vertical_flip(self):
        """Test RandomVerticalFlip transform."""
        from datasets.transforms import RandomVerticalFlip

        transform = RandomVerticalFlip(p=1.0)

        # Create asymmetric image
        images = torch.zeros(2, 4, 4)
        images[:, 0, :] = 1.0

        sample = {"images": images.clone()}
        result = transform(sample)

        # After vertical flip, row 0 should become row 3
        assert torch.allclose(result["images"][:, 3, :], torch.ones(2, 4))
        assert torch.allclose(result["images"][:, 0, :], torch.zeros(2, 4))

    def test_random_rotation90(self):
        """Test RandomRotation90 transform."""
        from datasets.transforms import RandomRotation90

        transform = RandomRotation90()

        images = torch.randn(2, 4, 4)
        sample = {"images": images.clone()}

        # Just verify it runs and preserves shape
        result = transform(sample)
        assert result["images"].shape == images.shape

    def test_normalize_minmax(self):
        """Test Normalize transform with minmax method."""
        from datasets.transforms import Normalize

        transform = Normalize(method="minmax")

        # NDVI range [-1, 1]
        images = torch.tensor([-1.0, 0.0, 1.0]).view(3, 1, 1)
        sample = {"images": images}
        result = transform(sample)

        # Should map to [0, 1]
        expected = torch.tensor([0.0, 0.5, 1.0]).view(3, 1, 1)
        assert torch.allclose(result["images"], expected)

    def test_get_train_transforms(self):
        """Test get_train_transforms factory function."""
        from datasets.transforms import get_train_transforms

        transform = get_train_transforms(
            temporal_dropout_prob=0.1,
            temporal_noise_std=0.03,
        )

        sample = {"images": torch.randn(10, 32, 32).clamp(-1, 1)}
        result = transform(sample)

        assert result["images"].shape == sample["images"].shape

    def test_get_val_transforms(self):
        """Test get_val_transforms factory function."""
        from datasets.transforms import get_val_transforms

        transform = get_val_transforms(normalize="minmax")

        sample = {"images": torch.randn(10, 32, 32).clamp(-1, 1)}
        result = transform(sample)

        # Should be in [0, 1] range after minmax
        assert result["images"].min() >= 0
        assert result["images"].max() <= 1


class TestSplits:
    """Tests for geographic splitting."""

    def test_create_geographic_splits(self, sample_tif_paths):
        """Test create_geographic_splits function."""
        from datasets.splits import create_geographic_splits

        splits = create_geographic_splits(
            file_paths=sample_tif_paths,
            test_size=0.2,
            val_size=0.2,
            n_blocks=5,
            random_state=42,
        )

        # Check all keys present
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

        # Check indices don't overlap
        all_idx = set()
        for split_name, indices in splits.items():
            for idx in indices:
                assert idx not in all_idx
                all_idx.add(idx)

        # Check all samples accounted for
        assert len(all_idx) == len(sample_tif_paths)

    def test_create_geographic_splits_from_coords(self):
        """Test create_geographic_splits with coordinates."""
        from datasets.splits import create_geographic_splits

        # Create some fake coordinates
        coordinates = [
            (45.0, -122.0),
            (45.1, -122.1),
            (30.0, -90.0),
            (30.1, -90.1),
            (60.0, 10.0),
            (60.1, 10.1),
        ]

        splits = create_geographic_splits(
            coordinates=coordinates,
            test_size=0.2,
            val_size=0.2,
            n_blocks=3,
            random_state=42,
        )

        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 6

    def test_get_split_statistics(self):
        """Test get_split_statistics function."""
        from datasets.splits import create_geographic_splits, get_split_statistics

        coordinates = [
            (45.0, -122.0),
            (45.1, -122.1),
            (30.0, -90.0),
            (30.1, -90.1),
            (60.0, 10.0),
            (60.1, 10.1),
        ]

        splits = create_geographic_splits(
            coordinates=coordinates,
            n_blocks=3,
            random_state=42,
        )

        stats = get_split_statistics(splits, coordinates)

        # Check structure
        for split_name in ["train", "val", "test"]:
            if len(splits[split_name]) > 0:
                assert "n_samples" in stats[split_name]
                assert "lat_min" in stats[split_name]
                assert "lat_max" in stats[split_name]


class TestResearchSplits:
    """Tests for research-grade geographic + temporal splits."""

    @pytest.fixture
    def mock_files(self, tmp_path):
        """Create mock TIF files with coordinate-based names."""
        files = []
        # Create 100 mock files spread across different locations
        np.random.seed(42)
        for i in range(100):
            lat = np.random.uniform(-60, 60)
            lon = np.random.uniform(-180, 180)
            # Format: sample_id_latdXXX_londXXX.tif
            lat_str = f"{int(lat)}d{abs(int((lat % 1) * 10000)):04d}"
            lon_str = f"{int(lon)}d{abs(int((lon % 1) * 10000)):04d}"
            filename = f"{i:06d}_{lat_str}_{lon_str}.tif"
            filepath = tmp_path / filename
            filepath.touch()
            files.append(filepath)
        return files

    def test_creates_all_splits(self, mock_files):
        """Test that all 5 splits are created."""
        from datasets.splits import create_research_splits

        splits = create_research_splits(
            mock_files,
            train_size=50,
            val_size=10,
            ood_t_size=15,
            ood_spatial_size=20,
            n_blocks=10,
        )

        assert "train" in splits
        assert "val_iid" in splits
        assert "ood_t" in splits
        assert "ood_s" in splits
        assert "ood_st" in splits

    def test_no_location_overlap_between_groups(self, mock_files):
        """Group A (train/val/ood-t) and Group B (ood-s/ood-st) don't share locations."""
        from datasets.splits import create_research_splits

        splits = create_research_splits(
            mock_files,
            train_size=50,
            val_size=10,
            ood_t_size=15,
            ood_spatial_size=20,
            n_blocks=10,
        )

        # Get filenames (which encode locations)
        group_a = set(Path(f).name for f in splits["train"]["files"])
        group_a |= set(Path(f).name for f in splits["val_iid"]["files"])
        group_a |= set(Path(f).name for f in splits["ood_t"]["files"])

        group_b = set(Path(f).name for f in splits["ood_s"]["files"])

        # No overlap
        assert len(group_a & group_b) == 0

    def test_val_iid_subset_of_ood_t_locations(self, mock_files):
        """Val-IID locations should be a subset of OOD-t locations."""
        from datasets.splits import create_research_splits

        splits = create_research_splits(
            mock_files,
            train_size=50,
            val_size=10,
            ood_t_size=15,
            ood_spatial_size=20,
            n_blocks=10,
        )

        val_files = set(Path(f).name for f in splits["val_iid"]["files"])
        ood_t_files = set(Path(f).name for f in splits["ood_t"]["files"])

        assert val_files <= ood_t_files

    def test_ood_s_and_ood_st_same_locations(self, mock_files):
        """OOD-s and OOD-st use identical locations."""
        from datasets.splits import create_research_splits

        splits = create_research_splits(
            mock_files,
            train_size=50,
            val_size=10,
            ood_t_size=15,
            ood_spatial_size=20,
            n_blocks=10,
        )

        ood_s_files = set(Path(f).name for f in splits["ood_s"]["files"])
        ood_st_files = set(Path(f).name for f in splits["ood_st"]["files"])

        assert ood_s_files == ood_st_files

    def test_year_configurations(self, mock_files):
        """Verify input/predict year configs are correct."""
        from datasets.splits import create_research_splits

        splits = create_research_splits(
            mock_files,
            train_size=50,
            val_size=10,
            ood_t_size=15,
            ood_spatial_size=20,
            n_blocks=10,
        )

        # Train: predict 2022 from 1990-2021
        assert splits["train"]["input_years"] == (1990, 2021)
        assert splits["train"]["predict_years"] == (2022, 2022)

        # Val-IID: same as train
        assert splits["val_iid"]["input_years"] == (1990, 2021)
        assert splits["val_iid"]["predict_years"] == (2022, 2022)

        # OOD-t: predict 2023-2024 from 1990-2022
        assert splits["ood_t"]["input_years"] == (1990, 2022)
        assert splits["ood_t"]["predict_years"] == (2023, 2024)

        # OOD-s: same years as train
        assert splits["ood_s"]["input_years"] == (1990, 2021)
        assert splits["ood_s"]["predict_years"] == (2022, 2022)

        # OOD-st: predict 2023-2024 from 1990-2022
        assert splits["ood_st"]["input_years"] == (1990, 2022)
        assert splits["ood_st"]["predict_years"] == (2023, 2024)

    def test_deterministic_with_same_seed(self, mock_files):
        """Same random_state produces identical splits."""
        from datasets.splits import create_research_splits

        splits1 = create_research_splits(
            mock_files,
            train_size=50,
            val_size=10,
            ood_t_size=15,
            ood_spatial_size=20,
            n_blocks=10,
            random_state=42,
        )
        splits2 = create_research_splits(
            mock_files,
            train_size=50,
            val_size=10,
            ood_t_size=15,
            ood_spatial_size=20,
            n_blocks=10,
            random_state=42,
        )

        for split_name in splits1:
            assert splits1[split_name]["files"] == splits2[split_name]["files"]

    def test_different_seeds_produce_different_splits(self, mock_files):
        """Different random_state produces different splits."""
        from datasets.splits import create_research_splits

        splits1 = create_research_splits(
            mock_files,
            train_size=50,
            val_size=10,
            ood_t_size=15,
            ood_spatial_size=20,
            n_blocks=10,
            random_state=42,
        )
        splits2 = create_research_splits(
            mock_files,
            train_size=50,
            val_size=10,
            ood_t_size=15,
            ood_spatial_size=20,
            n_blocks=10,
            random_state=123,
        )

        # At least one split should differ
        any_different = False
        for split_name in splits1:
            if splits1[split_name]["files"] != splits2[split_name]["files"]:
                any_different = True
                break
        assert any_different

    def test_save_splits_to_json(self, mock_files, tmp_path):
        """Splits are correctly saved to JSON manifest."""
        from datasets.splits import create_research_splits, save_splits_to_json

        splits = create_research_splits(
            mock_files,
            train_size=50,
            val_size=10,
            ood_t_size=15,
            ood_spatial_size=20,
            n_blocks=10,
        )

        json_path = tmp_path / "splits.json"
        save_splits_to_json(splits, json_path)

        # Check JSON exists
        assert json_path.exists()

        # Check JSON content
        import json
        with open(json_path) as f:
            manifest = json.load(f)

        assert "source_dir" in manifest
        assert "splits" in manifest
        for split_name in ["train", "val_iid", "ood_t", "ood_s", "ood_st"]:
            assert split_name in manifest["splits"]
            assert "files" in manifest["splits"][split_name]
            assert "input_years" in manifest["splits"][split_name]
            assert "predict_years" in manifest["splits"][split_name]

    def test_load_splits_from_json(self, mock_files, tmp_path):
        """Test loading splits back from JSON manifest."""
        from datasets.splits import (
            create_research_splits,
            save_splits_to_json,
            load_splits_from_json,
        )

        splits = create_research_splits(
            mock_files,
            train_size=50,
            val_size=10,
            ood_t_size=15,
            ood_spatial_size=20,
            n_blocks=10,
        )

        json_path = tmp_path / "splits.json"
        save_splits_to_json(splits, json_path)

        # Load back
        loaded = load_splits_from_json(json_path)

        # Check structure
        assert set(loaded.keys()) == {"train", "val_iid", "ood_t", "ood_s", "ood_st"}

        # Check counts match
        for split_name in splits:
            assert len(loaded[split_name]["files"]) == len(splits[split_name]["files"])

        # Check year configs preserved
        assert loaded["train"]["input_years"] == (1990, 2021)
        assert loaded["ood_t"]["predict_years"] == (2023, 2024)

    def test_load_splits_from_json_with_source_override(self, mock_files, tmp_path):
        """Test loading splits with custom source directory."""
        from datasets.splits import (
            create_research_splits,
            save_splits_to_json,
            load_splits_from_json,
        )

        splits = create_research_splits(
            mock_files,
            train_size=50,
            val_size=10,
            ood_t_size=15,
            ood_spatial_size=20,
            n_blocks=10,
        )

        json_path = tmp_path / "splits.json"
        save_splits_to_json(splits, json_path)

        # Load with different source dir
        new_source = tmp_path / "different_location"
        loaded = load_splits_from_json(json_path, source_dir=new_source)

        # Check paths use new source dir
        for f in loaded["train"]["files"]:
            assert str(new_source) in str(f)

    def test_load_splits_from_folder_handles_json(self, mock_files, tmp_path):
        """load_splits_from_folder() works with JSON file path."""
        from datasets.splits import (
            create_research_splits,
            save_splits_to_json,
            load_splits_from_folder,
        )

        splits = create_research_splits(
            mock_files,
            train_size=50,
            val_size=10,
            ood_t_size=15,
            ood_spatial_size=20,
            n_blocks=10,
        )

        json_path = tmp_path / "splits.json"
        save_splits_to_json(splits, json_path)

        # load_splits_from_folder should handle JSON path
        loaded = load_splits_from_folder(json_path)

        assert set(loaded.keys()) == {"train", "val_iid", "ood_t", "ood_s", "ood_st"}
        assert loaded["train"]["input_years"] == (1990, 2021)

    def test_get_research_split_statistics(self, mock_files):
        """Test statistics computation."""
        from datasets.splits import (
            create_research_splits,
            get_research_split_statistics,
        )

        splits = create_research_splits(
            mock_files,
            train_size=50,
            val_size=10,
            ood_t_size=15,
            ood_spatial_size=20,
            n_blocks=10,
        )

        stats = get_research_split_statistics(splits)

        for split_name in splits:
            assert "n_samples" in stats[split_name]
            assert "input_years" in stats[split_name]
            assert "predict_years" in stats[split_name]
            assert "lat_min" in stats[split_name]
            assert "lat_max" in stats[split_name]
            assert stats[split_name]["n_samples"] == len(splits[split_name]["files"])


class TestShardPrefetchLoader:
    """Tests for ShardPrefetchLoader."""

    @pytest.fixture(scope="class")
    def shard_dir(self, tmp_path_factory):
        """Create temporary shard directory with test shards."""
        shard_path = tmp_path_factory.mktemp("shards_prefetch")

        # Create 3 small test shards (5 samples each)
        for i in range(3):
            shard_data = {
                "images": torch.randn(5, 140, 256, 256, dtype=torch.float16),
                "nodata_masks": torch.zeros(5, 140, 256, 256, dtype=torch.bool),
                "lats": torch.randn(5),
                "lons": torch.randn(5),
                "files": [f"sample_{i}_{j}.tif" for j in range(5)],
            }
            _save_test_shard(shard_data, shard_path / f"shard_{i:03d}.safetensors")

        return shard_path

    def test_init_discovers_shards(self, shard_dir):
        """Test that initialization discovers all shards."""
        from datasets.prefetch_dataset import ShardPrefetchLoader

        loader = ShardPrefetchLoader(
            shard_dir=shard_dir,
            batch_size=2,
            num_workers=0,
            samples_per_shard=5,
        )

        assert len(loader.shard_paths) == 3
        assert loader.total_samples == 15  # 3 shards * 5 samples

    def test_iteration_yields_batches(self, shard_dir):
        """Test that iteration yields batches correctly."""
        from datasets.prefetch_dataset import ShardPrefetchLoader

        loader = ShardPrefetchLoader(
            shard_dir=shard_dir,
            batch_size=2,
            num_workers=0,
            drop_last=False,
            samples_per_shard=5,
        )
        loader.reset()

        batches = list(loader)

        # Total samples: 15, batch_size: 2, drop_last=False
        # Each shard has 5 samples -> ceil(5/2) = 3 batches per shard
        # Total: 3 shards * 3 batches = 9 batches (but last batch of each shard has 1 sample)
        assert len(batches) > 0

        # Check batch structure
        batch = batches[0]
        assert "images" in batch
        assert "season_ids" in batch
        assert "year_indices" in batch
        assert batch["images"].shape[0] <= 2  # batch_size

    def test_len_returns_total_batches(self, shard_dir):
        """Test that __len__ returns total batches."""
        from datasets.prefetch_dataset import ShardPrefetchLoader

        loader = ShardPrefetchLoader(
            shard_dir=shard_dir,
            batch_size=2,
            num_workers=0,
            drop_last=True,
            samples_per_shard=5,
        )

        # 3 shards * 5 samples = 15 total samples
        # With batch_size=2 and drop_last=True: 15 // 2 = 7 batches
        assert len(loader) == 7

    def test_reset_shuffles_shards(self, shard_dir):
        """Test that reset() shuffles shard order."""
        from datasets.prefetch_dataset import ShardPrefetchLoader

        loader = ShardPrefetchLoader(
            shard_dir=shard_dir,
            batch_size=2,
            shuffle_shards=True,
            num_workers=0,
            samples_per_shard=5,
        )

        # Reset with different epochs should produce different orders
        loader.reset(epoch=0)
        order1 = loader._shard_order.copy()

        loader.reset(epoch=1)
        order2 = loader._shard_order.copy()

        # Orders should be permutations but possibly different
        assert sorted(order1) == sorted(order2)  # Same elements

    def test_sampler_set_epoch(self, shard_dir):
        """Test that sampler.set_epoch() works for DDP compatibility."""
        from datasets.prefetch_dataset import ShardPrefetchLoader

        loader = ShardPrefetchLoader(
            shard_dir=shard_dir,
            batch_size=2,
            num_workers=0,
            samples_per_shard=5,
        )

        # Should have sampler attribute
        assert hasattr(loader, "sampler")
        assert hasattr(loader.sampler, "set_epoch")

        # set_epoch should call reset
        loader.sampler.set_epoch(5)
        assert len(loader._shard_order) == 3  # Reset was called

    def test_distributed_shard_distribution(self, shard_dir):
        """Test that shards are distributed across ranks."""
        from datasets.prefetch_dataset import ShardPrefetchLoader

        # Rank 0 of 2
        loader0 = ShardPrefetchLoader(
            shard_dir=shard_dir,
            batch_size=2,
            num_workers=0,
            rank=0,
            world_size=2,
            samples_per_shard=5,
        )

        # Rank 1 of 2
        loader1 = ShardPrefetchLoader(
            shard_dir=shard_dir,
            batch_size=2,
            num_workers=0,
            rank=1,
            world_size=2,
            samples_per_shard=5,
        )

        # Each rank should have a subset of shards
        # With 3 shards and 2 ranks:
        # Rank 0 gets shards 0, 2 (indices 0, 2)
        # Rank 1 gets shard 1 (index 1)
        assert len(loader0.shard_paths) == 2
        assert len(loader1.shard_paths) == 1

    def test_shard_range_limits_shards(self, shard_dir):
        """Test that shard_range limits which shards are loaded."""
        from datasets.prefetch_dataset import ShardPrefetchLoader

        # Load only first 2 shards (indices 0, 1)
        loader = ShardPrefetchLoader(
            shard_dir=shard_dir,
            batch_size=2,
            num_workers=0,
            shard_range=(0, 2),
            samples_per_shard=5,
        )

        # Should only have 2 shards
        assert len(loader.shard_paths) == 2
        # 2 shards * 5 samples = 10 samples
        assert loader.total_samples == 10

    def test_no_shards_raises_error(self, tmp_path):
        """Test that empty directory raises ValueError."""
        from datasets.prefetch_dataset import ShardPrefetchLoader

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No shard_.*files found"):
            ShardPrefetchLoader(shard_dir=empty_dir, batch_size=2)

    def test_reads_samples_per_shard_from_metadata(self, tmp_path):
        """Test that samples_per_shard is read from metadata.json if present."""
        import json
        from datasets.prefetch_dataset import ShardPrefetchLoader

        shard_path = tmp_path / "shards_with_meta"
        shard_path.mkdir()

        # Create metadata with custom samples_per_shard
        metadata = {"samples_per_shard": 1024, "num_samples": 2048}
        (shard_path / "metadata.json").write_text(json.dumps(metadata))

        # Create dummy shard files
        for i in range(2):
            shard_data = {
                "images": torch.randn(5, 140, 256, 256, dtype=torch.float16),
                "nodata_masks": torch.zeros(5, 140, 256, 256, dtype=torch.bool),
                "lats": torch.randn(5),
                "lons": torch.randn(5),
                "files": [f"sample_{i}_{j}.tif" for j in range(5)],
            }
            _save_test_shard(shard_data, shard_path / f"shard_{i:03d}.safetensors")

        # Without explicit samples_per_shard, should read from metadata
        loader = ShardPrefetchLoader(shard_dir=shard_path, batch_size=16, num_workers=0)
        assert loader.samples_per_shard == 1024

    def test_explicit_samples_per_shard_overrides_metadata(self, tmp_path):
        """Test that explicit samples_per_shard takes precedence over metadata."""
        import json
        from datasets.prefetch_dataset import ShardPrefetchLoader

        shard_path = tmp_path / "shards_override"
        shard_path.mkdir()

        # Create metadata with samples_per_shard
        metadata = {"samples_per_shard": 1024}
        (shard_path / "metadata.json").write_text(json.dumps(metadata))

        # Create dummy shard
        shard_data = {
            "images": torch.randn(5, 140, 256, 256, dtype=torch.float16),
            "nodata_masks": torch.zeros(5, 140, 256, 256, dtype=torch.bool),
            "lats": torch.randn(5),
            "lons": torch.randn(5),
            "files": [f"sample_{j}.tif" for j in range(5)],
        }
        _save_test_shard(shard_data, shard_path / "shard_00000.safetensors")

        # Explicit value takes precedence over metadata
        loader = ShardPrefetchLoader(
            shard_dir=shard_path, batch_size=16, num_workers=0, samples_per_shard=500
        )
        assert loader.samples_per_shard == 500


class TestSingleShardDatasetSharedMemory:
    """Tests for SingleShardDataset shared memory functionality."""

    def test_init_creates_shared_memory_buffers(self):
        """Test constructor creates shared memory tensors."""
        from datasets.prefetch_dataset import SingleShardDataset

        dataset = SingleShardDataset(max_samples=10)

        # Check buffers are allocated
        assert dataset.images is not None
        assert dataset.images.shape == (10, 140, 256, 256)
        assert dataset.images.dtype == torch.float16
        assert dataset.images.is_shared()

        assert dataset.nodata_masks is not None
        assert dataset.nodata_masks.shape == (10, 140, 256, 256)
        assert dataset.nodata_masks.dtype == torch.bool
        assert dataset.nodata_masks.is_shared()

        assert dataset.lats is not None
        assert dataset.lats.shape == (10,)
        assert dataset.lats.is_shared()

        assert dataset.lons is not None
        assert dataset.lons.shape == (10,)
        assert dataset.lons.is_shared()

        # Season and year indices should also be shared
        assert dataset._season_ids.is_shared()
        assert dataset._year_indices.is_shared()

    def test_update_data_copies_to_buffers(self):
        """Test update_data copies data into shared memory buffers."""
        from datasets.prefetch_dataset import SingleShardDataset

        dataset = SingleShardDataset(max_samples=10)

        # Create test shard data with 5 samples
        shard_data = {
            "images": torch.randn(5, 140, 256, 256, dtype=torch.float16),
            "nodata_masks": torch.zeros(5, 140, 256, 256, dtype=torch.bool),
            "lats": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32),
            "lons": torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=torch.float32),
            "files": [f"sample_{i}.tif" for i in range(5)],
        }

        dataset.update_data(shard_data)

        # Check data was copied
        assert len(dataset) == 5
        assert torch.allclose(dataset.images[:5], shard_data["images"])
        assert torch.allclose(dataset.lats[:5], shard_data["lats"])
        assert torch.allclose(dataset.lons[:5], shard_data["lons"])

    def test_update_data_raises_if_too_many_samples(self):
        """Test update_data raises error if shard exceeds buffer size."""
        from datasets.prefetch_dataset import SingleShardDataset

        dataset = SingleShardDataset(max_samples=5)

        # Create shard data with 10 samples (exceeds buffer)
        shard_data = {
            "images": torch.randn(10, 140, 256, 256, dtype=torch.float16),
            "nodata_masks": torch.zeros(10, 140, 256, 256, dtype=torch.bool),
            "lats": torch.randn(10, dtype=torch.float32),
            "lons": torch.randn(10, dtype=torch.float32),
            "files": [f"sample_{i}.tif" for i in range(10)],
        }

        with pytest.raises(ValueError, match="buffer only holds"):
            dataset.update_data(shard_data)

    def test_getitem_works_with_shared_memory(self):
        """Test __getitem__ returns correct data from shared memory buffers."""
        from datasets.prefetch_dataset import SingleShardDataset

        dataset = SingleShardDataset(max_samples=10)

        # Create and load test data
        shard_data = {
            "images": torch.randn(3, 140, 256, 256, dtype=torch.float16),
            "nodata_masks": torch.zeros(3, 140, 256, 256, dtype=torch.bool),
            "lats": torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32),
            "lons": torch.tensor([-10.0, -20.0, -30.0], dtype=torch.float32),
            "files": ["a.tif", "b.tif", "c.tif"],
        }
        dataset.update_data(shard_data)

        # Check __getitem__
        sample = dataset[1]
        assert torch.allclose(sample["images"], shard_data["images"][1])
        assert sample["lat"].item() == pytest.approx(20.0)
        assert sample["lon"].item() == pytest.approx(-20.0)

    def test_multiple_updates(self):
        """Test that multiple update_data calls work correctly."""
        from datasets.prefetch_dataset import SingleShardDataset

        dataset = SingleShardDataset(max_samples=10)

        # First shard: 5 samples
        shard1 = {
            "images": torch.ones(5, 140, 256, 256, dtype=torch.float16),
            "nodata_masks": torch.zeros(5, 140, 256, 256, dtype=torch.bool),
            "lats": torch.ones(5, dtype=torch.float32),
            "lons": torch.ones(5, dtype=torch.float32) * -1,
            "files": [f"shard1_{i}.tif" for i in range(5)],
        }
        dataset.update_data(shard1)
        assert len(dataset) == 5
        assert dataset[0]["lat"].item() == pytest.approx(1.0)

        # Second shard: 3 samples (fewer)
        shard2 = {
            "images": torch.ones(3, 140, 256, 256, dtype=torch.float16) * 2,
            "nodata_masks": torch.zeros(3, 140, 256, 256, dtype=torch.bool),
            "lats": torch.ones(3, dtype=torch.float32) * 2,
            "lons": torch.ones(3, dtype=torch.float32) * -2,
            "files": [f"shard2_{i}.tif" for i in range(3)],
        }
        dataset.update_data(shard2)
        assert len(dataset) == 3  # Now only 3 samples
        assert dataset[0]["lat"].item() == pytest.approx(2.0)
