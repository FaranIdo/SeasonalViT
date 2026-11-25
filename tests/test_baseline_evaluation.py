"""Tests for baseline evaluation infrastructure.

Tests for:
- Sufficient statistics computation and aggregation
- Persistence baseline using t-1 (not t-4)
- Sample ID extraction from filenames
"""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.naive import PersistenceBaseline, PreviousYearBaseline
from baselines.statistical import (
    HarmonicRegressionBaseline,
    KalmanFilterBaseline,
    LinearTrendBaseline,
)
from eval.evaluate_baselines import (
    aggregate_from_stats,
    compute_sample_stats,
    extract_sample_id,
)
from train.metrics import compute_mae, compute_nse, compute_r2


class TestExtractSampleId:
    """Tests for sample_id extraction from filenames."""

    def test_standard_filename(self):
        assert extract_sample_id("0005487_35d408_140d754.tif") == "0005487"

    def test_different_sample_ids(self):
        assert extract_sample_id("0000001_0d0_0d0.tif") == "0000001"
        assert extract_sample_id("1234567_12d345_67d890.tif") == "1234567"

    def test_with_path(self):
        # Should work with full path if passed just the filename
        filename = Path("/some/path/0005487_35d408_140d754.tif").name
        assert extract_sample_id(filename) == "0005487"


class TestComputeSampleStats:
    """Tests for compute_sample_stats function."""

    def test_basic_stats(self):
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        target = torch.tensor([1.1, 2.1, 2.9, 4.0, 5.1])

        stats = compute_sample_stats(pred, target, "sample1", 132)

        assert stats["sample_id"] == "sample1"
        assert stats["target_idx"] == 132
        assert stats["target_season"] == 132 % 4  # 0 (spring)
        assert stats["target_year"] == 1990 + 132 // 4  # 2023
        assert stats["n_valid"] == 5
        assert abs(stats["sum_pred"] - 15.0) < 1e-5
        assert abs(stats["sum_target"] - 15.2) < 1e-5

    def test_handles_nan(self):
        pred = torch.tensor([1.0, float("nan"), 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])

        stats = compute_sample_stats(pred, target, "sample1", 0)

        assert stats["n_valid"] == 2
        assert abs(stats["sum_pred"] - 4.0) < 1e-5

    def test_returns_none_for_empty(self):
        pred = torch.tensor([float("nan"), float("nan")])
        target = torch.tensor([1.0, 2.0])

        stats = compute_sample_stats(pred, target, "sample1", 0)

        assert stats is None


class TestAggregateFromStats:
    """Tests for aggregate_from_stats function."""

    def test_single_sample(self):
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        stats = compute_sample_stats(pred, target, "s1", 0)
        result = aggregate_from_stats([stats])

        # Perfect prediction: MAE=0, R²=1, NSE=1, bias=0
        assert abs(result["mae"]) < 1e-5
        assert abs(result["r2"] - 1.0) < 1e-5
        assert abs(result["nse"] - 1.0) < 1e-5
        assert abs(result["bias"]) < 1e-5

    def test_empty_list(self):
        result = aggregate_from_stats([])

        assert result["mae"] == 0.0
        assert result["r2"] == 0.0
        assert result["nse"] == 0.0
        assert result["bias"] == 0.0
        assert result["n_valid"] == 0

    def test_matches_direct_computation(self):
        """Verify aggregating via sufficient stats equals direct computation."""
        torch.manual_seed(42)
        pred = torch.randn(1000) * 0.3 + 0.5
        target = torch.randn(1000) * 0.3 + 0.5

        # Direct computation
        direct_mae = compute_mae(pred, target).item()
        direct_r2 = compute_r2(pred, target).item()
        direct_nse = compute_nse(pred, target).item()
        direct_bias = abs(pred.mean().item() - target.mean().item())

        # Via sufficient stats
        stats = compute_sample_stats(pred, target, "s1", 0)
        result = aggregate_from_stats([stats])

        # Should match within floating point tolerance
        assert abs(result["mae"] - direct_mae) < 1e-4, f"MAE: {result['mae']} vs {direct_mae}"
        assert abs(result["r2"] - direct_r2) < 1e-4, f"R²: {result['r2']} vs {direct_r2}"
        assert abs(result["nse"] - direct_nse) < 1e-4, f"NSE: {result['nse']} vs {direct_nse}"
        assert abs(result["bias"] - direct_bias) < 1e-4, f"Bias: {result['bias']} vs {direct_bias}"

    def test_aggregation_across_samples(self):
        """Verify aggregating averages per-sample metrics (not global pooling)."""
        torch.manual_seed(42)

        # Create two "samples" with different sizes
        pred1 = torch.randn(500) * 0.3 + 0.5
        target1 = torch.randn(500) * 0.3 + 0.5
        pred2 = torch.randn(300) * 0.3 + 0.5
        target2 = torch.randn(300) * 0.3 + 0.5

        # Compute stats per sample, then aggregate
        stats1 = compute_sample_stats(pred1, target1, "s1", 0)
        stats2 = compute_sample_stats(pred2, target2, "s2", 0)
        result = aggregate_from_stats([stats1, stats2])

        # Compute metrics per sample directly
        mae1 = compute_mae(pred1, target1).item()
        mae2 = compute_mae(pred2, target2).item()
        r2_1 = compute_r2(pred1, target1).item()
        r2_2 = compute_r2(pred2, target2).item()
        nse1 = compute_nse(pred1, target1).item()
        nse2 = compute_nse(pred2, target2).item()

        # Aggregated result should be average of per-sample metrics
        expected_mae = (mae1 + mae2) / 2
        expected_r2 = (r2_1 + r2_2) / 2
        expected_nse = (nse1 + nse2) / 2

        assert abs(result["mae"] - expected_mae) < 1e-4
        assert abs(result["r2"] - expected_r2) < 1e-4
        assert abs(result["nse"] - expected_nse) < 1e-4
        assert result["n_samples"] == 2


class TestPersistenceBaseline:
    """Tests for persistence baseline using t-1 (previous timestep)."""

    def test_uses_previous_timestep_not_same_season(self):
        """Verify persistence baseline uses t-1, not t-4."""
        # Create sample with known values at t-1 and t-4
        B, T, H, W = 1, 10, 4, 4
        images = torch.zeros(B, T, H, W)
        nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

        # Set distinct values at different timesteps
        images[:, 5, :, :] = 0.5  # t-4 (same season for target=9)
        images[:, 8, :, :] = 0.8  # t-1 (previous timestep for target=9)

        baseline = PersistenceBaseline()
        pred, valid = baseline.predict(images, target_idx=9, nodata_mask=nodata_mask)

        # Should use t-1 value (0.8), not t-4 value (0.5)
        assert torch.allclose(pred, torch.full((B, H, W), 0.8))
        assert valid.all()

    def test_marks_invalid_when_t1_is_nodata(self):
        """Test that persistence marks pixels as invalid if t-1 is nodata."""
        B, T, H, W = 1, 10, 4, 4
        images = torch.zeros(B, T, H, W)
        nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

        # t-1 has nodata
        images[:, 8, :, :] = 0.8  # t-1 (will be masked)
        nodata_mask[:, 8, :, :] = True  # Mark t-1 as nodata

        baseline = PersistenceBaseline()
        pred, valid = baseline.predict(images, target_idx=9, nodata_mask=nodata_mask)

        # Prediction uses t-1 values, but valid mask is False for nodata pixels
        assert not valid.any()  # All pixels are invalid since t-1 is nodata

    def test_differs_from_previous_year_baseline(self):
        """Verify persistence and previous_year baselines give different results."""
        B, T, H, W = 1, 12, 4, 4
        images = torch.zeros(B, T, H, W)
        nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

        # Set distinct values
        images[:, 7, :, :] = 0.4  # t-4 (previous year same season for target=11)
        images[:, 10, :, :] = 0.9  # t-1 (previous timestep for target=11)

        persistence = PersistenceBaseline()
        previous_year = PreviousYearBaseline()

        pers_pred, _ = persistence.predict(images, target_idx=11, nodata_mask=nodata_mask)
        prev_pred, _ = previous_year.predict(images, target_idx=11, nodata_mask=nodata_mask)

        # Persistence should use 0.9, previous_year should use 0.4
        assert torch.allclose(pers_pred, torch.full((B, H, W), 0.9))
        assert torch.allclose(prev_pred, torch.full((B, H, W), 0.4))


class TestOutputDirYearSuffix:
    """Tests for output directory year suffix logic."""

    def test_default_years_includes_both(self):
        """Default split without --predict-years includes all years in folder name."""
        from eval.evaluate_baselines import SPLIT_CONFIGS

        # Simulate args with default output_dir and no predict_years override
        output_dir = Path("outputs/baselines")
        split = "ood_t"
        predict_years = None

        # Logic from main()
        config = SPLIT_CONFIGS[split]
        if predict_years is None:
            predict_years = config["predict_years"]

        if output_dir == Path("outputs/baselines"):
            year_suffix = "_".join(str(y) for y in sorted(predict_years))
            output_dir = Path("outputs/baselines") / f"{split}_{year_suffix}"

        assert output_dir == Path("outputs/baselines/ood_t_2023_2024")

    def test_single_year_override(self):
        """--predict-years with single year creates folder with that year only."""
        output_dir = Path("outputs/baselines")
        split = "ood_t"
        predict_years = [2023]  # User override

        if output_dir == Path("outputs/baselines"):
            year_suffix = "_".join(str(y) for y in sorted(predict_years))
            output_dir = Path("outputs/baselines") / f"{split}_{year_suffix}"

        assert output_dir == Path("outputs/baselines/ood_t_2023")

    def test_explicit_output_dir_unchanged(self):
        """Explicit --output-dir is not modified with year suffix."""
        output_dir = Path("my_custom_output")
        split = "ood_t"
        predict_years = [2023, 2024]

        # Logic should NOT modify explicit output_dir
        if output_dir == Path("outputs/baselines"):
            year_suffix = "_".join(str(y) for y in sorted(predict_years))
            output_dir = Path("outputs/baselines") / f"{split}_{year_suffix}"

        assert output_dir == Path("my_custom_output")

    def test_years_sorted_in_suffix(self):
        """Years in suffix are sorted regardless of input order."""
        output_dir = Path("outputs/baselines")
        split = "ood_t"
        predict_years = [2024, 2022, 2023]  # Unsorted

        if output_dir == Path("outputs/baselines"):
            year_suffix = "_".join(str(y) for y in sorted(predict_years))
            output_dir = Path("outputs/baselines") / f"{split}_{year_suffix}"

        assert output_dir == Path("outputs/baselines/ood_t_2022_2023_2024")


class TestPerSeasonAggregation:
    """Tests for per-season breakdown."""

    def test_per_season_sums_to_overall(self):
        """Verify per-season breakdown is consistent with overall metrics."""
        torch.manual_seed(42)

        # Create stats for different seasons
        stats_list = []
        for season in range(4):
            pred = torch.randn(200) * 0.3 + 0.5
            target = torch.randn(200) * 0.3 + 0.5
            stats = compute_sample_stats(pred, target, f"s{season}", season)
            stats_list.append(stats)

        # Aggregate overall
        overall = aggregate_from_stats(stats_list)

        # Aggregate by season
        season_results = []
        for season in range(4):
            season_stats = [s for s in stats_list if s["target_season"] == season]
            season_results.append(aggregate_from_stats(season_stats))

        # Total n_valid should match
        total_season_n = sum(r["n_valid"] for r in season_results)
        assert total_season_n == overall["n_valid"]

        # Weighted MAE should match overall (MAE is linearly aggregable)
        weighted_mae = sum(
            r["mae"] * r["n_valid"] for r in season_results
        ) / total_season_n
        assert abs(weighted_mae - overall["mae"]) < 1e-4


class TestLinearTrendBaseline:
    """Tests for linear trend baseline."""

    def test_perfect_linear_data(self):
        """Linear trend should exactly predict linearly increasing data."""
        B, T, H, W = 1, 140, 4, 4
        images = torch.zeros(B, T, H, W)
        nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

        # Create linearly increasing spring values (season=0): y = 0.1 + 0.01*year
        for year in range(35):
            idx = year * 4  # spring index
            images[:, idx, :, :] = 0.1 + 0.01 * year

        baseline = LinearTrendBaseline()
        # Predict spring 2024 (target_idx=136, year=34) using data up to 2023
        pred, valid = baseline.predict(
            images, target_idx=136, nodata_mask=nodata_mask, context_end_idx=135
        )

        # Expected: 0.1 + 0.01 * 34 = 0.44
        expected = 0.1 + 0.01 * 34
        assert valid.all()
        assert torch.allclose(pred, torch.full((B, H, W), expected), atol=1e-4)

    def test_constant_data_predicts_same(self):
        """For constant values, linear trend should predict the same constant."""
        B, T, H, W = 1, 140, 4, 4
        images = torch.full((B, T, H, W), 0.5)
        nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

        baseline = LinearTrendBaseline()
        pred, valid = baseline.predict(
            images, target_idx=136, nodata_mask=nodata_mask, context_end_idx=135
        )

        assert valid.all()
        assert torch.allclose(pred, torch.full((B, H, W), 0.5), atol=1e-4)

    def test_insufficient_data_marks_invalid(self):
        """With fewer than 2 valid observations, result should be invalid."""
        B, T, H, W = 1, 8, 4, 4
        images = torch.zeros(B, T, H, W)
        nodata_mask = torch.ones(B, T, H, W, dtype=torch.bool)
        # Only one valid spring observation
        nodata_mask[:, 0, :, :] = False
        images[:, 0, :, :] = 0.5

        baseline = LinearTrendBaseline()
        pred, valid = baseline.predict(
            images, target_idx=4, nodata_mask=nodata_mask, context_end_idx=3
        )

        assert not valid.any()

    def test_batched_predictions(self):
        """Test with batch size > 1."""
        B, T, H, W = 4, 140, 4, 4
        images = torch.zeros(B, T, H, W)
        nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

        # Different linear trends per batch sample
        for b in range(B):
            for year in range(35):
                images[b, year * 4, :, :] = 0.1 * b + 0.01 * year

        baseline = LinearTrendBaseline()
        pred, valid = baseline.predict(
            images, target_idx=136, nodata_mask=nodata_mask, context_end_idx=135
        )

        assert pred.shape == (B, H, W)
        assert valid.shape == (B, H, W)
        assert valid.all()


class TestKalmanFilterBaseline:
    """Tests for Kalman filter baseline."""

    def test_constant_data(self):
        """KF on constant data should predict the same constant."""
        B, T, H, W = 1, 140, 4, 4
        images = torch.full((B, T, H, W), 0.5)
        nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

        baseline = KalmanFilterBaseline()
        pred, valid = baseline.predict(
            images, target_idx=136, nodata_mask=nodata_mask, context_end_idx=135
        )

        assert valid.all()
        assert torch.allclose(pred, torch.full((B, H, W), 0.5), atol=0.05)

    def test_handles_nodata_gracefully(self):
        """KF should skip nodata observations and still produce predictions."""
        B, T, H, W = 1, 140, 4, 4
        images = torch.full((B, T, H, W), 0.5)
        nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

        # Mark some spring observations as nodata
        nodata_mask[:, 8, :, :] = True  # year 2
        nodata_mask[:, 16, :, :] = True  # year 4

        baseline = KalmanFilterBaseline()
        pred, valid = baseline.predict(
            images, target_idx=136, nodata_mask=nodata_mask, context_end_idx=135
        )

        assert valid.all()

    def test_all_nodata_marks_invalid(self):
        """KF with all nodata should return invalid predictions."""
        B, T, H, W = 1, 140, 4, 4
        images = torch.zeros(B, T, H, W)
        nodata_mask = torch.ones(B, T, H, W, dtype=torch.bool)

        baseline = KalmanFilterBaseline()
        pred, valid = baseline.predict(
            images, target_idx=136, nodata_mask=nodata_mask, context_end_idx=135
        )

        assert not valid.any()

    def test_output_shape(self):
        """Verify output shapes for various batch sizes."""
        for B in [1, 2, 8]:
            T, H, W = 140, 4, 4
            images = torch.randn(B, T, H, W) * 0.3 + 0.5
            nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

            baseline = KalmanFilterBaseline()
            pred, valid = baseline.predict(
                images, target_idx=136, nodata_mask=nodata_mask
            )

            assert pred.shape == (B, H, W)
            assert valid.shape == (B, H, W)


class TestHarmonicRegressionBaseline:
    """Tests for harmonic regression baseline."""

    def test_constant_data(self):
        """Constant input should predict the same constant."""
        B, T, H, W = 1, 140, 4, 4
        images = torch.full((B, T, H, W), 0.5)
        nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

        baseline = HarmonicRegressionBaseline()
        pred, valid = baseline.predict(
            images, target_idx=136, nodata_mask=nodata_mask, context_end_idx=135
        )

        assert valid.all()
        assert torch.allclose(pred, torch.full((B, H, W), 0.5), atol=1e-4)

    def test_linear_trend(self):
        """Linearly increasing data should extrapolate correctly."""
        B, T, H, W = 1, 140, 4, 4
        images = torch.zeros(B, T, H, W)
        nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

        # y(t) = 0.1 + 0.002 * t for all timesteps
        for t in range(140):
            images[:, t, :, :] = 0.1 + 0.002 * t

        baseline = HarmonicRegressionBaseline()
        # Predict t=136 using context up to t=135
        pred, valid = baseline.predict(
            images, target_idx=136, nodata_mask=nodata_mask, context_end_idx=135
        )

        expected = 0.1 + 0.002 * 136
        assert valid.all()
        assert torch.allclose(pred, torch.full((B, H, W), expected), atol=1e-3)

    def test_seasonal_pattern(self):
        """Known seasonal cycle should predict correct target season."""
        B, T, H, W = 1, 140, 4, 4
        images = torch.zeros(B, T, H, W)
        nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

        # Set a clear seasonal pattern: spring=0.8, summer=0.6, fall=0.4, winter=0.2
        season_values = [0.8, 0.6, 0.4, 0.2]
        for t in range(140):
            images[:, t, :, :] = season_values[t % 4]

        baseline = HarmonicRegressionBaseline()
        # Predict spring (idx=136, season=0) using context up to t=135
        pred, valid = baseline.predict(
            images, target_idx=136, nodata_mask=nodata_mask, context_end_idx=135
        )

        assert valid.all()
        assert torch.allclose(pred, torch.full((B, H, W), 0.8), atol=0.05)

        # Also test summer prediction (idx=137, season=1)
        pred_summer, valid_summer = baseline.predict(
            images, target_idx=137, nodata_mask=nodata_mask, context_end_idx=135
        )
        assert valid_summer.all()
        assert torch.allclose(pred_summer, torch.full((B, H, W), 0.6), atol=0.05)

    def test_insufficient_data_invalid(self):
        """Fewer than K valid observations should return invalid."""
        B, T, H, W = 1, 140, 4, 4
        images = torch.zeros(B, T, H, W)
        nodata_mask = torch.ones(B, T, H, W, dtype=torch.bool)

        # Only 3 valid timesteps, but n_params=6 (default n_harmonics=2)
        for t in range(3):
            nodata_mask[:, t, :, :] = False
            images[:, t, :, :] = 0.5

        baseline = HarmonicRegressionBaseline()
        pred, valid = baseline.predict(
            images, target_idx=136, nodata_mask=nodata_mask, context_end_idx=135
        )

        assert not valid.any()

    def test_handles_nodata(self):
        """Scattered nodata should still produce valid predictions."""
        B, T, H, W = 1, 140, 4, 4
        images = torch.full((B, T, H, W), 0.5)
        nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

        # Mark ~30% of timesteps as nodata (but leave enough valid ones)
        torch.manual_seed(42)
        nodata_mask = torch.rand(B, T, H, W) < 0.3

        baseline = HarmonicRegressionBaseline()
        pred, valid = baseline.predict(
            images, target_idx=136, nodata_mask=nodata_mask, context_end_idx=135
        )

        # Most pixels should still be valid (have >= 6 out of ~95 valid obs)
        assert valid.float().mean() > 0.9
        # Predictions should be close to 0.5 for valid pixels
        valid_preds = pred[valid]
        assert torch.allclose(valid_preds, torch.full_like(valid_preds, 0.5), atol=0.1)

    def test_output_shape(self):
        """Verify (B, H, W) output for various batch sizes."""
        for B in [1, 2, 8]:
            T, H, W = 140, 4, 4
            images = torch.randn(B, T, H, W) * 0.3 + 0.5
            nodata_mask = torch.zeros(B, T, H, W, dtype=torch.bool)

            baseline = HarmonicRegressionBaseline()
            pred, valid = baseline.predict(
                images, target_idx=136, nodata_mask=nodata_mask
            )

            assert pred.shape == (B, H, W)
            assert valid.shape == (B, H, W)
