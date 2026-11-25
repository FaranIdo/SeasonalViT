"""Tests for evaluation metrics."""

import pytest
import torch


class TestBasicMetrics:
    """Tests for basic metric functions."""

    def test_compute_mae(self):
        """Test Mean Absolute Error computation."""
        from train.metrics import compute_mae

        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 2.5])

        mae = compute_mae(pred, target)

        # |1-1.5| + |2-2.5| + |3-2.5| = 0.5 + 0.5 + 0.5 = 1.5, mean = 0.5
        assert torch.allclose(mae, torch.tensor(0.5))

    def test_compute_mse(self):
        """Test Mean Squared Error computation."""
        from train.metrics import compute_mse

        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 2.0, 2.0])

        mse = compute_mse(pred, target)

        # (1-2)^2 + (2-2)^2 + (3-2)^2 = 1 + 0 + 1 = 2, mean = 0.667
        assert torch.allclose(mse, torch.tensor(2.0 / 3), atol=1e-4)

    def test_compute_rmse(self):
        """Test Root Mean Squared Error computation."""
        from train.metrics import compute_rmse

        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 2.0, 2.0])

        rmse = compute_rmse(pred, target)

        expected = (2.0 / 3) ** 0.5
        assert torch.allclose(rmse, torch.tensor(expected), atol=1e-4)

    def test_compute_r2_perfect(self):
        """Test R² for perfect predictions."""
        from train.metrics import compute_r2

        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])

        r2 = compute_r2(pred, target)

        assert torch.allclose(r2, torch.tensor(1.0))

    def test_compute_r2_mean_prediction(self):
        """Test R² when prediction equals mean (should be 0)."""
        from train.metrics import compute_r2

        target = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.ones(3) * target.mean()  # Predict mean

        r2 = compute_r2(pred, target)

        assert torch.allclose(r2, torch.tensor(0.0), atol=1e-6)

    def test_compute_r2_scaled_prediction(self):
        """Test R² is 1.0 for perfectly correlated but scaled predictions.

        R² = corr²(pred, target), so scaling doesn't affect it.
        """
        from train.metrics import compute_r2

        target = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.tensor([10.0, 20.0, 30.0])  # Scaled by 10x

        r2 = compute_r2(pred, target)

        # Perfect correlation despite scale difference
        assert torch.allclose(r2, torch.tensor(1.0), atol=1e-6)

    def test_compute_nse_negative(self):
        """Test NSE can be negative for bad predictions.

        NSE penalizes bias/scale errors unlike R².
        """
        from train.metrics import compute_nse

        target = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.tensor([10.0, 20.0, 30.0])  # Very wrong

        nse = compute_nse(pred, target)

        assert nse.item() < 0


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_returns_all_metrics(self):
        """Test that all expected metrics are returned."""
        from train.metrics import compute_metrics

        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)

        metrics = compute_metrics(pred, target)

        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "n_valid" in metrics

    def test_with_mask(self):
        """Test metrics computation with mask."""
        from train.metrics import compute_metrics

        pred = torch.zeros(2, 2, 2)
        target = torch.ones(2, 2, 2)

        # Mask half the pixels
        mask = torch.zeros(2, 2, 2, dtype=torch.bool)
        mask[0, 0, 0] = True
        mask[1, 1, 1] = True

        metrics = compute_metrics(pred, target, mask)

        # MAE should be 1.0 (|0 - 1| = 1)
        assert abs(metrics["mae"] - 1.0) < 1e-6
        assert metrics["n_valid"] == 2

    def test_empty_mask(self):
        """Test with all-False mask."""
        from train.metrics import compute_metrics

        pred = torch.randn(2, 2, 2)
        target = torch.randn(2, 2, 2)
        mask = torch.zeros(2, 2, 2, dtype=torch.bool)

        metrics = compute_metrics(pred, target, mask)

        assert metrics["n_valid"] == 0
        assert metrics["mae"] == 0.0


class TestPerSeasonMetrics:
    """Tests for per-season metrics computation."""

    def test_basic_per_season(self):
        """Test per-season metrics computation."""
        from train.metrics import compute_per_season_metrics

        B, T, H, W = 2, 8, 4, 4
        pred = torch.randn(B, T, H, W)
        target = torch.randn(B, T, H, W)
        season_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]] * B)

        metrics = compute_per_season_metrics(pred, target, season_ids)

        assert "spring" in metrics
        assert "summer" in metrics
        assert "fall" in metrics
        assert "winter" in metrics

        # Each season should have metrics
        for season in ["spring", "summer", "fall", "winter"]:
            assert "mae" in metrics[season]
            assert "rmse" in metrics[season]
            assert metrics[season]["n_valid"] > 0

    def test_1d_season_ids(self):
        """Test with 1D season IDs (broadcast)."""
        from train.metrics import compute_per_season_metrics

        B, T, H, W = 2, 4, 4, 4
        pred = torch.randn(B, T, H, W)
        target = torch.randn(B, T, H, W)
        season_ids = torch.tensor([0, 1, 2, 3])  # 1D

        metrics = compute_per_season_metrics(pred, target, season_ids)

        # Should still work
        for season in ["spring", "summer", "fall", "winter"]:
            assert metrics[season]["n_valid"] > 0


class TestMetricsTracker:
    """Tests for MetricsTracker class."""

    def test_single_batch(self):
        """Test tracking single batch."""
        from train.metrics import MetricsTracker

        tracker = MetricsTracker()

        pred = torch.zeros(2, 1, 4, 4)
        target = torch.ones(2, 1, 4, 4)

        tracker.update(pred, target)
        metrics = tracker.compute()

        assert abs(metrics["mae"] - 1.0) < 1e-6

    def test_multiple_batches(self):
        """Test tracking multiple batches."""
        from train.metrics import MetricsTracker

        tracker = MetricsTracker()

        # Add two batches of zeros vs ones
        tracker.update(torch.zeros(2, 4, 4), torch.ones(2, 4, 4))
        tracker.update(torch.zeros(2, 4, 4), torch.ones(2, 4, 4))

        metrics = tracker.compute()

        assert abs(metrics["mae"] - 1.0) < 1e-6
        assert metrics["n_valid"] == 2 * 2 * 4 * 4

    def test_with_mask(self):
        """Test tracking with masks."""
        from train.metrics import MetricsTracker

        tracker = MetricsTracker()

        pred = torch.zeros(2, 4, 4)
        target = torch.ones(2, 4, 4)
        mask = torch.ones(2, 4, 4, dtype=torch.bool)
        mask[0] = False  # Mask out first batch

        tracker.update(pred, target, mask)
        metrics = tracker.compute()

        # Only second batch should count
        assert metrics["n_valid"] == 4 * 4

    def test_reset(self):
        """Test reset clears accumulated data."""
        from train.metrics import MetricsTracker

        tracker = MetricsTracker()

        tracker.update(torch.randn(2, 4, 4), torch.randn(2, 4, 4))
        tracker.reset()
        metrics = tracker.compute()

        assert metrics["n_valid"] == 0

    def test_empty_tracker(self):
        """Test computing on empty tracker."""
        from train.metrics import MetricsTracker

        tracker = MetricsTracker()
        metrics = tracker.compute()

        assert metrics["n_valid"] == 0
        assert metrics["mae"] == 0.0


class TestEdgeCases:
    """Edge case tests for metrics."""

    def test_constant_target(self):
        """Test R² with constant target (all same value)."""
        from train.metrics import compute_r2

        target = torch.ones(10)
        pred = torch.randn(10)

        r2 = compute_r2(pred, target)

        # R² should be 0 when target has no variance (unless pred is perfect)
        assert not torch.isnan(r2)

    def test_single_value(self):
        """Test metrics with single value."""
        from train.metrics import compute_metrics

        pred = torch.tensor([0.5])
        target = torch.tensor([0.7])

        metrics = compute_metrics(pred, target)

        assert abs(metrics["mae"] - 0.2) < 1e-6
        assert metrics["n_valid"] == 1

    def test_gpu_tensors(self):
        """Test that metrics work with GPU tensors (if available)."""
        from train.metrics import MetricsTracker

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tracker = MetricsTracker()
        pred = torch.randn(2, 4, 4, device=device)
        target = torch.randn(2, 4, 4, device=device)

        tracker.update(pred, target)
        metrics = tracker.compute()

        # Should be computed on CPU
        assert isinstance(metrics["mae"], float)

    def test_float16_produces_same_r2_as_float32(self):
        """Test that float16 input produces same R² as float32 via upcast."""
        from train.metrics import PerSampleMetricsTracker

        torch.manual_seed(42)
        pred_f32 = torch.randn(4, 1, 64, 64)
        target_f32 = pred_f32 + torch.randn_like(pred_f32) * 0.1

        # Float32 tracker
        tracker32 = PerSampleMetricsTracker()
        tracker32.update(pred_f32, target_f32)
        metrics32 = tracker32.compute()

        # Float16 tracker (simulating AMP)
        tracker16 = PerSampleMetricsTracker()
        tracker16.update(pred_f32.half(), target_f32.half())
        metrics16 = tracker16.compute()

        # Should produce very similar results due to float() upcast
        assert metrics16["r2"] == pytest.approx(metrics32["r2"], abs=1e-3)
        assert metrics16["mae"] == pytest.approx(metrics32["mae"], abs=1e-3)
