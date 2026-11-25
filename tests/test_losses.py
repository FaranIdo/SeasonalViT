"""Tests for loss functions."""

import pytest
import torch


class TestPretrainLoss:
    """Tests for pre-training loss function."""

    def test_basic_loss(self):
        """Test basic masked reconstruction loss."""
        from train.losses import pretrain_loss

        prediction = torch.randn(2, 4, 8, 32)  # (B, T, N, P)
        target = torch.randn(2, 4, 8, 32)
        mask = torch.ones(2, 4, 8, dtype=torch.bool)  # All masked

        loss = pretrain_loss(prediction, target, mask)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_partial_mask(self):
        """Test loss with partial masking."""
        from train.losses import pretrain_loss

        prediction = torch.zeros(2, 4, 8, 32)
        target = torch.ones(2, 4, 8, 32)
        mask = torch.zeros(2, 4, 8, dtype=torch.bool)
        mask[0, 0, 0] = True  # Only mask one position

        loss = pretrain_loss(prediction, target, mask)

        # Loss should be 1.0 (MSE of 0 vs 1)
        assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)

    def test_no_mask(self):
        """Test loss with no masked positions."""
        from train.losses import pretrain_loss

        prediction = torch.randn(2, 4, 8, 32)
        target = torch.randn(2, 4, 8, 32)
        mask = torch.zeros(2, 4, 8, dtype=torch.bool)  # Nothing masked

        loss = pretrain_loss(prediction, target, mask)

        # Should return zero (no positions to compute loss on)
        assert loss.item() == 0.0

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        from train.losses import pretrain_loss

        prediction = torch.randn(2, 4, 8, 32, requires_grad=True)
        target = torch.randn(2, 4, 8, 32)
        mask = torch.ones(2, 4, 8, dtype=torch.bool)

        loss = pretrain_loss(prediction, target, mask)
        loss.backward()

        assert prediction.grad is not None
        assert not torch.isnan(prediction.grad).any()

    def test_loss_decreases_when_closer(self):
        """Test that loss is lower when prediction is closer to target."""
        from train.losses import pretrain_loss

        target = torch.randn(2, 4, 8, 32)
        mask = torch.ones(2, 4, 8, dtype=torch.bool)

        # Prediction close to target
        pred_close = target + torch.randn_like(target) * 0.1
        loss_close = pretrain_loss(pred_close, target, mask)

        # Prediction far from target
        pred_far = target + torch.randn_like(target) * 2.0
        loss_far = pretrain_loss(pred_far, target, mask)

        assert loss_close.item() < loss_far.item()

    def test_nodata_mask_excludes_positions(self):
        """Test that nodata_mask excludes positions from loss computation."""
        from train.losses import pretrain_loss

        # Create prediction and target with known values
        prediction = torch.zeros(2, 4, 8, 32)  # (B, T, N, P)
        target = torch.ones(2, 4, 8, 32)
        mask = torch.ones(2, 4, 8, dtype=torch.bool)  # All masked for reconstruction

        # Without nodata mask: loss should be 1.0 (MSE of 0 vs 1)
        loss_no_nodata = pretrain_loss(prediction, target, mask)
        assert torch.allclose(loss_no_nodata, torch.tensor(1.0), atol=1e-6)

        # With nodata mask excluding all positions: loss should be 0
        nodata_mask = torch.ones(2, 4, 8, dtype=torch.bool)  # All nodata
        loss_all_nodata = pretrain_loss(prediction, target, mask, nodata_mask=nodata_mask)
        assert loss_all_nodata.item() == 0.0

        # With nodata mask excluding half: loss still 1.0 but computed on fewer positions
        nodata_mask_half = torch.zeros(2, 4, 8, dtype=torch.bool)
        nodata_mask_half[:, :, :4] = True  # First 4 patches are nodata
        loss_half_nodata = pretrain_loss(prediction, target, mask, nodata_mask=nodata_mask_half)
        assert torch.allclose(loss_half_nodata, torch.tensor(1.0), atol=1e-6)

    def test_nodata_mask_pixel_level_conversion(self):
        """Test that pixel-level nodata_mask is converted to patch-level."""
        from train.losses import pretrain_loss

        # (B, T, N, P) where N=64 patches of 32x32 pixels
        prediction = torch.zeros(1, 1, 64, 1024)
        target = torch.ones(1, 1, 64, 1024)
        mask = torch.ones(1, 1, 64, dtype=torch.bool)

        # Pixel-level nodata mask (B, T, H, W) where H=W=256
        # Mark top-left 32x32 region as nodata (first patch)
        nodata_mask_pixels = torch.zeros(1, 1, 256, 256, dtype=torch.bool)
        nodata_mask_pixels[:, :, :32, :32] = True  # First patch has nodata

        loss = pretrain_loss(prediction, target, mask, nodata_mask=nodata_mask_pixels)

        # Loss should be computed on 63 patches (excluding first)
        # Since all predictions are 0 and targets are 1, MSE = 1.0
        assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)


class TestForecastLoss:
    """Tests for forecasting loss function."""

    def test_basic_loss(self):
        """Test basic forecasting loss."""
        from train.losses import forecast_loss

        output = {
            "prediction": torch.randn(2, 1, 32, 32),
            "delta": torch.randn(2, 1, 32, 32),
        }
        target = torch.randn(2, 1, 32, 32)

        total_loss, components = forecast_loss(output, target)

        assert total_loss.shape == ()
        assert total_loss.item() > 0
        assert "recon_loss" in components
        assert "delta_reg" in components
        assert "total_loss" in components

    def test_with_nodata_mask(self):
        """Test loss with nodata mask."""
        from train.losses import forecast_loss

        output = {
            "prediction": torch.zeros(2, 1, 4, 4),
            "delta": torch.zeros(2, 1, 4, 4),
        }
        target = torch.ones(2, 1, 4, 4)

        # Mask half the pixels
        nodata_mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
        nodata_mask[:, :, :2, :] = True  # Top half is nodata

        total_loss, components = forecast_loss(output, target, nodata_mask)

        # Loss should only be computed on bottom half (where mask is False)
        # prediction=0, target=1 on valid pixels -> MSE = 1.0
        assert torch.allclose(components["recon_loss"], torch.tensor(1.0), atol=1e-6)

    def test_delta_regularization(self):
        """Test that delta regularization penalizes large deltas."""
        from train.losses import forecast_loss

        target = torch.randn(2, 1, 32, 32)

        # Small delta
        output_small = {
            "prediction": target.clone(),  # Perfect prediction
            "delta": torch.zeros(2, 1, 32, 32) + 0.1,
        }
        _, comp_small = forecast_loss(output_small, target)

        # Large delta
        output_large = {
            "prediction": target.clone(),  # Perfect prediction
            "delta": torch.zeros(2, 1, 32, 32) + 1.0,
        }
        _, comp_large = forecast_loss(output_large, target)

        # Larger delta should have larger delta_reg
        assert comp_small["delta_reg"].item() < comp_large["delta_reg"].item()

    def test_delta_weight(self):
        """Test that delta_reg_weight scales regularization."""
        from train.losses import forecast_loss

        output = {
            "prediction": torch.randn(2, 1, 32, 32),
            "delta": torch.ones(2, 1, 32, 32),  # delta^2 = 1
        }
        target = output["prediction"].clone()  # Perfect prediction

        _, comp_w01 = forecast_loss(output, target, delta_reg_weight=0.01)
        _, comp_w1 = forecast_loss(output, target, delta_reg_weight=1.0)

        # Delta reg contribution should differ by factor of 100
        ratio = comp_w1["total_loss"] / comp_w01["total_loss"]
        assert 90 < ratio < 110  # Allow some tolerance

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        from train.losses import forecast_loss

        prediction = torch.randn(2, 1, 32, 32, requires_grad=True)
        output = {
            "prediction": prediction,
            "delta": torch.randn(2, 1, 32, 32, requires_grad=True),
        }
        target = torch.randn(2, 1, 32, 32)

        total_loss, _ = forecast_loss(output, target)
        total_loss.backward()

        assert prediction.grad is not None
        assert output["delta"].grad is not None

    def test_delta_reg_excludes_nodata(self):
        """Test that delta regularization excludes nodata pixels."""
        from train.losses import forecast_loss

        delta = torch.ones(2, 1, 4, 4) * 0.5
        # Set nodata region to have large delta (should be excluded)
        delta[:, :, :2, :] = 10.0

        output = {
            "prediction": torch.zeros(2, 1, 4, 4),
            "delta": delta,
        }
        target = torch.zeros(2, 1, 4, 4)

        # Top half is nodata
        nodata_mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
        nodata_mask[:, :, :2, :] = True

        _, components = forecast_loss(output, target, nodata_mask)

        # Delta reg should only consider bottom half (value=0.5), so 0.5^2 = 0.25
        assert components["delta_reg"].item() == pytest.approx(0.25, abs=1e-6)


class TestMaskedMSELoss:
    """Tests for general masked MSE loss."""

    def test_basic_loss(self):
        """Test basic MSE loss without mask."""
        from train.losses import masked_mse_loss

        pred = torch.zeros(2, 4, 4)
        target = torch.ones(2, 4, 4)

        loss = masked_mse_loss(pred, target)

        assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)

    def test_with_mask(self):
        """Test MSE loss with mask."""
        from train.losses import masked_mse_loss

        pred = torch.zeros(2, 4, 4)
        target = torch.ones(2, 4, 4)

        # Only compute on first pixel
        mask = torch.zeros(2, 4, 4, dtype=torch.bool)
        mask[0, 0, 0] = True

        loss = masked_mse_loss(pred, target, mask)

        assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)

    def test_empty_mask(self):
        """Test with empty mask (all False)."""
        from train.losses import masked_mse_loss

        pred = torch.randn(2, 4, 4)
        target = torch.randn(2, 4, 4)
        mask = torch.zeros(2, 4, 4, dtype=torch.bool)

        loss = masked_mse_loss(pred, target, mask)

        assert loss.item() == 0.0


class TestPretrainingLossClass:
    """Tests for PretrainingLoss class."""

    def test_callable(self):
        """Test that class is callable like a function."""
        from train.losses import PretrainingLoss

        loss_fn = PretrainingLoss()

        prediction = torch.randn(2, 4, 8, 32)
        target = torch.randn(2, 4, 8, 32)
        mask = torch.ones(2, 4, 8, dtype=torch.bool)

        loss = loss_fn(prediction, target, mask)

        assert loss.shape == ()
        assert loss.item() > 0


class TestForecastingLossClass:
    """Tests for ForecastingLoss class."""

    def test_callable(self):
        """Test that class is callable like a function."""
        from train.losses import ForecastingLoss

        loss_fn = ForecastingLoss(delta_reg_weight=0.01)

        output = {
            "prediction": torch.randn(2, 1, 32, 32),
            "delta": torch.randn(2, 1, 32, 32),
        }
        target = torch.randn(2, 1, 32, 32)

        total_loss, components = loss_fn(output, target)

        assert total_loss.shape == ()
        assert "recon_loss" in components

    def test_custom_weight(self):
        """Test custom delta_reg_weight."""
        from train.losses import ForecastingLoss

        loss_fn = ForecastingLoss(delta_reg_weight=0.5)

        output = {
            "prediction": torch.zeros(2, 1, 32, 32),
            "delta": torch.ones(2, 1, 32, 32),
        }
        target = torch.zeros(2, 1, 32, 32)

        _, components = loss_fn(output, target)

        # With delta_reg_weight=0.5 and delta=1, delta_reg contribution = 0.5
        assert torch.allclose(components["delta_reg"], torch.tensor(1.0), atol=1e-6)


class TestSmoothL1ForecastLoss:
    """Tests for smooth L1 forecasting loss."""

    def test_basic_loss(self):
        """Test basic smooth L1 loss."""
        from train.losses import smooth_l1_forecast_loss

        output = {
            "prediction": torch.randn(2, 1, 32, 32),
            "delta": torch.randn(2, 1, 32, 32),
        }
        target = torch.randn(2, 1, 32, 32)

        total_loss, components = smooth_l1_forecast_loss(output, target)

        assert total_loss.shape == ()
        assert "recon_loss" in components

    def test_outlier_robustness(self):
        """Test that smooth L1 is more robust to outliers than MSE."""
        from train.losses import forecast_loss, smooth_l1_forecast_loss

        target = torch.zeros(2, 1, 32, 32)
        target[0, 0, 0, 0] = 10.0  # One outlier

        output = {
            "prediction": torch.zeros(2, 1, 32, 32),
            "delta": torch.zeros(2, 1, 32, 32),
        }

        _, comp_mse = forecast_loss(output, target, delta_reg_weight=0)
        _, comp_smooth = smooth_l1_forecast_loss(output, target, delta_reg_weight=0)

        # Smooth L1 should have lower loss due to outlier
        assert comp_smooth["recon_loss"].item() < comp_mse["recon_loss"].item()


class TestPatchNormalization:
    """Tests for MAE-style per-patch target normalization."""

    def test_normalize_patches_zero_mean_unit_var(self):
        """Verify normalized patches have zero mean and unit variance."""
        from train.losses import _normalize_patches

        patches = torch.randn(2, 4, 8, 32) * 5 + 10  # Non-zero mean, large var
        normalized = _normalize_patches(patches)

        # Check per-patch stats along last dim (P)
        means = normalized.mean(dim=-1)
        vars_ = normalized.var(dim=-1)

        assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
        assert torch.allclose(vars_, torch.ones_like(vars_), atol=1e-2)

    def test_pretrain_loss_with_normalization(self):
        """Verify loss changes when normalization is enabled."""
        from train.losses import pretrain_loss

        prediction = torch.randn(2, 4, 8, 32)
        target = torch.randn(2, 4, 8, 32) * 3 + 5  # Offset target
        mask = torch.ones(2, 4, 8, dtype=torch.bool)

        loss_raw = pretrain_loss(prediction, target, mask, normalize_target=False)
        loss_norm = pretrain_loss(prediction, target, mask, normalize_target=True)

        # Losses should differ (normalization changes the target)
        assert not torch.allclose(loss_raw, loss_norm)

    def test_normalization_gradient_flow(self):
        """Verify gradients flow through normalized loss."""
        from train.losses import pretrain_loss

        prediction = torch.randn(2, 4, 8, 32, requires_grad=True)
        target = torch.randn(2, 4, 8, 32)
        mask = torch.ones(2, 4, 8, dtype=torch.bool)

        loss = pretrain_loss(prediction, target, mask, normalize_target=True)
        loss.backward()

        assert prediction.grad is not None
        assert not torch.isnan(prediction.grad).any()
        assert prediction.grad.abs().sum() > 0

    def test_uniform_patch_penalized(self):
        """Predicting per-patch mean should give HIGH loss with normalization.

        This is the key test: without normalization, predicting the patch
        mean gives near-zero loss (the plateau problem). With normalization,
        the target becomes zero-mean, so predicting the original mean maps
        to a non-zero prediction against a zero-mean target.
        """
        from train.losses import pretrain_loss

        B, T, N, P = 2, 4, 8, 32
        target = torch.randn(B, T, N, P) * 0.5 + 0.35  # Typical NDVI range

        # Prediction = per-patch mean (the plateau behavior)
        patch_means = target.mean(dim=-1, keepdim=True).expand_as(target)
        mask = torch.ones(B, T, N, dtype=torch.bool)

        loss_no_norm = pretrain_loss(patch_means, target, mask, normalize_target=False)
        loss_with_norm = pretrain_loss(patch_means, target, mask, normalize_target=True)

        # Without normalization: predicting mean gives low loss (just variance)
        # With normalization: predicting mean gives HIGH loss (~1.0)
        # because normalized target has unit var, and prediction of raw mean
        # maps to a constant offset from zero
        assert loss_with_norm.item() > loss_no_norm.item() * 2

    def test_efficient_loss_with_normalization(self):
        """Verify normalization works with efficient temporal masking loss."""
        from train.losses import pretrain_loss_efficient

        prediction = torch.randn(2, 4, 8, 32, requires_grad=True)
        target = torch.randn(2, 4, 8, 32)
        temporal_mask = torch.ones(2, 4, dtype=torch.bool)

        loss = pretrain_loss_efficient(
            prediction, target, temporal_mask, normalize_target=True
        )
        loss.backward()

        assert prediction.grad is not None
        assert not torch.isnan(prediction.grad).any()


class TestEdgeCases:
    """Edge case tests for loss functions."""

    def test_single_batch(self):
        """Test with single batch."""
        from train.losses import pretrain_loss, forecast_loss

        # Pre-training
        pred = torch.randn(1, 4, 8, 32)
        target = torch.randn(1, 4, 8, 32)
        mask = torch.ones(1, 4, 8, dtype=torch.bool)
        loss = pretrain_loss(pred, target, mask)
        assert loss.shape == ()

        # Forecasting
        output = {
            "prediction": torch.randn(1, 1, 32, 32),
            "delta": torch.randn(1, 1, 32, 32),
        }
        total_loss, _ = forecast_loss(output, torch.randn(1, 1, 32, 32))
        assert total_loss.shape == ()

    def test_all_nodata(self):
        """Test forecast loss when all pixels are nodata."""
        from train.losses import forecast_loss

        output = {
            "prediction": torch.randn(2, 1, 32, 32),
            "delta": torch.randn(2, 1, 32, 32),
        }
        target = torch.randn(2, 1, 32, 32)
        nodata_mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)  # All nodata

        total_loss, components = forecast_loss(output, target, nodata_mask)

        # Recon loss should be 0 (no valid pixels)
        assert components["recon_loss"].item() == 0.0
        # Delta reg should also be 0 (nodata pixels excluded)
        assert components["delta_reg"].item() == 0.0

    def test_large_tensors(self):
        """Test with larger tensor sizes."""
        from train.losses import forecast_loss

        output = {
            "prediction": torch.randn(4, 1, 256, 256),
            "delta": torch.randn(4, 1, 256, 256),
        }
        target = torch.randn(4, 1, 256, 256)

        total_loss, _ = forecast_loss(output, target)

        assert not torch.isnan(total_loss)
        assert not torch.isinf(total_loss)


class TestPerSampleCorrelationLoss:
    """Tests for per-sample correlation loss."""

    def test_perfect_correlation_gives_zero_loss(self):
        """Identical pred and target should produce loss near 0."""
        from train.losses import per_sample_correlation_loss

        target = torch.randn(2, 1, 32, 32)
        valid_mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)

        loss = per_sample_correlation_loss(target, target, valid_mask)

        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_scaled_prediction_gives_zero_loss(self):
        """Prediction = a*target + b should give loss near 0 (correlation is 1)."""
        from train.losses import per_sample_correlation_loss

        target = torch.randn(2, 1, 32, 32)
        pred = target * 3.0 + 0.5  # Affine transform
        valid_mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)

        loss = per_sample_correlation_loss(pred, target, valid_mask)

        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_uncorrelated_gives_high_loss(self):
        """Uncorrelated pred and target should produce loss near 1."""
        from train.losses import per_sample_correlation_loss

        torch.manual_seed(42)
        pred = torch.randn(4, 1, 64, 64)
        target = torch.randn(4, 1, 64, 64)
        valid_mask = torch.ones(4, 1, 64, 64, dtype=torch.bool)

        loss = per_sample_correlation_loss(pred, target, valid_mask)

        # Random tensors have near-zero correlation, so loss should be near 1
        assert loss.item() > 0.8

    def test_negative_correlation(self):
        """Negatively correlated should produce loss near 2."""
        from train.losses import per_sample_correlation_loss

        target = torch.randn(2, 1, 32, 32)
        pred = -target  # Perfect negative correlation: r = -1, loss = 1 - (-1) = 2
        valid_mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)

        loss = per_sample_correlation_loss(pred, target, valid_mask)

        assert loss.item() == pytest.approx(2.0, abs=1e-5)

    def test_nodata_mask_respected(self):
        """Only valid pixels should contribute to correlation."""
        from train.losses import per_sample_correlation_loss

        target = torch.randn(2, 1, 32, 32)
        pred = target.clone()
        # Corrupt the masked region; should not affect loss
        pred[:, :, :16, :] = torch.randn(2, 1, 16, 32)

        valid_mask = torch.zeros(2, 1, 32, 32, dtype=torch.bool)
        valid_mask[:, :, 16:, :] = True  # Only bottom half is valid

        loss = per_sample_correlation_loss(pred, target, valid_mask)

        # Bottom half is identical, so loss should be near 0
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_gradient_flow(self):
        """Gradients should flow through correlation loss."""
        from train.losses import per_sample_correlation_loss

        pred = torch.randn(2, 1, 32, 32, requires_grad=True)
        target = torch.randn(2, 1, 32, 32)
        valid_mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)

        loss = per_sample_correlation_loss(pred, target, valid_mask)
        loss.backward()

        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()
        assert pred.grad.abs().sum() > 0

    def test_all_nodata_returns_zero(self):
        """All-nodata samples should return zero loss."""
        from train.losses import per_sample_correlation_loss

        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        valid_mask = torch.zeros(2, 1, 32, 32, dtype=torch.bool)

        loss = per_sample_correlation_loss(pred, target, valid_mask)

        assert loss.item() == 0.0

    def test_constant_prediction_excluded(self):
        """Constant prediction (zero variance) should be excluded, not crash."""
        from train.losses import per_sample_correlation_loss

        pred = torch.ones(2, 1, 32, 32) * 0.5  # Constant
        target = torch.randn(2, 1, 32, 32)
        valid_mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)

        loss = per_sample_correlation_loss(pred, target, valid_mask)

        # Should return 0 (no valid samples) rather than NaN
        assert not torch.isnan(loss)


class TestForecastingLossWithCorrelation:
    """Tests for ForecastingLoss with correlation loss enabled."""

    def test_corr_loss_in_components(self):
        """Correlation loss should appear in components dict."""
        from train.losses import ForecastingLoss

        loss_fn = ForecastingLoss(correlation_loss_weight=0.1)
        output = {
            "prediction": torch.randn(2, 1, 32, 32),
            "delta": torch.randn(2, 1, 32, 32),
        }
        target = torch.randn(2, 1, 32, 32)

        _, components = loss_fn(output, target)

        assert "corr_loss" in components
        assert components["corr_loss"].item() >= 0

    def test_corr_weight_zero_disables(self):
        """With weight=0, correlation loss should be zero and not affect total."""
        from train.losses import ForecastingLoss

        output = {
            "prediction": torch.randn(2, 1, 32, 32),
            "delta": torch.randn(2, 1, 32, 32),
        }
        target = torch.randn(2, 1, 32, 32)

        loss_fn_off = ForecastingLoss(correlation_loss_weight=0.0, delta_reg_weight=0.0)
        loss_fn_on = ForecastingLoss(correlation_loss_weight=0.1, delta_reg_weight=0.0)

        total_off, comp_off = loss_fn_off(output, target)
        total_on, comp_on = loss_fn_on(output, target)

        assert comp_off["corr_loss"].item() == 0.0
        # With correlation enabled, total loss should be higher (adds penalty)
        assert total_on.item() >= total_off.item()

    def test_corr_weight_scales_contribution(self):
        """Higher correlation weight should increase total loss."""
        from train.losses import ForecastingLoss

        torch.manual_seed(42)
        output = {
            "prediction": torch.randn(2, 1, 32, 32),
            "delta": torch.zeros(2, 1, 32, 32),
        }
        target = torch.randn(2, 1, 32, 32)

        loss_fn_low = ForecastingLoss(correlation_loss_weight=0.05, delta_reg_weight=0)
        loss_fn_high = ForecastingLoss(correlation_loss_weight=0.5, delta_reg_weight=0)

        total_low, _ = loss_fn_low(output, target)
        total_high, _ = loss_fn_high(output, target)

        assert total_high.item() > total_low.item()

    def test_gradient_with_correlation(self):
        """Gradients should flow through combined loss with correlation."""
        from train.losses import ForecastingLoss

        loss_fn = ForecastingLoss(correlation_loss_weight=0.1)
        pred = torch.randn(2, 1, 32, 32, requires_grad=True)
        output = {
            "prediction": pred,
            "delta": torch.randn(2, 1, 32, 32, requires_grad=True),
        }
        target = torch.randn(2, 1, 32, 32)

        total_loss, _ = loss_fn(output, target)
        total_loss.backward()

        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


class TestDeltaSupervisionLoss:
    """Tests for the delta supervision loss."""

    def test_zero_delta_gives_nonzero_loss(self):
        """Delta=0 with nonzero true delta should produce positive loss."""
        from train.losses import _compute_delta_supervision

        delta = torch.zeros(2, 1, 32, 32)
        target = torch.ones(2, 1, 32, 32) * 0.5
        anchor = torch.ones(2, 1, 32, 32) * 0.48  # true delta = 0.02
        valid_mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)

        loss = _compute_delta_supervision(delta, target, anchor, valid_mask, delta_std=0.025)

        # Normalized: true_delta/std = 0.02/0.025 = 0.8, pred/std = 0
        # L1 loss = 0.8
        assert loss.item() == pytest.approx(0.8, abs=0.01)

    def test_correct_delta_gives_zero_loss(self):
        """Delta matching true delta should give near-zero loss."""
        from train.losses import _compute_delta_supervision

        anchor = torch.ones(2, 1, 32, 32) * 0.5
        target = torch.ones(2, 1, 32, 32) * 0.52  # true delta = 0.02
        delta = torch.ones(2, 1, 32, 32) * 0.02   # matches true delta
        valid_mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)

        loss = _compute_delta_supervision(delta, target, anchor, valid_mask, delta_std=0.025)

        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_normalization_amplifies_gradient(self):
        """Smaller delta_std should produce larger loss for same absolute error."""
        from train.losses import _compute_delta_supervision

        delta = torch.zeros(2, 1, 32, 32)
        anchor = torch.ones(2, 1, 32, 32) * 0.5
        target = torch.ones(2, 1, 32, 32) * 0.52  # true delta = 0.02
        valid_mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)

        loss_small_std = _compute_delta_supervision(
            delta, target, anchor, valid_mask, delta_std=0.01,
        )
        loss_large_std = _compute_delta_supervision(
            delta, target, anchor, valid_mask, delta_std=0.1,
        )

        # Smaller std = larger normalized error = larger loss
        assert loss_small_std.item() > loss_large_std.item() * 5

    def test_gradient_flows_to_delta(self):
        """Gradient should flow through delta supervision loss."""
        from train.losses import _compute_delta_supervision

        delta = torch.randn(2, 1, 32, 32, requires_grad=True)
        anchor = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        valid_mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)

        loss = _compute_delta_supervision(delta, target, anchor, valid_mask, delta_std=0.025)
        loss.backward()

        assert delta.grad is not None
        assert not torch.isnan(delta.grad).any()
        assert delta.grad.abs().sum() > 0

    def test_nodata_excluded(self):
        """Nodata pixels should not contribute to delta supervision loss."""
        from train.losses import _compute_delta_supervision

        delta = torch.zeros(2, 1, 4, 4)
        anchor = torch.ones(2, 1, 4, 4) * 0.5
        target = torch.ones(2, 1, 4, 4) * 0.52
        valid_mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
        valid_mask[:, :, 2:, :] = True  # Only bottom half valid

        loss = _compute_delta_supervision(delta, target, anchor, valid_mask, delta_std=0.025)

        # Should only use bottom half
        assert loss.item() > 0

    def test_anchor_nodata_excluded(self):
        """Pixels where the anchor is the nodata sentinel (-9999) must be
        excluded from the loss. Previously leaked in under `last_step` anchor
        mode whose history-walk returns -9999 for entirely-nodata pixels,
        producing (target - anchor) ≈ 10000 → loss blow-up → NaN training.
        """
        from train.losses import _compute_delta_supervision

        delta = torch.zeros(1, 1, 4, 4)
        target = torch.full((1, 1, 4, 4), 0.5)
        anchor = torch.full((1, 1, 4, 4), 0.48)  # valid anchor, true delta=0.02
        # Poison one pixel with the nodata sentinel
        anchor[0, 0, 0, 0] = -9999.0
        valid_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)  # target is valid

        loss = _compute_delta_supervision(delta, target, anchor, valid_mask, delta_std=0.025)

        # If the nodata pixel leaked in, loss would be ~10000/0.025 = ~400000.
        # With the fix it's excluded and loss reflects only the other 15 valid
        # pixels: |0 - 0.02/0.025| = 0.8
        assert loss.item() == pytest.approx(0.8, abs=0.01)

    def test_forecasting_loss_with_delta_supervision(self):
        """ForecastingLoss should include delta supervision when enabled."""
        from train.losses import ForecastingLoss

        loss_fn = ForecastingLoss(
            delta_supervision_weight=1.0, delta_std=0.025, delta_reg_weight=0.0,
        )
        anchor = torch.ones(2, 1, 32, 32) * 0.5
        delta = torch.zeros(2, 1, 32, 32)
        output = {
            "prediction": anchor + delta,
            "delta": delta,
            "anchor": anchor,
        }
        target = torch.ones(2, 1, 32, 32) * 0.52

        total_loss, components = loss_fn(output, target)

        assert "delta_sup_loss" in components
        assert components["delta_sup_loss"].item() > 0
        # Total loss should be higher with delta supervision
        assert total_loss.item() > components["recon_loss"].item()

    def test_delta_supervision_disabled_when_weight_zero(self):
        """Delta supervision should not contribute when weight=0."""
        from train.losses import ForecastingLoss

        loss_fn = ForecastingLoss(
            delta_supervision_weight=0.0, delta_reg_weight=0.0,
        )
        output = {
            "prediction": torch.randn(2, 1, 32, 32),
            "delta": torch.randn(2, 1, 32, 32),
            "anchor": torch.randn(2, 1, 32, 32),
        }
        target = torch.randn(2, 1, 32, 32)

        _, components = loss_fn(output, target)

        assert components["delta_sup_loss"].item() == 0.0
