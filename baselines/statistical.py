"""Statistical baseline methods for NDVI prediction.

Implements intermediate-complexity baselines between naive methods and deep learning:
- Linear Trend: Per-pixel OLS on same-season history, extrapolate to target
- Kalman Filter: Per-pixel local linear trend model with Kalman filtering
- Harmonic Regression: Fourier harmonics + linear trend on full time series (CCDC-style)
"""

import math
from typing import Tuple

import torch

from baselines.naive import BaselineMethod


class LinearTrendBaseline(BaselineMethod):
    """Linear trend baseline: per-pixel OLS regression on same-season history.

    Fits a simple linear model (value = a + b * year_index) to each pixel's
    same-season observations, then extrapolates to the target timestep.
    Pixels with fewer than 2 valid same-season observations are marked invalid.
    """

    name = "linear_trend"

    def predict(
        self,
        images: torch.Tensor,
        target_idx: int,
        nodata_mask: torch.Tensor,
        context_end_idx: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict using per-pixel linear trend extrapolation."""
        B, T, H, W = images.shape
        device = images.device
        dtype = images.dtype

        season = target_idx % 4
        end_idx = target_idx if context_end_idx is None else context_end_idx + 1
        same_season_indices = list(range(season, end_idx, 4))

        if len(same_season_indices) < 2:
            return (
                torch.zeros(B, H, W, device=device, dtype=dtype),
                torch.zeros(B, H, W, device=device, dtype=torch.bool),
            )

        obs, valid, x_vals = self._gather_observations(
            images, nodata_mask, same_season_indices
        )
        slope, intercept, valid_mask = self._fit_ols(obs, valid, x_vals)
        target_x = self._compute_target_x(target_idx, season)
        prediction = intercept + slope * target_x

        return prediction.to(dtype), valid_mask

    def _gather_observations(
        self,
        images: torch.Tensor,
        nodata_mask: torch.Tensor,
        indices: list,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack same-season observations and build x-coordinates.

        Returns:
            Tuple of (obs_stack, valid_stack, x_values) where
            obs_stack and valid_stack are (B, N, H, W) and x_values is (N,).
        """
        obs_stack = images[:, indices]  # (B, N, H, W)
        valid_stack = ~nodata_mask[:, indices]  # (B, N, H, W)
        x_vals = torch.tensor(
            [idx // 4 for idx in indices],
            device=images.device,
            dtype=images.dtype,
        )
        return obs_stack, valid_stack, x_vals

    def _fit_ols(
        self,
        obs: torch.Tensor,
        valid: torch.Tensor,
        x_vals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fit per-pixel OLS regression using sufficient statistics.

        Args:
            obs: Observations (B, N, H, W).
            valid: Valid mask (B, N, H, W).
            x_vals: X-coordinates for each observation (N,).

        Returns:
            Tuple of (slope, intercept, valid_mask) each (B, H, W).
        """
        B, N, H, W = obs.shape

        # Always compute OLS in float32 to avoid overflow in sufficient
        # statistics (e.g. count * sum_x2 can reach ~377K, exceeding fp16 max)
        obs = obs.float()
        x = x_vals.float().view(1, N, 1, 1).expand(B, N, H, W)

        # Zero out invalid observations
        obs_masked = obs.masked_fill(~valid, 0.0)
        x_masked = x.masked_fill(~valid, 0.0)
        count = valid.float().sum(dim=1)  # (B, H, W)

        # Sufficient statistics for OLS
        sum_x = x_masked.sum(dim=1)
        sum_y = obs_masked.sum(dim=1)
        sum_x2 = (x_masked ** 2).sum(dim=1)
        sum_xy = (x_masked * obs_masked).sum(dim=1)

        # Denominator: n * sum(x^2) - sum(x)^2
        denom = count * sum_x2 - sum_x ** 2
        valid_mask = (count >= 2) & (denom.abs() > 1e-10)

        # Compute slope and intercept (safe division)
        safe_denom = denom.clamp(min=1e-10)
        slope = (count * sum_xy - sum_x * sum_y) / safe_denom
        intercept = (sum_y * sum_x2 - sum_x * sum_xy) / safe_denom

        # Zero out invalid pixels
        slope = slope.masked_fill(~valid_mask, 0.0)
        intercept = intercept.masked_fill(~valid_mask, 0.0)

        return slope, intercept, valid_mask

    def _compute_target_x(self, target_idx: int, season: int) -> float:
        """Compute x-coordinate for the target timestep."""
        return target_idx // 4


class KalmanFilterBaseline(BaselineMethod):
    """Kalman filter baseline: per-pixel local linear trend model.

    State vector: [level, trend] where prediction = level + trend * dt.
    Forward pass on same-season history; nodata timesteps skip the update step.
    Extrapolates the final state to predict the target.
    """

    name = "kalman_filter"

    def __init__(
        self,
        process_noise_level: float = 0.01,
        process_noise_trend: float = 0.001,
        obs_noise: float = 0.05,
    ):
        """Initialize Kalman filter parameters.

        Args:
            process_noise_level: Process noise variance for level state.
            process_noise_trend: Process noise variance for trend state.
            obs_noise: Observation noise variance.
        """
        self.q_level = process_noise_level
        self.q_trend = process_noise_trend
        self.r = obs_noise

    def predict(
        self,
        images: torch.Tensor,
        target_idx: int,
        nodata_mask: torch.Tensor,
        context_end_idx: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict using Kalman filter with local linear trend model."""
        B, T, H, W = images.shape
        device = images.device
        dtype = images.dtype

        season = target_idx % 4
        end_idx = target_idx if context_end_idx is None else context_end_idx + 1
        same_season_indices = list(range(season, end_idx, 4))

        if not same_season_indices:
            return (
                torch.zeros(B, H, W, device=device, dtype=dtype),
                torch.zeros(B, H, W, device=device, dtype=torch.bool),
            )

        level, trend, valid_mask = self._run_filter(
            images, nodata_mask, same_season_indices
        )

        # Extrapolate to target: years ahead from last observation
        last_year = same_season_indices[-1] // 4
        target_year = target_idx // 4
        dt = float(target_year - last_year)
        prediction = level + trend * dt

        return prediction.to(dtype), valid_mask

    def _run_filter(
        self,
        images: torch.Tensor,
        nodata_mask: torch.Tensor,
        indices: list,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Kalman filter forward pass over same-season observations.

        Returns:
            Tuple of (level, trend, valid_mask) each (B, H, W).
        """
        B, T, H, W = images.shape
        device = images.device

        # Initialize state from first valid observation
        level, trend, P, initialized = self._initialize_state(
            images, nodata_mask, indices
        )

        # Process remaining observations
        for i in range(1, len(indices)):
            idx = indices[i]
            obs = images[:, idx]  # (B, H, W)
            obs_valid = ~nodata_mask[:, idx]  # (B, H, W)

            dt = float((indices[i] // 4) - (indices[i - 1] // 4))
            level, trend, P = self._predict_step(level, trend, P, dt)
            level, trend, P, initialized = self._update_step(
                level, trend, P, obs, obs_valid, initialized,
            )

        valid_mask = initialized
        return level, trend, valid_mask

    def _initialize_state(
        self,
        images: torch.Tensor,
        nodata_mask: torch.Tensor,
        indices: list,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize Kalman state from first observation.

        Returns:
            Tuple of (level, trend, P_sym, initialized_mask).
            P stores the symmetric 2x2 covariance as (B, 3, H, W):
            [P00=level var, P01=cross-cov, P11=trend var].
        """
        B, T, H, W = images.shape
        device = images.device
        dtype = images.dtype

        first_obs = images[:, indices[0]]
        first_valid = ~nodata_mask[:, indices[0]]

        level = torch.zeros(B, H, W, device=device, dtype=dtype)
        trend = torch.zeros(B, H, W, device=device, dtype=dtype)
        P = torch.zeros(B, 3, H, W, device=device, dtype=dtype)

        level[first_valid] = first_obs[first_valid]
        P[:, 0] = 1.0   # P00: level variance
        # P[:, 1] = 0.0  # P01: cross-covariance (zero initially)
        P[:, 2] = 0.1   # P11: trend variance

        initialized = first_valid
        return level, trend, P, initialized

    def _predict_step(
        self,
        level: torch.Tensor,
        trend: torch.Tensor,
        P: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Kalman prediction step: propagate state forward by dt years.

        State transition: F = [[1, dt], [0, 1]].
        Covariance: P_pred = F @ P @ F' + Q.
        """
        level_pred = level + trend * dt
        P_new = torch.empty_like(P)
        P_new[:, 0] = P[:, 0] + 2 * dt * P[:, 1] + dt * dt * P[:, 2] + self.q_level
        P_new[:, 1] = P[:, 1] + dt * P[:, 2]
        P_new[:, 2] = P[:, 2] + self.q_trend
        return level_pred, trend, P_new

    def _update_step(
        self,
        level: torch.Tensor,
        trend: torch.Tensor,
        P: torch.Tensor,
        obs: torch.Tensor,
        obs_valid: torch.Tensor,
        initialized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Kalman update step: incorporate new observation.

        With H = [1, 0], K = P @ H' / S gives gains for both level and trend.
        For pixels not yet initialized, sets level directly from observation.
        """
        innovation = obs - level  # (B, H, W)

        # Kalman gains: K = [P00, P01]' / S
        S = P[:, 0] + self.r
        S_safe = S.clamp(min=1e-10)
        K_level = P[:, 0] / S_safe
        K_trend = P[:, 1] / S_safe

        # State update
        update_mask = obs_valid & initialized
        level = torch.where(update_mask, level + K_level * innovation, level)
        trend = torch.where(update_mask, trend + K_trend * innovation, trend)

        # Covariance update: P_new = (I - K @ H) @ P
        P_new = P.clone()
        P_new[:, 0] = torch.where(update_mask, (1 - K_level) * P[:, 0], P[:, 0])
        P_new[:, 1] = torch.where(update_mask, (1 - K_level) * P[:, 1], P[:, 1])
        P_new[:, 2] = torch.where(update_mask, P[:, 2] - K_trend * P[:, 1], P[:, 2])

        # Initialize from first valid observation for uninitialized pixels
        init_mask = obs_valid & ~initialized
        level = torch.where(init_mask, obs, level)
        initialized = initialized | obs_valid

        return level, trend, P_new, initialized


class HarmonicRegressionBaseline(BaselineMethod):
    """Harmonic regression baseline: Fourier harmonics + linear trend on full time series.

    Fits y(t) = a0 + a1*t + sum_k[a_k*cos(2*pi*k*t/4) + b_k*sin(2*pi*k*t/4)]
    to ALL timesteps (not same-season only), modeling seasonality explicitly.
    This is the standard CCDC-style approach in remote sensing.

    Period = 4 (seasonal cycle), t = timestep index (0-139).
    Default n_harmonics=2 gives 6 parameters: intercept, trend,
    cos/sin(annual), cos/sin(semi-annual).
    """

    name = "harmonic_regression"

    def __init__(self, n_harmonics: int = 2):
        self.n_harmonics = n_harmonics
        self.n_params = 2 + 2 * n_harmonics  # intercept + trend + 2 per harmonic

    def predict(
        self,
        images: torch.Tensor,
        target_idx: int,
        nodata_mask: torch.Tensor,
        context_end_idx: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict using harmonic regression on full time series."""
        B, T, H, W = images.shape
        device = images.device
        dtype = images.dtype
        end_idx = target_idx if context_end_idx is None else context_end_idx + 1

        indices = list(range(end_idx))
        if len(indices) < self.n_params:
            return (
                torch.zeros(B, H, W, device=device, dtype=dtype),
                torch.zeros(B, H, W, device=device, dtype=torch.bool),
            )

        X = self._build_design_matrix(indices, device)
        coeffs, valid_mask = self._solve_normal_equations(
            X, images, nodata_mask, indices
        )
        prediction = self._extrapolate(coeffs, target_idx, device)

        return prediction.to(dtype), valid_mask

    def _build_design_matrix(
        self, indices: list, device: torch.device
    ) -> torch.Tensor:
        """Build design matrix X of shape (N, K).

        Columns: [1, t, cos(2*pi*1*t/4), sin(2*pi*1*t/4), cos(2*pi*2*t/4), ...].
        """
        t = torch.tensor(indices, device=device, dtype=torch.float32)
        N = len(indices)
        K = self.n_params
        X = torch.zeros(N, K, device=device, dtype=torch.float32)
        X[:, 0] = 1.0  # intercept
        X[:, 1] = t  # linear trend
        for k in range(1, self.n_harmonics + 1):
            freq = 2.0 * math.pi * k / 4.0
            X[:, 2 + 2 * (k - 1)] = torch.cos(freq * t)
            X[:, 2 + 2 * (k - 1) + 1] = torch.sin(freq * t)
        return X

    def _solve_normal_equations(
        self,
        X: torch.Tensor,
        images: torch.Tensor,
        nodata_mask: torch.Tensor,
        indices: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve per-pixel weighted OLS via normal equations.

        Accumulates X'X and X'y one timestep at a time to avoid materializing
        large intermediate tensors. Peak GPU memory: O(B*K^2*H*W) instead of
        O(B*N*K*H*W), which saves ~50 GB for typical batch sizes.

        Args:
            X: Design matrix (N, K).
            images: Full image tensor (B, T, H, W).
            nodata_mask: Nodata mask (B, T, H, W).
            indices: Context timestep indices.

        Returns:
            Tuple of (coeffs, valid_mask) where coeffs is (B, K, H, W)
            and valid_mask is (B, H, W).
        """
        B, T, H, W = images.shape
        K = self.n_params
        P = H * W
        device = images.device

        # Accumulate normal equations one timestep at a time
        XtX = torch.zeros(B, K * K, P, device=device, dtype=torch.float32)
        Xty = torch.zeros(B, K, P, device=device, dtype=torch.float32)
        count = torch.zeros(B, P, device=device, dtype=torch.float32)

        for i, t_idx in enumerate(indices):
            v = (~nodata_mask[:, t_idx]).reshape(B, P).float()  # (B, P)
            y = images[:, t_idx].reshape(B, P).float() * v  # (B, P)
            x_t = X[i]  # (K,)
            xx = (x_t.unsqueeze(1) * x_t.unsqueeze(0)).reshape(K * K)  # (K²,)
            XtX += v.unsqueeze(1) * xx.view(1, K * K, 1)
            Xty += y.unsqueeze(1) * x_t.view(1, K, 1)
            count += v

        valid_mask = (count >= K).reshape(B, H, W)
        XtX = XtX.reshape(B, K, K, P)

        # Solve (B*P) independent K×K systems
        XtX_solve = XtX.permute(0, 3, 1, 2).reshape(B * P, K, K)
        del XtX
        Xty_solve = Xty.permute(0, 2, 1).reshape(B * P, K, 1)
        del Xty

        # Add small regularization for numerical stability (in-place)
        XtX_solve.diagonal(dim1=-2, dim2=-1).add_(1e-6)

        coeffs_flat = torch.linalg.solve(XtX_solve, Xty_solve)  # (B*P, K, 1)
        del XtX_solve, Xty_solve

        coeffs = coeffs_flat.squeeze(-1).reshape(B, P, K).permute(0, 2, 1)
        coeffs = coeffs.reshape(B, K, H, W)
        coeffs = coeffs.masked_fill(~valid_mask.unsqueeze(1), 0.0)

        return coeffs, valid_mask

    def _extrapolate(
        self, coeffs: torch.Tensor, target_idx: int, device: torch.device
    ) -> torch.Tensor:
        """Extrapolate to target timestep using fitted coefficients.

        Args:
            coeffs: Fitted coefficients (B, K, H, W).
            target_idx: Target timestep index.
            device: Torch device.

        Returns:
            Prediction (B, H, W).
        """
        X_target = self._build_design_matrix([target_idx], device)  # (1, K)
        # prediction = sum_k X_target[k] * coeffs[b, k, h, w]
        prediction = torch.einsum("nk,bkhw->bhw", X_target, coeffs)
        return prediction
