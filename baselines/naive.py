"""Naive baseline methods for NDVI prediction.

Implements four simple baselines:
- Persistence: Use most recent observation (previous timestep)
- Previous Year: Use same season from previous years (t-4, t-8, t-12, ...)
- Climatology: Mean of all same-season observations before target
- Recent Climatology: Mean of last 5 same-season observations before target
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class BaselineMethod(ABC):
    """Base class for naive baseline methods."""

    name: str = "base"

    @abstractmethod
    def predict(
        self,
        images: torch.Tensor,
        target_idx: int,
        nodata_mask: torch.Tensor,
        context_end_idx: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate prediction for a target timestep.

        Args:
            images: NDVI images of shape (B, T, H, W).
            target_idx: Index of timestep to predict (0-139).
            nodata_mask: Boolean mask where True = nodata, shape (B, T, H, W).
            context_end_idx: Last index that can be used as context (inclusive).
                If None, defaults to target_idx - 1 (use all data before target).

        Returns:
            Tuple of:
                - prediction: Shape (B, H, W)
                - valid_mask: Boolean mask where True = valid prediction
        """
        raise NotImplementedError


class PersistenceBaseline(BaselineMethod):
    """Persistence baseline: use the last available observation.

    Simply returns the value at context_end_idx (or target_idx - 1).
    Pixels with nodata at that timestep are marked invalid.
    """

    name = "persistence"

    def predict(
        self,
        images: torch.Tensor,
        target_idx: int,
        nodata_mask: torch.Tensor,
        context_end_idx: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the last available observation."""
        # Determine which timestep to use
        if context_end_idx is None:
            src_idx = target_idx - 1
        else:
            src_idx = min(target_idx - 1, context_end_idx)

        prediction = images[:, src_idx]  # (B, H, W)
        valid_mask = ~nodata_mask[:, src_idx]  # (B, H, W)

        return prediction, valid_mask


class PreviousYearBaseline(BaselineMethod):
    """Previous year baseline: use same season from previous year.

    Returns the most recent same-season observation before the target.
    Pixels with nodata at that timestep are marked invalid.
    """

    name = "previous_year"

    def predict(
        self,
        images: torch.Tensor,
        target_idx: int,
        nodata_mask: torch.Tensor,
        context_end_idx: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return observation from same season in previous year."""
        # Find the most recent same-season index
        src_idx = target_idx - 4
        if context_end_idx is not None:
            # Align to same season and ensure <= context_end_idx
            while src_idx > context_end_idx:
                src_idx -= 4

        if src_idx < 0:
            # No previous same-season observation available
            B, T, H, W = images.shape
            return (
                torch.zeros(B, H, W, device=images.device, dtype=images.dtype),
                torch.zeros(B, H, W, device=images.device, dtype=torch.bool),
            )

        prediction = images[:, src_idx]  # (B, H, W)
        valid_mask = ~nodata_mask[:, src_idx]  # (B, H, W)

        return prediction, valid_mask


class ClimatologyBaseline(BaselineMethod):
    """Climatology baseline: mean of all same-season observations before target.

    For target at index t with season s = t % 4, compute mean of observations
    at indices [s, s+4, s+8, ..., t-4], handling nodata per-pixel.
    """

    name = "climatology"

    def predict(
        self,
        images: torch.Tensor,
        target_idx: int,
        nodata_mask: torch.Tensor,
        context_end_idx: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute climatological mean for same season."""
        B, T, H, W = images.shape
        device = images.device
        dtype = images.dtype

        season = target_idx % 4

        # Determine end index for context
        end_idx = target_idx if context_end_idx is None else context_end_idx + 1

        # Collect all same-season indices up to end_idx (exclusive)
        same_season_indices = list(range(season, end_idx, 4))

        if not same_season_indices:
            # No prior observations for this season
            return (
                torch.zeros(B, H, W, device=device, dtype=dtype),
                torch.zeros(B, H, W, device=device, dtype=torch.bool),
            )

        # Stack observations and masks for vectorized computation
        obs_stack = images[:, same_season_indices]  # (B, N, H, W)
        mask_stack = nodata_mask[:, same_season_indices]  # (B, N, H, W)

        # Compute mean ignoring nodata (masked mean)
        valid_stack = ~mask_stack  # (B, N, H, W)

        # Sum of valid values (zero out nodata without copying)
        value_sum = obs_stack.masked_fill(mask_stack, 0.0).sum(dim=1)  # (B, H, W)

        # Count of valid observations per pixel
        valid_count = valid_stack.sum(dim=1).float()  # (B, H, W)

        # Compute mean where we have valid observations
        prediction = torch.zeros(B, H, W, device=device, dtype=dtype)
        has_valid = valid_count > 0
        prediction[has_valid] = (value_sum[has_valid] / valid_count[has_valid]).to(
            dtype
        )

        return prediction, has_valid


class RecentClimatologyBaseline(BaselineMethod):
    """Recent climatology baseline: mean of last N same-season observations.

    Like climatology, but only uses the most recent N years (default 5) instead
    of all historical data. This can be more relevant for detecting trends
    since climate patterns shift over time.
    """

    name = "recent_climatology"

    def __init__(self, n_years: int = 5):
        """Initialize with number of recent years to use."""
        self.n_years = n_years

    def predict(
        self,
        images: torch.Tensor,
        target_idx: int,
        nodata_mask: torch.Tensor,
        context_end_idx: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean of last N same-season observations."""
        B, T, H, W = images.shape
        device = images.device
        dtype = images.dtype

        season = target_idx % 4

        # Determine end index for context
        end_idx = target_idx if context_end_idx is None else context_end_idx + 1

        # Collect last N same-season indices up to end_idx
        all_same_season = list(range(season, end_idx, 4))
        same_season_indices = all_same_season[-self.n_years:]  # Take last N

        if not same_season_indices:
            return (
                torch.zeros(B, H, W, device=device, dtype=dtype),
                torch.zeros(B, H, W, device=device, dtype=torch.bool),
            )

        # Stack observations and masks for vectorized computation
        obs_stack = images[:, same_season_indices]  # (B, N, H, W)
        mask_stack = nodata_mask[:, same_season_indices]  # (B, N, H, W)

        # Compute mean ignoring nodata (masked mean)
        valid_stack = ~mask_stack  # (B, N, H, W)

        # Sum of valid values (zero out nodata without copying)
        value_sum = obs_stack.masked_fill(mask_stack, 0.0).sum(dim=1)  # (B, H, W)

        # Count of valid observations per pixel
        valid_count = valid_stack.sum(dim=1).float()  # (B, H, W)

        # Compute mean where we have valid observations
        prediction = torch.zeros(B, H, W, device=device, dtype=dtype)
        has_valid = valid_count > 0
        prediction[has_valid] = (value_sum[has_valid] / valid_count[has_valid]).to(
            dtype
        )

        return prediction, has_valid


def get_baseline(name: str) -> BaselineMethod:
    """Get baseline method by name.

    Args:
        name: One of "persistence", "previous_year", "climatology".

    Returns:
        Baseline method instance.
    """
    baselines = {
        "persistence": PersistenceBaseline,
        "previous_year": PreviousYearBaseline,
        "climatology": ClimatologyBaseline,
        "recent_climatology": RecentClimatologyBaseline,
    }

    if name not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(baselines)}")

    return baselines[name]()


ALL_BASELINES = ["persistence", "previous_year", "climatology", "recent_climatology"]
