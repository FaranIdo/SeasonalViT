"""Baseline methods for NDVI prediction evaluation.

Provides simple baselines to compare against SeasonalViT:
- PersistenceBaseline: Most recent same-season observation
- PreviousYearBaseline: Exactly one year ago (t-4)
- ClimatologyBaseline: Mean of all prior same-season observations
- RecentClimatologyBaseline: Mean of last 5 same-season observations
- LinearTrendBaseline: Per-pixel OLS trend extrapolation
- KalmanFilterBaseline: Per-pixel Kalman filter with local linear trend
- HarmonicRegressionBaseline: Fourier harmonics + linear trend (CCDC-style)
"""

from baselines.naive import (
    ALL_BASELINES as _NAIVE_BASELINES,
    BaselineMethod,
    ClimatologyBaseline,
    PersistenceBaseline,
    PreviousYearBaseline,
    RecentClimatologyBaseline,
    get_baseline as _get_naive_baseline,
)
from baselines.statistical import (
    HarmonicRegressionBaseline,
    KalmanFilterBaseline,
    LinearTrendBaseline,
)

STATISTICAL_BASELINES = ["linear_trend", "kalman_filter", "harmonic_regression"]
ALL_BASELINES = _NAIVE_BASELINES + STATISTICAL_BASELINES


def get_baseline(name: str) -> BaselineMethod:
    """Get baseline method by name.

    Args:
        name: Baseline method name.

    Returns:
        Baseline method instance.
    """
    if name in _NAIVE_BASELINES:
        return _get_naive_baseline(name)

    statistical = {
        "linear_trend": LinearTrendBaseline,
        "kalman_filter": KalmanFilterBaseline,
        "harmonic_regression": HarmonicRegressionBaseline,
    }
    if name in statistical:
        return statistical[name]()

    raise ValueError(f"Unknown baseline: {name}. Choose from {ALL_BASELINES}")


__all__ = [
    "ALL_BASELINES",
    "STATISTICAL_BASELINES",
    "BaselineMethod",
    "ClimatologyBaseline",
    "HarmonicRegressionBaseline",
    "KalmanFilterBaseline",
    "LinearTrendBaseline",
    "PersistenceBaseline",
    "PreviousYearBaseline",
    "RecentClimatologyBaseline",
    "get_baseline",
]
