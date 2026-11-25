"""Evaluation utilities for SeasonalViT.

Components:
- visualize: Prediction visualization utilities
- analyze_temporal_importance: Temporal attention analysis

Imports are lazy to avoid errors during incremental development.
"""

__all__ = [
    # Visualization
    "visualize_predictions",
    "visualize_error_map",
    "visualize_temporal_sequence",
    "visualize_training_history",
    "visualize_per_season_metrics",
    "create_prediction_grid",
    # Temporal importance analysis
    "analyze_temporal_importance",
    "TemporalImportanceAnalyzer",
    "TemporalImportanceReport",
    "format_report_summary",
]


def __getattr__(name: str):
    """Lazy import to allow incremental development."""
    # Visualization functions
    if name in (
        "visualize_predictions",
        "visualize_error_map",
        "visualize_temporal_sequence",
        "visualize_training_history",
        "visualize_per_season_metrics",
        "create_prediction_grid",
    ):
        from eval import visualize
        return getattr(visualize, name)

    # Temporal importance analysis functions
    if name in (
        "analyze_temporal_importance",
        "TemporalImportanceAnalyzer",
        "TemporalImportanceReport",
        "format_report_summary",
    ):
        from eval import analyze_temporal_importance
        return getattr(analyze_temporal_importance, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
