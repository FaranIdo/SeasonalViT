"""Tests for temporal importance analysis."""

import pytest
import torch

from model.decoder import TemporalAttentionAnalysis
from eval.analyze_temporal_importance import (
    TemporalImportanceAnalyzer,
    TemporalImportanceReport,
    format_report_summary,
)


class TestTemporalImportanceAnalyzer:
    """Tests for TemporalImportanceAnalyzer."""

    def test_basic_collection(self) -> None:
        """Test basic collection of attention analysis."""
        analyzer = TemporalImportanceAnalyzer(num_timesteps=140, num_seasons=4)

        # Create mock attention analysis
        batch_size = 4
        num_timesteps = 140

        analysis = TemporalAttentionAnalysis(
            attention_weights=torch.rand(batch_size, num_timesteps).softmax(dim=-1),
            effective_history_90=torch.randint(10, 50, (batch_size,)),
            effective_history_50=torch.randint(5, 25, (batch_size,)),
            recent_5yr_weight=torch.rand(batch_size),
            attention_entropy=torch.rand(batch_size) * 3,
            peak_timestep=torch.randint(0, num_timesteps, (batch_size,)),
        )
        target_season = torch.tensor([0, 1, 2, 3])
        lat = torch.tensor([45.0, -30.0, 10.0, 60.0])

        analyzer.update(analysis, target_season, lat)

        assert analyzer.attention_count == batch_size

    def test_compute_report(self) -> None:
        """Test computing final report."""
        analyzer = TemporalImportanceAnalyzer(num_timesteps=140, num_seasons=4)

        # Add multiple batches
        for _ in range(3):
            batch_size = 4
            num_timesteps = 140

            # Create attention weights that sum to 1
            raw_weights = torch.rand(batch_size, num_timesteps)
            attention_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)

            analysis = TemporalAttentionAnalysis(
                attention_weights=attention_weights,
                effective_history_90=torch.randint(10, 50, (batch_size,)),
                effective_history_50=torch.randint(5, 25, (batch_size,)),
                recent_5yr_weight=torch.rand(batch_size),
                attention_entropy=torch.rand(batch_size) * 3,
                peak_timestep=torch.randint(0, num_timesteps, (batch_size,)),
            )
            target_season = torch.tensor([0, 1, 2, 3])
            lat = torch.tensor([45.0, -30.0, 10.0, 60.0])

            analyzer.update(analysis, target_season, lat)

        report = analyzer.compute_report()

        assert isinstance(report, TemporalImportanceReport)
        assert report.num_samples == 12
        assert report.num_timesteps == 140
        assert report.effective_history_90_mean > 0
        assert report.effective_years_90_mean == report.effective_history_90_mean / 4
        assert len(report.mean_attention_by_timestep) == 140
        assert len(report.mean_attention_by_year) == 35

    def test_by_season_breakdown(self) -> None:
        """Test per-season breakdown."""
        analyzer = TemporalImportanceAnalyzer(num_timesteps=140, num_seasons=4)

        # Add data for all seasons
        for season in range(4):
            batch_size = 2
            num_timesteps = 140

            raw_weights = torch.rand(batch_size, num_timesteps)
            attention_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)

            analysis = TemporalAttentionAnalysis(
                attention_weights=attention_weights,
                effective_history_90=torch.tensor([20, 25]),
                effective_history_50=torch.tensor([10, 12]),
                recent_5yr_weight=torch.tensor([0.6, 0.7]),
                attention_entropy=torch.tensor([2.5, 2.8]),
                peak_timestep=torch.tensor([130, 135]),
            )
            target_season = torch.tensor([season, season])
            lat = torch.tensor([45.0, 45.0])

            analyzer.update(analysis, target_season, lat)

        report = analyzer.compute_report()

        assert "spring" in report.by_season
        assert "summer" in report.by_season
        assert "fall" in report.by_season
        assert "winter" in report.by_season

        # Each season should have 2 samples
        for stats in report.by_season.values():
            assert stats["count"] == 2

    def test_by_latitude_band(self) -> None:
        """Test per-latitude band breakdown."""
        analyzer = TemporalImportanceAnalyzer(num_timesteps=140, num_seasons=4)

        # Add data for different latitudes
        latitudes = [10.0, 30.0, 45.0, 60.0, 75.0]  # tropical, subtropical, temperate, subarctic, arctic
        for lat_val in latitudes:
            batch_size = 2
            num_timesteps = 140

            raw_weights = torch.rand(batch_size, num_timesteps)
            attention_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)

            analysis = TemporalAttentionAnalysis(
                attention_weights=attention_weights,
                effective_history_90=torch.tensor([20, 25]),
                effective_history_50=torch.tensor([10, 12]),
                recent_5yr_weight=torch.tensor([0.6, 0.7]),
                attention_entropy=torch.tensor([2.5, 2.8]),
                peak_timestep=torch.tensor([130, 135]),
            )
            target_season = torch.tensor([0, 0])
            lat = torch.tensor([lat_val, lat_val])

            analyzer.update(analysis, target_season, lat)

        report = analyzer.compute_report()

        assert "tropical" in report.by_latitude_band
        assert "subtropical" in report.by_latitude_band
        assert "temperate" in report.by_latitude_band
        assert "subarctic" in report.by_latitude_band
        assert "arctic" in report.by_latitude_band

    def test_by_hemisphere(self) -> None:
        """Test per-hemisphere breakdown."""
        analyzer = TemporalImportanceAnalyzer(num_timesteps=140, num_seasons=4)

        # Add data for both hemispheres
        for lat_val in [45.0, -45.0]:
            batch_size = 2
            num_timesteps = 140

            raw_weights = torch.rand(batch_size, num_timesteps)
            attention_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)

            analysis = TemporalAttentionAnalysis(
                attention_weights=attention_weights,
                effective_history_90=torch.tensor([20, 25]),
                effective_history_50=torch.tensor([10, 12]),
                recent_5yr_weight=torch.tensor([0.6, 0.7]),
                attention_entropy=torch.tensor([2.5, 2.8]),
                peak_timestep=torch.tensor([130, 135]),
            )
            target_season = torch.tensor([0, 0])
            lat = torch.tensor([lat_val, lat_val])

            analyzer.update(analysis, target_season, lat)

        report = analyzer.compute_report()

        assert "northern" in report.by_hemisphere
        assert "southern" in report.by_hemisphere


class TestFormatReport:
    """Tests for report formatting."""

    def test_format_summary(self) -> None:
        """Test that format_report_summary produces valid output."""
        report = TemporalImportanceReport(
            num_samples=100,
            num_timesteps=140,
            effective_history_90_mean=40.0,
            effective_history_90_std=10.0,
            effective_history_90_median=38.0,
            effective_history_50_mean=15.0,
            effective_history_50_std=5.0,
            effective_years_90_mean=10.0,
            effective_years_50_mean=3.75,
            recent_5yr_weight_mean=0.65,
            recent_5yr_weight_std=0.1,
            recent_10yr_weight_mean=0.85,
            attention_entropy_mean=3.5,
            attention_entropy_std=0.5,
            max_possible_entropy=4.94,
            peak_timestep_mean=120.0,
            peak_year_mean=30.0,
            by_season={
                "spring": {"count": 25, "effective_history_90_mean": 38.0, "effective_years_90_mean": 9.5, "recent_5yr_weight_mean": 0.62},
                "summer": {"count": 25, "effective_history_90_mean": 42.0, "effective_years_90_mean": 10.5, "recent_5yr_weight_mean": 0.68},
            },
            by_hemisphere={
                "northern": {"count": 60, "effective_history_90_mean": 39.0, "effective_years_90_mean": 9.75, "recent_5yr_weight_mean": 0.64},
                "southern": {"count": 40, "effective_history_90_mean": 41.0, "effective_years_90_mean": 10.25, "recent_5yr_weight_mean": 0.66},
            },
            by_latitude_band={
                "tropical": {"count": 30, "effective_history_90_mean": 45.0, "effective_years_90_mean": 11.25, "recent_5yr_weight_mean": 0.58},
                "temperate": {"count": 50, "effective_history_90_mean": 38.0, "effective_years_90_mean": 9.5, "recent_5yr_weight_mean": 0.70},
            },
            mean_attention_by_timestep=[0.007] * 140,
            mean_attention_by_year=[0.0286] * 35,
        )

        summary = format_report_summary(report)

        assert "TEMPORAL IMPORTANCE ANALYSIS REPORT" in summary
        assert "100" in summary  # num_samples
        assert "10.0 years" in summary  # effective_years_90
        assert "65.0%" in summary  # recent_5yr_weight
        assert "BY TARGET SEASON" in summary
        assert "BY HEMISPHERE" in summary
        assert "BY LATITUDE BAND" in summary
