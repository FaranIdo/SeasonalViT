"""Temporal importance analysis for SeasonalViT.

Analyzes attention weights from the decoder's temporal pooling to understand
how far back in history the model uses for predictions. This provides insights
for model interpretability and potential simplification.

Usage:
    python -m eval.analyze_temporal_importance \
        --checkpoint path/to/checkpoint.pt \
        --data-dir path/to/data \
        --output-dir path/to/output

Output:
    - temporal_importance_report.json: Full analysis results
    - temporal_importance_summary.txt: Human-readable summary
    - temporal_attention_heatmap.png: Visualization (if matplotlib available)
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.config import SeasonalViTConfig
from model.decoder import TemporalAttentionAnalysis
from model.seasonal_vit import SeasonalViT

logger = logging.getLogger(__name__)


@dataclass
class TemporalImportanceReport:
    """Aggregated temporal importance analysis results."""

    # Overall statistics
    num_samples: int = 0
    num_timesteps: int = 140

    # Effective history (timesteps from most recent needed for X% attention)
    effective_history_90_mean: float = 0.0
    effective_history_90_std: float = 0.0
    effective_history_90_median: float = 0.0
    effective_history_50_mean: float = 0.0
    effective_history_50_std: float = 0.0

    # Converted to years (divide by 4 seasons)
    effective_years_90_mean: float = 0.0
    effective_years_50_mean: float = 0.0

    # Recent history weight
    recent_5yr_weight_mean: float = 0.0
    recent_5yr_weight_std: float = 0.0
    recent_10yr_weight_mean: float = 0.0

    # Attention distribution
    attention_entropy_mean: float = 0.0
    attention_entropy_std: float = 0.0
    max_possible_entropy: float = 0.0  # log(T) for uniform distribution

    # Peak attention
    peak_timestep_mean: float = 0.0
    peak_year_mean: float = 0.0  # Converted to years from start

    # Per-group breakdowns
    by_season: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_hemisphere: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_latitude_band: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Raw attention weights (averaged across samples)
    mean_attention_by_timestep: List[float] = field(default_factory=list)
    mean_attention_by_year: List[float] = field(default_factory=list)


class TemporalImportanceAnalyzer:
    """Collects and aggregates temporal attention analysis across samples."""

    def __init__(self, num_timesteps: int = 140, num_seasons: int = 4):
        self.num_timesteps = num_timesteps
        self.num_seasons = num_seasons
        self.num_years = num_timesteps // num_seasons

        # Storage for raw values
        self.effective_history_90: List[float] = []
        self.effective_history_50: List[float] = []
        self.recent_5yr_weight: List[float] = []
        self.attention_entropy: List[float] = []
        self.peak_timestep: List[float] = []

        # Per-timestep attention weights
        self.attention_weights_sum = np.zeros(num_timesteps)
        self.attention_count = 0

        # Metadata for grouping
        self.target_seasons: List[int] = []
        self.latitudes: List[float] = []
        self.hemispheres: List[int] = []  # 0=south, 1=north

    def update(
        self,
        analysis: TemporalAttentionAnalysis,
        target_season: torch.Tensor,
        lat: torch.Tensor,
    ) -> None:
        """Update analyzer with a batch of results.

        Args:
            analysis: TemporalAttentionAnalysis from decoder.
            target_season: Target season indices (B,).
            lat: Latitude values (B,).
        """
        batch_size = analysis.effective_history_90.shape[0]

        # Convert to python lists and store (avoid tensor.numpy() ABI issues)
        self.effective_history_90.extend(
            analysis.effective_history_90.cpu().tolist()
        )
        self.effective_history_50.extend(
            analysis.effective_history_50.cpu().tolist()
        )
        self.recent_5yr_weight.extend(
            analysis.recent_5yr_weight.cpu().tolist()
        )
        self.attention_entropy.extend(
            analysis.attention_entropy.cpu().tolist()
        )
        self.peak_timestep.extend(
            analysis.peak_timestep.cpu().tolist()
        )

        # Accumulate attention weights
        weights = analysis.attention_weights.cpu()  # (B, T)
        self.attention_weights_sum += np.array(weights.tolist()).sum(axis=0)
        self.attention_count += batch_size

        # Store metadata
        self.target_seasons.extend(target_season.cpu().tolist())
        self.latitudes.extend(lat.cpu().tolist())
        self.hemispheres.extend((lat >= 0).int().cpu().tolist())

    def compute_report(self) -> TemporalImportanceReport:
        """Compute final aggregated report."""
        if self.attention_count == 0:
            raise ValueError("No samples collected")

        report = TemporalImportanceReport()
        report.num_samples = self.attention_count
        report.num_timesteps = self.num_timesteps

        # Overall statistics
        eff_90 = np.array(self.effective_history_90)
        eff_50 = np.array(self.effective_history_50)
        recent_5yr = np.array(self.recent_5yr_weight)
        entropy = np.array(self.attention_entropy)
        peak = np.array(self.peak_timestep)

        report.effective_history_90_mean = float(eff_90.mean())
        report.effective_history_90_std = float(eff_90.std())
        report.effective_history_90_median = float(np.median(eff_90))
        report.effective_history_50_mean = float(eff_50.mean())
        report.effective_history_50_std = float(eff_50.std())

        report.effective_years_90_mean = report.effective_history_90_mean / 4
        report.effective_years_50_mean = report.effective_history_50_mean / 4

        report.recent_5yr_weight_mean = float(recent_5yr.mean())
        report.recent_5yr_weight_std = float(recent_5yr.std())

        # Compute 10-year weight from raw attention
        mean_attn = self.attention_weights_sum / self.attention_count
        report.recent_10yr_weight_mean = float(mean_attn[-40:].sum())

        report.attention_entropy_mean = float(entropy.mean())
        report.attention_entropy_std = float(entropy.std())
        report.max_possible_entropy = float(np.log(self.num_timesteps))

        report.peak_timestep_mean = float(peak.mean())
        report.peak_year_mean = float(peak.mean() / 4)

        # Mean attention by timestep and year
        report.mean_attention_by_timestep = mean_attn.tolist()
        year_attn = mean_attn.reshape(self.num_years, self.num_seasons).sum(axis=1)
        report.mean_attention_by_year = year_attn.tolist()

        # Per-group breakdowns
        report.by_season = self._compute_by_group(
            self.target_seasons,
            {0: "spring", 1: "summer", 2: "fall", 3: "winter"},
        )
        report.by_hemisphere = self._compute_by_group(
            self.hemispheres,
            {0: "southern", 1: "northern"},
        )
        report.by_latitude_band = self._compute_by_latitude_band()

        return report

    def _compute_by_group(
        self, groups: List[int], group_names: Dict[int, str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute statistics grouped by a categorical variable."""
        result = {}
        groups_arr = np.array(groups)
        eff_90 = np.array(self.effective_history_90)
        recent_5yr = np.array(self.recent_5yr_weight)

        for group_id, name in group_names.items():
            mask = groups_arr == group_id
            if mask.sum() == 0:
                continue
            result[name] = {
                "count": int(mask.sum()),
                "effective_history_90_mean": float(eff_90[mask].mean()),
                "effective_years_90_mean": float(eff_90[mask].mean() / 4),
                "recent_5yr_weight_mean": float(recent_5yr[mask].mean()),
            }
        return result

    def _compute_by_latitude_band(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics by latitude band."""
        lat_arr = np.abs(np.array(self.latitudes))
        eff_90 = np.array(self.effective_history_90)
        recent_5yr = np.array(self.recent_5yr_weight)

        bands = {
            "tropical": (0, 23.5),
            "subtropical": (23.5, 35),
            "temperate": (35, 55),
            "subarctic": (55, 66.5),
            "arctic": (66.5, 90),
        }

        result = {}
        for name, (lat_min, lat_max) in bands.items():
            mask = (lat_arr >= lat_min) & (lat_arr < lat_max)
            if mask.sum() == 0:
                continue
            result[name] = {
                "count": int(mask.sum()),
                "effective_history_90_mean": float(eff_90[mask].mean()),
                "effective_years_90_mean": float(eff_90[mask].mean() / 4),
                "recent_5yr_weight_mean": float(recent_5yr[mask].mean()),
            }
        return result


def analyze_temporal_importance(
    model: SeasonalViT,
    dataloader: DataLoader,
    device: torch.device,
) -> TemporalImportanceReport:
    """Run temporal importance analysis on a dataset.

    Args:
        model: Trained SeasonalViT model.
        dataloader: DataLoader for analysis dataset.
        device: Device for computation.

    Returns:
        TemporalImportanceReport with aggregated results.
    """
    model.eval()
    model.to(device)

    config = model.config
    analyzer = TemporalImportanceAnalyzer(
        num_timesteps=config.num_timesteps,
        num_seasons=config.num_seasons,
    )

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            season_ids = batch["season_ids"].to(device)
            year_indices = batch["year_indices"].to(device)
            lat = batch["lat"].to(device)
            lon = batch["lon"].to(device)

            # Split input and target
            target_season = season_ids[:, -1]
            input_images = images[:, :-1, :, :]
            input_season_ids = season_ids[:, :-1]
            input_year_indices = year_indices[:, :-1]

            # Forward pass with attention analysis
            output = model(
                input_images,
                input_season_ids,
                input_year_indices,
                lat,
                lon,
                target_season,
                return_attention_analysis=True,
            )

            if "attention_analysis" in output:
                analyzer.update(
                    output["attention_analysis"],
                    target_season,
                    lat,
                )

    return analyzer.compute_report()


def format_report_summary(report: TemporalImportanceReport) -> str:
    """Format report as human-readable summary."""
    lines = [
        "=" * 60,
        "TEMPORAL IMPORTANCE ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Samples analyzed: {report.num_samples:,}",
        f"Timesteps per sample: {report.num_timesteps} ({report.num_timesteps // 4} years)",
        "",
        "-" * 60,
        "EFFECTIVE HISTORY LENGTH",
        "-" * 60,
        "",
        "How far back the model looks for 90% of attention mass:",
        f"  Mean:   {report.effective_history_90_mean:.1f} timesteps = {report.effective_years_90_mean:.1f} years",
        f"  Median: {report.effective_history_90_median:.1f} timesteps",
        f"  Std:    {report.effective_history_90_std:.1f} timesteps",
        "",
        "For 50% of attention mass:",
        f"  Mean:   {report.effective_history_50_mean:.1f} timesteps = {report.effective_years_50_mean:.1f} years",
        "",
        "-" * 60,
        "RECENT HISTORY WEIGHT",
        "-" * 60,
        "",
        f"Last 5 years:  {report.recent_5yr_weight_mean:.1%} of attention (std: {report.recent_5yr_weight_std:.1%})",
        f"Last 10 years: {report.recent_10yr_weight_mean:.1%} of attention",
        "",
        "-" * 60,
        "ATTENTION DISTRIBUTION",
        "-" * 60,
        "",
        f"Entropy: {report.attention_entropy_mean:.2f} (max possible: {report.max_possible_entropy:.2f})",
        f"  Ratio: {report.attention_entropy_mean / report.max_possible_entropy:.1%} of maximum",
        f"  (Higher = more uniform, lower = more concentrated)",
        "",
        f"Peak attention timestep: {report.peak_timestep_mean:.1f} = year {report.peak_year_mean:.1f}",
        "",
    ]

    # Per-season breakdown
    if report.by_season:
        lines.extend([
            "-" * 60,
            "BY TARGET SEASON",
            "-" * 60,
            "",
        ])
        for season, stats in report.by_season.items():
            lines.append(
                f"  {season.capitalize():8s}: {stats['effective_years_90_mean']:.1f} years (90%), "
                f"{stats['recent_5yr_weight_mean']:.1%} recent-5yr (n={stats['count']:,})"
            )
        lines.append("")

    # Per-hemisphere breakdown
    if report.by_hemisphere:
        lines.extend([
            "-" * 60,
            "BY HEMISPHERE",
            "-" * 60,
            "",
        ])
        for hemi, stats in report.by_hemisphere.items():
            lines.append(
                f"  {hemi.capitalize():10s}: {stats['effective_years_90_mean']:.1f} years (90%), "
                f"{stats['recent_5yr_weight_mean']:.1%} recent-5yr (n={stats['count']:,})"
            )
        lines.append("")

    # Per-latitude band breakdown
    if report.by_latitude_band:
        lines.extend([
            "-" * 60,
            "BY LATITUDE BAND",
            "-" * 60,
            "",
        ])
        for band, stats in report.by_latitude_band.items():
            lines.append(
                f"  {band.capitalize():12s}: {stats['effective_years_90_mean']:.1f} years (90%), "
                f"{stats['recent_5yr_weight_mean']:.1%} recent-5yr (n={stats['count']:,})"
            )
        lines.append("")

    lines.extend([
        "=" * 60,
        "INTERPRETATION",
        "=" * 60,
        "",
        f"The model uses approximately {report.effective_years_90_mean:.0f} years of history",
        f"for 90% of its attention, with {report.recent_5yr_weight_mean:.0%} focused on",
        "the most recent 5 years.",
        "",
    ])

    if report.effective_years_90_mean < 15:
        lines.append(
            "SUGGESTION: Consider reducing input history to speed up training/inference."
        )
    elif report.effective_years_90_mean > 25:
        lines.append(
            "NOTE: Model uses most of the available 35-year history."
        )

    lines.append("")

    return "\n".join(lines)


def save_attention_heatmap(
    report: TemporalImportanceReport,
    output_path: Path,
) -> None:
    """Save attention heatmap visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping heatmap")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # By timestep
    ax1 = axes[0]
    timesteps = np.arange(len(report.mean_attention_by_timestep))
    ax1.bar(timesteps, report.mean_attention_by_timestep, width=1.0, alpha=0.7)
    ax1.set_xlabel("Timestep (0 = 1990 Spring)")
    ax1.set_ylabel("Mean Attention Weight")
    ax1.set_title("Temporal Attention by Timestep")
    ax1.axvline(
        x=len(timesteps) - report.effective_history_90_mean,
        color="red",
        linestyle="--",
        label=f"90% threshold ({report.effective_years_90_mean:.1f} years)",
    )
    ax1.legend()

    # By year
    ax2 = axes[1]
    years = np.arange(len(report.mean_attention_by_year)) + 1990
    ax2.bar(years, report.mean_attention_by_year, width=0.8, alpha=0.7)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Mean Attention Weight (sum of 4 seasons)")
    ax2.set_title("Temporal Attention by Year")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved attention heatmap to {output_path}")


def main():
    """Main entry point for temporal importance analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze temporal importance in SeasonalViT predictions"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for analysis",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to analyze (for quick testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu/mps)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", SeasonalViTConfig())
    model = SeasonalViT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Create dataloader
    # Note: User needs to implement dataset loading based on their setup
    logger.info(f"Loading data from {args.data_dir}")
    try:
        from train.dataset import NDVITimeSeriesDataset, create_dataloader
        dataset = NDVITimeSeriesDataset(args.data_dir)
        if args.max_samples:
            from torch.utils.data import Subset
            indices = list(range(min(args.max_samples, len(dataset))))
            dataset = Subset(dataset, indices)
        dataloader = create_dataloader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
    except ImportError:
        logger.error(
            "Could not import dataset. Please ensure train/dataset.py exists "
            "or modify this script to use your dataset implementation."
        )
        return

    # Run analysis
    logger.info("Running temporal importance analysis...")
    report = analyze_temporal_importance(model, dataloader, device)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    json_path = args.output_dir / "temporal_importance_report.json"
    with open(json_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info(f"Saved JSON report to {json_path}")

    # Save text summary
    summary = format_report_summary(report)
    txt_path = args.output_dir / "temporal_importance_summary.txt"
    with open(txt_path, "w") as f:
        f.write(summary)
    logger.info(f"Saved text summary to {txt_path}")

    # Print summary
    print(summary)

    # Save heatmap
    heatmap_path = args.output_dir / "temporal_attention_heatmap.png"
    save_attention_heatmap(report, heatmap_path)


if __name__ == "__main__":
    main()
