"""Tests for scripts/geo_strata/_geo_strata_helpers.py.

These cover the per-tile gather logic: per-tile
metric aggregation, NoData-aware temporal-std computation, and TSV float
formatting. They run on CPU in milliseconds and have no model / shard /
GPU dependency.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# Make scripts/geo_strata importable for the helper module under test.
GEO_STRATA_DIR = Path(__file__).resolve().parents[1] / "scripts" / "geo_strata"
sys.path.insert(0, str(GEO_STRATA_DIR))

from _geo_strata_helpers import (  # type: ignore[import-not-found]  # noqa: E402
    SEASON_NAMES,
    format_tsv_value,
    per_tile_stats_one_season,
    sigma_tile_temporal,
)


# ---------------------------------------------------------------------------
# per_tile_stats_one_season
# ---------------------------------------------------------------------------


def test_per_tile_stats_uniform_positive_error():
    pred = torch.full((1, 2, 2), 0.5)
    gt = torch.full((1, 2, 2), 0.4)
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    out = per_tile_stats_one_season(pred, gt, valid)
    assert math.isclose(out["mae"][0].item(), 0.1, abs_tol=1e-6)
    assert math.isclose(out["rmse"][0].item(), 0.1, abs_tol=1e-6)
    assert math.isclose(out["bias"][0].item(), 0.1, abs_tol=1e-6)
    assert math.isclose(out["pred_mean"][0].item(), 0.5, abs_tol=1e-6)
    assert math.isclose(out["gt_mean"][0].item(), 0.4, abs_tol=1e-6)
    assert out["n_valid"][0].item() == 4


def test_per_tile_stats_mixed_signs_with_nodata():
    # Tile has 3 valid pixels: +0.1, -0.1, 0.0; one NoData pixel.
    pred = torch.tensor([[[0.6, 0.4], [0.5, 0.5]]])
    gt = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
    valid = torch.tensor([[[True, True], [True, False]]])
    out = per_tile_stats_one_season(pred, gt, valid)

    expected_mae = (0.1 + 0.1 + 0.0) / 3
    expected_rmse = math.sqrt((0.01 + 0.01 + 0.0) / 3)
    assert math.isclose(out["mae"][0].item(), expected_mae, abs_tol=1e-6)
    assert math.isclose(out["bias"][0].item(), 0.0, abs_tol=1e-6)
    assert math.isclose(out["rmse"][0].item(), expected_rmse, abs_tol=1e-6)
    assert out["n_valid"][0].item() == 3


def test_per_tile_stats_zero_valid_yields_nan():
    pred = torch.zeros(1, 2, 2)
    gt = torch.zeros(1, 2, 2)
    valid = torch.zeros(1, 2, 2, dtype=torch.bool)
    out = per_tile_stats_one_season(pred, gt, valid)
    assert math.isnan(out["mae"][0].item())
    assert math.isnan(out["rmse"][0].item())
    assert math.isnan(out["bias"][0].item())
    assert math.isnan(out["pred_mean"][0].item())
    assert math.isnan(out["gt_mean"][0].item())
    assert out["n_valid"][0].item() == 0


def test_per_tile_stats_batched_independence():
    # Tile 0: clean +0.1 bias. Tile 1: zero error. Verify per-tile isolation.
    pred = torch.stack([torch.full((2, 2), 0.6), torch.full((2, 2), 0.5)]).unsqueeze(0).reshape(2, 2, 2)
    gt = torch.full((2, 2, 2), 0.5)
    valid = torch.ones(2, 2, 2, dtype=torch.bool)
    out = per_tile_stats_one_season(pred, gt, valid)
    assert math.isclose(out["mae"][0].item(), 0.1, abs_tol=1e-6)
    assert math.isclose(out["mae"][1].item(), 0.0, abs_tol=1e-6)
    assert out["n_valid"][0].item() == 4
    assert out["n_valid"][1].item() == 4


# ---------------------------------------------------------------------------
# sigma_tile_temporal
# ---------------------------------------------------------------------------


def test_sigma_tile_temporal_constant_series_is_zero():
    images = torch.full((1, 4, 2, 2), 0.5)
    nodata = torch.zeros(1, 4, 2, 2, dtype=torch.bool)
    sigma = sigma_tile_temporal(images, nodata)
    assert math.isclose(sigma[0].item(), 0.0, abs_tol=1e-6)


def test_sigma_tile_temporal_two_value_known_std():
    # One tile, T=2, H=W=1. Values [0.0, 1.0]. Sample sigma = sqrt(0.5).
    images = torch.zeros(1, 2, 1, 1)
    images[0, 1, 0, 0] = 1.0
    nodata = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
    sigma = sigma_tile_temporal(images, nodata)
    assert math.isclose(sigma[0].item(), math.sqrt(0.5), abs_tol=1e-6)


def test_sigma_tile_temporal_ignores_nodata_value():
    # Inject -9999 NoData "values" but mark them invalid; result must equal
    # the std of the remaining valid pixels (which are constant, so 0).
    images = torch.full((1, 4, 2, 2), 0.5)
    images[0, 0, 0, 0] = -9999.0
    nodata = torch.zeros(1, 4, 2, 2, dtype=torch.bool)
    nodata[0, 0, 0, 0] = True
    sigma = sigma_tile_temporal(images, nodata)
    assert math.isclose(sigma[0].item(), 0.0, abs_tol=1e-6)


def test_sigma_tile_temporal_all_nodata_yields_nan():
    images = torch.zeros(1, 2, 2, 2)
    nodata = torch.ones(1, 2, 2, 2, dtype=torch.bool)
    sigma = sigma_tile_temporal(images, nodata)
    assert math.isnan(sigma[0].item())


def test_sigma_tile_temporal_one_valid_yields_nan():
    # Sample std is undefined for n=1 (denominator is 0).
    images = torch.zeros(1, 2, 1, 1)
    nodata = torch.tensor([[[[True]], [[False]]]], dtype=torch.bool)
    sigma = sigma_tile_temporal(images, nodata)
    assert math.isnan(sigma[0].item())


def test_sigma_tile_temporal_rejects_out_of_range_when_mask_misses_it():
    # In practice, shard nodata_masks occasionally miss individual -9999
    # sentinels. A single missed pixel out of 8M would produce sigma ~3.5
    # without value-based filtering. Verify the helper is robust to that.
    images = torch.full((1, 4, 4, 4), 0.5)
    images[0, 0, 0, 0] = -9999.0
    nodata = torch.zeros(1, 4, 4, 4, dtype=torch.bool)  # mask DOES NOT flag the -9999
    sigma = sigma_tile_temporal(images, nodata)
    assert torch.isfinite(sigma[0])
    assert sigma[0].item() < 1.0  # physically valid NDVI std
    assert math.isclose(sigma[0].item(), 0.0, abs_tol=1e-6)


def test_sigma_tile_temporal_fp16_inputs_safe():
    # gather passes fp16 images; helper must upcast and not overflow on
    # NoData -9999 placeholders.
    images = torch.full((1, 4, 2, 2), 0.5, dtype=torch.float16)
    images[0, 0, 0, 0] = -9999.0
    nodata = torch.zeros(1, 4, 2, 2, dtype=torch.bool)
    nodata[0, 0, 0, 0] = True
    sigma = sigma_tile_temporal(images, nodata)
    assert torch.isfinite(sigma[0])
    assert math.isclose(sigma[0].item(), 0.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# format_tsv_value
# ---------------------------------------------------------------------------


def test_format_tsv_value_floats_and_nan():
    assert format_tsv_value(0.123456789) == "0.123457"
    assert format_tsv_value(float("nan")) == "nan"
    assert format_tsv_value(1.0) == "1.000000"


def test_format_tsv_value_int_and_str():
    assert format_tsv_value(5) == "5"
    assert format_tsv_value("0000029") == "0000029"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_season_names_match_target_indices_2022():
    # Drift guard: the gather script assumes target_idx % 4 maps to spring/
    # summer/fall/winter in this exact order.
    assert SEASON_NAMES == ["spring", "summer", "fall", "winter"]
    target_indices_2022 = [128, 129, 130, 131]
    seasons_from_idx = [SEASON_NAMES[i % 4] for i in target_indices_2022]
    assert seasons_from_idx == SEASON_NAMES


# ---------------------------------------------------------------------------
# Gather row emit + TSV write (end-to-end on a stub dataset)
# ---------------------------------------------------------------------------


from dataclasses import dataclass  # noqa: E402

from geo_strata_gather import emit_rows, write_tsv  # type: ignore[import-not-found]  # noqa: E402


@dataclass
class _StubDataset:
    """Minimal stand-in for InMemoryNDVIDataset; only `.files` is needed by emit_rows."""
    files: list


def _make_batch_out(n: int) -> dict:
    """Construct a fake `process_batch` output for `n` tiles with deterministic values."""
    per_season = {}
    for s_idx, season in enumerate(SEASON_NAMES):
        per_season[season] = {
            "mae": torch.full((n,), 0.10 + s_idx * 0.01),
            "rmse": torch.full((n,), 0.20 + s_idx * 0.01),
            "bias": torch.full((n,), 0.03 + s_idx * 0.01),
            "pred_mean": torch.full((n,), 0.50 + s_idx * 0.01),
            "gt_mean": torch.full((n,), 0.45 + s_idx * 0.01),
            "n_valid": torch.full((n,), 65000, dtype=torch.long),
        }
    return {
        "lat": torch.tensor([10.0 + i for i in range(n)]),
        "lon": torch.tensor([-50.0 + i for i in range(n)]),
        "sigma": torch.tensor([0.12 + 0.01 * i for i in range(n)]),
        "per_season": per_season,
    }


def test_emit_rows_extracts_sample_id_and_metadata():
    dataset = _StubDataset(files=[
        "0000029_27d5668_118d4049.tif",
        "0000069_26d1026_-104d3181.tif",
    ])
    batch_out = _make_batch_out(n=2)

    rows = emit_rows(batch_out, dataset, sample_offset=0, split_name="ood_s")

    assert len(rows) == 2
    assert rows[0]["sample_id"] == "0000029"
    assert rows[1]["sample_id"] == "0000069"
    assert rows[0]["filename"] == "0000029_27d5668_118d4049.tif"
    assert rows[0]["split"] == "ood_s"
    assert rows[1]["split"] == "ood_s"
    assert math.isclose(rows[0]["lat"], 10.0, abs_tol=1e-6)
    assert math.isclose(rows[1]["lon"], -49.0, abs_tol=1e-6)
    assert math.isclose(rows[0]["sigma_tile"], 0.12, abs_tol=1e-6)
    assert math.isclose(rows[0]["mae_spring"], 0.10, abs_tol=1e-6)
    assert math.isclose(rows[0]["bias_winter"], 0.06, abs_tol=1e-6)
    assert rows[0]["n_valid_summer"] == 65000


def test_emit_rows_uses_sample_offset_for_filename_lookup():
    dataset = _StubDataset(files=[
        "0000001_0_0.tif",
        "0000002_0_0.tif",
        "0000003_0_0.tif",
        "0000004_0_0.tif",
    ])
    batch_out = _make_batch_out(n=2)

    rows = emit_rows(batch_out, dataset, sample_offset=2, split_name="val_iid")

    assert [r["sample_id"] for r in rows] == ["0000003", "0000004"]
    assert all(r["split"] == "val_iid" for r in rows)


def test_write_tsv_round_trips(tmp_path):
    dataset = _StubDataset(files=[
        "0000029_27d5668_118d4049.tif",
        "0000069_26d1026_-104d3181.tif",
    ])
    rows = emit_rows(
        _make_batch_out(n=2), dataset, sample_offset=0, split_name="ood_t",
    )
    output = tmp_path / "out.tsv"

    write_tsv(rows, output)
    text = output.read_text()
    lines = text.strip().split("\n")

    header = lines[0].split("\t")
    assert header[:6] == [
        "sample_id", "filename", "split", "lat", "lon", "sigma_tile",
    ]
    assert "mae_spring" in header
    assert "n_valid_winter" in header
    assert len(lines) == 1 + 2
    first_data = lines[1].split("\t")
    assert first_data[0] == "0000029"
    assert first_data[2] == "ood_t"


def test_write_tsv_handles_empty_rows(tmp_path):
    output = tmp_path / "empty.tsv"
    write_tsv([], output)
    assert output.exists()
    assert output.read_text() == ""
