"""Shared helpers for the geographic-stratification scripts.

All functions are kept short (<=25 lines of logic) per project conventions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

LAT_BANDS = [
    ("polar_s", -90.0, -66.0),
    ("boreal_s", -66.0, -55.0),
    ("temperate_s", -55.0, -35.0),
    ("subtropical_s", -35.0, -23.0),
    ("tropical", -23.0, 23.0),
    ("subtropical_n", 23.0, 35.0),
    ("temperate_n", 35.0, 55.0),
    ("boreal_n", 55.0, 66.0),
    ("polar_n", 66.0, 90.0),
]

# Coarse latitude bands as named in the plan (collapse N/S together for tables).
LAT_BANDS_PLAN = [
    ("tropical", 0.0, 23.0),
    ("subtropical", 23.0, 35.0),
    ("temperate", 35.0, 55.0),
    ("boreal", 55.0, 66.0),
    ("polar", 66.0, 90.0),
]

SEASONS = ["spring", "summer", "autumn", "winter"]

# Sensor-era bins per the plan (Part 1, output B).
SENSOR_ERAS = [
    ("1990-1998_L5", 1990, 1998),
    ("1999-2002_L5+L7healthy", 1999, 2002),
    ("2003-2012_L5+L7slc", 2003, 2012),
    ("2012-2013_gap", 2012, 2013),
    ("2013-2021_L8+L7slc", 2013, 2021),
    ("2022-2024_L8+L9", 2022, 2024),
]

NODATA_TIERS = [("low", -1.0, 0.05), ("medium", 0.05, 0.15), ("high", 0.15, 1.01)]


def latitude_band(lat: float) -> str:
    abs_lat = abs(lat)
    for name, lo, hi in LAT_BANDS_PLAN:
        if lo <= abs_lat < hi:
            return name
    return "polar"


def signed_latitude_band(lat: float) -> str:
    for name, lo, hi in LAT_BANDS:
        if lo <= lat < hi:
            return name
    return "polar_n"


def nodata_tier(frac: float) -> str:
    for name, lo, hi in NODATA_TIERS:
        if lo <= frac < hi:
            return name
    return "high"


def sample_id_from_filename(filename: str) -> str:
    """`0000029_27d5668_118d4049.tif` -> `0000029`."""
    return Path(filename).stem.split("_")[0]


def load_locations(path: str | Path) -> dict[str, tuple[float, float]]:
    raw = json.loads(Path(path).read_text())
    return {sid: (float(latlon[0]), float(latlon[1])) for sid, latlon in raw.items()}


def load_landcover_stats(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def load_split_files(splits_path: str | Path) -> dict[str, list[str]]:
    raw = json.loads(Path(splits_path).read_text())
    return {k: list(v["files"]) for k, v in raw["splits"].items()}


class KoppenLookup:
    """Wraps a Beck-style Köppen-Geiger raster and returns the 5-class major label.

    The Beck et al. 2018 1-km raster encodes classes 1-30; we collapse them
    into the standard major groups A/B/C/D/E. If no raster is provided, the
    lookup returns "unknown" so callers can degrade gracefully.
    """

    MAJOR_BY_CODE = {
        **{c: "A" for c in (1, 2, 3)},
        **{c: "B" for c in (4, 5, 6, 7)},
        **{c: "C" for c in (8, 9, 10, 11, 12, 13, 14, 15, 16)},
        **{c: "D" for c in (17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)},
        **{c: "E" for c in (29, 30)},
    }

    def __init__(self, raster_path: Optional[str | Path]):
        self.available = False
        self._src = None
        self._array = None
        self._transform = None
        if raster_path is None:
            return
        path = Path(raster_path)
        if not path.exists():
            return
        self._open_raster(path)

    def _open_raster(self, path: Path) -> None:
        import rasterio

        src = rasterio.open(path)
        self._src = src
        self._transform = src.transform
        self._array = src.read(1)
        self.available = True

    def lookup(self, lat: float, lon: float) -> str:
        if not self.available or self._array is None or self._transform is None:
            return "unknown"
        col, row = ~self._transform * (lon, lat)
        c, r = int(round(col)), int(round(row))
        h, w = self._array.shape
        if not (0 <= r < h and 0 <= c < w):
            return "unknown"
        code = int(self._array[r, c])
        if code <= 0:
            return "unknown"
        return self.MAJOR_BY_CODE.get(code, "unknown")

    def lookup_many(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        if not self.available or self._array is None or self._transform is None:
            return np.array(["unknown"] * len(lats), dtype=object)
        inv = ~self._transform
        cols = np.empty_like(lats, dtype=np.float64)
        rows = np.empty_like(lats, dtype=np.float64)
        for i, (la, lo) in enumerate(zip(lats, lons)):
            x, y = inv * (lo, la)
            cols[i], rows[i] = x, y
        return self._codes_to_labels(rows, cols)

    def _codes_to_labels(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        assert self._array is not None
        h, w = self._array.shape
        ri = np.clip(np.round(rows).astype(np.int32), 0, h - 1)
        ci = np.clip(np.round(cols).astype(np.int32), 0, w - 1)
        codes = self._array[ri, ci]
        in_bounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        out = np.empty(len(codes), dtype=object)
        for i, code in enumerate(codes):
            if not in_bounds[i] or code <= 0:
                out[i] = "unknown"
            else:
                out[i] = self.MAJOR_BY_CODE.get(int(code), "unknown")
        return out


def parse_band_label(label: str) -> tuple[int, str]:
    """`1990_spring` -> (1990, 'spring')."""
    year_str, season = label.split("_")
    return int(year_str), season


def era_for_year(year: int) -> str:
    for name, lo, hi in SENSOR_ERAS:
        if lo <= year <= hi:
            return name
    return "other"
