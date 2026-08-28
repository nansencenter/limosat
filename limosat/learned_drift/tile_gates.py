"""Conservative pre-matcher gates for projected SAR tiles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from pyproj import Transformer


SIC_FILENAME_RE = re.compile(
    r"^ice_conc_nh_polstere-100_multi_(\d{8})1200\.nc$"
)


@dataclass(frozen=True)
class TileValidityGate:
    skip: bool
    reason: str | None
    source_support_pixels: int
    target_support_pixels: int
    minimum_bounds_distance_m: float | None


@dataclass(frozen=True)
class SicField:
    values_percent: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    crs: str
    source_path: Path
    variable: str


@dataclass(frozen=True)
class OpenWaterEvidence:
    confidently_open: bool
    valid_samples: int
    maximum_sic_percent: float | None


class SicFileIndex:
    """Resolve daily OSI SAF files without rescanning the tree for every pair."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        paths = [self.root] if self.root.is_file() else self.root.rglob("*.nc")
        self._by_day: dict[datetime, Path] = {}
        for path in paths:
            match = SIC_FILENAME_RE.match(path.name)
            if match is None or path.name.startswith("._"):
                continue
            day = datetime.strptime(match.group(1), "%Y%m%d")
            previous = self._by_day.get(day)
            if previous is not None and previous != path:
                raise ValueError(
                    f"multiple OSI SAF SIC files found for {day:%Y-%m-%d}"
                )
            self._by_day[day] = path.resolve()
        if not self._by_day:
            raise ValueError(f"no OSI SAF SIC files found below {self.root}")

    def resolve(
        self, acquisition_time: datetime | None, max_age_days: int = 1
    ) -> Path | None:
        if acquisition_time is None:
            return None
        day = acquisition_time.replace(hour=0, minute=0, second=0, microsecond=0)
        for age in range(max_age_days + 1):
            path = self._by_day.get(day - timedelta(days=age))
            if path is not None:
                return path
        return None

@lru_cache(maxsize=8)
def load_sic_field(path: str | Path) -> SicField:
    """Load one OSI SAF field, preferring unfiltered SIC for safe skipping."""
    from netCDF4 import Dataset

    path = Path(path)
    with Dataset(path) as dataset:
        variable = (
            "ice_conc_unfiltered"
            if "ice_conc_unfiltered" in dataset.variables
            else "ice_conc"
        )
        values = (
            np.ma.asarray(dataset.variables[variable][0])
            .filled(np.nan)
            .astype(np.float32)
        )
        x_variable = dataset.variables["xc"]
        y_variable = dataset.variables["yc"]
        if str(x_variable.units) != "km" or str(y_variable.units) != "km":
            raise ValueError("OSI SAF x/y coordinates must use kilometres")
        x_m = np.asarray(x_variable[:], dtype=float) * 1_000.0
        y_m = np.asarray(y_variable[:], dtype=float) * 1_000.0
        crs = str(dataset.variables["Polar_Stereographic_Grid"].proj4_string)

    if values.shape != (len(y_m), len(x_m)):
        raise ValueError(
            f"SIC shape {values.shape} does not match y/x axes {(len(y_m), len(x_m))}"
        )
    return SicField(values, x_m, y_m, crs, path, variable)


def _support_bounds(
    support: np.ndarray,
    center_xy_m: tuple[float, float],
    pixel_size_m: float,
) -> tuple[float, float, float, float] | None:
    rows, columns = np.nonzero(support)
    if not len(rows):
        return None
    center_px = (support.shape[0] - 1) / 2.0
    return (
        center_xy_m[0] + (columns.min() - center_px - 0.5) * pixel_size_m,
        center_xy_m[1] - (rows.max() - center_px + 0.5) * pixel_size_m,
        center_xy_m[0] + (columns.max() - center_px + 0.5) * pixel_size_m,
        center_xy_m[1] - (rows.min() - center_px - 0.5) * pixel_size_m,
    )


def _bounds_distance_m(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    dx = max(first[0] - second[2], second[0] - first[2], 0.0)
    dy = max(first[1] - second[3], second[1] - first[3], 0.0)
    return float(np.hypot(dx, dy))


def valid_tile_overlap_gate(
    source_support: np.ndarray,
    target_support: np.ndarray,
    source_center_xy_m: tuple[float, float],
    target_center_xy_m: tuple[float, float],
    pixel_size_m: float,
    maximum_displacement_m: float,
) -> TileValidityGate:
    """Skip only when no endpoint pair can satisfy the physical motion gate."""
    source_support = np.asarray(source_support, dtype=bool)
    target_support = np.asarray(target_support, dtype=bool)
    if source_support.shape != target_support.shape:
        raise ValueError("source and target support masks must have equal shapes")

    source_pixels = int(source_support.sum())
    target_pixels = int(target_support.sum())
    if source_pixels == 0:
        return TileValidityGate(
            True, "no_source_core_support", 0, target_pixels, None
        )
    if target_pixels == 0:
        return TileValidityGate(True, "no_target_support", source_pixels, 0, None)

    source_bounds = _support_bounds(
        source_support, source_center_xy_m, pixel_size_m
    )
    target_bounds = _support_bounds(
        target_support, target_center_xy_m, pixel_size_m
    )
    assert source_bounds is not None and target_bounds is not None
    distance_m = _bounds_distance_m(source_bounds, target_bounds)
    if distance_m > maximum_displacement_m:
        return TileValidityGate(
            True,
            "no_physics_reachable_valid_overlap",
            source_pixels,
            target_pixels,
            distance_m,
        )
    return TileValidityGate(False, None, source_pixels, target_pixels, distance_m)


def _nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    values = np.asarray(values, dtype=float)
    spacing = float(axis[1] - axis[0])
    if not np.allclose(np.diff(axis), spacing):
        raise ValueError("SIC coordinate axes must be regularly spaced")
    finite = np.isfinite(values)
    safe_values = np.where(finite, values, axis[0])
    indices = np.rint((safe_values - axis[0]) / spacing).astype(int)
    inside = finite & (indices >= 0) & (indices < len(axis))
    return np.where(inside, indices, -1)


def sic_at_points(
    field: SicField,
    points_xy_m: np.ndarray,
    analysis_epsg: int,
) -> np.ndarray:
    points = np.asarray(points_xy_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("SIC sample points must have shape (n, 2)")
    transformer = _sic_transformer(analysis_epsg, field.crs)
    x_sic, y_sic = transformer.transform(points[:, 0], points[:, 1])
    columns = _nearest_axis_indices(field.x_m, np.asarray(x_sic))
    rows = _nearest_axis_indices(field.y_m, np.asarray(y_sic))
    samples = np.full(len(points), np.nan, dtype=np.float32)
    inside = (rows >= 0) & (columns >= 0)
    samples[inside] = field.values_percent[rows[inside], columns[inside]]
    return samples


@lru_cache(maxsize=8)
def _sic_transformer(analysis_epsg: int, sic_crs: str) -> Transformer:
    return Transformer.from_crs(analysis_epsg, sic_crs, always_xy=True)


def tile_open_water_evidence(
    field: SicField | None,
    center_xy_m: tuple[float, float],
    extent_m: float,
    analysis_epsg: int,
    threshold_percent: float = 15.0,
    samples_per_axis: int = 5,
) -> OpenWaterEvidence:
    """Require complete, unfiltered SIC evidence before calling a tile open water."""
    total = samples_per_axis**2
    if field is None:
        return OpenWaterEvidence(False, 0, None)

    spacing = extent_m / samples_per_axis
    offsets = np.linspace(
        -extent_m / 2.0 + spacing / 2.0,
        extent_m / 2.0 - spacing / 2.0,
        samples_per_axis,
    )
    x, y = np.meshgrid(center_xy_m[0] + offsets, center_xy_m[1] + offsets)
    samples = sic_at_points(
        field, np.column_stack((x.ravel(), y.ravel())), analysis_epsg
    )
    finite = np.isfinite(samples)
    valid = int(finite.sum())
    maximum = float(samples[finite].max()) if valid else None
    return OpenWaterEvidence(
        confidently_open=(
            valid == total
            and maximum is not None
            and maximum < threshold_percent
        ),
        valid_samples=valid,
        maximum_sic_percent=maximum,
    )
