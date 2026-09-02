"""Conservative pre-inference gates for projected SAR tiles."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio import Affine
from rasterio.errors import RasterioIOError
from rasterio.transform import rowcol


SIC_FILENAME_RE = re.compile(
    r"^ice_conc_nh_polstere-100_multi_(\d{8})1200\.(?:nc|tif|tiff)$",
    re.IGNORECASE,
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
    transform: object
    crs: object
    source_path: Path
    variable: str


@dataclass(frozen=True)
class OpenWaterEvidence:
    confidently_open: bool
    valid_samples: int
    maximum_sic_percent: float | None


class SicFileIndex:
    """Resolve daily OSI SAF files without rescanning for every image pair."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        paths = (self.root,) if self.root.is_file() else self.root.rglob("*")
        self._by_day: dict[date, Path] = {}
        for path in paths:
            match = SIC_FILENAME_RE.match(path.name)
            if match is None or path.name.startswith("._"):
                continue
            day = datetime.strptime(match.group(1), "%Y%m%d").date()
            previous = self._by_day.get(day)
            if previous is not None and previous != path:
                raise ValueError(f"multiple SIC files found for {day.isoformat()}")
            self._by_day[day] = path.resolve()
        if not self._by_day:
            raise ValueError(f"no supported OSI SAF SIC files found below {self.root}")

    def resolve(
        self, acquisition_time: datetime | None, maximum_age_days: int = 1
    ) -> Path | None:
        if acquisition_time is None:
            return None
        day = acquisition_time.date()
        for age in range(maximum_age_days + 1):
            path = self._by_day.get(day - timedelta(days=age))
            if path is not None:
                return path
        return None


def load_sic_field(path: str | Path) -> SicField:
    """Read OSI SAF SIC with rasterio, preferring the unfiltered variable."""
    source_path = Path(path).resolve()
    stat = source_path.stat()
    return _load_sic_field_cached(
        str(source_path), stat.st_size, stat.st_mtime_ns
    )


@lru_cache(maxsize=8)
def _load_sic_field_cached(
    path: str, _size: int, _mtime_ns: int
) -> SicField:
    source_path = Path(path)
    try:
        with rasterio.open(source_path) as root:
            subdatasets = tuple(root.subdatasets)
        preferred = next(
            (
                item
                for item in subdatasets
                if item.endswith(":ice_conc_unfiltered")
            ),
            None,
        )
        fallback = next(
            (item for item in subdatasets if item.endswith(":ice_conc")), None
        )
        dataset_name = preferred or fallback or str(source_path)
        variable = (
            "ice_conc_unfiltered"
            if preferred is not None
            else "ice_conc" if fallback is not None else source_path.stem
        )
        with rasterio.open(dataset_name) as dataset:
            values = dataset.read(1, masked=True).filled(np.nan).astype(np.float64)
            scale = float(dataset.scales[0]) if dataset.scales else 1.0
            offset = float(dataset.offsets[0]) if dataset.offsets else 0.0
            if scale != 1.0 or offset != 0.0:
                values = values * scale + offset
            if dataset.crs is None:
                raise ValueError(f"SIC raster has no CRS: {source_path}")
            return SicField(
                values,
                dataset.transform,
                dataset.crs,
                source_path,
                variable,
            )
    except RasterioIOError:
        if source_path.suffix.lower() != ".nc":
            raise
        return _load_sic_netcdf4(source_path)


def _load_sic_netcdf4(source_path: Path) -> SicField:
    try:
        from netCDF4 import Dataset
    except ImportError as error:  # pragma: no cover - GPU environment check
        raise RuntimeError(
            "SIC NetCDF requires either a Rasterio NetCDF driver or an existing "
            "netCDF4 runtime"
        ) from error
    with Dataset(source_path) as dataset:
        variable = (
            "ice_conc_unfiltered"
            if "ice_conc_unfiltered" in dataset.variables
            else "ice_conc"
        )
        if variable not in dataset.variables:
            raise ValueError(f"SIC variable not found in {source_path}")
        raw = dataset.variables[variable]
        values = np.ma.asarray(raw[0] if raw.ndim == 3 else raw[:])
        values = values.filled(np.nan).astype(np.float64)
        x_variable = dataset.variables["xc"]
        y_variable = dataset.variables["yc"]
        x = np.asarray(x_variable[:], dtype=np.float64)
        y = np.asarray(y_variable[:], dtype=np.float64)
        if str(x_variable.units) == "km":
            x *= 1_000.0
        if str(y_variable.units) == "km":
            y *= 1_000.0
        if len(x) < 2 or len(y) < 2:
            raise ValueError("SIC coordinate axes require at least two values")
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
        if not np.allclose(np.diff(x), dx) or not np.allclose(np.diff(y), dy):
            raise ValueError("SIC coordinate axes must be regularly spaced")
        crs = str(dataset.variables["Polar_Stereographic_Grid"].proj4_string)
    if values.shape != (len(y), len(x)):
        raise ValueError(
            f"SIC shape {values.shape} does not match y/x axes {(len(y), len(x))}"
        )
    transform = Affine.translation(x[0] - dx / 2, y[0] - dy / 2) * Affine.scale(
        dx, dy
    )
    return SicField(values, transform, crs, source_path, variable)


def sic_file_sha256(path: str | Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    return _sic_file_sha256_cached(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=16)
def _sic_file_sha256_cached(path: str, _size: int, _mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_tile_overlap_gate(
    source_support: np.ndarray,
    target_support: np.ndarray,
    source_center_xy_m: tuple[float, float],
    target_center_xy_m: tuple[float, float],
    pixel_size_m: float,
    maximum_displacement_m: float,
) -> TileValidityGate:
    """Skip only if no supported endpoint can satisfy the speed limit."""
    source_support = np.asarray(source_support, dtype=bool)
    target_support = np.asarray(target_support, dtype=bool)
    if source_support.shape != target_support.shape:
        raise ValueError("source and target support masks must have equal shapes")
    source_pixels = int(source_support.sum())
    target_pixels = int(target_support.sum())
    if source_pixels == 0:
        return TileValidityGate(True, "no_source_core_support", 0, target_pixels, None)
    if target_pixels == 0:
        return TileValidityGate(True, "no_target_support", source_pixels, 0, None)
    source_bounds = _support_bounds(source_support, source_center_xy_m, pixel_size_m)
    target_bounds = _support_bounds(target_support, target_center_xy_m, pixel_size_m)
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


def tile_open_water_evidence(
    field: SicField | None,
    center_xy_m: tuple[float, float],
    extent_m: float,
    analysis_epsg: int,
    threshold_percent: float = 15.0,
    samples_per_axis: int = 5,
) -> OpenWaterEvidence:
    """Require complete SIC evidence before classifying a tile as open water."""
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
        valid == total and maximum is not None and maximum < threshold_percent,
        valid,
        maximum,
    )


def sic_at_points(
    field: SicField, points_xy_m: np.ndarray, analysis_epsg: int
) -> np.ndarray:
    points = np.asarray(points_xy_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("SIC sample points must have shape (n, 2)")
    transformer = _sic_transformer(analysis_epsg, str(field.crs))
    x, y = transformer.transform(points[:, 0], points[:, 1])
    rows, columns = rowcol(field.transform, x, y)
    rows = np.asarray(rows, dtype=int)
    columns = np.asarray(columns, dtype=int)
    inside = (
        (rows >= 0)
        & (rows < field.values_percent.shape[0])
        & (columns >= 0)
        & (columns < field.values_percent.shape[1])
    )
    samples = np.full(len(points), np.nan, dtype=np.float64)
    samples[inside] = field.values_percent[rows[inside], columns[inside]]
    return samples


@lru_cache(maxsize=8)
def _sic_transformer(analysis_epsg: int, sic_crs: str) -> Transformer:
    return Transformer.from_crs(analysis_epsg, sic_crs, always_xy=True)


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
