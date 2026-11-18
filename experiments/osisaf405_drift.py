"""Sample and integrate operational OSI-405-d Southern Hemisphere drift.

OSI-405-d provides a 48-hour displacement field each day.  For an arbitrary
SAR interval, this module treats each product as a mean velocity estimate and
integrates that velocity only over the overlapping part of its trailing
noon-to-noon day.  The overlapping 48-hour products are therefore not added as
full displacements.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

from experiments.osisaf_routing_prior_audit import daily_segments, utc_datetime


USABLE_OSI405_FLAGS = frozenset({20, 21, 22, 30})
OSI405_FILENAME = (
    "ice_drift_sh_polstere-625_multi-oi_{start}1200-{end}1200.nc"
)


def osi405_path(cache_dir: Path, product_end: datetime) -> Path:
    """Return the local path for the trailing 48-hour product."""
    product_end = utc_datetime(product_end)
    product_start = product_end - timedelta(days=2)
    return cache_dir / OSI405_FILENAME.format(
        start=product_start.strftime("%Y%m%d"),
        end=product_end.strftime("%Y%m%d"),
    )


def required_product_paths(
    start: datetime, end: datetime, cache_dir: Path
) -> list[Path]:
    return [
        osi405_path(cache_dir, segment.product_end)
        for segment in daily_segments(start, end)
    ]


def _sample_variable(
    dataset: xr.Dataset,
    name: str,
    coordinates: dict[str, xr.DataArray],
    method: str,
) -> np.ndarray:
    variable = dataset[name]
    if "time" in variable.dims:
        variable = variable.isel(time=0)
    if method == "linear":
        return np.asarray(variable.interp(coordinates, method=method).values)
    return np.asarray(variable.sel(coordinates, method=method).values)


def sample_osi405_field(
    dataset: xr.Dataset,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """Bilinearly sample displacement, with a valid nearest-cell fallback."""
    coordinates = {
        "xc": xr.DataArray(np.asarray(x_m, dtype=np.float64) / 1_000.0, dims="point"),
        "yc": xr.DataArray(np.asarray(y_m, dtype=np.float64) / 1_000.0, dims="point"),
    }
    names = ("dX", "dY", "uncert_dX_and_dY", "dt0", "dt1")
    linear = {
        name: _sample_variable(dataset, name, coordinates, "linear")
        for name in names
    }
    nearest = {
        name: _sample_variable(dataset, name, coordinates, "nearest")
        for name in names
    }
    flags = _sample_variable(
        dataset, "status_flag", coordinates, "nearest"
    ).astype(np.int16)
    linear_valid = np.logical_and.reduce(
        [np.isfinite(linear[name]) for name in names]
    )
    nearest_valid = np.isin(flags, list(USABLE_OSI405_FLAGS)) & np.logical_and.reduce(
        [np.isfinite(nearest[name]) for name in names]
    )
    fallback = ~linear_valid & nearest_valid
    for name in names:
        linear[name][fallback] = nearest[name][fallback]
    valid = nearest_valid & np.logical_and.reduce(
        [np.isfinite(linear[name]) for name in names]
    )
    linear["flag"] = np.where(valid, flags, -1)
    linear["available"] = valid
    return linear


def _product_duration_seconds(dataset: xr.Dataset) -> float:
    bounds = np.asarray(dataset["time_bnds"].isel(time=0).values)
    duration = (bounds[1] - bounds[0]) / np.timedelta64(1, "s")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("OSI-405-d product has an invalid time interval")
    return float(duration)


def advect_with_osi405(
    source_xy_m: np.ndarray,
    start: datetime,
    end: datetime,
    cache_dir: Path,
    analysis_epsg: int = 3412,
) -> dict[str, np.ndarray]:
    """Integrate trailing 48-hour OSI-405-d velocities over a SAR interval."""
    source_xy_m = np.asarray(source_xy_m, dtype=np.float64)
    if source_xy_m.ndim != 2 or source_xy_m.shape[1] != 2:
        raise ValueError("source positions must have shape (n, 2)")
    segments = daily_segments(start, end)
    if not segments:
        raise ValueError("target time must follow source time")
    first_path = osi405_path(cache_dir, segments[0].product_end)
    with xr.open_dataset(first_path, decode_timedelta=False) as first:
        product_id = str(first.attrs.get("product_id", ""))
        if product_id not in {"OSI-405-c", "OSI-405-d"}:
            raise ValueError(f"unexpected product {product_id!r} in {first_path}")
        osi_crs = CRS.from_cf(first["Polar_Stereographic_Grid"].attrs)
    to_osi = Transformer.from_crs(analysis_epsg, osi_crs, always_xy=True)
    from_osi = Transformer.from_crs(osi_crs, analysis_epsg, always_xy=True)
    current_x, current_y = to_osi.transform(source_xy_m[:, 0], source_xy_m[:, 1])
    current_x = np.asarray(current_x, dtype=np.float64)
    current_y = np.asarray(current_y, dtype=np.float64)
    available = np.ones(len(source_xy_m), dtype=bool)
    uncertainty_m = np.zeros(len(source_xy_m), dtype=np.float64)
    flags: list[list[int]] = [[] for _ in range(len(source_xy_m))]
    product_files: list[str] = []
    for segment in segments:
        path = osi405_path(cache_dir, segment.product_end)
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path, decode_timedelta=False) as dataset:
            sample = sample_osi405_field(dataset, current_x, current_y)
            nominal_duration_s = _product_duration_seconds(dataset)
        product_files.append(path.name)
        step_valid = sample["available"]
        effective_duration_s = (
            nominal_duration_s + sample["dt1"] - sample["dt0"]
        )
        step_valid &= np.isfinite(effective_duration_s) & (effective_duration_s > 0)
        available &= step_valid
        overlap_seconds = (
            segment.overlap_end - segment.overlap_start
        ).total_seconds()
        scale = np.zeros(len(source_xy_m), dtype=np.float64)
        scale[step_valid] = overlap_seconds / effective_duration_s[step_valid]
        indices = np.flatnonzero(step_valid)
        current_x[indices] += sample["dX"][indices] * 1_000.0 * scale[indices]
        current_y[indices] += sample["dY"][indices] * 1_000.0 * scale[indices]
        uncertainty_m[indices] += (
            sample["uncert_dX_and_dY"][indices] * 1_000.0 * scale[indices]
        )
        for index in indices:
            flags[index].append(int(sample["flag"][index]))
    target_x, target_y = from_osi.transform(current_x, current_y)
    target_xy = np.column_stack((target_x, target_y)).astype(np.float64)
    displacement = target_xy - source_xy_m
    displacement[~available] = np.nan
    uncertainty_m[~available] = np.nan
    return {
        "displacement_m": displacement,
        "available": available,
        "uncertainty_m": uncertainty_m,
        "flags": np.asarray(
            [";".join(map(str, values)) if values else "" for values in flags],
            dtype=object,
        ),
        "product_files": np.asarray(product_files, dtype=object),
    }
