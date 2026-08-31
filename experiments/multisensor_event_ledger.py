#!/usr/bin/env python3
"""Shared, outcome-neutral ledger helpers for Arctic altimetry comparisons.

The scientific validators keep their existing selection and metric code.  This
module records the contract around those results: coordinates, time reference,
selection counts, support identity, field provenance, and deterministic audit
checkpoints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


LEDGER_SCHEMA = "limosat.multisensor-event-ledger/v1"
SELECTION_STAGES = (
    "candidate_observations",
    "temporally_eligible_observations",
    "product_qc_survivors",
    "spatially_supported_observations",
    "common_method_observations",
    "final_bins",
)


@dataclass(frozen=True)
class DeformationFieldIdentity:
    """Identity and interpolation contract for one deformation field."""

    method: str
    field_id: str
    path: str | None
    sha256: str | None
    vector_count: int
    source_image_id: str
    target_image_id: str
    source_time_utc: str
    target_time_utc: str
    interpolation: str
    boundary_rule: str
    vector_units: str = "metres over the complete SAR interval"
    deformation_units: str = "per day"
    temporal_reference: str = "source-time material coordinate"


def utc_timestamp(value: Any, name: str) -> pd.Timestamp:
    """Return an explicit UTC timestamp and reject naive input."""
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return result.tz_convert("UTC")


def interval_fraction(
    observation_times: Sequence[Any] | pd.Series,
    pair_start: Any,
    pair_end: Any,
) -> np.ndarray:
    """Fractional observation time in a strictly forward SAR interval."""
    start = utc_timestamp(pair_start, "pair_start")
    end = utc_timestamp(pair_end, "pair_end")
    if end <= start:
        raise ValueError("pair_end must be later than pair_start")
    times = pd.DatetimeIndex(pd.to_datetime(observation_times, utc=True))
    return np.asarray((times - start).total_seconds() / (end - start).total_seconds())


def reverse_displacement_vectors(vectors: pd.DataFrame) -> pd.DataFrame:
    """Reverse source-to-target vectors without changing their trajectories."""
    required = {"source_x", "source_y", "dx_m", "dy_m"}
    missing = required.difference(vectors.columns)
    if missing:
        raise ValueError(f"Missing reverse-vector columns: {sorted(missing)}")
    result = vectors.copy()
    result["source_x"] = vectors["source_x"].to_numpy(float) + vectors[
        "dx_m"
    ].to_numpy(float)
    result["source_y"] = vectors["source_y"].to_numpy(float) + vectors[
        "dy_m"
    ].to_numpy(float)
    result["dx_m"] = -vectors["dx_m"].to_numpy(float)
    result["dy_m"] = -vectors["dy_m"].to_numpy(float)
    return result


def assign_along_track_bins(
    along_track_m: Sequence[float] | pd.Series, bin_size_m: float
) -> pd.Series:
    """Assign each finite coordinate to one left-closed along-track bin."""
    if not np.isfinite(bin_size_m) or bin_size_m <= 0:
        raise ValueError("bin_size_m must be finite and positive")
    values = pd.Series(along_track_m, dtype=float)
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    finite = np.isfinite(values)
    result.loc[finite] = np.floor(values.loc[finite] / bin_size_m).astype("int64")
    return result


def exact_common_support(
    observations: pd.DataFrame,
    availability_columns: Sequence[str],
    eligibility: Sequence[bool] | pd.Series | None = None,
) -> pd.Series:
    """Return the exact row identity shared by all named methods."""
    if len(availability_columns) < 2:
        raise ValueError("At least two availability columns are required")
    missing = set(availability_columns).difference(observations.columns)
    if missing:
        raise ValueError(f"Missing availability columns: {sorted(missing)}")
    common = pd.Series(True, index=observations.index, dtype=bool)
    for column in availability_columns:
        common &= observations[column].fillna(False).astype(bool)
    if eligibility is not None:
        eligible = pd.Series(eligibility, index=observations.index).fillna(False)
        common &= eligible.astype(bool)
    return common


def selection_flow_table(
    counts: Mapping[str, int], event_id: str
) -> pd.DataFrame:
    """Build and validate the mandatory ordered selection-flow table."""
    missing = set(SELECTION_STAGES).difference(counts)
    if missing:
        raise ValueError(f"Missing selection stages: {sorted(missing)}")
    values = [int(counts[stage]) for stage in SELECTION_STAGES]
    if any(value < 0 for value in values):
        raise ValueError("Selection counts must be non-negative")
    # Final bins are a different unit and need not be bounded by observations.
    if any(after > before for before, after in zip(values[:4], values[1:5])):
        raise ValueError("Observation selection counts must be non-increasing")
    return pd.DataFrame(
        {
            "event_id": event_id,
            "stage_order": np.arange(len(SELECTION_STAGES), dtype=int),
            "stage": SELECTION_STAGES,
            "count": values,
            "count_unit": ["observations"] * 5 + ["along_track_bins"],
        }
    )


def file_sha256(path: str | Path, block_size: int = 1024 * 1024) -> str:
    """Hash a frozen input or output without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash selected scientific values in a stable row and column order."""
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing hash columns: {sorted(missing)}")
    ordered = frame.loc[:, list(columns)].sort_values(list(columns)).reset_index(
        drop=True
    )
    payload = ordered.to_csv(index=False, float_format="%.17g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def epsg3413_coordinate_contract() -> dict[str, Any]:
    """Machine-readable coordinate, unit, axis, and transform conventions."""
    source = CRS.from_epsg(4326)
    target = CRS.from_epsg(3413)
    forward = Transformer.from_crs(source, target, always_xy=True)
    inverse = Transformer.from_crs(target, source, always_xy=True)
    return {
        "source_crs": source.to_string(),
        "target_crs": target.to_string(),
        "target_crs_name": target.name,
        "axis_order": "always_xy",
        "geographic_input": {
            "x": "longitude",
            "y": "latitude",
            "units": "degrees",
        },
        "projected_coordinates": {
            "x": "EPSG:3413 projected x/easting",
            "y": "EPSG:3413 projected y/northing",
            "units": "metres",
            "shape": "(observation, 2) ordered [x, y]",
            "dtype": "float64",
        },
        "epsg3413_parameters": {
            "latitude_of_origin_degrees_north": 90.0,
            "standard_parallel_degrees_north": 70.0,
            "central_meridian_degrees_east": -45.0,
            "false_easting_m": 0.0,
            "false_northing_m": 0.0,
        },
        "forward_transform": forward.description,
        "inverse_transform": inverse.description,
        "transform_implementation": (
            "pyproj.Transformer.from_crs(..., always_xy=True)"
        ),
        "displacement_shape_dtype_units": {
            "shape": "(observation, 2) ordered [dx, dy]",
            "dtype": "float64",
            "units": "metres over the complete SAR interval",
        },
        "deformation_shape_dtype_units": {
            "shape": "one scalar per observation or along-track bin",
            "dtype": "float64",
            "units": "per day",
        },
    }


def deterministic_checkpoints(
    observations: pd.DataFrame,
    track_column: str,
    common_mask: Sequence[bool] | pd.Series,
    methods: Sequence[str] = ("orb", "aliked"),
    per_track: int = 3,
) -> pd.DataFrame:
    """Choose labelled start/middle/end common-support points per track."""
    if per_track < 1:
        raise ValueError("per_track must be positive")
    if track_column not in observations:
        raise ValueError(f"Missing track column: {track_column}")
    mask = pd.Series(common_mask, index=observations.index).fillna(False).astype(bool)
    selected = observations.loc[mask].copy()
    if selected.empty:
        return pd.DataFrame(columns=["checkpoint_id", track_column])
    order_columns = [track_column]
    if "along_track_m" in selected:
        order_columns.append("along_track_m")
    elif "time_utc" in selected:
        order_columns.append("time_utc")
    selected = selected.sort_values(order_columns)
    rows: list[pd.Series] = []
    for track, group in selected.groupby(track_column, sort=True):
        positions = np.unique(
            np.linspace(0, len(group) - 1, min(per_track, len(group))).round().astype(int)
        )
        for sequence, position in enumerate(positions):
            row = group.iloc[int(position)].copy()
            row["checkpoint_id"] = f"{track}:{sequence + 1}"
            rows.append(row)
    checkpoints = pd.DataFrame(rows)
    base_columns = [
        "checkpoint_id",
        track_column,
        "time_utc",
        "along_track_m",
        "longitude",
        "latitude",
        "laser_x",
        "laser_y",
    ]
    method_columns = [
        f"{method}_{suffix}"
        for method in methods
        for suffix in (
            "interval_fraction",
            "source_x",
            "source_y",
            "target_x",
            "target_y",
            "pair_dx_m",
            "pair_dy_m",
            "drift_to_laser_dx_m",
            "drift_to_laser_dy_m",
            "drift_correction_m",
            "inversion_residual_m",
        )
    ]
    columns = [column for column in base_columns + method_columns if column in checkpoints]
    return checkpoints.loc[:, columns].reset_index(drop=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def write_event_ledger(
    output_dir: str | Path,
    *,
    event_id: str,
    sensor: str,
    product_identifiers: Sequence[str],
    product_time_start_utc: Any,
    product_time_end_utc: Any,
    pair_start_utc: Any,
    pair_end_utc: Any,
    source_image_id: str,
    target_image_id: str,
    inclusion_reason: str,
    analysis_role: str,
    result_status: str,
    selection_counts: Mapping[str, int],
    reporting_resolution_m: float,
    minimum_observations_per_bin: int,
    deformation_fields: Sequence[DeformationFieldIdentity],
    point_ledger_path: str | Path,
    bin_ledger_path: str | Path,
    checkpoints: pd.DataFrame,
    missing_support_reasons: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Write one event ledger, flow table, and numerical checkpoints."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pair_start = utc_timestamp(pair_start_utc, "pair_start_utc")
    pair_end = utc_timestamp(pair_end_utc, "pair_end_utc")
    if pair_end <= pair_start:
        raise ValueError("pair_end_utc must be later than pair_start_utc")
    product_start = utc_timestamp(product_time_start_utc, "product_time_start_utc")
    product_end = utc_timestamp(product_time_end_utc, "product_time_end_utc")
    if product_end < product_start:
        raise ValueError("product_time_end_utc must not precede product_time_start_utc")
    if reporting_resolution_m <= 0 or minimum_observations_per_bin <= 0:
        raise ValueError("Reporting resolution and minimum bin count must be positive")
    if analysis_role not in {
        "development",
        "confirmation",
        "independent_evaluation",
        "insufficient_support_control",
    }:
        raise ValueError(f"Unsupported analysis_role: {analysis_role}")

    flow = selection_flow_table(selection_counts, event_id)
    point_path = Path(point_ledger_path)
    bin_path = Path(bin_ledger_path)
    checkpoint_path = output / "audit_checkpoints.csv"
    flow_path = output / "selection_flow.csv"
    checkpoints.to_csv(checkpoint_path, index=False)
    flow.to_csv(flow_path, index=False)

    ledger = {
        "schema": LEDGER_SCHEMA,
        "event_id": event_id,
        "sensor": sensor,
        "analysis_role": analysis_role,
        "result_status": result_status,
        "candidate_inclusion_reason": inclusion_reason,
        "products": {
            "identifiers": list(product_identifiers),
            "time_start_utc": product_start.isoformat(),
            "time_end_utc": product_end.isoformat(),
        },
        "sar_interval": {
            "source_image_id": str(source_image_id),
            "target_image_id": str(target_image_id),
            "start_utc": pair_start.isoformat(),
            "end_utc": pair_end.isoformat(),
            "elapsed_seconds": float((pair_end - pair_start).total_seconds()),
        },
        "coordinates_and_units": epsg3413_coordinate_contract(),
        "advection": {
            "fraction": "alpha = (observation_time - pair_start) / (pair_end - pair_start)",
            "mapping": "observed_xy = source_xy + alpha * pair_displacement(source_xy)",
            "method": "fixed-point inversion of the direct pairwise displacement field",
            "reference_time": "SAR pair start",
            "outside_interval_rule": "excluded",
        },
        "point_coordinate_columns": {
            "observed": ["laser_x", "laser_y"],
            "observed_time": "time_utc",
            "material_reference": ["<method>_source_x", "<method>_source_y"],
            "pair_end": ["<method>_target_x", "<method>_target_y"],
            "pair_displacement": ["<method>_pair_dx_m", "<method>_pair_dy_m"],
            "advection_to_observation": [
                "<method>_drift_to_laser_dx_m",
                "<method>_drift_to_laser_dy_m",
            ],
            "advection_fraction": "<method>_interval_fraction",
            "numerical_residual_m": "<method>_inversion_residual_m",
            "method_placeholder_values": ["orb", "aliked"],
        },
        "deformation_fields": [asdict(field) for field in deformation_fields],
        "interpolation_and_boundary": {
            "missing_values": "unsupported observations remain unavailable and are not extrapolated",
            "common_support": "logical intersection of the exact same observation row identities",
            "bin_boundary": "left-closed [k*resolution, (k+1)*resolution); exact upper edge enters the next bin",
        },
        "reporting": {
            "resolution_m": float(reporting_resolution_m),
            "minimum_observations_per_bin": int(minimum_observations_per_bin),
        },
        "selection_counts": {stage: int(selection_counts[stage]) for stage in SELECTION_STAGES},
        "missing_support_reasons": dict(missing_support_reasons or {}),
        "files": {
            "point_ledger": {
                "path": str(point_path),
                "sha256": file_sha256(point_path),
            },
            "bin_ledger": {
                "path": str(bin_path),
                "sha256": file_sha256(bin_path),
            },
            "selection_flow": str(flow_path),
            "audit_checkpoints": str(checkpoint_path),
        },
    }
    encoded = json.dumps(_json_safe(ledger), indent=2, allow_nan=False)
    (output / "event_ledger.json").write_text(encoded + "\n")
    return ledger
