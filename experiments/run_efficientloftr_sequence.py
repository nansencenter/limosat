#!/usr/bin/env python3
"""Run optimized EfficientLoFTR on a frozen adjacent-image SAR chain.

The matcher receives one north-up source tile and one target tile per source
core. Target tiles can be colocated or shifted by the local velocity sampled
from the immediately preceding accepted field. All downstream field and
trajectory stages are matcher-neutral ``limosat.learned_drift`` components.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import torch
from shapely.geometry import MultiPoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from limosat.learned_drift.config import EfficientLoFTRConfig
from limosat.learned_drift.efficientloftr import (
    load_optimized_model,
    matcher_inputs,
    run_optimized_matcher,
    source_core_mask,
    speed_limit_mask,
    synchronize,
    valid_endpoints,
    valid_support,
)
from limosat.learned_drift.features import tile_layout
from limosat.learned_drift.field import (
    estimate_field,
    estimate_queries,
    reject_folds,
    topology_summary,
)
from limosat.learned_drift.imagery import north_up_patch, projected_footprint
from limosat.learned_drift.routing import (
    coarse_phase_translation,
    preceding_field_shifts,
)
from limosat.learned_drift.trajectory import advect_trajectories
from limosat.learned_drift.trajectory_graph import (
    FieldEdge,
    advect_trajectory_graph,
)
from limosat.learned_drift.tile_gates import (
    OpenWaterEvidence,
    SicField,
    SicFileIndex,
    load_sic_field,
    tile_open_water_evidence,
    valid_tile_overlap_gate,
)
from limosat.learned_drift.types import DriftField, MotionMatches


SENTINEL1_TIME_RE = re.compile(r"_(\d{8}T\d{6})_")
ROUTING_RECOVERY_MINIMUM_MATCHES = 12
ROUTING_RECOVERY_EDGE_BAND_PX = 32
ROUTING_RECOVERY_MINIMUM_EDGE_IMBALANCE = 0.05


@dataclass(frozen=True)
class PairSpec:
    source_image_id: int
    target_image_id: int
    source_path: str
    target_path: str
    elapsed_hours: float
    buoy_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-pair-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--aliked-sequence-dir", type=Path)
    parser.add_argument("--efficientloftr-repo", type=Path, required=True)
    parser.add_argument("--efficientloftr-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--routing-mode",
        choices=("same_center", "sequential", "sequential_global", "sequential_local"),
        required=True,
    )
    parser.add_argument(
        "--initial-routing",
        choices=("same_center", "phase_correlation"),
        default="same_center",
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument(
        "--maximum-speed-km-per-day",
        type=float,
        default=30.0,
        help="Radial physics gate; test 30, 40, and 50 before changing the default.",
    )
    parser.add_argument("--tile-size-px", type=int, default=512)
    parser.add_argument("--tile-margin-px", type=int, default=32)
    parser.add_argument(
        "--routing-recovery",
        choices=("none", "residual_edge"),
        default="none",
        help="Optionally recenter and rematch tiles with aligned residual and edge pressure.",
    )
    parser.add_argument(
        "--sic-root",
        type=Path,
        help="Optional OSI SAF OSI-401-d root for conservative open-water skipping.",
    )
    parser.add_argument("--sic-open-water-threshold-percent", type=float, default=15.0)
    parser.add_argument("--sic-max-age-days", type=int, default=1)
    parser.add_argument("--sic-samples-per-axis", type=int, default=5)
    return parser.parse_args()


def acquisition_time_from_path(path: str) -> datetime | None:
    """Return the first Sentinel-1 acquisition time encoded in a filename."""
    match = SENTINEL1_TIME_RE.search(Path(path).name)
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S") if match else None


def open_water_gate_policy(
    threshold_percent: float, samples_per_axis: int
) -> dict:
    """Return the shared scientific policy recorded at every output level."""
    if not 0 <= threshold_percent <= 100:
        raise ValueError("SIC open-water threshold must be in [0, 100]")
    if samples_per_axis < 2:
        raise ValueError("at least two SIC samples per axis are required")
    return {
        "variable_policy": "prefer_ice_conc_unfiltered",
        "threshold_percent": threshold_percent,
        "samples_per_axis": samples_per_axis,
        "policy": "skip only when all samples on both dates are below threshold",
        "missing_sic_policy": "keep_tile",
    }


def sic_evidence_metrics(
    prefix: str, evidence: OpenWaterEvidence | None
) -> dict[str, int | float | None]:
    return {
        f"{prefix}_sic_valid_samples": (
            None if evidence is None else evidence.valid_samples
        ),
        f"{prefix}_sic_max_percent": (
            None if evidence is None else evidence.maximum_sic_percent
        ),
    }


def load_specs(args: argparse.Namespace) -> list[PairSpec]:
    specs = []
    for run_dir in args.reference_pair_run_dir:
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        source_id = int(manifest["source_image_id"])
        target_id = int(manifest["target_image_id"])
        buoy_candidates = [run_dir / "buoy_results.csv"]
        if args.aliked_sequence_dir is not None:
            buoy_candidates.insert(
                0,
                args.aliked_sequence_dir
                / f"pair_{source_id}_{target_id}"
                / "buoy_nearest12.csv",
            )
        specs.append(
            PairSpec(
                source_image_id=source_id,
                target_image_id=target_id,
                source_path=manifest["source_image_filepath"],
                target_path=manifest["target_image_filepath"],
                elapsed_hours=float(manifest["elapsed_hours"]),
                buoy_path=next(
                    (path for path in buoy_candidates if path.exists()), None
                ),
            )
        )
    for previous, current in zip(specs, specs[1:]):
        if previous.target_image_id != current.source_image_id:
            raise ValueError("reference pairs do not form a contiguous chain")
        if previous.target_path != current.source_path:
            raise ValueError("adjacent pair image paths disagree")
    return specs


def pair_domains(spec: PairSpec, config: EfficientLoFTRConfig):
    maximum_displacement_m = config.maximum_displacement_m(spec.elapsed_hours)
    source_footprint = projected_footprint(spec.source_path, config.analysis_epsg)
    target_footprint = projected_footprint(spec.target_path, config.analysis_epsg)
    source_domain = source_footprint.intersection(
        target_footprint.buffer(maximum_displacement_m)
    )
    target_domain = target_footprint.intersection(
        source_domain.buffer(maximum_displacement_m)
    )
    if source_domain.is_empty or target_domain.is_empty:
        raise ValueError("source and target have no physics-reachable overlap")
    return source_domain, target_domain


def projected_xy(
    points_px: np.ndarray,
    center_xy_m: tuple[float, float],
    config: EfficientLoFTRConfig,
) -> np.ndarray:
    center_px = (config.tile_size_px - 1) / 2.0
    return np.column_stack(
        (
            center_xy_m[0] + (points_px[:, 0] - center_px) * config.pixel_size_m,
            center_xy_m[1] - (points_px[:, 1] - center_px) * config.pixel_size_m,
        )
    )


def field_sha256(field: DriftField | None) -> str | None:
    """Return a stable hash of the field values used to route the next pair."""
    if field is None:
        return None
    digest = hashlib.sha256()
    for values in (
        field.source_xy_m,
        field.displacement_m,
        field.available,
        field.selected_matches,
        field.support_radius_m,
        field.maximum_residual_m,
    ):
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.floating):
            finite = np.isfinite(array)
            digest.update(np.ascontiguousarray(finite, dtype=np.uint8).tobytes())
            array = np.round(
                np.where(finite, array, 0.0), decimals=6
            ).astype("<f8", copy=False)
        elif np.issubdtype(array.dtype, np.bool_):
            array = array.astype(np.uint8, copy=False)
        else:
            array = array.astype("<i8", copy=False)
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def field_from_csv(path: Path) -> DriftField:
    rows = pd.read_csv(path)
    return DriftField(
        grid_row=rows["grid_row"].to_numpy(np.int64),
        grid_column=rows["grid_column"].to_numpy(np.int64),
        source_xy_m=rows[["source_x", "source_y"]].to_numpy(np.float64),
        displacement_m=rows[
            ["proposal_dx_m", "proposal_dy_m"]
        ].to_numpy(np.float64),
        available=rows["available"].fillna(False).to_numpy(bool),
        selected_matches=rows["selected_vectors"].to_numpy(np.int32),
        candidate_matches=rows["candidate_count"].to_numpy(np.int32),
        support_radius_m=rows["support_radius_m"].to_numpy(np.float64),
        maximum_residual_m=rows[
            "maximum_vector_residual_m"
        ].to_numpy(np.float64),
    )


def pair_identity(
    spec: PairSpec,
    config: EfficientLoFTRConfig,
    routing_mode: str,
    initial_routing: str,
    initial_displacement_m: tuple[float, float] | None,
    checkpoint_sha256: str,
    previous_field: DriftField | None,
    previous_elapsed_days: float | None,
    source_selection_xy_m: np.ndarray | None = None,
    source_selection_buffer_m: float | None = None,
    source_sic_path: Path | None = None,
    target_sic_path: Path | None = None,
    sic_open_water_threshold_percent: float = 15.0,
    sic_samples_per_axis: int = 5,
    routing_recovery: str = "none",
) -> str:
    identity = {
        "source_image_id": spec.source_image_id,
        "target_image_id": spec.target_image_id,
        "source_path": spec.source_path,
        "target_path": spec.target_path,
        "elapsed_hours": spec.elapsed_hours,
        "config": asdict(config),
        "matcher": "official EfficientLoFTR optimized",
        "checkpoint_sha256": checkpoint_sha256,
        "routing_mode": routing_mode,
        "initial_routing": initial_routing,
        "initial_displacement_m": initial_displacement_m,
        "previous_field_sha256": field_sha256(previous_field),
        "previous_elapsed_days": previous_elapsed_days,
        "pre_match_tile_gates": {
            "valid_overlap": "endpoint_support_bounds_v1",
            "open_water": {
                **open_water_gate_policy(
                    sic_open_water_threshold_percent, sic_samples_per_axis
                ),
                "enabled": source_sic_path is not None and target_sic_path is not None,
                "source_sic_path": None if source_sic_path is None else str(source_sic_path),
                "target_sic_path": None if target_sic_path is None else str(target_sic_path),
            },
        },
    }
    if source_selection_xy_m is not None:
        identity["source_selection_sha256"] = array_sha256(source_selection_xy_m)
        identity["source_selection_buffer_m"] = source_selection_buffer_m
    if routing_recovery != "none":
        identity["routing_recovery"] = routing_recovery
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def routing_recovery_diagnostic(
    source_xy_m: np.ndarray,
    target_xy_m: np.ndarray,
    target_px: np.ndarray,
    routing_shift_m: np.ndarray,
    config: EfficientLoFTRConfig,
) -> dict:
    """Diagnose a target-window routing error from accepted image matches."""

    match_count = len(source_xy_m)
    slack_px = config.tile_margin_px - config.endpoint_support_radius_px
    slack_m = max(0.0, slack_px * config.pixel_size_m)
    result = {
        "eligible": False,
        "triggered": False,
        "match_count": int(match_count),
        "usable_routing_slack_m": float(slack_m),
        "median_residual_dx_m": None,
        "median_residual_dy_m": None,
        "left_edge_fraction": None,
        "right_edge_fraction": None,
        "top_edge_fraction": None,
        "bottom_edge_fraction": None,
        "aligned_axes": [],
    }
    if match_count < ROUTING_RECOVERY_MINIMUM_MATCHES:
        return result
    displacement = target_xy_m - source_xy_m
    residual = np.median(displacement, axis=0) - np.asarray(
        routing_shift_m, dtype=np.float64
    )
    lower_edge = (
        config.endpoint_support_radius_px + ROUTING_RECOVERY_EDGE_BAND_PX
    )
    upper_edge = (
        config.tile_size_px
        - config.endpoint_support_radius_px
        - ROUTING_RECOVERY_EDGE_BAND_PX
    )
    left = float(np.mean(target_px[:, 0] < lower_edge))
    right = float(np.mean(target_px[:, 0] >= upper_edge))
    top = float(np.mean(target_px[:, 1] < lower_edge))
    bottom = float(np.mean(target_px[:, 1] >= upper_edge))
    aligned_axes = []
    threshold = ROUTING_RECOVERY_MINIMUM_EDGE_IMBALANCE
    if residual[0] < -slack_m and left - right >= threshold:
        aligned_axes.append("x_negative")
    if residual[0] > slack_m and right - left >= threshold:
        aligned_axes.append("x_positive")
    # Projected y increases upward while image-row y increases downward.
    if residual[1] > slack_m and top - bottom >= threshold:
        aligned_axes.append("y_positive")
    if residual[1] < -slack_m and bottom - top >= threshold:
        aligned_axes.append("y_negative")
    result.update(
        {
            "eligible": True,
            "triggered": bool(aligned_axes),
            "median_residual_dx_m": float(residual[0]),
            "median_residual_dy_m": float(residual[1]),
            "left_edge_fraction": left,
            "right_edge_fraction": right,
            "top_edge_fraction": top,
            "bottom_edge_fraction": bottom,
            "aligned_axes": aligned_axes,
        }
    )
    return result


def array_sha256(values: np.ndarray | None) -> str | None:
    if values is None:
        return None
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def regions_near_points(
    regions,
    points_xy_m: np.ndarray,
    buffer_m: float,
):
    """Select stable source-tile cores near dormant trajectory positions."""
    points = np.asarray(points_xy_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError("source selection points must have shape (n, 2)")
    if not np.isfinite(points).all():
        raise ValueError("source selection points must be finite")
    if not np.isfinite(buffer_m) or buffer_m <= 0:
        raise ValueError("source selection buffer must be finite and positive")
    selection = MultiPoint(points).buffer(buffer_m)
    return tuple(region for region in regions if region.core.intersects(selection))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_completed_pair(
    output_dir: Path, expected_identity: str
) -> tuple[DriftField, dict] | None:
    summary_path = output_dir / "summary.json"
    field_path = output_dir / "field_4km.csv"
    if not summary_path.exists() or not field_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete":
        return None
    if summary.get("pair_identity_sha256") != expected_identity:
        raise ValueError(
            f"completed output has a different scientific identity: {output_dir}"
        )
    field = field_from_csv(field_path)
    stored_field_hash = summary.get("field_sha256")
    if stored_field_hash is not None and field_sha256(field) != stored_field_hash:
        raise ValueError(f"completed field failed its content hash: {output_dir}")
    return field, summary


def save_matches(path: Path, matches: MotionMatches) -> None:
    np.savez_compressed(
        path,
        source_feature_id=matches.source_feature_id,
        source_tile_id=matches.source_tile_id,
        target_tile_id=matches.target_tile_id,
        source_xy_m=matches.source_xy_m,
        target_xy_m=matches.target_xy_m,
        score=matches.score,
    )


def summarize_buoys(rows: pd.DataFrame) -> dict:
    available = rows["available"].fillna(False)
    errors = rows.loc[available, "error_m"].dropna().to_numpy(float)
    return {
        "expected": int(len(rows)),
        "available": int(available.sum()),
        "correct_within_2km": int((available & rows["error_m"].le(2_000.0)).sum()),
        "median_error_m": float(np.median(errors)) if len(errors) else None,
        "p90_error_m": float(np.quantile(errors, 0.90)) if len(errors) else None,
        "maximum_error_m": float(np.max(errors)) if len(errors) else None,
    }


def track_pair(
    spec: PairSpec,
    model,
    device: torch.device,
    config: EfficientLoFTRConfig,
    routing_mode: str,
    previous_field: DriftField | None,
    previous_elapsed_days: float | None,
    initial_displacement_m: tuple[float, float] | None,
    identity_sha256: str,
    output_dir: Path,
    source_selection_xy_m: np.ndarray | None = None,
    source_selection_buffer_m: float | None = None,
    source_sic: SicField | None = None,
    target_sic: SicField | None = None,
    sic_open_water_threshold_percent: float = 15.0,
    sic_samples_per_axis: int = 5,
    routing_recovery: str = "none",
) -> tuple[DriftField, dict]:
    if routing_recovery not in {"none", "residual_edge"}:
        raise ValueError(f"unsupported routing recovery: {routing_recovery}")
    pair_started = time.perf_counter()
    source_domain, target_domain = pair_domains(spec, config)
    source_rows = []
    target_rows = []
    score_rows = []
    tile_rows = []
    tile_metrics = []
    sampling_seconds = 0.0
    matching_seconds = 0.0
    all_regions = tile_layout(source_domain, config)
    regions = all_regions
    if source_selection_xy_m is not None:
        if source_selection_buffer_m is None:
            raise ValueError("targeted matching requires a source selection buffer")
        regions = regions_near_points(
            all_regions,
            source_selection_xy_m,
            source_selection_buffer_m,
        )
        if not regions:
            raise ValueError("no source tiles intersect the targeted positions")
    tile_shifts_m, routing_sources = preceding_field_shifts(
        np.asarray([region.center_xy_m for region in regions]),
        routing_mode,
        previous_field,
        previous_elapsed_days,
        spec.elapsed_hours / 24.0,
        config.minimum_agreeing_matches,
        config.grid_spacing_m,
        initial_displacement_m,
    )

    for region, shift_m, routing_source in zip(
        regions, tile_shifts_m, routing_sources, strict=True
    ):
        target_center = (
            region.center_xy_m[0] + shift_m[0],
            region.center_xy_m[1] + shift_m[1],
        )
        started = time.perf_counter()
        source_patch, source_valid = north_up_patch(
            spec.source_path,
            region.center_xy_m,
            config.tile_size_px,
            config.pixel_size_m,
            config.analysis_epsg,
            config.transform_grid_spacing_px,
        )
        target_patch, target_valid = north_up_patch(
            spec.target_path,
            target_center,
            config.tile_size_px,
            config.pixel_size_m,
            config.analysis_epsg,
            config.transform_grid_spacing_px,
        )
        source_support = valid_support(
            source_valid, config.endpoint_support_radius_px
        )
        target_support = valid_support(
            target_valid, config.endpoint_support_radius_px
        )
        source_core_support = source_support.copy()
        margin = config.tile_margin_px
        source_core_support[:margin] = False
        source_core_support[config.tile_size_px - margin :] = False
        source_core_support[:, :margin] = False
        source_core_support[:, config.tile_size_px - margin :] = False

        validity_gate = valid_tile_overlap_gate(
            source_core_support,
            target_support,
            region.center_xy_m,
            target_center,
            config.pixel_size_m,
            config.maximum_displacement_m(spec.elapsed_hours),
        )
        source_open_water = None
        target_open_water = None
        skip_reason = validity_gate.reason
        if not validity_gate.skip and source_sic is not None and target_sic is not None:
            source_open_water = tile_open_water_evidence(
                source_sic,
                region.center_xy_m,
                config.tile_core_size_m,
                config.analysis_epsg,
                sic_open_water_threshold_percent,
                sic_samples_per_axis,
            )
            target_open_water = tile_open_water_evidence(
                target_sic,
                target_center,
                config.tile_size_px * config.pixel_size_m,
                config.analysis_epsg,
                sic_open_water_threshold_percent,
                sic_samples_per_axis,
            )
            if source_open_water.confidently_open and target_open_water.confidently_open:
                skip_reason = "open_water_both_dates"
        sampling_seconds += time.perf_counter() - started

        tile_metric = {
            "tile_id": region.tile_id,
            "row": region.row,
            "column": region.column,
            "source_center_x_m": region.center_xy_m[0],
            "source_center_y_m": region.center_xy_m[1],
            "target_center_x_m": target_center[0],
            "target_center_y_m": target_center[1],
            "routing_dx_m": shift_m[0],
            "routing_dy_m": shift_m[1],
            "routing_source": routing_source,
            "skip_reason": skip_reason,
            "source_support_pixels": validity_gate.source_support_pixels,
            "target_support_pixels": validity_gate.target_support_pixels,
            "minimum_support_bounds_distance_m": (
                validity_gate.minimum_bounds_distance_m
            ),
            **sic_evidence_metrics("source", source_open_water),
            **sic_evidence_metrics("target", target_open_water),
        }
        if skip_reason is not None:
            tile_metrics.append(
                {
                    **tile_metric,
                    "raw_matches": 0,
                    "physics_valid_matches": 0,
                    "matching_seconds": 0.0,
                }
            )
            continue

        inputs = matcher_inputs(source_patch, target_patch, device)
        started = time.perf_counter()
        with torch.inference_mode():
            source_px, target_px, scores = run_optimized_matcher(model, inputs)
        synchronize(device)
        tile_matching_seconds = time.perf_counter() - started
        matching_seconds += tile_matching_seconds

        source_xy_m = projected_xy(source_px, region.center_xy_m, config)
        target_xy_m = projected_xy(target_px, target_center, config)
        in_source_core = source_core_mask(
            source_px,
            config.tile_size_px,
            config.tile_margin_px,
        )
        in_domains = shapely.intersects_xy(
            source_domain, source_xy_m[:, 0], source_xy_m[:, 1]
        ) & shapely.intersects_xy(
            target_domain, target_xy_m[:, 0], target_xy_m[:, 1]
        )
        speed_valid = speed_limit_mask(
            source_xy_m,
            target_xy_m,
            spec.elapsed_hours,
            config.maximum_speed_m_per_day,
        )
        accepted = (
            in_source_core
            & valid_endpoints(source_px, source_support)
            & valid_endpoints(target_px, target_support)
            & in_domains
            & speed_valid
        )
        recovery = routing_recovery_diagnostic(
            source_xy_m[accepted],
            target_xy_m[accepted],
            target_px[accepted],
            shift_m,
            config,
        )
        recovery_matching_seconds = 0.0
        recovery_sampling_seconds = 0.0
        first_pass_physics_valid_matches = int(accepted.sum())
        first_pass_matching_seconds = tile_matching_seconds
        if routing_recovery == "residual_edge" and recovery["triggered"]:
            correction_m = np.array(
                [
                    recovery["median_residual_dx_m"],
                    recovery["median_residual_dy_m"],
                ],
                dtype=np.float64,
            )
            recovered_center = tuple(np.asarray(target_center) + correction_m)
            started = time.perf_counter()
            recovered_patch, recovered_valid = north_up_patch(
                spec.target_path,
                recovered_center,
                config.tile_size_px,
                config.pixel_size_m,
                config.analysis_epsg,
                config.transform_grid_spacing_px,
            )
            recovered_support = valid_support(
                recovered_valid, config.endpoint_support_radius_px
            )
            recovered_gate = valid_tile_overlap_gate(
                source_core_support,
                recovered_support,
                region.center_xy_m,
                recovered_center,
                config.pixel_size_m,
                config.maximum_displacement_m(spec.elapsed_hours),
            )
            recovery_sampling_seconds = time.perf_counter() - started
            sampling_seconds += recovery_sampling_seconds
            if not recovered_gate.skip:
                recovered_inputs = matcher_inputs(
                    source_patch, recovered_patch, device
                )
                started = time.perf_counter()
                with torch.inference_mode():
                    recovered_source_px, recovered_target_px, recovered_scores = (
                        run_optimized_matcher(model, recovered_inputs)
                    )
                synchronize(device)
                recovery_matching_seconds = time.perf_counter() - started
                matching_seconds += recovery_matching_seconds
                recovered_source_xy_m = projected_xy(
                    recovered_source_px, region.center_xy_m, config
                )
                recovered_target_xy_m = projected_xy(
                    recovered_target_px, recovered_center, config
                )
                recovered_accepted = (
                    source_core_mask(
                        recovered_source_px,
                        config.tile_size_px,
                        config.tile_margin_px,
                    )
                    & valid_endpoints(recovered_source_px, source_support)
                    & valid_endpoints(recovered_target_px, recovered_support)
                    & shapely.intersects_xy(
                        source_domain,
                        recovered_source_xy_m[:, 0],
                        recovered_source_xy_m[:, 1],
                    )
                    & shapely.intersects_xy(
                        target_domain,
                        recovered_target_xy_m[:, 0],
                        recovered_target_xy_m[:, 1],
                    )
                    & speed_limit_mask(
                        recovered_source_xy_m,
                        recovered_target_xy_m,
                        spec.elapsed_hours,
                        config.maximum_speed_m_per_day,
                    )
                )
                recovery["post_recovery"] = routing_recovery_diagnostic(
                    recovered_source_xy_m[recovered_accepted],
                    recovered_target_xy_m[recovered_accepted],
                    recovered_target_px[recovered_accepted],
                    np.asarray(shift_m) + correction_m,
                    config,
                )
                source_xy_m = recovered_source_xy_m
                target_xy_m = recovered_target_xy_m
                source_px = recovered_source_px
                target_px = recovered_target_px
                scores = recovered_scores
                accepted = recovered_accepted
                target_center = recovered_center
                recovery["applied"] = True
                recovery["correction_dx_m"] = float(correction_m[0])
                recovery["correction_dy_m"] = float(correction_m[1])
            else:
                recovery["applied"] = False
                recovery["failure_reason"] = recovered_gate.reason
        else:
            recovery["applied"] = False
        source_rows.append(source_xy_m[accepted])
        target_rows.append(target_xy_m[accepted])
        score_rows.append(scores[accepted].astype(np.float32))
        tile_rows.append(np.full(int(accepted.sum()), region.tile_id, dtype=np.int32))
        tile_metrics.append(
            {
                **tile_metric,
                "target_center_x_m": target_center[0],
                "target_center_y_m": target_center[1],
                "raw_matches": int(len(scores)),
                "physics_valid_matches": int(accepted.sum()),
                "matching_seconds": (
                    tile_matching_seconds + recovery_matching_seconds
                ),
                "first_pass_matching_seconds": first_pass_matching_seconds,
                "first_pass_physics_valid_matches": first_pass_physics_valid_matches,
                "recovery_matching_seconds": recovery_matching_seconds,
                "recovery_sampling_seconds": recovery_sampling_seconds,
                "routing_recovery": json.dumps(recovery, sort_keys=True),
            }
        )

    source_xy_m = np.concatenate(source_rows) if source_rows else np.empty((0, 2))
    target_xy_m = np.concatenate(target_rows) if target_rows else np.empty((0, 2))
    scores = np.concatenate(score_rows) if score_rows else np.empty(0, dtype=np.float32)
    tile_ids = np.concatenate(tile_rows) if tile_rows else np.empty(0, dtype=np.int32)
    matches = MotionMatches(
        source_feature_id=np.arange(len(scores), dtype=np.int64),
        source_tile_id=tile_ids,
        target_tile_id=tile_ids.copy(),
        source_xy_m=source_xy_m,
        target_xy_m=target_xy_m,
        score=scores,
    )

    started = time.perf_counter()
    raw_field = estimate_field(matches, source_domain, config)
    field, rejected = reject_folds(
        raw_field,
        config.grid_spacing_m,
        config.maximum_triangle_edge_m,
    )
    field_seconds = time.perf_counter() - started

    buoy = None
    if spec.buoy_path is not None:
        buoy_queries = pd.read_csv(spec.buoy_path, dtype={"buoy_id": str})
        buoy = buoy_queries.drop(
            columns=[
                "available",
                "selected_vectors",
                "proposal_dx_m",
                "proposal_dy_m",
                "maximum_source_distance_m",
                "maximum_vector_residual_m",
                "error_m",
                "candidate_count",
                "support_radius_m",
            ],
            errors="ignore",
        ).copy()
        estimates = estimate_queries(
            matches,
            buoy[["source_x", "source_y"]].to_numpy(np.float64),
            config,
        )
        buoy["available"] = estimates["available"]
        buoy["selected_vectors"] = estimates["selected_matches"]
        buoy["proposal_dx_m"] = estimates["displacement_m"][:, 0]
        buoy["proposal_dy_m"] = estimates["displacement_m"][:, 1]
        buoy["maximum_source_distance_m"] = estimates["support_radius_m"]
        buoy["maximum_vector_residual_m"] = estimates["maximum_residual_m"]
        buoy["candidate_count"] = estimates["candidate_matches"]
        buoy["support_radius_m"] = estimates["support_radius_m"]
        buoy["error_m"] = np.hypot(
            buoy["proposal_dx_m"] - buoy["truth_dx_m"],
            buoy["proposal_dy_m"] - buoy["truth_dy_m"],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_matches(output_dir / "matches.npz", matches)
    field_path = output_dir / "field_4km.csv"
    field.to_frame().to_csv(field_path, index=False)
    persisted_field = field_from_csv(field_path)
    finite = np.isfinite(field.displacement_m) & np.isfinite(
        persisted_field.displacement_m
    )
    persistence_max_abs_difference_m = (
        float(
            np.max(
                np.abs(
                    field.displacement_m[finite]
                    - persisted_field.displacement_m[finite]
                )
            )
        )
        if finite.any()
        else 0.0
    )
    if persistence_max_abs_difference_m > 1.0e-6:
        raise ValueError(
            "persisted field changed displacement by more than one micrometre"
        )
    field = persisted_field
    if buoy is not None:
        buoy.to_csv(output_dir / "buoy_results.csv", index=False)
    pd.DataFrame(tile_metrics).to_csv(output_dir / "tiles.csv", index=False)
    skip_counts = (
        pd.Series(
            [item["skip_reason"] for item in tile_metrics if item["skip_reason"]]
        )
        .value_counts()
        .to_dict()
    )

    summary = {
        "status": "complete",
        "pair_identity_sha256": identity_sha256,
        "field_sha256": field_sha256(field),
        "persistence_max_abs_difference_m": persistence_max_abs_difference_m,
        "source_image_id": spec.source_image_id,
        "target_image_id": spec.target_image_id,
        "elapsed_hours": spec.elapsed_hours,
        "routing": {
            "mode": routing_mode,
            "source_counts": pd.Series(routing_sources).value_counts().to_dict(),
            "shift_median": np.median(tile_shifts_m, axis=0).tolist(),
            "shift_p90_magnitude": float(
                np.quantile(np.linalg.norm(tile_shifts_m, axis=1), 0.90)
            ),
            "recovery": {
                "mode": routing_recovery,
                "triggered_tiles": int(
                    sum(
                        json.loads(item.get("routing_recovery", "{}"))
                        .get("triggered", False)
                        for item in tile_metrics
                    )
                ),
                "applied_tiles": int(
                    sum(
                        json.loads(item.get("routing_recovery", "{}"))
                        .get("applied", False)
                        for item in tile_metrics
                    )
                ),
                "minimum_matches": ROUTING_RECOVERY_MINIMUM_MATCHES,
                "edge_band_px": ROUTING_RECOVERY_EDGE_BAND_PX,
                "minimum_edge_imbalance": ROUTING_RECOVERY_MINIMUM_EDGE_IMBALANCE,
            },
        },
        "source_tiles": len(tile_metrics),
        "matched_source_tiles": len(tile_metrics) - sum(skip_counts.values()),
        "skipped_source_tiles": sum(skip_counts.values()),
        "tile_skip_counts": skip_counts,
        "open_water_gate": {
            **open_water_gate_policy(
                sic_open_water_threshold_percent, sic_samples_per_axis
            ),
            "enabled": source_sic is not None and target_sic is not None,
            "source_sic_path": (
                None if source_sic is None else str(source_sic.source_path)
            ),
            "target_sic_path": (
                None if target_sic is None else str(target_sic.source_path)
            ),
            "source_variable": None if source_sic is None else source_sic.variable,
            "target_variable": None if target_sic is None else target_sic.variable,
        },
        "full_source_tiles": len(all_regions),
        "targeted_recovery": (
            {
                "source_positions": int(len(source_selection_xy_m)),
                "selection_buffer_m": float(source_selection_buffer_m),
                "selected_source_tiles": len(regions),
                "full_source_tiles": len(all_regions),
                "selected_tile_fraction": float(len(regions) / len(all_regions)),
            }
            if source_selection_xy_m is not None
            else None
        ),
        "physics_valid_matches": len(matches),
        "grid_nodes": len(field),
        "available_before_fold_rejection": int(raw_field.available.sum()),
        "available_after_fold_rejection": int(field.available.sum()),
        "coverage_after_fold_rejection": float(field.available.mean()),
        "fold_rejected_nodes": int(len(rejected)),
        "topology_after_rejection": topology_summary(
            field,
            config.grid_spacing_m,
            config.maximum_triangle_edge_m,
        ),
        "buoys": summarize_buoys(buoy) if buoy is not None else None,
        "timing_seconds": {
            "sampling": sampling_seconds,
            "matching": matching_seconds,
            "field_estimation_and_topology": field_seconds,
            "pair_total": time.perf_counter() - pair_started,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return field, summary


def save_trajectory_products(
    specs: list[PairSpec],
    fields: list[DriftField],
    output_dir: Path,
    config: EfficientLoFTRConfig,
) -> dict:
    image_ids = [specs[0].source_image_id] + [
        spec.target_image_id for spec in specs
    ]
    trajectories = advect_trajectories(
        fields,
        image_ids,
        config.grid_spacing_m,
        maximum_triangle_edge_m=config.maximum_triangle_edge_m,
    )
    trajectories.to_csv(output_dir / "trajectories_4km.csv", index=False)
    survival = (
        trajectories.groupby("image_index", sort=True)["active"]
        .agg(["sum", "count"])
        .reset_index()
    )
    survival["fraction"] = survival["sum"] / survival["count"]
    survival.to_csv(output_dir / "trajectory_survival.csv", index=False)
    summary = {
        "seeded": int(survival.iloc[0]["count"]),
        "complete": int(survival.iloc[-1]["sum"]),
        "complete_fraction": float(survival.iloc[-1]["fraction"]),
        "active_by_image": survival["sum"].astype(int).tolist(),
        "active_fraction_by_image": survival["fraction"].tolist(),
    }
    observed_edges = [
        FieldEdge(
            spec.source_image_id,
            spec.target_image_id,
            spec.elapsed_hours,
            field,
        )
        for spec, field in zip(specs, fields, strict=True)
    ]
    points_graph = advect_trajectory_graph(
        observed_edges,
        image_ids,
        config.grid_spacing_m,
        maximum_triangle_edge_m=config.maximum_triangle_edge_m,
        add_new_trajectories=True,
        new_point_exclusion_radius_m=config.new_point_exclusion_radius_m,
    )
    points_graph.to_csv(
        output_dir / "trajectories_with_new_points_adjacent_graph.csv", index=False
    )
    points_by_image = (
        points_graph.groupby("image_index", sort=True)["active"]
        .agg(["sum", "count"])
        .reset_index()
    )
    points_by_image["fraction"] = (
        points_by_image["sum"] / points_by_image["count"]
    )
    points_by_image.to_csv(
        output_dir / "trajectory_coverage_with_new_points_adjacent_graph.csv",
        index=False,
    )
    final_points = points_graph.loc[
        points_graph.image_index == len(image_ids) - 1, "active"
    ]
    summary["adjacent_observed_graph_with_new_points"] = {
        "initial_trajectories": int(
            points_graph.loc[
                points_graph.seed_image_index == 0, "trajectory_id"
            ].nunique()
        ),
        "new_trajectories": int(
            points_graph.loc[
                points_graph.trajectory_state == "new_trajectory", "trajectory_id"
            ].nunique()
        ),
        "trajectory_count": int(points_graph.trajectory_id.nunique()),
        "final_active": int(final_points.sum()),
        "final_active_fraction": (
            float(final_points.mean()) if len(final_points) else None
        ),
        "active_by_image": points_by_image["sum"].astype(int).tolist(),
        "trajectory_count_by_image": points_by_image["count"].astype(int).tolist(),
        "active_fraction_by_image": points_by_image["fraction"].tolist(),
        "dormant_rows": int((points_graph.trajectory_state == "dormant").sum()),
        "reconnected_rows": int(points_graph.reconnected_after_gap.sum()),
        "observed_skip_edge_rows": int(
            (points_graph.trajectory_state == "observed_skip_edge").sum()
        ),
        "new_point_exclusion_radius_m": config.new_point_exclusion_radius_m,
        "note": "Adjacent observed edges only; add independently matched skip edges for reconnection.",
    }
    gap_aware = advect_trajectories(
        fields,
        image_ids,
        config.grid_spacing_m,
        elapsed_hours=[spec.elapsed_hours for spec in specs],
        maximum_prediction_gap_hours=96.0,
        maximum_triangle_edge_m=config.maximum_triangle_edge_m,
    )
    gap_aware.to_csv(output_dir / "trajectories_gap96h.csv", index=False)
    gap_survival = (
        gap_aware.groupby("image_index", sort=True)["active"]
        .agg(["sum", "count"])
        .reset_index()
    )
    gap_survival["fraction"] = gap_survival["sum"] / gap_survival["count"]
    gap_survival.to_csv(
        output_dir / "trajectory_survival_gap96h.csv", index=False
    )
    summary["gap_aware_96h"] = {
        "complete": int(gap_survival.iloc[-1]["sum"]),
        "complete_fraction": float(gap_survival.iloc[-1]["fraction"]),
        "active_by_image": gap_survival["sum"].astype(int).tolist(),
        "active_fraction_by_image": gap_survival["fraction"].tolist(),
        "predicted_rows": int((gap_aware.trajectory_state == "predicted").sum()),
        "field_resupported_rows": int(
            (gap_aware.trajectory_state == "field_resupported").sum()
        ),
    }

    buoy_specs = [spec for spec in specs if spec.buoy_path is not None]
    if not buoy_specs or buoy_specs[0] is not specs[0]:
        summary["buoy_trajectory_validation"] = None
        return summary
    first_buoys = pd.read_csv(specs[0].buoy_path, dtype={"buoy_id": str})
    required = {"buoy_id", "source_x", "source_y"}
    if not required.issubset(first_buoys.columns):
        summary["buoy_trajectory_validation"] = None
        return summary
    first_buoys = first_buoys.drop_duplicates("buoy_id").reset_index(drop=True)
    buoy_trajectories = advect_trajectories(
        fields,
        image_ids,
        config.grid_spacing_m,
        first_buoys[["source_x", "source_y"]].to_numpy(np.float64),
        maximum_triangle_edge_m=config.maximum_triangle_edge_m,
    )
    buoy_trajectories["buoy_id"] = first_buoys.loc[
        buoy_trajectories["trajectory_id"], "buoy_id"
    ].to_numpy()
    truth_rows = []
    for step, spec in enumerate(specs, start=1):
        if spec.buoy_path is None:
            continue
        reference = pd.read_csv(spec.buoy_path, dtype={"buoy_id": str})
        required = {"buoy_id", "source_x", "source_y", "truth_dx_m", "truth_dy_m"}
        if not required.issubset(reference.columns):
            continue
        reference = reference.drop_duplicates("buoy_id").copy()
        reference["truth_x_m"] = reference["source_x"] + reference["truth_dx_m"]
        reference["truth_y_m"] = reference["source_y"] + reference["truth_dy_m"]
        reference["image_index"] = step
        truth_rows.append(
            reference[["buoy_id", "image_index", "truth_x_m", "truth_y_m"]]
        )
    if not truth_rows:
        summary["buoy_trajectory_validation"] = None
        return summary

    validation = buoy_trajectories.merge(
        pd.concat(truth_rows, ignore_index=True),
        on=["buoy_id", "image_index"],
        how="left",
        validate="many_to_one",
    )
    validation["error_m"] = np.where(
        validation["active"] & validation["truth_x_m"].notna(),
        np.hypot(
            validation["x_m"] - validation["truth_x_m"],
            validation["y_m"] - validation["truth_y_m"],
        ),
        np.nan,
    )
    validation.to_csv(output_dir / "buoy_trajectories.csv", index=False)
    comparisons = validation["error_m"].dropna().to_numpy(float)
    final = validation.loc[
        validation.image_index == len(specs), "error_m"
    ].dropna().to_numpy(float)
    summary["buoy_trajectory_validation"] = {
        "seeded": len(first_buoys),
        "comparisons": len(comparisons),
        "complete": int(
            validation.loc[validation.image_index == len(specs), "active"].sum()
        ),
        "median_error_m": float(np.median(comparisons)) if len(comparisons) else None,
        "p90_error_m": float(np.quantile(comparisons, 0.90)) if len(comparisons) else None,
        "final_median_error_m": float(np.median(final)) if len(final) else None,
        "final_p90_error_m": float(np.quantile(final, 0.90)) if len(final) else None,
        "final_maximum_error_m": float(np.max(final)) if len(final) else None,
    }
    gap_buoy_trajectories = advect_trajectories(
        fields,
        image_ids,
        config.grid_spacing_m,
        first_buoys[["source_x", "source_y"]].to_numpy(np.float64),
        elapsed_hours=[spec.elapsed_hours for spec in specs],
        maximum_prediction_gap_hours=96.0,
        maximum_triangle_edge_m=config.maximum_triangle_edge_m,
    )
    gap_buoy_trajectories["buoy_id"] = first_buoys.loc[
        gap_buoy_trajectories["trajectory_id"], "buoy_id"
    ].to_numpy()
    gap_validation = gap_buoy_trajectories.merge(
        pd.concat(truth_rows, ignore_index=True),
        on=["buoy_id", "image_index"],
        how="left",
        validate="many_to_one",
    )
    gap_validation["error_m"] = np.where(
        gap_validation["active"] & gap_validation["truth_x_m"].notna(),
        np.hypot(
            gap_validation["x_m"] - gap_validation["truth_x_m"],
            gap_validation["y_m"] - gap_validation["truth_y_m"],
        ),
        np.nan,
    )
    gap_validation.to_csv(
        output_dir / "buoy_trajectories_gap96h.csv", index=False
    )
    gap_comparisons = gap_validation["error_m"].dropna().to_numpy(float)
    gap_final = gap_validation.loc[
        gap_validation.image_index == len(specs), "error_m"
    ].dropna().to_numpy(float)
    summary["gap_aware_96h"]["buoy_validation"] = {
        "comparisons": len(gap_comparisons),
        "complete": int(
            gap_validation.loc[
                gap_validation.image_index == len(specs), "active"
            ].sum()
        ),
        "median_error_m": (
            float(np.median(gap_comparisons)) if len(gap_comparisons) else None
        ),
        "p90_error_m": (
            float(np.quantile(gap_comparisons, 0.90))
            if len(gap_comparisons)
            else None
        ),
        "final_median_error_m": (
            float(np.median(gap_final)) if len(gap_final) else None
        ),
        "final_p90_error_m": (
            float(np.quantile(gap_final, 0.90)) if len(gap_final) else None
        ),
        "final_maximum_error_m": (
            float(np.max(gap_final)) if len(gap_final) else None
        ),
        "predicted_rows": int(
            (gap_validation.trajectory_state == "predicted").sum()
        ),
        "field_resupported_rows": int(
            (gap_validation.trajectory_state == "field_resupported").sum()
        ),
    }
    return summary


def main() -> int:
    args = parse_args()
    specs = load_specs(args)
    open_water_policy = open_water_gate_policy(
        args.sic_open_water_threshold_percent, args.sic_samples_per_axis
    )
    if args.sic_max_age_days < 0:
        raise ValueError("maximum SIC age cannot be negative")
    sic_index = SicFileIndex(args.sic_root) if args.sic_root is not None else None
    config = EfficientLoFTRConfig(
        maximum_speed_m_per_day=args.maximum_speed_km_per_day * 1_000.0,
        tile_size_px=args.tile_size_px,
        tile_margin_px=args.tile_margin_px,
    )
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_report = None
    report_path = args.output_dir / "run_manifest.json"
    if report_path.exists():
        previous_report = json.loads(report_path.read_text())
    sequence_started = time.perf_counter()
    checkpoint_sha256 = file_sha256(args.efficientloftr_checkpoint)
    initial_translation = None
    initial_routing_seconds = 0.0
    if args.initial_routing == "phase_correlation":
        started = time.perf_counter()
        initial_domain, _ = pair_domains(specs[0], config)
        initial_translation = coarse_phase_translation(
            specs[0].source_path,
            specs[0].target_path,
            initial_domain,
            config.maximum_displacement_m(specs[0].elapsed_hours),
            config.analysis_epsg,
            config.transform_grid_spacing_px,
        )
        initial_routing_seconds = time.perf_counter() - started
    model = None
    model_setup_seconds = 0.0

    previous_field = None
    previous_elapsed_days = None
    pair_summaries = []
    fields = []
    resumed_pairs = 0
    for pair_index, spec in enumerate(specs):
        source_sic_path = (
            None
            if sic_index is None
            else sic_index.resolve(
                acquisition_time_from_path(spec.source_path),
                args.sic_max_age_days,
            )
        )
        target_sic_path = (
            None
            if sic_index is None
            else sic_index.resolve(
                acquisition_time_from_path(spec.target_path),
                args.sic_max_age_days,
            )
        )
        initial_displacement_m = (
            initial_translation.displacement_m
            if pair_index == 0 and initial_translation is not None
            else None
        )
        pair_dir = args.output_dir / f"pair_{spec.source_image_id}_{spec.target_image_id}"
        identity = pair_identity(
            spec,
            config,
            args.routing_mode,
            args.initial_routing,
            initial_displacement_m,
            checkpoint_sha256,
            previous_field,
            previous_elapsed_days,
            source_sic_path=source_sic_path,
            target_sic_path=target_sic_path,
            sic_open_water_threshold_percent=(
                args.sic_open_water_threshold_percent
            ),
            sic_samples_per_axis=args.sic_samples_per_axis,
            routing_recovery=args.routing_recovery,
        )
        completed = load_completed_pair(pair_dir, identity)
        if completed is not None:
            previous_field, summary = completed
            resumed_pairs += 1
        else:
            if model is None:
                setup_started = time.perf_counter()
                model = load_optimized_model(
                    args.efficientloftr_repo,
                    args.efficientloftr_checkpoint,
                    device,
                )
                model_setup_seconds = time.perf_counter() - setup_started
            previous_field, summary = track_pair(
                spec,
                model,
                device,
                config,
                args.routing_mode,
                previous_field,
                previous_elapsed_days,
                initial_displacement_m,
                identity,
                pair_dir,
                source_sic=(
                    None if source_sic_path is None else load_sic_field(source_sic_path)
                ),
                target_sic=(
                    None if target_sic_path is None else load_sic_field(target_sic_path)
                ),
                sic_open_water_threshold_percent=(
                    args.sic_open_water_threshold_percent
                ),
                sic_samples_per_axis=args.sic_samples_per_axis,
                routing_recovery=args.routing_recovery,
            )
        previous_elapsed_days = spec.elapsed_hours / 24.0
        pair_summaries.append(summary)
        fields.append(previous_field)
        print(json.dumps(summary), flush=True)

    trajectory_summary = save_trajectory_products(
        specs, fields, args.output_dir, config
    )

    execution_elapsed_seconds = time.perf_counter() - sequence_started
    previous_initial_elapsed = (
        previous_report.get(
            "initial_elapsed_seconds", previous_report.get("elapsed_seconds")
        )
        if previous_report is not None
        else None
    )
    previous_initial_setup = (
        previous_report.get(
            "initial_model_setup_seconds",
            previous_report.get("model_setup_seconds"),
        )
        if previous_report is not None
        else None
    )
    initial_elapsed_seconds = (
        previous_initial_elapsed
        if resumed_pairs == len(specs) and previous_initial_elapsed is not None
        else execution_elapsed_seconds
    )
    initial_model_setup_seconds = (
        previous_initial_setup
        if resumed_pairs == len(specs) and previous_initial_setup is not None
        else model_setup_seconds
    )
    report = {
        "status": "complete",
        "matcher": "official EfficientLoFTR optimized",
        "routing_mode": args.routing_mode,
        "initial_routing": args.initial_routing,
        "routing_recovery": args.routing_recovery,
        "initial_translation": (
            asdict(initial_translation) if initial_translation is not None else None
        ),
        "initial_routing_seconds": initial_routing_seconds,
        "device": device.type,
        "images": len(specs) + 1,
        "pairs": len(specs),
        "resumed_pairs": resumed_pairs,
        "checkpoint_sha256": checkpoint_sha256,
        "model_setup_seconds": initial_model_setup_seconds,
        "initial_model_setup_seconds": initial_model_setup_seconds,
        "current_execution_model_setup_seconds": model_setup_seconds,
        "elapsed_seconds": initial_elapsed_seconds,
        "initial_elapsed_seconds": initial_elapsed_seconds,
        "current_execution_elapsed_seconds": execution_elapsed_seconds,
        "pair_compute_seconds": float(
            sum(
                pair["timing_seconds"]["pair_total"]
                for pair in pair_summaries
            )
        ),
        "config": {
            "analysis_epsg": config.analysis_epsg,
            "pixel_size_m": config.pixel_size_m,
            "tile_size_px": config.tile_size_px,
            "tile_margin_px": config.tile_margin_px,
            "endpoint_support_radius_px": config.endpoint_support_radius_px,
            "grid_spacing_m": config.grid_spacing_m,
            "maximum_speed_m_per_day": config.maximum_speed_m_per_day,
            "neighbour_count": config.neighbour_count,
            "minimum_agreeing_matches": config.minimum_agreeing_matches,
            "maximum_neighbour_distance_m": config.maximum_neighbour_distance_m,
            "agreement_distance_m": config.agreement_distance_m,
            "score_weighting": config.score_weighting,
            "maximum_triangle_edge_m": config.maximum_triangle_edge_m,
            "new_point_exclusion_radius_m": config.new_point_exclusion_radius_m,
            "confidence_threshold": None,
        },
        "pre_match_tile_gates": {
            "valid_overlap": "endpoint_support_bounds_v1",
            "open_water": {
                **open_water_policy,
                "enabled": sic_index is not None,
                "sic_root": None if args.sic_root is None else str(args.sic_root),
                "product": "EUMETSAT OSI SAF OSI-401-d",
                "max_age_days": args.sic_max_age_days,
            },
        },
        "pairs_summary": pair_summaries,
        "trajectories": trajectory_summary,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
