"""Lagrangian trajectories from consecutive drift fields."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError

from .types import DriftField


@dataclass(frozen=True)
class FieldSamples:
    """Displacements and field diagnostics sampled at projected metre coordinates."""

    displacement_m: np.ndarray
    available: np.ndarray
    selected_matches: np.ndarray
    support_radius_m: np.ndarray
    maximum_residual_m: np.ndarray


def sample_field(
    field: DriftField,
    query_xy_m: np.ndarray,
    maximum_triangle_edge_m: float,
) -> FieldSamples:
    """Interpolate one fold-free field without bridging unsupported spatial gaps."""
    query_xy_m = np.asarray(query_xy_m, dtype=np.float64)
    if query_xy_m.ndim != 2 or query_xy_m.shape[1] != 2:
        raise ValueError("query coordinates must have shape (n, 2)")
    if not np.isfinite(maximum_triangle_edge_m) or maximum_triangle_edge_m <= 0:
        raise ValueError("maximum triangle edge must be finite and positive")

    count = len(query_xy_m)
    displacement_m = np.full((count, 2), np.nan, dtype=np.float64)
    available = np.zeros(count, dtype=bool)
    selected_matches = np.zeros(count, dtype=np.float64)
    support_radius_m = np.full(count, np.nan, dtype=np.float64)
    maximum_residual_m = np.full(count, np.nan, dtype=np.float64)
    valid = field.available & np.isfinite(field.displacement_m).all(axis=1)
    if valid.sum() < 3 or count == 0:
        return FieldSamples(
            displacement_m,
            available,
            selected_matches,
            support_radius_m,
            maximum_residual_m,
        )

    source = np.asarray(field.source_xy_m[valid], dtype=np.float64)
    target = source + np.asarray(field.displacement_m[valid], dtype=np.float64)
    try:
        triangulation = Delaunay(source)
    except QhullError:
        return FieldSamples(
            displacement_m,
            available,
            selected_matches,
            support_radius_m,
            maximum_residual_m,
        )

    vertices = triangulation.simplices
    source_triangles = source[vertices]
    target_triangles = target[vertices]
    source_edges = np.stack(
        (
            source_triangles[:, 1] - source_triangles[:, 0],
            source_triangles[:, 2] - source_triangles[:, 1],
            source_triangles[:, 0] - source_triangles[:, 2],
        )
    )
    maximum_edge = np.linalg.norm(source_edges, axis=2).max(axis=0)
    source_area = _signed_twice_area(source_triangles)
    target_area = _signed_twice_area(target_triangles)
    usable_triangle = (
        (maximum_edge <= maximum_triangle_edge_m)
        & (source_area * target_area > 0)
    )

    simplex = triangulation.find_simplex(query_xy_m)
    inside = simplex >= 0
    if not np.any(inside):
        return FieldSamples(
            displacement_m,
            available,
            selected_matches,
            support_radius_m,
            maximum_residual_m,
        )
    query_indices = np.flatnonzero(inside)
    query_simplex = simplex[inside]
    keep = usable_triangle[query_simplex]
    query_indices = query_indices[keep]
    query_simplex = query_simplex[keep]
    if not len(query_indices):
        return FieldSamples(
            displacement_m,
            available,
            selected_matches,
            support_radius_m,
            maximum_residual_m,
        )

    transform = triangulation.transform[query_simplex]
    first_weights = np.einsum(
        "nij,nj->ni",
        transform[:, :2],
        query_xy_m[query_indices] - transform[:, 2],
    )
    weights = np.column_stack(
        (first_weights, 1.0 - first_weights.sum(axis=1))
    )
    triangle_vertices = vertices[query_simplex]
    field_displacement = np.asarray(field.displacement_m[valid], dtype=np.float64)
    displacement_m[query_indices] = np.einsum(
        "ni,nij->nj", weights, field_displacement[triangle_vertices]
    )

    selected = np.asarray(field.selected_matches[valid], dtype=np.float64)
    radius = np.asarray(field.support_radius_m[valid], dtype=np.float64)
    residual = np.asarray(field.maximum_residual_m[valid], dtype=np.float64)
    selected_matches[query_indices] = np.einsum(
        "ni,ni->n", weights, selected[triangle_vertices]
    )
    support_radius_m[query_indices] = _weighted_diagnostic(
        weights, radius[triangle_vertices]
    )
    maximum_residual_m[query_indices] = _weighted_diagnostic(
        weights, residual[triangle_vertices]
    )
    available[query_indices] = True
    return FieldSamples(
        displacement_m,
        available,
        selected_matches,
        support_radius_m,
        maximum_residual_m,
    )


def advect_trajectories(
    fields: Sequence[DriftField],
    image_ids: Sequence[str | int],
    grid_spacing_m: float,
    seed_xy_m: np.ndarray | None = None,
    maximum_triangle_edge_factor: float = 1.6,
    elapsed_hours: Sequence[float] | None = None,
    maximum_prediction_gap_hours: float = 0.0,
    maximum_triangle_edge_m: float | None = None,
) -> pd.DataFrame:
    """Advect fixed trajectory IDs through consecutive pair fields.

    Coordinates and displacements are float64 projected metres. A point becomes
    inactive when it leaves locally supported, fold-free field triangles and is
    not silently reintroduced on later images.
    """
    if len(image_ids) != len(fields) + 1:
        raise ValueError("image_ids must contain one more item than fields")
    if not fields:
        raise ValueError("at least one field is required")
    if not np.isfinite(grid_spacing_m) or grid_spacing_m <= 0:
        raise ValueError("grid spacing must be finite and positive")
    if maximum_triangle_edge_factor <= 0:
        raise ValueError("maximum triangle edge factor must be positive")
    if maximum_triangle_edge_m is not None and (
        not np.isfinite(maximum_triangle_edge_m) or maximum_triangle_edge_m <= 0
    ):
        raise ValueError("maximum triangle edge must be finite and positive")
    if maximum_prediction_gap_hours < 0:
        raise ValueError("maximum prediction gap cannot be negative")
    if elapsed_hours is not None and len(elapsed_hours) != len(fields):
        raise ValueError("elapsed_hours must contain one item per field")
    if maximum_prediction_gap_hours > 0 and elapsed_hours is None:
        raise ValueError("elapsed_hours are required to bridge field gaps")
    if elapsed_hours is not None and (
        not np.isfinite(elapsed_hours).all() or np.any(np.asarray(elapsed_hours) <= 0)
    ):
        raise ValueError("elapsed hours must be finite and positive")

    if seed_xy_m is None:
        valid = fields[0].available & np.isfinite(fields[0].displacement_m).all(axis=1)
        seed_xy_m = fields[0].source_xy_m[valid]
    positions = np.asarray(seed_xy_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("seed coordinates must have shape (n, 2)")
    if not np.isfinite(positions).all():
        raise ValueError("seed coordinates must be finite")
    positions = positions.copy()
    active = np.ones(len(positions), dtype=bool)
    last_velocity_m_per_hour = np.full_like(positions, np.nan)
    prediction_gap_hours = np.zeros(len(positions), dtype=np.float64)
    state = np.full(len(positions), "seed", dtype=object)
    rows = [
        _trajectory_rows(
            0,
            image_ids[0],
            positions,
            active,
            field_observed=np.zeros(len(positions), dtype=bool),
            trajectory_state=state,
            prediction_gap_hours=prediction_gap_hours,
        )
    ]
    maximum_edge_m = (
        grid_spacing_m * maximum_triangle_edge_factor
        if maximum_triangle_edge_m is None
        else float(maximum_triangle_edge_m)
    )

    for step, (field, image_id) in enumerate(
        zip(fields, image_ids[1:], strict=True), start=1
    ):
        hours = float(elapsed_hours[step - 1]) if elapsed_hours is not None else None
        step_displacement = np.full_like(positions, np.nan)
        selected_matches = np.full(len(positions), np.nan)
        support_radius_m = np.full(len(positions), np.nan)
        maximum_residual_m = np.full(len(positions), np.nan)
        was_active = active.copy()
        was_predicted = state == "predicted"
        field_observed = np.zeros(len(positions), dtype=bool)
        state = np.full(len(positions), "dropped", dtype=object)
        active_indices = np.flatnonzero(was_active)
        if len(active_indices):
            sampled = sample_field(
                field, positions[active_indices], maximum_edge_m
            )
            supported_indices = active_indices[sampled.available]
            step_displacement[supported_indices] = sampled.displacement_m[
                sampled.available
            ]
            selected_matches[supported_indices] = sampled.selected_matches[
                sampled.available
            ]
            support_radius_m[supported_indices] = sampled.support_radius_m[
                sampled.available
            ]
            maximum_residual_m[supported_indices] = sampled.maximum_residual_m[
                sampled.available
            ]
            active[active_indices] = sampled.available
            positions[supported_indices] += step_displacement[supported_indices]
            field_observed[supported_indices] = True
            state[supported_indices] = np.where(
                was_predicted[supported_indices], "field_resupported", "observed"
            )
            prediction_gap_hours[supported_indices] = 0.0
            if hours is not None:
                last_velocity_m_per_hour[supported_indices] = (
                    step_displacement[supported_indices] / hours
                )

            unsupported_indices = active_indices[~sampled.available]
            if hours is not None and maximum_prediction_gap_hours > 0:
                predictable = (
                    np.isfinite(last_velocity_m_per_hour[unsupported_indices]).all(axis=1)
                    & (
                        prediction_gap_hours[unsupported_indices] + hours
                        <= maximum_prediction_gap_hours
                    )
                )
                predicted_indices = unsupported_indices[predictable]
                positions[predicted_indices] += (
                    last_velocity_m_per_hour[predicted_indices] * hours
                )
                step_displacement[predicted_indices] = (
                    last_velocity_m_per_hour[predicted_indices] * hours
                )
                prediction_gap_hours[predicted_indices] += hours
                active[predicted_indices] = True
                state[predicted_indices] = "predicted"

        failure_reason = np.full(len(positions), "", dtype=object)
        failure_reason[was_active & ~active] = "outside_supported_field"
        failure_reason[~was_active] = "inactive_previous_step"
        failure_reason[state == "predicted"] = "field_gap_velocity_prediction"
        rows.append(
            _trajectory_rows(
                step,
                image_id,
                positions,
                active,
                step_displacement,
                selected_matches,
                support_radius_m,
                maximum_residual_m,
                failure_reason,
                field_observed,
                state,
                prediction_gap_hours,
            )
        )
    return pd.concat(rows, ignore_index=True)


def _trajectory_rows(
    step: int,
    image_id: str | int,
    positions: np.ndarray,
    active: np.ndarray,
    displacement: np.ndarray | None = None,
    selected_matches: np.ndarray | None = None,
    support_radius_m: np.ndarray | None = None,
    maximum_residual_m: np.ndarray | None = None,
    failure_reason: np.ndarray | None = None,
    field_observed: np.ndarray | None = None,
    trajectory_state: np.ndarray | None = None,
    prediction_gap_hours: np.ndarray | None = None,
) -> pd.DataFrame:
    count = len(positions)
    if displacement is None:
        displacement = np.full((count, 2), np.nan)
    if selected_matches is None:
        selected_matches = np.full(count, np.nan)
    if support_radius_m is None:
        support_radius_m = np.full(count, np.nan)
    if maximum_residual_m is None:
        maximum_residual_m = np.full(count, np.nan)
    if failure_reason is None:
        failure_reason = np.full(count, "", dtype=object)
    if field_observed is None:
        field_observed = np.zeros(count, dtype=bool)
    if trajectory_state is None:
        trajectory_state = np.where(active, "observed", "dropped")
    if prediction_gap_hours is None:
        prediction_gap_hours = np.zeros(count, dtype=np.float64)
    return pd.DataFrame(
        {
            "trajectory_id": np.arange(count, dtype=np.int64),
            "image_index": step,
            "image_id": str(image_id),
            "x_m": positions[:, 0],
            "y_m": positions[:, 1],
            "active": active,
            "field_observed": field_observed,
            "trajectory_state": trajectory_state,
            "prediction_gap_hours": prediction_gap_hours,
            "step_dx_m": displacement[:, 0],
            "step_dy_m": displacement[:, 1],
            "field_selected_matches": selected_matches,
            "field_support_radius_m": support_radius_m,
            "field_maximum_residual_m": maximum_residual_m,
            "failure_reason": failure_reason,
        }
    )


def _weighted_diagnostic(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values).all(axis=1)
    result = np.full(len(values), np.nan)
    result[finite] = np.einsum(
        "ni,ni->n", weights[finite], values[finite]
    )
    return result


def _signed_twice_area(triangles: np.ndarray) -> np.ndarray:
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
