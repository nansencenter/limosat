"""Observed Lagrangian trajectories on a time-directed drift-field graph."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .trajectory import sample_field
from .types import DriftField


@dataclass(frozen=True)
class FieldEdge:
    source_image_id: str | int
    target_image_id: str | int
    elapsed_hours: float
    field: DriftField


def advect_trajectory_graph(
    edges: Sequence[FieldEdge],
    image_ids: Sequence[str | int],
    grid_spacing_m: float,
    seed_xy_m: np.ndarray | None = None,
    maximum_triangle_edge_m: float | None = None,
    add_new_trajectories: bool = False,
    new_point_exclusion_radius_m: float | None = None,
) -> pd.DataFrame:
    """Use the shortest available observed edge to reach each later image.

    Earlier trajectory states remain available to skip edges. Therefore a
    failed adjacent edge does not permanently remove a trajectory hypothesis.
    No position is created by temporal extrapolation. Optional new trajectories
    start only at measured source points of fields leaving the current image.
    """
    if len(image_ids) < 2 or not edges:
        raise ValueError("a trajectory graph needs images and field edges")
    if not np.isfinite(grid_spacing_m) or grid_spacing_m <= 0:
        raise ValueError("grid spacing must be finite and positive")
    if maximum_triangle_edge_m is not None and (
        not np.isfinite(maximum_triangle_edge_m) or maximum_triangle_edge_m <= 0
    ):
        raise ValueError("maximum triangle edge must be finite and positive")
    if add_new_trajectories and new_point_exclusion_radius_m is None:
        raise ValueError("adding trajectories requires a new-point exclusion radius")
    if new_point_exclusion_radius_m is not None and (
        not np.isfinite(new_point_exclusion_radius_m)
        or new_point_exclusion_radius_m < 0
    ):
        raise ValueError("new-point exclusion radius must be finite and non-negative")
    image_keys = [str(image_id) for image_id in image_ids]
    if len(set(image_keys)) != len(image_keys):
        raise ValueError("image IDs must be unique and ordered")
    index = {image_id: step for step, image_id in enumerate(image_keys)}
    indexed_edges = []
    for edge in edges:
        source = index.get(str(edge.source_image_id))
        target = index.get(str(edge.target_image_id))
        if source is None or target is None or source >= target:
            raise ValueError("every edge must point forward between listed images")
        if not np.isfinite(edge.elapsed_hours) or edge.elapsed_hours <= 0:
            raise ValueError("edge elapsed hours must be finite and positive")
        indexed_edges.append((source, target, edge))

    first_edges = [edge for source, _target, edge in indexed_edges if source == 0]
    if seed_xy_m is None:
        if not first_edges:
            raise ValueError("the first image has no outgoing field")
        if add_new_trajectories:
            seed_xy_m = _outgoing_source_points(indexed_edges, 0)
        else:
            first = min(
                first_edges,
                key=lambda edge: index[str(edge.target_image_id)],
            ).field
            valid = first.available & np.isfinite(first.displacement_m).all(axis=1)
            seed_xy_m = first.source_xy_m[valid]
    seed_xy_m = np.asarray(seed_xy_m, dtype=np.float64)
    if seed_xy_m.ndim != 2 or seed_xy_m.shape[1] != 2:
        raise ValueError("seed coordinates must have shape (n, 2)")
    if not np.isfinite(seed_xy_m).all():
        raise ValueError("seed coordinates must be finite")

    count = len(seed_xy_m)
    positions = [np.full((count, 2), np.nan) for _ in image_ids]
    available = [np.zeros(count, dtype=bool) for _ in image_ids]
    positions[0][:] = seed_xy_m
    available[0][:] = True
    trajectory_ids = np.arange(count, dtype=np.int64)
    seed_image_index = np.zeros(count, dtype=np.int32)
    seed_image_id = np.full(count, image_keys[0], dtype=object)
    rows = [
        _graph_rows(
            0,
            image_ids[0],
            positions[0],
            available[0],
            np.full(count, "seed", dtype=object),
            trajectory_ids=trajectory_ids,
            seed_image_index=seed_image_index,
            seed_image_id=seed_image_id,
        )
    ]
    maximum_edge_m = (
        grid_spacing_m * 1.6
        if maximum_triangle_edge_m is None
        else float(maximum_triangle_edge_m)
    )

    for target_step in range(1, len(image_ids)):
        target_edges = sorted(
            (
                (source_step, edge)
                for source_step, target, edge in indexed_edges
                if target == target_step
            ),
            key=lambda item: target_step - item[0],
        )
        chosen_skip = np.full(count, np.iinfo(np.int32).max, dtype=np.int32)
        chosen_selected = np.full(count, -np.inf)
        chosen_source_step = np.full(count, -1, dtype=np.int32)
        chosen_displacement = np.full((count, 2), np.nan)
        chosen_radius = np.full(count, np.nan)
        chosen_residual = np.full(count, np.nan)
        for source_step, edge in target_edges:
            source_indices = np.flatnonzero(available[source_step])
            if not len(source_indices):
                continue
            sampled = sample_field(
                edge.field,
                positions[source_step][source_indices],
                maximum_edge_m,
            )
            candidate_indices = source_indices[sampled.available]
            if not len(candidate_indices):
                continue
            candidate_selected = sampled.selected_matches[sampled.available]
            skipped_images = target_step - source_step - 1
            better = (skipped_images < chosen_skip[candidate_indices]) | (
                (skipped_images == chosen_skip[candidate_indices])
                & (candidate_selected > chosen_selected[candidate_indices])
            )
            selected_indices = candidate_indices[better]
            sampled_indices = np.flatnonzero(sampled.available)[better]
            chosen_skip[selected_indices] = skipped_images
            chosen_selected[selected_indices] = sampled.selected_matches[
                sampled_indices
            ]
            chosen_source_step[selected_indices] = source_step
            chosen_displacement[selected_indices] = sampled.displacement_m[
                sampled_indices
            ]
            chosen_radius[selected_indices] = sampled.support_radius_m[
                sampled_indices
            ]
            chosen_residual[selected_indices] = sampled.maximum_residual_m[
                sampled_indices
            ]

        reached = chosen_source_step >= 0
        available[target_step][reached] = True
        source_positions = np.full((count, 2), np.nan)
        for source_step in np.unique(chosen_source_step[reached]):
            selected = chosen_source_step == source_step
            source_positions[selected] = positions[source_step][selected]
        positions[target_step][reached] = (
            source_positions[reached] + chosen_displacement[reached]
        )
        reconnected_after_gap = reached & ~available[target_step - 1]
        missing_state = "dormant" if add_new_trajectories else "unreached"
        state = np.full(count, missing_state, dtype=object)
        state[reached & (chosen_skip == 0)] = "observed_adjacent"
        state[reached & (chosen_skip > 0)] = "observed_skip_edge"
        source_image = np.full(count, "", dtype=object)
        for source_step in np.unique(chosen_source_step[reached]):
            source_image[chosen_source_step == source_step] = image_keys[source_step]
        rows.append(
            _graph_rows(
                target_step,
                image_ids[target_step],
                positions[target_step],
                available[target_step],
                state,
                source_image,
                chosen_skip,
                chosen_displacement,
                chosen_selected,
                chosen_radius,
                chosen_residual,
                trajectory_ids,
                seed_image_index,
                seed_image_id,
                reconnected_after_gap,
            )
        )

        if add_new_trajectories:
            candidates = _outgoing_source_points(indexed_edges, target_step)
            occupancy_points = _occupancy_positions(
                positions,
                available,
                target_step,
                reached,
            )
            points_new_xy_m = _exclude_occupied_points(
                candidates,
                occupancy_points,
                float(new_point_exclusion_radius_m),
            )
            new_count = len(points_new_xy_m)
            if new_count:
                new_ids = np.arange(count, count + new_count, dtype=np.int64)
                for step in range(len(image_ids)):
                    positions[step] = np.vstack(
                        (positions[step], np.full((new_count, 2), np.nan))
                    )
                    available[step] = np.concatenate(
                        (available[step], np.zeros(new_count, dtype=bool))
                    )
                positions[target_step][-new_count:] = points_new_xy_m
                available[target_step][-new_count:] = True
                trajectory_ids = np.concatenate((trajectory_ids, new_ids))
                seed_image_index = np.concatenate(
                    (
                        seed_image_index,
                        np.full(new_count, target_step, dtype=np.int32),
                    )
                )
                seed_image_id = np.concatenate(
                    (seed_image_id, np.full(new_count, image_keys[target_step], dtype=object))
                )
                rows.append(
                    _graph_rows(
                        target_step,
                        image_ids[target_step],
                        points_new_xy_m,
                        np.ones(new_count, dtype=bool),
                        np.full(new_count, "new_trajectory", dtype=object),
                        trajectory_ids=new_ids,
                        seed_image_index=np.full(
                            new_count, target_step, dtype=np.int32
                        ),
                        seed_image_id=np.full(
                            new_count, image_keys[target_step], dtype=object
                        ),
                    )
                )
                count += new_count
    return pd.concat(rows, ignore_index=True)


def _outgoing_source_points(
    indexed_edges: Sequence[tuple[int, int, FieldEdge]], source_step: int
) -> np.ndarray:
    candidates = []
    for edge_source, _edge_target, edge in indexed_edges:
        if edge_source != source_step:
            continue
        valid = edge.field.available & np.isfinite(edge.field.displacement_m).all(axis=1)
        candidates.append(np.asarray(edge.field.source_xy_m[valid], dtype=np.float64))
    if not candidates:
        return np.empty((0, 2), dtype=np.float64)
    return np.unique(np.vstack(candidates), axis=0)


def _occupancy_positions(
    positions: Sequence[np.ndarray],
    available: Sequence[np.ndarray],
    target_step: int,
    reached: np.ndarray,
) -> np.ndarray:
    """Use current observations and dormant points' last observed positions."""
    occupancy_points = [positions[target_step][reached]]
    dormant_indices = np.flatnonzero(~reached)
    unresolved = np.ones(len(dormant_indices), dtype=bool)
    for previous_step in range(target_step - 1, -1, -1):
        if not np.any(unresolved):
            break
        candidate_indices = dormant_indices[unresolved]
        observed = available[previous_step][candidate_indices]
        if np.any(observed):
            occupancy_points.append(
                positions[previous_step][candidate_indices[observed]]
            )
            unresolved[np.flatnonzero(unresolved)[observed]] = False
    finite = [
        values[np.isfinite(values).all(axis=1)]
        for values in occupancy_points
        if len(values)
    ]
    return np.vstack(finite) if finite else np.empty((0, 2), dtype=np.float64)


def _exclude_occupied_points(
    candidates_xy_m: np.ndarray,
    occupancy_xy_m: np.ndarray,
    exclusion_radius_m: float,
) -> np.ndarray:
    if not len(candidates_xy_m) or not len(occupancy_xy_m):
        return candidates_xy_m
    distance, _ = cKDTree(occupancy_xy_m).query(candidates_xy_m, k=1)
    return candidates_xy_m[distance > exclusion_radius_m]


def _graph_rows(
    image_index: int,
    image_id: str | int,
    positions: np.ndarray,
    active: np.ndarray,
    state: np.ndarray,
    edge_source_image_id: np.ndarray | None = None,
    skipped_images: np.ndarray | None = None,
    displacement_m: np.ndarray | None = None,
    selected_matches: np.ndarray | None = None,
    support_radius_m: np.ndarray | None = None,
    maximum_residual_m: np.ndarray | None = None,
    trajectory_ids: np.ndarray | None = None,
    seed_image_index: np.ndarray | None = None,
    seed_image_id: np.ndarray | None = None,
    reconnected_after_gap: np.ndarray | None = None,
) -> pd.DataFrame:
    count = len(positions)
    if edge_source_image_id is None:
        edge_source_image_id = np.full(count, "", dtype=object)
    if skipped_images is None:
        skipped_images = np.full(count, -1, dtype=np.int32)
    if displacement_m is None:
        displacement_m = np.full((count, 2), np.nan)
    if selected_matches is None:
        selected_matches = np.full(count, np.nan)
    if support_radius_m is None:
        support_radius_m = np.full(count, np.nan)
    if maximum_residual_m is None:
        maximum_residual_m = np.full(count, np.nan)
    if trajectory_ids is None:
        trajectory_ids = np.arange(count, dtype=np.int64)
    if seed_image_index is None:
        seed_image_index = np.zeros(count, dtype=np.int32)
    if seed_image_id is None:
        seed_image_id = np.full(count, str(image_id), dtype=object)
    if reconnected_after_gap is None:
        reconnected_after_gap = np.zeros(count, dtype=bool)
    return pd.DataFrame(
        {
            "trajectory_id": trajectory_ids,
            "image_index": image_index,
            "image_id": str(image_id),
            "seed_image_index": seed_image_index,
            "seed_image_id": seed_image_id,
            "x_m": positions[:, 0],
            "y_m": positions[:, 1],
            "active": active,
            "trajectory_state": state,
            "reconnected_after_gap": reconnected_after_gap,
            "edge_source_image_id": edge_source_image_id,
            "skipped_images": skipped_images,
            "step_dx_m": displacement_m[:, 0],
            "step_dy_m": displacement_m[:, 1],
            "field_selected_matches": selected_matches,
            "field_support_radius_m": support_radius_m,
            "field_maximum_residual_m": maximum_residual_m,
        }
    )
