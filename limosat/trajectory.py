"""Observed Lagrangian identities composed from pair-field edges."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .catalog import ImageRecord
from .config import FieldConfig, TrajectoryConfig
from .field import sample_field
from .models import FieldEdge


@dataclass(frozen=True)
class TrajectoryPoint:
    trajectory_id: str
    image_id: str
    time_utc: datetime
    state: str
    position_basis: str
    x_m: float | None
    y_m: float | None
    source_pair_id: str | None
    selected_matches: float | None = None
    support_radius_m: float | None = None
    maximum_residual_m: float | None = None

    @property
    def available(self) -> bool:
        return self.x_m is not None and self.y_m is not None


def build_trajectories(
    edges: Sequence[FieldEdge],
    images: Sequence[ImageRecord],
    field_config: FieldConfig,
    trajectory_config: TrajectoryConfig,
) -> tuple[TrajectoryPoint, ...]:
    """Build stable identities with creation, dormancy, and observed reappearance."""
    if len(images) < 2 or not edges:
        raise ValueError("trajectory construction requires images and field edges")
    image_keys = [image.image_id for image in images]
    if len(image_keys) != len(set(image_keys)):
        raise ValueError("trajectory image identity must be unique")
    if any(right.time_utc <= left.time_utc for left, right in zip(images, images[1:])):
        raise ValueError("trajectory images must be strictly chronological")
    image_index = {image_id: index for index, image_id in enumerate(image_keys)}
    indexed: list[tuple[int, int, FieldEdge]] = []
    for edge in edges:
        source = image_index.get(edge.source_image_id)
        target = image_index.get(edge.target_image_id)
        if source is None or target is None or source >= target:
            raise ValueError("every field edge must point forward between listed images")
        indexed.append((source, target, edge))

    initial = _outgoing_points(indexed, 0)
    if not len(initial):
        raise ValueError("first image has no supported outgoing field")
    positions: list[dict[str, np.ndarray]] = [dict() for _ in images]
    points: list[TrajectoryPoint] = []
    for xy in initial:
        identity = stable_trajectory_id(images[0].image_id, xy)
        positions[0][identity] = xy
        points.append(_point(identity, images[0], "created", "seed_grid", xy, None))
    identities = list(positions[0])

    for target_step in range(1, len(images)):
        target_edges = sorted(
            ((source, edge) for source, target, edge in indexed if target == target_step),
            key=lambda item: target_step - item[0],
        )
        chosen: dict[str, tuple[int, FieldEdge, np.ndarray, float, float, float]] = {}
        for source_step, edge in target_edges:
            eligible = [identity for identity in identities if identity in positions[source_step]]
            if not eligible:
                continue
            queries = np.vstack([positions[source_step][identity] for identity in eligible])
            sampled = sample_field(
                edge.field, queries, field_config.maximum_triangle_edge_m
            )
            for local_index in np.flatnonzero(sampled.available):
                identity = eligible[local_index]
                candidate = (
                    source_step,
                    edge,
                    sampled.displacement_m[local_index],
                    float(sampled.selected_matches[local_index]),
                    float(sampled.support_radius_m[local_index]),
                    float(sampled.maximum_residual_m[local_index]),
                )
                current = chosen.get(identity)
                if current is None or _edge_better(
                    candidate, current, target_step
                ):
                    chosen[identity] = candidate

        previous_available = positions[target_step - 1]
        for identity in identities:
            if identity not in chosen:
                points.append(
                    _point(identity, images[target_step], "dormant", "missing", None, None)
                )
                continue
            source_step, edge, displacement, selected, radius, residual = chosen[identity]
            xy = positions[source_step][identity] + displacement
            positions[target_step][identity] = xy
            reappeared = identity not in previous_available
            points.append(
                _point(
                    identity,
                    images[target_step],
                    "reappeared" if reappeared else "observed",
                    (
                        "field_advected_adjacent"
                        if target_step - source_step == 1
                        else "field_advected_skip"
                    ),
                    xy,
                    edge.field.pair_id,
                    selected,
                    radius,
                    residual,
                )
            )

        if trajectory_config.add_as_coverage_enters:
            candidates = _outgoing_points(indexed, target_step)
            occupancy = _occupancy(positions, identities, target_step)
            for xy in _exclude_occupied(
                candidates,
                occupancy,
                trajectory_config.new_point_exclusion_radius_m,
            ):
                identity = stable_trajectory_id(images[target_step].image_id, xy)
                if identity in identities:
                    continue
                identities.append(identity)
                positions[target_step][identity] = xy
                points.append(
                    _point(
                        identity,
                        images[target_step],
                        "created",
                        "seed_grid",
                        xy,
                        None,
                    )
                )
    return tuple(points)


def targeted_recovery_positions(
    points: Sequence[TrajectoryPoint],
    source_image_id: str,
    target_image_id: str,
) -> np.ndarray:
    """Return measured source positions for identities dormant at a later target."""
    source = {
        point.trajectory_id: point
        for point in points
        if point.image_id == source_image_id and point.available
    }
    dormant = {
        point.trajectory_id
        for point in points
        if point.image_id == target_image_id and point.state == "dormant"
    }
    selected = [source[identity] for identity in sorted(dormant & source.keys())]
    if not selected:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray([[point.x_m, point.y_m] for point in selected], dtype=np.float64)


def stable_trajectory_id(seed_image_id: str, xy_m: np.ndarray) -> str:
    """Derive an identity from the exact seed image and millimetre-rounded position."""
    xy = np.asarray(xy_m, dtype=np.float64)
    encoded = f"{seed_image_id}|{xy[0]:.3f}|{xy[1]:.3f}".encode("utf-8")
    return "trj_" + hashlib.sha256(encoded).hexdigest()[:20]


def _edge_better(candidate, current, target_step: int) -> bool:
    candidate_skip = target_step - candidate[0] - 1
    current_skip = target_step - current[0] - 1
    return candidate_skip < current_skip or (
        candidate_skip == current_skip and candidate[3] > current[3]
    )


def _outgoing_points(indexed, source_step: int) -> np.ndarray:
    values = []
    for source, _target, edge in indexed:
        if source == source_step:
            valid = edge.field.available
            values.append(edge.field.source_xy_m[valid])
    return (
        np.unique(np.vstack(values), axis=0)
        if values
        else np.empty((0, 2), dtype=np.float64)
    )


def _occupancy(positions, identities, target_step: int) -> np.ndarray:
    values = []
    for identity in identities:
        for step in range(target_step, -1, -1):
            if identity in positions[step]:
                values.append(positions[step][identity])
                break
    return np.asarray(values, dtype=np.float64) if values else np.empty((0, 2))


def _exclude_occupied(candidates, occupancy, radius_m: float) -> np.ndarray:
    if not len(candidates) or not len(occupancy):
        return candidates
    distance, _ = cKDTree(occupancy).query(candidates, k=1)
    return candidates[distance > radius_m]


def _point(
    identity,
    image,
    state,
    basis,
    xy,
    source_pair_id,
    selected=None,
    radius=None,
    residual=None,
) -> TrajectoryPoint:
    return TrajectoryPoint(
        trajectory_id=identity,
        image_id=image.image_id,
        time_utc=image.time_utc,
        state=state,
        position_basis=basis,
        x_m=None if xy is None else float(xy[0]),
        y_m=None if xy is None else float(xy[1]),
        source_pair_id=source_pair_id,
        selected_matches=selected,
        support_radius_m=radius,
        maximum_residual_m=residual,
    )
