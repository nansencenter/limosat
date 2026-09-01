"""Global observed Lagrangian identities composed from completed pair fields."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Iterator
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

    def __post_init__(self) -> None:
        if self.time_utc.tzinfo is None or self.time_utc.utcoffset() is None:
            raise ValueError("trajectory times must be timezone-aware")
        object.__setattr__(self, "time_utc", self.time_utc.astimezone(timezone.utc))
        if self.state not in {"created", "observed", "dormant", "reappeared"}:
            raise ValueError(f"unknown trajectory state: {self.state}")
        if (self.x_m is None) != (self.y_m is None):
            raise ValueError("trajectory coordinates must both be present or both be NULL")
        if self.state == "dormant" and self.available:
            raise ValueError("dormant trajectory coordinates must be NULL")
        if self.state != "dormant" and not self.available:
            raise ValueError("measured trajectory states require coordinates")

    @property
    def available(self) -> bool:
        return self.x_m is not None and self.y_m is not None


@dataclass(frozen=True)
class _Continuation:
    source_step: int
    edge: FieldEdge
    displacement_m: np.ndarray
    selected_matches: float
    support_radius_m: float
    maximum_residual_m: float


def compose_global_trajectories(
    edges: Sequence[FieldEdge],
    images: Sequence[ImageRecord],
    field_config: FieldConfig,
    trajectory_config: TrajectoryConfig,
) -> tuple[TrajectoryPoint, ...]:
    """Recompose the catalogue without treating compute labels as boundaries."""
    return tuple(
        point
        for batch in iter_global_trajectory_points(
            edges, images, field_config, trajectory_config
        )
        for point in batch
    )


def iter_global_trajectory_points(
    edges: Sequence[FieldEdge],
    images: Sequence[ImageRecord],
    field_config: FieldConfig,
    trajectory_config: TrajectoryConfig,
) -> Iterator[tuple[TrajectoryPoint, ...]]:
    """Yield deterministic per-image rows while retaining only needed positions."""
    ordered_images = tuple(
        sorted(images, key=lambda image: (image.time_utc, image.image_id))
    )
    if not ordered_images:
        raise ValueError("trajectory composition requires catalogue images")
    image_keys = [image.image_id for image in ordered_images]
    if len(image_keys) != len(set(image_keys)):
        raise ValueError("trajectory image identity must be globally unique")
    image_index = {image_id: index for index, image_id in enumerate(image_keys)}
    indexed: list[tuple[int, int, FieldEdge]] = []
    for edge in edges:
        source = image_index.get(edge.source_image_id)
        target = image_index.get(edge.target_image_id)
        if source is None or target is None or source >= target:
            raise ValueError("every pair field must point forward between listed images")
        indexed.append((source, target, edge))
    indexed.sort(
        key=lambda item: (
            item[1],
            item[0],
            item[2].pair_kind,
            item[2].field.pair_id,
        )
    )
    incoming_by_target: dict[int, list[tuple[int, int, FieldEdge]]] = {}
    for item in indexed:
        incoming_by_target.setdefault(item[1], []).append(item)

    positions: list[dict[str, np.ndarray]] = [dict() for _ in ordered_images]
    identities: set[str] = set()
    last_use: dict[int, int] = {}
    for source, target, _edge in indexed:
        last_use[source] = max(last_use.get(source, target), target)

    for step, image in enumerate(ordered_images):
        points: list[TrajectoryPoint] = []
        if step:
            incoming = incoming_by_target.get(step, ())
            eligible = sorted(
                {
                    identity
                    for source_step, _target_step, _edge in incoming
                    for identity in positions[source_step]
                }
            )
            chosen = _choose_continuations(
                incoming, positions, eligible, field_config
            )
            for identity in eligible:
                continuation = chosen.get(identity)
                if continuation is None:
                    points.append(
                        _point(identity, image, "dormant", "missing", None, None)
                    )
                    continue
                xy = (
                    positions[continuation.source_step][identity]
                    + continuation.displacement_m
                )
                positions[step][identity] = xy
                state = (
                    "reappeared"
                    if continuation.edge.pair_kind == "recovery"
                    else "observed"
                )
                points.append(
                    _point(
                        identity,
                        image,
                        state,
                        f"{continuation.edge.pair_kind}_pair_field",
                        xy,
                        continuation.edge.field.pair_id,
                        continuation.selected_matches,
                        continuation.support_radius_m,
                        continuation.maximum_residual_m,
                    )
                )

        if trajectory_config.add_as_coverage_enters:
            candidates = _outgoing_primary_points(indexed, step)
            occupancy = _active_positions(positions[step])
            for xy in _exclude_occupied(
                candidates,
                occupancy,
                trajectory_config.new_point_exclusion_radius_m,
            ):
                identity = stable_trajectory_id(image.image_id, xy)
                if identity in identities:
                    continue
                identities.add(identity)
                positions[step][identity] = xy
                points.append(
                    _point(identity, image, "created", "seed_grid", xy, None)
                )
        yield tuple(
            sorted(
                points,
                key=lambda point: point.trajectory_id,
            )
        )
        for source_step, target_step in tuple(last_use.items()):
            if target_step == step:
                positions[source_step].clear()
        if step not in last_use:
            positions[step].clear()


def build_trajectories(
    edges: Sequence[FieldEdge],
    images: Sequence[ImageRecord],
    field_config: FieldConfig,
    trajectory_config: TrajectoryConfig,
) -> tuple[TrajectoryPoint, ...]:
    """Compatibility name for the global catalogue composer."""
    return compose_global_trajectories(
        edges, images, field_config, trajectory_config
    )


def targeted_recovery_positions(
    points: Sequence[TrajectoryPoint],
    source_image_id: str,
    target_image_id: str,
) -> np.ndarray:
    """Return measured source positions for parcels dormant at a later target."""
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
    return np.asarray(
        [[point.x_m, point.y_m] for point in selected], dtype=np.float64
    )


def stable_trajectory_id(seed_image_id: str, xy_m: np.ndarray) -> str:
    """Derive a global identity from seed image and millimetre-rounded position."""
    xy = np.asarray(xy_m, dtype=np.float64)
    encoded = f"{seed_image_id}|{xy[0]:.3f}|{xy[1]:.3f}".encode("utf-8")
    return "trj_" + hashlib.sha256(encoded).hexdigest()[:20]


def _choose_continuations(
    incoming: Sequence[tuple[int, int, FieldEdge]],
    positions: Sequence[dict[str, np.ndarray]],
    identities: Sequence[str],
    field_config: FieldConfig,
) -> dict[str, _Continuation]:
    primary = _supported_continuations(
        [item for item in incoming if item[2].pair_kind == "primary"],
        positions,
        identities,
        field_config,
    )
    remaining = [identity for identity in identities if identity not in primary]
    recovery = _supported_continuations(
        [item for item in incoming if item[2].pair_kind == "recovery"],
        positions,
        remaining,
        field_config,
    )
    return {**primary, **recovery}


def _supported_continuations(
    incoming: Sequence[tuple[int, int, FieldEdge]],
    positions: Sequence[dict[str, np.ndarray]],
    identities: Sequence[str],
    field_config: FieldConfig,
) -> dict[str, _Continuation]:
    chosen: dict[str, _Continuation] = {}
    for source_step, _target_step, edge in incoming:
        eligible = [
            identity for identity in identities if identity in positions[source_step]
        ]
        if not eligible:
            continue
        queries = np.vstack([positions[source_step][identity] for identity in eligible])
        sampled = sample_field(
            edge.field, queries, field_config.maximum_triangle_edge_m
        )
        for local_index in np.flatnonzero(sampled.available):
            identity = eligible[local_index]
            candidate = _Continuation(
                source_step,
                edge,
                sampled.displacement_m[local_index],
                float(sampled.selected_matches[local_index]),
                float(sampled.support_radius_m[local_index]),
                float(sampled.maximum_residual_m[local_index]),
            )
            current = chosen.get(identity)
            if current is None or _continuation_key(candidate) > _continuation_key(
                current
            ):
                chosen[identity] = candidate
    return chosen


def _continuation_key(candidate: _Continuation) -> tuple:
    residual = candidate.maximum_residual_m
    radius = candidate.support_radius_m
    return (
        candidate.edge.field.source_time_utc,
        candidate.selected_matches,
        -residual if np.isfinite(residual) else -np.inf,
        -radius if np.isfinite(radius) else -np.inf,
        _reverse_text(candidate.edge.field.pair_id),
    )


def _reverse_text(value: str) -> tuple[int, ...]:
    return tuple(-ord(character) for character in value)


def _outgoing_primary_points(indexed, source_step: int) -> np.ndarray:
    values = [
        edge.field.source_xy_m[edge.field.available]
        for source, _target, edge in indexed
        if source == source_step and edge.pair_kind == "primary"
    ]
    values = [value for value in values if len(value)]
    return (
        np.unique(np.vstack(values), axis=0)
        if values
        else np.empty((0, 2), dtype=np.float64)
    )


def _active_positions(positions: dict[str, np.ndarray]) -> np.ndarray:
    return (
        np.vstack([positions[identity] for identity in sorted(positions)])
        if positions
        else np.empty((0, 2), dtype=np.float64)
    )


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
