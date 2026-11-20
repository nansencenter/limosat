"""Global observed Lagrangian identities composed from completed pair fields."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime, timezone
from collections.abc import Callable, Iterator
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .catalog import ImageRecord
from .config import FieldConfig, TrajectoryConfig
from .field import sample_field_with_fallback
from .models import FieldEdge


TargetValidity = Callable[[np.ndarray], np.ndarray]
TargetValidityFactory = Callable[[ImageRecord], TargetValidity]
# ``image -> (xy -> keep mask)`` for new seeds, e.g. excluding land and open water.
SeedFilterFactory = Callable[[ImageRecord], Callable[[np.ndarray], np.ndarray]]


@dataclass(frozen=True)
class TrajectoryPoint:
    trajectory_id: str
    image_id: str
    time_utc: datetime
    state: str
    position_basis: str
    x_m: float
    y_m: float
    source_pair_id: str | None
    selected_matches: float | None = None
    support_radius_m: float | None = None
    maximum_residual_m: float | None = None

    def __post_init__(self) -> None:
        if self.time_utc.tzinfo is None or self.time_utc.utcoffset() is None:
            raise ValueError("trajectory times must be timezone-aware")
        object.__setattr__(self, "time_utc", self.time_utc.astimezone(timezone.utc))
        if self.state not in {"created", "observed"}:
            raise ValueError(f"unknown trajectory state: {self.state}")
        if not np.isfinite([self.x_m, self.y_m]).all():
            raise ValueError("trajectory coordinates must be finite")


@dataclass(frozen=True)
class TrajectoryExtension:
    """One directly measured source-to-target trajectory extension."""

    trajectory_id: str
    pair_id: str
    source_image_id: str
    target_image_id: str
    start_time_utc: datetime
    end_time_utc: datetime
    x0_m: float
    y0_m: float
    x1_m: float
    y1_m: float
    target_state: str
    target_position_basis: str
    selected_matches: float
    support_radius_m: float
    maximum_residual_m: float

    def __post_init__(self) -> None:
        start = _utc(self.start_time_utc, "start_time_utc")
        end = _utc(self.end_time_utc, "end_time_utc")
        object.__setattr__(self, "start_time_utc", start)
        object.__setattr__(self, "end_time_utc", end)
        if end <= start:
            raise ValueError("trajectory extension time must increase")
        if not np.isfinite([self.x0_m, self.y0_m, self.x1_m, self.y1_m]).all():
            raise ValueError("trajectory extension coordinates must be finite")

    @property
    def elapsed_seconds(self) -> float:
        return (self.end_time_utc - self.start_time_utc).total_seconds()

    @property
    def displacement_m(self) -> np.ndarray:
        return np.asarray(
            [self.x1_m - self.x0_m, self.y1_m - self.y0_m],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class MergeEvent:
    """A trajectory ended because another one occupies the same ice (merge_radius_m)."""

    trajectory_id: str
    merged_into: str
    image_id: str
    time_utc: datetime
    separation_m: float


@dataclass(frozen=True)
class TrajectoryBatch:
    """Per-image points and their directly measured incoming extensions."""

    points: tuple[TrajectoryPoint, ...]
    extensions: tuple[TrajectoryExtension, ...]
    merges: tuple[MergeEvent, ...] = ()


@dataclass(frozen=True)
class _Continuation:
    source_step: int
    pair_id: str
    source_time_utc: datetime
    displacement_m: np.ndarray
    selected_matches: float
    support_radius_m: float
    maximum_residual_m: float
    edge: FieldEdge | None = None


@dataclass(frozen=True)
class PairSpec:
    """A primary pair as composition schedules it (no field data)."""

    pair_id: str
    source_image_id: str
    target_image_id: str
    source_time_utc: datetime


@dataclass(frozen=True)
class PairSamples:
    """One pair's field sampled at the positions of its source image: the tracks it can
    continue (ascending identity, target validity applied) and their step values."""

    pair_id: str
    identities: tuple[str, ...]
    displacement_m: np.ndarray
    selected_matches: np.ndarray
    support_radius_m: np.ndarray
    maximum_residual_m: np.ndarray
    product: dict | None = None


def sample_edge(
    edge: FieldEdge,
    identities: Sequence[str],
    queries: np.ndarray,
    field_config: FieldConfig,
    target_validity: TargetValidity | None = None,
) -> PairSamples:
    """Sample a pair field at ``queries`` (the positions of ``identities``, ascending)."""
    empty = PairSamples(edge.field.pair_id, (), np.empty((0, 2)), np.empty(0), np.empty(0), np.empty(0))
    if not len(identities):
        return empty
    sampled = sample_field_with_fallback(
        edge.field, edge.fallback_field, queries,
        field_config.maximum_triangle_edge_m,
        field_config=field_config, gap_matches=edge.gap_matches,
    )
    indices = np.flatnonzero(sampled.available)
    if target_validity is not None and len(indices):
        target_xy = queries[indices] + sampled.displacement_m[indices]
        valid = np.asarray(target_validity(target_xy), dtype=bool)
        if valid.shape != (len(indices),):
            raise ValueError("target validity must return one flag per endpoint")
        indices = indices[valid]
    return PairSamples(
        edge.field.pair_id,
        tuple(identities[i] for i in indices),
        np.asarray(sampled.displacement_m[indices], dtype=np.float64),
        np.asarray(sampled.selected_matches[indices], dtype=np.float64),
        np.asarray(sampled.support_radius_m[indices], dtype=np.float64),
        np.asarray(sampled.maximum_residual_m[indices], dtype=np.float64),
    )


def _choose(
    samples: Sequence[tuple[int, datetime, PairSamples, FieldEdge | None]],
    excluded: set[str] | frozenset[str] = frozenset(),
) -> dict[str, _Continuation]:
    """The best supported continuation per track over the incoming pairs, visited in
    their composition order (``_continuation_key``; ties keep the earlier one)."""
    chosen: dict[str, _Continuation] = {}
    for source_step, source_time, values, edge in samples:
        for k, identity in enumerate(values.identities):
            if identity in excluded:
                continue
            candidate = _Continuation(
                source_step, values.pair_id, source_time,
                values.displacement_m[k],
                float(values.selected_matches[k]),
                float(values.support_radius_m[k]),
                float(values.maximum_residual_m[k]),
                edge,
            )
            current = chosen.get(identity)
            if current is None or _continuation_key(candidate) > _continuation_key(current):
                chosen[identity] = candidate
    return chosen


def iter_global_trajectory_batches(
    edges: Sequence[FieldEdge],
    images: Sequence[ImageRecord],
    field_config: FieldConfig,
    trajectory_config: TrajectoryConfig,
    *,
    target_validity_factory: TargetValidityFactory | None = None,
    seed_filter_factory: SeedFilterFactory | None = None,
) -> Iterator[TrajectoryBatch]:
    """Compose in-memory pair fields (see ``compose_batches``), sampling each pair in this
    process as soon as its source image is complete."""
    by_id = {edge.field.pair_id: edge for edge in edges}
    ordered = tuple(sorted(images, key=lambda image: (image.time_utc, image.image_id)))
    validity: dict[int, TargetValidity | None] = {}

    def submit(pair_id, identities, queries, target_step):
        if target_validity_factory is not None and target_step not in validity and len(identities):
            validity[target_step] = target_validity_factory(ordered[target_step])
        return _Done(sample_edge(by_id[pair_id], identities, queries, field_config, validity.get(target_step)))

    def nodes(pair_id):
        field = by_id[pair_id].field
        return field.source_xy_m[field.available]

    specs = [
        PairSpec(edge.field.pair_id, edge.field.source_image_id, edge.field.target_image_id,
                 edge.field.source_time_utc)
        for edge in edges
    ]
    yield from compose_batches(specs, images, trajectory_config, nodes=nodes, submit=submit,
                               seed_filter_factory=seed_filter_factory, edges=by_id)


class _Done:
    """A completed result with the ``Future.result()`` interface."""

    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


def compose_batches(
    pairs: Sequence[PairSpec],
    images: Sequence[ImageRecord],
    trajectory_config: TrajectoryConfig,
    *,
    nodes: Callable[[str], np.ndarray],
    submit: Callable[[str, tuple[str, ...], np.ndarray, int], object],
    seed_filter_factory: SeedFilterFactory | None = None,
    edges: dict[str, FieldEdge] | None = None,
) -> Iterator[TrajectoryBatch]:
    """Yield deterministic points and direct extensions for each image.

    At each image a trajectory continues from its most recent position through the best
    supported incoming primary pair field. Gaps are not stored: a trajectory simply has
    no row on an image that a pair from its last position reaches without supporting it.

    Pair fields are not held here. Once an image is complete (continuations, merges and
    new seeds), each pair leaving it is handed to ``submit(pair_id, identities, queries,
    target_step)``, which returns a future of the pair's ``PairSamples`` at those
    positions; the futures are collected when the pair's target image is composed. So
    sampling can run in other processes, ahead of the sequential choice of continuations.
    ``nodes(pair_id)`` gives a pair's field nodes with a displacement (new seeds).
    """
    ordered_images = tuple(
        sorted(images, key=lambda image: (image.time_utc, image.image_id))
    )
    if not ordered_images:
        raise ValueError("trajectory composition requires catalogue images")
    image_keys = [image.image_id for image in ordered_images]
    if len(image_keys) != len(set(image_keys)):
        raise ValueError("trajectory image identity must be globally unique")
    image_index = {image_id: index for index, image_id in enumerate(image_keys)}
    indexed: list[tuple[int, int, PairSpec]] = []
    for spec in pairs:
        source = image_index.get(spec.source_image_id)
        target = image_index.get(spec.target_image_id)
        if source is None or target is None or source >= target:
            raise ValueError("every pair field must point forward between listed images")
        indexed.append((source, target, spec))
    indexed.sort(key=lambda item: (item[1], item[0], item[2].pair_id))
    incoming_by_target: dict[int, list[tuple[int, int, PairSpec]]] = {}
    outgoing_by_source: dict[int, list[tuple[int, int, PairSpec]]] = {}
    for item in indexed:
        incoming_by_target.setdefault(item[1], []).append(item)
        outgoing_by_source.setdefault(item[0], []).append(item)

    positions: list[dict[str, np.ndarray]] = [dict() for _ in ordered_images]
    pending: dict[str, object] = {}
    identities: set[str] = set()
    observations: dict[str, int] = {}
    merged: set[str] = set()
    merge_radius = trajectory_config.merge_radius_m
    young_hours = trajectory_config.merge_maximum_younger_age_hours
    created_step: dict[str, int] = {}
    young_steps: dict[str, list[int]] = {}
    last_use: dict[int, int] = {}
    for source, target, _spec in indexed:
        last_use[source] = max(last_use.get(source, target), target)

    for step, image in enumerate(ordered_images):
        points: list[TrajectoryPoint] = []
        extensions: list[TrajectoryExtension] = []
        merges: list[MergeEvent] = []
        if step:
            samples = [
                (source_step, spec.source_time_utc, pending.pop(spec.pair_id).result(),
                 edges.get(spec.pair_id) if edges else None)
                for source_step, _target_step, spec in incoming_by_target.get(step, ())
            ]
            chosen = _choose(samples, merged)
            for identity in sorted(chosen):
                continuation = chosen[identity]
                source_xy = positions[continuation.source_step][identity]
                xy = source_xy + continuation.displacement_m
                positions[step][identity] = xy
                points.append(
                    _point(
                        identity,
                        image,
                        "observed",
                        "primary_pair_field",
                        xy,
                        continuation.pair_id,
                        continuation.selected_matches,
                        continuation.support_radius_m,
                        continuation.maximum_residual_m,
                    )
                )
                source_image = ordered_images[continuation.source_step]
                extensions.append(
                    TrajectoryExtension(
                        trajectory_id=identity,
                        pair_id=continuation.pair_id,
                        source_image_id=source_image.image_id,
                        target_image_id=image.image_id,
                        start_time_utc=source_image.time_utc,
                        end_time_utc=image.time_utc,
                        x0_m=float(source_xy[0]),
                        y0_m=float(source_xy[1]),
                        x1_m=float(xy[0]),
                        y1_m=float(xy[1]),
                        target_state="observed",
                        target_position_basis="primary_pair_field",
                        selected_matches=continuation.selected_matches,
                        support_radius_m=continuation.support_radius_m,
                        maximum_residual_m=continuation.maximum_residual_m,
                    )
                )

        if merge_radius is not None and len(positions[step]) > 1:
            eligible_loser = None
            if young_hours is not None:
                limit = young_hours * 3_600.0

                def eligible_loser(loser: str, winner: str) -> bool:
                    born = created_step.get(loser)
                    if born is None:
                        return False
                    age = (image.time_utc - ordered_images[born].time_utc).total_seconds()
                    if age > limit:
                        return False
                    return not any(
                        winner in positions[earlier]
                        for earlier in young_steps.get(loser, ())
                        if earlier != step
                    )
            merges = _merge_duplicates(
                positions[step], points, observations, merge_radius, image,
                eligible_loser=eligible_loser,
            )
            if merges:
                ended = {event.trajectory_id for event in merges}
                merged.update(ended)
                points = [point for point in points if point.trajectory_id not in ended]
                extensions = [
                    extension for extension in extensions
                    if extension.trajectory_id not in ended
                ]
        for point in points:
            observations[point.trajectory_id] = observations.get(point.trajectory_id, 0) + 1
            if young_hours is not None and point.trajectory_id in young_steps:
                young_steps[point.trajectory_id].append(step)
        if young_hours is not None and young_steps:
            # Forget the history of trajectories too old to be merged.
            limit = young_hours * 3_600.0
            for identity in [
                identity for identity, steps in young_steps.items()
                if (image.time_utc - ordered_images[created_step[identity]].time_utc).total_seconds() > limit
            ]:
                del young_steps[identity]

        if trajectory_config.add_as_coverage_enters:
            candidates = _outgoing_points(outgoing_by_source.get(step, ()), nodes)
            if seed_filter_factory is not None and len(candidates):
                candidates = candidates[
                    np.asarray(seed_filter_factory(image)(candidates), dtype=bool)
                ]
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
                observations[identity] = 1
                if young_hours is not None:
                    created_step[identity] = step
                    young_steps[identity] = [step]
                points.append(
                    _point(identity, image, "created", "seed_grid", xy, None)
                )
        # The image is complete: sample every pair leaving it at its positions.
        if outgoing_by_source.get(step):
            identities_here = tuple(sorted(positions[step]))
            queries = (
                np.vstack([positions[step][identity] for identity in identities_here])
                if identities_here else np.empty((0, 2), dtype=np.float64)
            )
            for _source_step, target_step, spec in outgoing_by_source[step]:
                pending[spec.pair_id] = submit(spec.pair_id, identities_here, queries, target_step)
        yield TrajectoryBatch(
            points=tuple(sorted(points, key=lambda point: point.trajectory_id)),
            extensions=tuple(
                sorted(
                    extensions,
                    key=lambda extension: (
                        extension.trajectory_id,
                        extension.pair_id,
                    ),
                )
            ),
            merges=tuple(sorted(merges, key=lambda event: event.trajectory_id)),
        )
        for source_step, target_step in tuple(last_use.items()):
            if target_step == step:
                positions[source_step].clear()
        if step not in last_use:
            positions[step].clear()


def _merge_duplicates(
    current: dict[str, np.ndarray],
    points: Sequence[TrajectoryPoint],
    observations: dict[str, int],
    radius_m: float,
    image: ImageRecord,
    eligible_loser=None,
) -> list[MergeEvent]:
    """End all but one trajectory of each cluster of positions within ``radius_m`` in
    this image; ``current`` loses the ended ones. The survivor has the longest
    history, then the convergence-audit order. ``eligible_loser(loser, survivor)``,
    when given, keeps trajectories it rejects (merge_maximum_younger_age_hours)."""
    identities = sorted(current)
    coordinates = np.vstack([current[identity] for identity in identities])
    pairs = cKDTree(coordinates).query_pairs(radius_m, output_type="ndarray")
    if not len(pairs):
        return []
    by_identity = {point.trajectory_id: point for point in points}
    counts = {identity: observations.get(identity, 0) + 1 for identity in identities}
    events: list[MergeEvent] = []
    for cluster in _index_clusters(len(identities), pairs):
        ranked = sorted(
            cluster,
            key=lambda index: _convergence_rank(by_identity[identities[index]], counts),
        )
        winner = identities[ranked[0]]
        for index in ranked[1:]:
            loser = identities[index]
            if eligible_loser is not None and not eligible_loser(loser, winner):
                continue
            events.append(MergeEvent(
                trajectory_id=loser, merged_into=winner, image_id=image.image_id,
                time_utc=image.time_utc,
                separation_m=float(np.linalg.norm(coordinates[index] - coordinates[ranked[0]])),
            ))
            del current[loser]
    return events


def _convergence_rank(
    point: TrajectoryPoint, observation_count: dict[str, int]
) -> tuple:
    selected = point.selected_matches
    residual = point.maximum_residual_m
    return (
        -observation_count[point.trajectory_id],
        -(selected if selected is not None and np.isfinite(selected) else 0.0),
        residual if residual is not None and np.isfinite(residual) else np.inf,
        point.trajectory_id,
    )


def _index_clusters(count: int, pairs: np.ndarray) -> tuple[tuple[int, ...], ...]:
    parents = list(range(count))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left, right in np.asarray(pairs, dtype=int).reshape(-1, 2):
        left_root, right_root = root(int(left)), root(int(right))
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)
    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(root(index), []).append(index)
    return tuple(
        tuple(indices)
        for _label, indices in sorted(grouped.items())
        if len(indices) > 1
    )


def stable_trajectory_id(seed_image_id: str, xy_m: np.ndarray) -> str:
    """Derive a global identity from seed image and millimetre-rounded position."""
    xy = np.asarray(xy_m, dtype=np.float64)
    encoded = f"{seed_image_id}|{xy[0]:.3f}|{xy[1]:.3f}".encode("utf-8")
    return "trj_" + hashlib.sha256(encoded).hexdigest()[:20]


def _supported_continuations(
    incoming: Sequence[tuple[int, int, FieldEdge]],
    positions: Sequence[dict[str, np.ndarray]],
    identities: Sequence[str],
    field_config: FieldConfig,
    *,
    target_validity: TargetValidity | None = None,
) -> dict[str, _Continuation]:
    """Best continuation per identity over in-memory incoming pair fields."""
    identity_set = set(identities)
    samples = []
    for source_step, _target_step, edge in incoming:
        eligible = tuple(sorted(identity_set.intersection(positions[source_step])))
        if not eligible:
            continue
        queries = np.vstack([positions[source_step][identity] for identity in eligible])
        samples.append((source_step, edge.field.source_time_utc,
                        sample_edge(edge, eligible, queries, field_config, target_validity), edge))
    return _choose(samples)


def _continuation_key(candidate: _Continuation) -> tuple:
    residual = candidate.maximum_residual_m
    radius = candidate.support_radius_m
    return (
        candidate.source_time_utc,
        candidate.selected_matches,
        -residual if np.isfinite(residual) else -np.inf,
        -radius if np.isfinite(radius) else -np.inf,
        _reverse_text(candidate.pair_id),
    )


@lru_cache(maxsize=65536)
def _reverse_text(value: str) -> tuple[int, ...]:
    return tuple(-ord(character) for character in value)


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _outgoing_points(outgoing, nodes) -> np.ndarray:
    values = [nodes(spec.pair_id) for _source, _target, spec in outgoing]
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
        x_m=float(xy[0]),
        y_m=float(xy[1]),
        source_pair_id=source_pair_id,
        selected_matches=selected,
        support_radius_m=radius,
        maximum_residual_m=residual,
    )
