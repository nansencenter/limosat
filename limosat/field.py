"""Consensus fields, explicit missing support, and topology validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import shapely
from scipy.spatial import Delaunay, QhullError, cKDTree
from shapely.geometry.base import BaseGeometry

from .catalog import ImagePair
from .config import FieldConfig
from .models import DisplacementField, FieldEdge, MotionMatches


@dataclass(frozen=True)
class FieldSamples:
    displacement_m: np.ndarray
    available: np.ndarray
    selected_matches: np.ndarray
    support_radius_m: np.ndarray
    maximum_residual_m: np.ndarray


def regular_grid(
    domain: BaseGeometry, spacing_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if domain.is_empty:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty((0, 2))
    minx, miny, maxx, maxy = domain.bounds
    xs = np.arange(math.ceil(minx / spacing_m) * spacing_m, maxx, spacing_m)
    ys = np.arange(math.ceil(miny / spacing_m) * spacing_m, maxy, spacing_m)
    x_grid, y_grid = np.meshgrid(xs, ys)
    inside = shapely.intersects_xy(domain, x_grid.ravel(), y_grid.ravel())
    rows = np.repeat(np.arange(len(ys)), len(xs))[inside].astype(np.int32)
    columns = np.tile(np.arange(len(xs)), len(ys))[inside].astype(np.int32)
    return rows, columns, np.column_stack((x_grid.ravel()[inside], y_grid.ravel()[inside])).astype(np.float64)


def estimate_field(
    matches: MotionMatches,
    pair: ImagePair,
    domain: BaseGeometry,
    config: FieldConfig,
) -> DisplacementField:
    rows, columns, query = regular_grid(domain, config.grid_spacing_m)
    if config.strong_match_score is None:
        estimates = estimate_queries(matches, query, config)
    else:
        estimates = _strong_first_estimates(matches, query, config)
    return DisplacementField(
        pair_id=pair.pair_id,
        source_image_id=pair.source.image_id,
        target_image_id=pair.target.image_id,
        source_time_utc=pair.source.time_utc,
        target_time_utc=pair.target.time_utc,
        grid_row=rows,
        grid_column=columns,
        source_xy_m=query,
        **estimates,
    )


def _strong_first_estimates(
    matches: MotionMatches, query: np.ndarray, config: FieldConfig
) -> dict[str, np.ndarray]:
    """Node consensus from the strong matches, then from every match where that fails."""
    strong = np.flatnonzero(matches.score >= config.strong_match_score)
    first = estimate_queries(_match_subset(matches, strong), query, config)
    failed = np.flatnonzero(~first["available"])
    if len(failed):
        fallback = replace(
            config, strong_match_score=None,
            neighbour_count=config.fallback_neighbour_count or config.neighbour_count,
            minimum_agreeing_matches=config.fallback_minimum_agreeing_matches or config.minimum_agreeing_matches,
            agreement_distance_m=config.fallback_agreement_distance_m or config.agreement_distance_m,
        )
        second = estimate_queries(matches, query[failed], fallback)
        for name, values in first.items():
            values[failed] = second[name]
    return first


def _match_subset(matches: MotionMatches, keep: np.ndarray) -> MotionMatches:
    return MotionMatches(matches.source_xy_m[keep], matches.target_xy_m[keep], matches.score[keep],
                         matches.source_tile[keep], matches.target_tile[keep])


def estimate_queries(
    matches: MotionMatches, query_xy_m: np.ndarray, config: FieldConfig
) -> dict[str, np.ndarray]:
    query = np.asarray(query_xy_m, dtype=np.float64)
    count = len(query)
    displacement = np.full((count, 2), np.nan, dtype=np.float64)
    available = np.zeros(count, dtype=bool)
    selected = np.zeros(count, dtype=np.int32)
    candidates = np.zeros(count, dtype=np.int32)
    radius = np.full(count, np.nan, dtype=np.float64)
    residual = np.full(count, np.nan, dtype=np.float64)
    if not len(matches):
        return _estimates(displacement, available, selected, candidates, radius, residual)
    vectors = matches.displacement_m
    for index, (distances, neighbours) in enumerate(
        _canonical_neighbours(matches, query, config)
    ):
        candidates[index] = len(neighbours)
        if not len(neighbours):
            continue
        radius[index] = distances.max()
        local = vectors[neighbours]
        weights = np.maximum(matches.score[neighbours], 1.0e-12)
        separation = np.linalg.norm(local[:, None] - local[None, :], axis=2)
        agreeing = separation[int(np.argmax((separation <= config.agreement_distance_m) @ weights))] <= config.agreement_distance_m
        selected[index] = int(agreeing.sum())
        if selected[index] < config.minimum_agreeing_matches:
            continue
        estimate = weighted_geometric_median(local[agreeing], weights[agreeing])
        displacement[index] = estimate
        residual[index] = np.linalg.norm(local[agreeing] - estimate, axis=1).max()
        available[index] = True
    return _estimates(displacement, available, selected, candidates, radius, residual)


def missing_node_fallback_field(
    baseline: DisplacementField, matches: MotionMatches, config: FieldConfig
) -> DisplacementField:
    """Add measured support only where the completed baseline field is missing."""
    unresolved = np.flatnonzero(~baseline.available)
    if not len(unresolved) or not len(matches):
        return baseline
    values = {
        name: getattr(baseline, name).copy()
        for name in (
            "displacement_m", "available", "selected_matches", "candidate_matches",
            "support_radius_m", "maximum_residual_m",
        )
    }
    for neighbours, minimum, radius, agreement in (
        (8, 6, 3_000.0, 750.0),
        (16, 10, 8_000.0, 1_250.0),
    ):
        if not len(unresolved):
            break
        tier = replace(
            config,
            neighbour_count=neighbours,
            minimum_agreeing_matches=minimum,
            maximum_neighbour_distance_m=radius,
            agreement_distance_m=agreement,
        )
        result = estimate_queries(matches, baseline.source_xy_m[unresolved], tier)
        for name in values:
            values[name][unresolved] = result[name]
        unresolved = unresolved[~result["available"]]
    if not np.any(values["available"] & ~baseline.available):
        return baseline
    trial = DisplacementField(**{**baseline.__dict__, **values})
    available = trial.available.copy()
    while True:
        flipped = flipped_indices(trial.with_available(available),
                                  config.maximum_triangle_edge_m)
        if not len(flipped):
            break
        candidates = flipped[~baseline.available[flipped]]
        if not len(candidates):
            return baseline
        available[candidates] = False
    return trial.with_available(available) if np.any(
        available & ~baseline.available
    ) else baseline


def field_edge_with_fallback(
    baseline: DisplacementField,
    matches: MotionMatches,
    config: FieldConfig,
) -> FieldEdge:
    """Attach supplemental support without replacing the baseline field."""
    fallback = (
        missing_node_fallback_field(baseline, matches, config)
        if config.missing_node_fallback else baseline
    )
    return FieldEdge(
        baseline,
        fallback_field=fallback if fallback is not baseline else None,
        gap_matches=(
            gap_matches(baseline, matches, config)
            if config.gap_fill_point_neighbour_count is not None else None
        ),
    )


def gap_matches(
    field: DisplacementField, matches: MotionMatches, config: FieldConfig
) -> MotionMatches:
    """The raw matches point-local estimates can use, so composition need not hold
    every match of every pair. With shear point estimates every sample may need its
    own estimate, so all matches are kept (without their tile indices).

    A point lacks a usable triangle only if a corner of its field grid cell is a
    failed node or lies outside the field (a regular grid cell of accepted nodes
    triangulates into usable triangles), so it is within one cell diagonal of a
    failed or boundary node. Matches farther than that plus the neighbour radius
    from every such node can never be selected.
    """
    if not len(matches):
        return matches
    if config.shear_point_estimates:
        return _without_tiles(matches.source_xy_m, matches.target_xy_m, matches.score)
    rows = field.grid_row - field.grid_row.min() + 1
    columns = field.grid_column - field.grid_column.min() + 1
    present = np.zeros((rows.max() + 2, columns.max() + 2), dtype=bool)
    present[rows, columns] = True
    boundary = np.zeros(len(rows), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                boundary |= ~present[rows + dr, columns + dc]
    anchors = field.source_xy_m[~field.available | boundary]
    if not len(anchors):
        return MotionMatches.empty()
    reach = (
        config.gap_fill_point_neighbour_distance_m
        + math.sqrt(2.0) * config.grid_spacing_m
        + 1.0
    )
    near = cKDTree(anchors).query(
        matches.source_xy_m, distance_upper_bound=reach, workers=1
    )[0] < np.inf
    return MotionMatches(
        matches.source_xy_m[near], matches.target_xy_m[near], matches.score[near],
        matches.source_tile[near], matches.target_tile[near],
    )


def _without_tiles(source: np.ndarray, target: np.ndarray, score: np.ndarray) -> MotionMatches:
    """Matches for point estimates, which never read tile indices: those are a
    zero-stride view, so they take no memory."""
    zeros = np.broadcast_to(np.int32(0), (len(score),))
    return MotionMatches(source, target, score, zeros, zeros)


def weighted_geometric_median(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    estimate = _weighted_mean(vectors, weights)
    for _ in range(100):
        distance = np.linalg.norm(vectors - estimate, axis=1)
        if np.any(distance <= 1.0e-3):
            close = distance <= 1.0e-3
            return _weighted_mean(vectors[close], weights[close])
        updated = _weighted_mean(vectors, weights / distance)
        if np.linalg.norm(updated - estimate) <= 1.0e-3:
            return updated
        estimate = updated
    return estimate


def _canonical_neighbours(
    matches: MotionMatches, query: np.ndarray, config: FieldConfig
):
    selected_count = min(config.neighbour_count, len(matches))
    tree = cKDTree(matches.source_xy_m)
    distances, neighbours = tree.query(
        query,
        k=min(selected_count + 1, len(matches)),
        distance_upper_bound=config.maximum_neighbour_distance_m,
        workers=1,
    )
    distances = np.asarray(distances)
    neighbours = np.asarray(neighbours)
    if distances.ndim == 1:
        distances = distances[:, None]
        neighbours = neighbours[:, None]

    for point, row_distances, row_neighbours in zip(
        query, distances, neighbours, strict=True
    ):
        finite = np.isfinite(row_distances) & (row_neighbours < len(matches))
        selected = row_neighbours[finite].astype(np.int64, copy=False)
        squared = np.sum((matches.source_xy_m[selected] - point) ** 2, axis=1)
        order = _canonical_neighbour_order(matches, selected, squared)
        selected, squared = selected[order], squared[order]

        if len(selected) > selected_count:
            boundary = squared[selected_count - 1]
            tolerance = 100.0 * np.finfo(np.float64).eps * max(boundary, 1.0)
            if squared[selected_count] <= boundary + tolerance:
                selected = np.asarray(
                    tree.query_ball_point(point, np.sqrt(boundary + tolerance)),
                    dtype=np.int64,
                )
                squared = np.sum(
                    (matches.source_xy_m[selected] - point) ** 2, axis=1
                )
                keep = squared <= boundary + tolerance
                selected, squared = selected[keep], squared[keep]
                order = _canonical_neighbour_order(matches, selected, squared)
                selected, squared = selected[order], squared[order]

        yield np.sqrt(squared[:selected_count]), selected[:selected_count]


def _canonical_neighbour_order(
    matches: MotionMatches, neighbours: np.ndarray, squared_distances: np.ndarray
) -> np.ndarray:
    return np.lexsort(
        (
            neighbours,
            matches.target_tile[neighbours],
            matches.source_tile[neighbours],
            matches.target_xy_m[neighbours, 1],
            matches.target_xy_m[neighbours, 0],
            matches.source_xy_m[neighbours, 1],
            matches.source_xy_m[neighbours, 0],
            -matches.score[neighbours],
            squared_distances,
        )
    )


def _weighted_mean(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(vectors * weights[:, None], axis=0) / np.sum(weights)


def reject_folds(
    field: DisplacementField, maximum_triangle_edge_m: float
) -> tuple[DisplacementField, np.ndarray]:
    available = field.available.copy()
    rejected: list[np.ndarray] = []
    while True:
        selected = flipped_indices(field.with_available(available), maximum_triangle_edge_m)
        if not len(selected):
            break
        available[selected] = False
        rejected.append(selected)
    indices = np.unique(np.concatenate(rejected)) if rejected else np.empty(0, dtype=int)
    return field.with_available(available), indices


def flipped_indices(field: DisplacementField, maximum_triangle_edge_m: float) -> np.ndarray:
    indices = np.flatnonzero(field.available)
    if len(indices) < 3:
        return np.empty(0, dtype=int)
    source = field.source_xy_m[indices]
    target = source + field.displacement_m[indices]
    try:
        triangles = Delaunay(source).simplices
    except QhullError:
        return np.empty(0, dtype=int)
    source_triangles, target_triangles = source[triangles], target[triangles]
    local = _maximum_edge(source_triangles) <= maximum_triangle_edge_m
    flipped = _area(source_triangles[local]) * _area(target_triangles[local]) < 0
    return indices[np.unique(triangles[local][flipped].ravel())]


def sample_field_with_fallback(
    baseline: DisplacementField,
    fallback: DisplacementField | None,
    query_xy_m: np.ndarray,
    maximum_triangle_edge_m: float,
    *,
    field_config: FieldConfig | None = None,
    gap_matches: MotionMatches | None = None,
) -> FieldSamples:
    """Preserve all supported baseline samples; use fallback only for gaps.

    With gap filling configured (``field_config``), points still without a sample
    are filled by ``fill_gaps``; with shear point estimates, samples whose nodes
    disagree are then refined by ``refine_shear_samples``.
    """
    original = sample_field(baseline, query_xy_m, maximum_triangle_edge_m)
    if fallback is not None and not original.available.all():
        missing = np.flatnonzero(~original.available)
        recovered = sample_field(fallback, np.asarray(query_xy_m)[missing],
                                 maximum_triangle_edge_m)
        original = _merge_samples(original, missing, recovered)
    if field_config is not None and field_config.shear_point_estimates:
        # One rule for every point: interpolate where the nodes agree; use the point's
        # own matches where they disagree or the field has a hole; bridge what is left
        # with a rigid triangle (the only estimate not resting on nearby matches).
        original = refine_shear_samples(baseline, original, query_xy_m, field_config, gap_matches)
        if field_config.gap_fill_rigid_edge_m is not None:
            original = _fill_rigid(baseline, original, query_xy_m, field_config)
    elif field_config is not None and field_config.gap_filling:
        original = fill_gaps(baseline, original, query_xy_m, field_config, gap_matches)
    return original


def _fill_rigid(field, samples, query_xy_m, config):
    query = np.asarray(query_xy_m, dtype=np.float64)
    missing = np.flatnonzero(~samples.available)
    if not len(missing):
        return samples
    rigid = sample_field(
        field, query[missing], config.maximum_triangle_edge_m,
        rigid=(config.gap_fill_rigid_edge_m, config.gap_fill_rigid_spread_m),
    )
    return _merge_samples(samples, missing, rigid)


def node_triangle_spread(field: DisplacementField, query_xy_m: np.ndarray) -> np.ndarray:
    """For each point, the largest difference between the displacements of the three
    accepted nodes of the Delaunay triangle containing it (NaN outside the nodes' hull)."""
    query = np.asarray(query_xy_m, dtype=np.float64)
    spread = np.full(len(query), np.nan)
    valid = field.available & np.isfinite(field.displacement_m).all(axis=1)
    if valid.sum() < 3 or not len(query):
        return spread
    try:
        triangulation = Delaunay(field.source_xy_m[valid])
    except QhullError:
        return spread
    simplex = triangulation.find_simplex(query)
    inside = simplex >= 0
    vectors = field.displacement_m[valid][triangulation.simplices[simplex[inside]]]
    spread[inside] = _maximum_edge(vectors)
    return spread


def refine_shear_samples(
    field: DisplacementField,
    samples: FieldSamples,
    query_xy_m: np.ndarray,
    config: FieldConfig,
    matches: MotionMatches | None,
) -> FieldSamples:
    """Replace a sample by the point's own estimate where the field's nodes disagree
    more than the point's matches do, and give one to points without a sample.

    Linear interpolation between three nodes up to ``maximum_triangle_edge_m`` apart
    blends the two sides of a lead or shear line between them. The point-local
    estimate (``estimate_points`` with the ``gap_fill_point_*`` settings) uses the
    matches on the point's own side; its maximum residual measures how well those
    matches agree. Where the node triangle's displacement spread exceeds that
    residual, the nodes disagree more than the measurement noise allows and the
    point estimate is used; elsewhere the interpolated sample, whose error
    neighbouring points share, is kept. No threshold of its own.
    """
    if matches is None or not len(matches):
        return samples
    query = np.asarray(query_xy_m, dtype=np.float64)
    spread = np.full(len(query), np.inf)  # a point without a sample: any estimate wins
    sampled = np.flatnonzero(samples.available)
    spread[sampled] = node_triangle_spread(field, query[sampled])
    rows = np.flatnonzero(~np.isnan(spread))
    spread = spread[rows]
    if not len(rows):
        return samples
    estimate = estimate_points(
        matches, query[rows],
        config.gap_fill_point_neighbour_count,
        config.gap_fill_point_minimum_agreeing_matches,
        config.gap_fill_point_neighbour_distance_m,
        config.gap_fill_point_agreement_distance_m,
    )
    use = estimate["available"] & (spread > estimate["maximum_residual_m"])
    return _merge_samples(samples, rows, FieldSamples(
        estimate["displacement_m"], use,
        estimate["selected_matches"].astype(np.float64),
        estimate["support_radius_m"], estimate["maximum_residual_m"],
    ))


def fill_gaps(
    field: DisplacementField,
    samples: FieldSamples,
    query_xy_m: np.ndarray,
    config: FieldConfig,
    gap_matches: MotionMatches | None,
) -> FieldSamples:
    """Fill points without a usable triangle, in order: rigid large triangles, then
    the point's own raw matches (point-local consensus).

    A rigid triangle joins accepted nodes up to ``gap_fill_rigid_edge_m`` apart whose
    displacements differ by at most ``gap_fill_rigid_spread_m``: the ice inside moves
    as one block, so a hole between them is bridged. Its residual is at least that
    spread. Point-local consensus applies the node rule at the point itself with the
    ``gap_fill_point_*`` settings, so at a shear zone it uses the matches on the
    point's own side. Existing samples are never changed.
    """
    query = np.asarray(query_xy_m, dtype=np.float64)
    missing = np.flatnonzero(~samples.available)
    if len(missing) and config.gap_fill_rigid_edge_m is not None:
        rigid = sample_field(
            field, query[missing], config.maximum_triangle_edge_m,
            rigid=(config.gap_fill_rigid_edge_m, config.gap_fill_rigid_spread_m),
        )
        samples = _merge_samples(samples, missing, rigid)
        missing = np.flatnonzero(~samples.available)
    if (
        len(missing)
        and config.gap_fill_point_neighbour_count is not None
        and gap_matches is not None
        and len(gap_matches)
    ):
        estimate = estimate_points(
            gap_matches,
            query[missing],
            config.gap_fill_point_neighbour_count,
            config.gap_fill_point_minimum_agreeing_matches,
            config.gap_fill_point_neighbour_distance_m,
            config.gap_fill_point_agreement_distance_m,
        )
        samples = _merge_samples(samples, missing, FieldSamples(
            estimate["displacement_m"], estimate["available"],
            estimate["selected_matches"].astype(np.float64),
            estimate["support_radius_m"], estimate["maximum_residual_m"],
        ))
    return samples


def _merge_samples(
    samples: FieldSamples, rows: np.ndarray, update: FieldSamples
) -> FieldSamples:
    """Copy ``update`` into ``samples`` at ``rows`` where ``update`` is available."""
    values = {name: getattr(samples, name).copy() for name in samples.__dataclass_fields__}
    filled = rows[update.available]
    for name in values:
        values[name][filled] = getattr(update, name)[update.available]
    return FieldSamples(**values)


def estimate_points(
    matches: MotionMatches,
    query_xy_m: np.ndarray,
    neighbour_count: int,
    minimum_agreeing_matches: int,
    maximum_neighbour_distance_m: float,
    agreement_distance_m: float,
) -> dict[str, np.ndarray]:
    """Vectorized consensus at query points: the node rule of ``estimate_queries``
    applied to many points at once.

    For each point, the ``neighbour_count`` nearest matches within
    ``maximum_neighbour_distance_m`` vote; the largest score-weighted group agreeing
    within ``agreement_distance_m`` must have ``minimum_agreeing_matches`` members,
    and its weighted geometric median is the estimate. Equal-distance neighbours are
    ordered by match index rather than ``estimate_queries``' canonical order.
    """
    query = np.asarray(query_xy_m, dtype=np.float64)
    count = len(query)
    result = {
        "displacement_m": np.full((count, 2), np.nan),
        "available": np.zeros(count, dtype=bool),
        "selected_matches": np.zeros(count, dtype=np.int32),
        "candidate_matches": np.zeros(count, dtype=np.int32),
        "support_radius_m": np.full(count, np.nan),
        "maximum_residual_m": np.full(count, np.nan),
    }
    if not count or not len(matches):
        return result
    k = min(neighbour_count, len(matches))
    distances, neighbours = cKDTree(matches.source_xy_m).query(
        query, k=k, distance_upper_bound=maximum_neighbour_distance_m, workers=1
    )
    distances = np.asarray(distances, dtype=np.float64).reshape(count, k)
    neighbours = np.asarray(neighbours).reshape(count, k)
    valid = np.isfinite(distances) & (neighbours < len(matches))
    safe = np.where(valid, neighbours, 0)
    vectors = matches.displacement_m[safe]
    weights = np.where(valid, np.maximum(matches.score[safe], 1.0e-12), 0.0)
    separation = np.linalg.norm(vectors[:, :, None, :] - vectors[:, None, :, :], axis=3)
    agree = (separation <= agreement_distance_m) & valid[:, :, None] & valid[:, None, :]
    support = np.einsum("nij,nj->ni", agree, weights)
    support[~valid] = -1.0
    best = np.argmax(support, axis=1)
    members = agree[np.arange(count), best]
    selected = members.sum(axis=1)
    result["candidate_matches"] = valid.sum(axis=1).astype(np.int32)
    result["selected_matches"] = selected.astype(np.int32)
    rows = np.flatnonzero(selected >= minimum_agreeing_matches)
    if not len(rows):
        return result
    member_weights = np.where(members[rows], weights[rows], 0.0)
    estimate = _batched_geometric_median(vectors[rows], member_weights)
    residual = np.where(
        members[rows], np.linalg.norm(vectors[rows] - estimate[:, None, :], axis=2), 0.0
    ).max(axis=1)
    result["displacement_m"][rows] = estimate
    result["available"][rows] = True
    result["support_radius_m"][rows] = np.where(valid[rows], distances[rows], 0.0).max(axis=1)
    result["maximum_residual_m"][rows] = residual
    return result


def _batched_geometric_median(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``weighted_geometric_median`` for many point sets (zero weight = not a member)."""
    total = weights.sum(axis=1, keepdims=True)
    estimate = np.einsum("nk,nkj->nj", weights, vectors) / total
    active = np.ones(len(vectors), dtype=bool)
    member = weights > 0
    for _ in range(100):
        if not active.any():
            break
        rows = np.flatnonzero(active)
        distance = np.linalg.norm(vectors[rows] - estimate[rows, None, :], axis=2)
        close = (distance <= 1.0e-3) & member[rows]
        settled = close.any(axis=1)
        if settled.any():
            done = rows[settled]
            close_weights = np.where(close[settled], weights[done], 0.0)
            estimate[done] = np.einsum("nk,nkj->nj", close_weights, vectors[done]) / close_weights.sum(axis=1, keepdims=True)
            active[done] = False
        rows, distance = rows[~settled], distance[~settled]
        if not len(rows):
            continue
        inverse = np.where(member[rows], weights[rows] / np.maximum(distance, 1.0e-12), 0.0)
        updated = np.einsum("nk,nkj->nj", inverse, vectors[rows]) / inverse.sum(axis=1, keepdims=True)
        converged = np.linalg.norm(updated - estimate[rows], axis=1) <= 1.0e-3
        estimate[rows] = updated
        active[rows[converged]] = False
    return estimate


def sample_field(
    field: DisplacementField,
    query_xy_m: np.ndarray,
    maximum_triangle_edge_m: float,
    *,
    rigid: tuple[float, float] | None = None,
) -> FieldSamples:
    """Interpolate the field linearly in usable triangles of accepted nodes.

    ``rigid=(edge_m, spread_m)`` samples only triangles that are not usable under
    ``maximum_triangle_edge_m`` but have edges up to ``edge_m`` and node
    displacements within ``spread_m`` of each other (gap filling); their residual
    is at least the displacement spread.
    """
    query = np.asarray(query_xy_m, dtype=np.float64)
    count = len(query)
    displacement = np.full((count, 2), np.nan, dtype=np.float64)
    available = np.zeros(count, dtype=bool)
    selected = np.zeros(count, dtype=np.float64)
    radius = np.full(count, np.nan)
    residual = np.full(count, np.nan)
    valid = field.available & np.isfinite(field.displacement_m).all(axis=1)
    if valid.sum() < 3 or not count:
        return FieldSamples(displacement, available, selected, radius, residual)
    source = field.source_xy_m[valid]
    vectors = field.displacement_m[valid]
    target = source + vectors
    try:
        triangulation = Delaunay(source)
    except QhullError:
        return FieldSamples(displacement, available, selected, radius, residual)
    vertices = triangulation.simplices
    source_triangles = source[vertices]
    edges = _maximum_edge(source_triangles)
    unfolded = _area(source_triangles) * _area(target[vertices]) > 0
    usable = (edges <= maximum_triangle_edge_m) & unfolded
    spread = None
    if rigid is not None:
        spread = _maximum_edge(vectors[vertices])
        usable = ~usable & unfolded & (edges <= rigid[0]) & (spread <= rigid[1])
    del source_triangles, target
    simplex = triangulation.find_simplex(query)
    _recover_boundary_simplices(triangulation, usable, query, simplex)
    query_indices = np.flatnonzero(simplex >= 0)
    if not len(query_indices):
        return FieldSamples(displacement, available, selected, radius, residual)
    chosen = simplex[query_indices]
    keep = usable[chosen]
    query_indices, chosen = query_indices[keep], chosen[keep]
    if not len(query_indices):
        return FieldSamples(displacement, available, selected, radius, residual)
    affine = triangulation.transform[chosen]
    first = np.einsum("nij,nj->ni", affine[:, :2], query[query_indices] - affine[:, 2])
    weights = np.column_stack((first, 1.0 - first.sum(axis=1)))
    triangle_vertices = vertices[chosen]
    displacement[query_indices] = np.einsum("ni,nij->nj", weights, vectors[triangle_vertices])
    selected[query_indices] = np.einsum("ni,ni->n", weights, field.selected_matches[valid][triangle_vertices])
    radius[query_indices] = _weighted(weights, field.support_radius_m[valid][triangle_vertices])
    residual[query_indices] = _weighted(weights, field.maximum_residual_m[valid][triangle_vertices])
    if spread is not None:
        residual[query_indices] = np.fmax(residual[query_indices], spread[chosen])
    available[query_indices] = True
    return FieldSamples(displacement, available, selected, radius, residual)


def _recover_boundary_simplices(
    triangulation: Delaunay,
    usable: np.ndarray,
    query: np.ndarray,
    simplex: np.ndarray,
) -> None:
    """Fill rejected boundary lookups without replacing successful simplices.

    A shared edge has one neighbouring simplex; an exact vertex can have more
    than two incident simplices. Search only those existing local alternatives,
    keeping the same containment tolerance as the default simplex lookup.
    """
    tolerance = 100.0 * np.finfo(np.float64).eps
    inside = np.flatnonzero(simplex >= 0)
    rejected = inside[~usable[simplex[inside]]]
    vertex_queries = []
    vertex_indices = []
    if len(rejected):
        chosen = simplex[rejected]
        vertices = triangulation.simplices[chosen]
        exact = np.all(
            query[rejected, None, :] == triangulation.points[vertices], axis=2
        )
        at_vertex = exact.any(axis=1)
        vertex_queries.append(rejected[at_vertex])
        vertex_indices.append(vertices[at_vertex, exact[at_vertex].argmax(axis=1)])

        edge_queries = rejected[~at_vertex]
        chosen = chosen[~at_vertex]
        affine = triangulation.transform[chosen]
        first = np.einsum(
            "nij,nj->ni", affine[:, :2], query[edge_queries] - affine[:, 2]
        )
        weights = np.column_stack((first, 1.0 - first.sum(axis=1)))
        on_edge = (np.abs(weights) <= tolerance) & np.all(
            weights >= -tolerance, axis=1
        )[:, None]
        rows, facets = np.nonzero(on_edge)
        neighbours = triangulation.neighbors[chosen[rows], facets]
        keep = neighbours >= 0
        rows, neighbours = rows[keep], neighbours[keep]
        keep = usable[neighbours]
        rows, neighbours = rows[keep], neighbours[keep]
        affine = triangulation.transform[neighbours]
        first = np.einsum(
            "nij,nj->ni", affine[:, :2], query[edge_queries[rows]] - affine[:, 2]
        )
        weights = np.column_stack((first, 1.0 - first.sum(axis=1)))
        contains = np.all(weights >= -tolerance, axis=1)
        # Near a vertex more than one facet can qualify. Use a stable simplex
        # index rather than the order in which queries reached the boundary.
        replacement = np.full(len(edge_queries), len(usable), dtype=np.intp)
        np.minimum.at(replacement, rows[contains], neighbours[contains])
        found = replacement < len(usable)
        simplex[edge_queries[found]] = replacement[found]

    outside = np.flatnonzero(simplex < 0)
    if len(outside):
        points = triangulation.points
        in_bounds = np.all(
            (query[outside] >= points.min(axis=0))
            & (query[outside] <= points.max(axis=0)),
            axis=1,
        )
        outside = outside[in_bounds]
        if len(outside):
            # Directed lookup can also miss an exact hull vertex. The tree is
            # queried only for these rejected rows, with a numerical-radius
            # bound; only an exactly equal coordinate may be recovered.
            radius = np.finfo(np.float64).eps * max(np.abs(points).max(), 1.0)
            distance, vertex = cKDTree(points).query(
                query[outside], distance_upper_bound=radius, workers=1
            )
            exact = distance == 0.0
            outside, vertex = outside[exact], vertex[exact]
            exact = np.all(query[outside] == points[vertex], axis=1)
            vertex_queries.append(outside[exact])
            vertex_indices.append(vertex[exact])

    if vertex_indices and any(len(indices) for indices in vertex_indices):
        best = np.full(len(triangulation.points), len(usable), dtype=np.intp)
        selected = np.flatnonzero(usable)
        np.minimum.at(
            best, triangulation.simplices[selected].ravel(), np.repeat(selected, 3)
        )
        queries = np.concatenate(vertex_queries)
        chosen = best[np.concatenate(vertex_indices)]
        found = chosen < len(usable)
        simplex[queries[found]] = chosen[found]


def _estimates(displacement, available, selected, candidates, radius, residual):
    return {
        "displacement_m": displacement,
        "available": available,
        "selected_matches": selected,
        "candidate_matches": candidates,
        "support_radius_m": radius,
        "maximum_residual_m": residual,
    }


def _area(triangles: np.ndarray) -> np.ndarray:
    first, second = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]


def _maximum_edge(triangles: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        np.stack((triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 1], triangles[:, 0] - triangles[:, 2])),
        axis=2,
    ).max(axis=0)


def _weighted(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values).all(axis=1)
    result = np.full(len(values), np.nan)
    result[finite] = np.einsum("ni,ni->n", weights[finite], values[finite])
    return result
