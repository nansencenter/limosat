"""Consensus fields, explicit missing support, and topology validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.spatial import Delaunay, QhullError, cKDTree
from shapely.geometry.base import BaseGeometry

from .catalog import ImagePair
from .config import FieldConfig
from .models import DisplacementField, MotionMatches


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
    estimates = estimate_queries(matches, query, config)
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


def sample_field(
    field: DisplacementField,
    query_xy_m: np.ndarray,
    maximum_triangle_edge_m: float,
) -> FieldSamples:
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
    usable = (
        (_maximum_edge(source_triangles) <= maximum_triangle_edge_m)
        & (_area(source_triangles) * _area(target[vertices]) > 0)
    )
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
