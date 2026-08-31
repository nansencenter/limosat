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
    tree = cKDTree(matches.source_xy_m)
    vectors = matches.displacement_m
    for index, point in enumerate(query):
        distances, neighbours = tree.query(
            point,
            k=min(config.neighbour_count, len(matches)),
            distance_upper_bound=config.maximum_neighbour_distance_m,
        )
        distances = np.atleast_1d(distances)
        neighbours = np.atleast_1d(neighbours)
        finite = np.isfinite(distances) & (neighbours < len(matches))
        distances, neighbours = distances[finite], neighbours[finite]
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
    estimate = np.average(vectors, axis=0, weights=weights)
    for _ in range(100):
        distance = np.linalg.norm(vectors - estimate, axis=1)
        if np.any(distance <= 1.0e-3):
            return np.average(vectors[distance <= 1.0e-3], axis=0, weights=weights[distance <= 1.0e-3])
        updated = np.average(vectors, axis=0, weights=weights / distance)
        if np.linalg.norm(updated - estimate) <= 1.0e-3:
            return updated
        estimate = updated
    return estimate


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
    target = source + field.displacement_m[valid]
    try:
        triangulation = Delaunay(source)
    except QhullError:
        return FieldSamples(displacement, available, selected, radius, residual)
    vertices = triangulation.simplices
    usable = (
        (_maximum_edge(source[vertices]) <= maximum_triangle_edge_m)
        & (_area(source[vertices]) * _area(target[vertices]) > 0)
    )
    simplex = triangulation.find_simplex(query)
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
    displacement[query_indices] = np.einsum("ni,nij->nj", weights, field.displacement_m[valid][triangle_vertices])
    selected[query_indices] = np.einsum("ni,ni->n", weights, field.selected_matches[valid][triangle_vertices])
    radius[query_indices] = _weighted(weights, field.support_radius_m[valid][triangle_vertices])
    residual[query_indices] = _weighted(weights, field.maximum_residual_m[valid][triangle_vertices])
    available[query_indices] = True
    return FieldSamples(displacement, available, selected, radius, residual)


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
