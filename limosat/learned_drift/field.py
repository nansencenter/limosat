"""Regular-grid estimation and fold rejection for learned motion matches."""

from __future__ import annotations

import math

import numpy as np
import shapely
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry.base import BaseGeometry

from .config import ALIKEDConfig, EfficientLoFTRConfig
from .types import DriftField, MotionMatches


def regular_grid(domain: BaseGeometry, spacing_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row, column, and EPSG-projected coordinates inside a domain."""
    minx, miny, maxx, maxy = domain.bounds
    xs = np.arange(math.ceil(minx / spacing_m) * spacing_m, maxx, spacing_m)
    ys = np.arange(math.ceil(miny / spacing_m) * spacing_m, maxy, spacing_m)
    x_grid, y_grid = np.meshgrid(xs, ys)
    inside = shapely.intersects_xy(domain, x_grid.ravel(), y_grid.ravel())
    rows = np.repeat(np.arange(len(ys)), len(xs))[inside]
    columns = np.tile(np.arange(len(xs)), len(ys))[inside]
    xy_m = np.column_stack((x_grid.ravel()[inside], y_grid.ravel()[inside]))
    return rows, columns, xy_m


def weighted_geometric_median(
    vectors: np.ndarray,
    weights: np.ndarray,
    maximum_iterations: int = 100,
    tolerance_m: float = 1.0e-3,
) -> np.ndarray:
    """Return a deterministic two-dimensional weighted geometric median."""
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if not np.any(weights):
        weights = np.ones(len(vectors), dtype=float)
    estimate = np.average(vectors, axis=0, weights=weights)
    for _ in range(maximum_iterations):
        distance = np.linalg.norm(vectors - estimate, axis=1)
        coincident = distance <= tolerance_m
        if np.any(coincident):
            return np.average(
                vectors[coincident], axis=0, weights=weights[coincident]
            )
        updated = np.average(
            vectors,
            axis=0,
            weights=weights / np.maximum(distance, tolerance_m),
        )
        if np.linalg.norm(updated - estimate) <= tolerance_m:
            return updated
        estimate = updated
    return estimate


def estimate_field(
    matches: MotionMatches,
    domain: BaseGeometry,
    config: ALIKEDConfig | EfficientLoFTRConfig,
) -> DriftField:
    """Estimate the selected nearest-neighbour consensus field."""
    rows, columns, query_xy_m = regular_grid(domain, config.grid_spacing_m)
    estimates = estimate_queries(matches, query_xy_m, config)
    return DriftField(rows, columns, query_xy_m, **estimates)


def estimate_queries(
    matches: MotionMatches,
    query_xy_m: np.ndarray,
    config: ALIKEDConfig | EfficientLoFTRConfig,
) -> dict[str, np.ndarray]:
    """Estimate nearest-neighbour consensus at supplied projected positions."""
    query_xy_m = np.asarray(query_xy_m, dtype=float)
    if query_xy_m.ndim != 2 or query_xy_m.shape[1] != 2:
        raise ValueError("query positions must have shape (n, 2)")
    count = len(query_xy_m)
    displacement_m = np.full((count, 2), np.nan, dtype=float)
    available = np.zeros(count, dtype=bool)
    selected_matches = np.zeros(count, dtype=np.int32)
    candidate_matches = np.zeros(count, dtype=np.int32)
    support_radius_m = np.full(count, np.nan, dtype=float)
    maximum_residual_m = np.full(count, np.nan, dtype=float)
    if not len(matches):
        return {
            "displacement_m": displacement_m,
            "available": available,
            "selected_matches": selected_matches,
            "candidate_matches": candidate_matches,
            "support_radius_m": support_radius_m,
            "maximum_residual_m": maximum_residual_m,
        }

    tree = cKDTree(matches.source_xy_m)
    vectors = matches.displacement_m
    for query_index, query_xy in enumerate(query_xy_m):
        neighbour_count = min(config.neighbour_count, len(matches))
        distances, indices = tree.query(
            query_xy,
            k=neighbour_count,
            distance_upper_bound=config.maximum_neighbour_distance_m,
        )
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)
        finite = np.isfinite(distances) & (indices < len(matches))
        distances = distances[finite]
        indices = indices[finite]
        candidate_matches[query_index] = len(indices)
        if not len(indices):
            continue
        support_radius_m[query_index] = float(distances.max())

        local_vectors = vectors[indices]
        local_scores = np.maximum(matches.score[indices], 0.0)
        local_weights = (
            np.ones(len(local_scores), dtype=float)
            if getattr(config, "score_weighting", "raw") == "uniform"
            else np.maximum(local_scores, 1.0e-12)
        )
        separation = np.linalg.norm(
            local_vectors[:, None] - local_vectors[None, :], axis=2
        )
        support = (separation <= config.agreement_distance_m) @ local_weights
        agreeing = separation[int(np.argmax(support))] <= config.agreement_distance_m
        selected_matches[query_index] = int(agreeing.sum())
        if selected_matches[query_index] < config.minimum_agreeing_matches:
            continue

        estimate = weighted_geometric_median(
            local_vectors[agreeing],
            local_weights[agreeing],
        )
        displacement_m[query_index] = estimate
        maximum_residual_m[query_index] = float(
            np.linalg.norm(local_vectors[agreeing] - estimate, axis=1).max()
        )
        available[query_index] = True

    return {
        "displacement_m": displacement_m,
        "available": available,
        "selected_matches": selected_matches,
        "candidate_matches": candidate_matches,
        "support_radius_m": support_radius_m,
        "maximum_residual_m": maximum_residual_m,
    }


def _signed_twice_area(triangles: np.ndarray) -> np.ndarray:
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]


def flipped_field_indices(
    field: DriftField,
    spacing_m: float,
    maximum_triangle_edge_m: float | None = None,
) -> np.ndarray:
    """Return field indices belonging to locally folded triangles."""
    maximum_triangle_edge_m = _maximum_triangle_edge(
        spacing_m, maximum_triangle_edge_m
    )
    available_indices = np.flatnonzero(field.available)
    if len(available_indices) < 3:
        return np.empty(0, dtype=int)
    source = field.source_xy_m[available_indices]
    target = source + field.displacement_m[available_indices]
    triangles = Delaunay(source).simplices
    source_triangles = source[triangles]
    target_triangles = target[triangles]
    edge_lengths = np.max(
        np.stack(
            [
                np.linalg.norm(source_triangles[:, 0] - source_triangles[:, 1], axis=1),
                np.linalg.norm(source_triangles[:, 1] - source_triangles[:, 2], axis=1),
                np.linalg.norm(source_triangles[:, 2] - source_triangles[:, 0], axis=1),
            ]
        ),
        axis=0,
    )
    local = edge_lengths <= maximum_triangle_edge_m
    triangles = triangles[local]
    flipped = (
        _signed_twice_area(source_triangles[local])
        * _signed_twice_area(target_triangles[local])
        < 0
    )
    return available_indices[np.unique(triangles[flipped].ravel())]


def reject_folds(
    field: DriftField,
    spacing_m: float,
    maximum_triangle_edge_m: float | None = None,
) -> tuple[DriftField, np.ndarray]:
    """Remove fold vertices until retriangulation produces no new folds."""
    available = field.available.copy()
    rejected = []
    while True:
        selected = flipped_field_indices(
            field.with_available(available),
            spacing_m,
            maximum_triangle_edge_m,
        )
        if not len(selected):
            break
        available[selected] = False
        rejected.append(selected)
    rejected_indices = (
        np.unique(np.concatenate(rejected)) if rejected else np.empty(0, dtype=int)
    )
    return field.with_available(available), rejected_indices


def topology_summary(
    field: DriftField,
    spacing_m: float,
    maximum_triangle_edge_m: float | None = None,
) -> dict[str, float | int]:
    """Summarize local triangle orientation and area change."""
    maximum_triangle_edge_m = _maximum_triangle_edge(
        spacing_m, maximum_triangle_edge_m
    )
    available = field.available
    if available.sum() < 3:
        return {"triangles": 0}
    source = field.source_xy_m[available]
    target = source + field.displacement_m[available]
    triangles = Delaunay(source).simplices
    source_triangles = source[triangles]
    target_triangles = target[triangles]
    edge_lengths = np.max(
        np.stack(
            [
                np.linalg.norm(source_triangles[:, 0] - source_triangles[:, 1], axis=1),
                np.linalg.norm(source_triangles[:, 1] - source_triangles[:, 2], axis=1),
                np.linalg.norm(source_triangles[:, 2] - source_triangles[:, 0], axis=1),
            ]
        ),
        axis=0,
    )
    keep = edge_lengths <= maximum_triangle_edge_m
    source_area = _signed_twice_area(source_triangles[keep])
    target_area = _signed_twice_area(target_triangles[keep])
    if not len(source_area):
        return {"triangles": 0}
    flipped = source_area * target_area < 0
    area_ratio = np.abs(target_area / source_area)
    return {
        "triangles": int(len(source_area)),
        "flipped_triangles": int(flipped.sum()),
        "flipped_fraction": float(flipped.mean()),
        "area_ratio_p01": float(np.quantile(area_ratio, 0.01)),
        "area_ratio_median": float(np.median(area_ratio)),
        "area_ratio_p99": float(np.quantile(area_ratio, 0.99)),
    }


def _maximum_triangle_edge(
    spacing_m: float, maximum_triangle_edge_m: float | None
) -> float:
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("grid spacing must be finite and positive")
    if maximum_triangle_edge_m is None:
        return spacing_m * 1.6
    if not np.isfinite(maximum_triangle_edge_m) or maximum_triangle_edge_m <= 0:
        raise ValueError("maximum triangle edge must be finite and positive")
    return float(maximum_triangle_edge_m)
