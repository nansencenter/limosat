# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

"""Utilities for detecting and mitigating flipped triangles (triangle orientation changes)
caused by inconsistent displacement vectors, and interpolating masked vectors."""

import numpy as np
from matplotlib.tri import Triangulation
from .utils import logger


def jacobian(x0, y0, x1, y1, x2, y2):
    """Return Jacobian determinant (2 * signed area) for triangles defined by three points.
    Args:
        x0, y0, x1, y1, x2, y2 (ndarray): Coordinate arrays.
    Returns:
        ndarray: 2 * signed area (sign change indicates flipped orientation).
    """
    return (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)


def get_area(x, y, t):
    """Return signed areas of triangles.
    Args:
        x, y (ndarray): Vertex coordinate arrays.
        t (ndarray): Triangle indices (n_triangles x 3).
    Returns:
        ndarray: Signed area; negative sign indicates orientation flip.
    """
    return 0.5 * jacobian(x[t][:, 0], y[t][:, 0], x[t][:, 1], y[t][:, 1], x[t][:, 2], y[t][:, 2])


def find_triangle(x, y, t, point):
    """Return index of first triangle containing point, or -1 if none.
    Uses barycentric coordinate test (strictly inside -> all > 0).
    """
    vertices = np.dstack([x[t], y[t]])  # (n_triangles, 3, 2)
    v0 = vertices[:, 1] - vertices[:, 0]
    v1 = vertices[:, 2] - vertices[:, 0]
    v2 = np.array(point) - vertices[:, 0]

    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)
    denom = d00 * d11 - d01 * d01
    valid = denom != 0

    v = np.full_like(denom, np.inf, dtype=float)
    w = np.full_like(denom, np.inf, dtype=float)
    v[valid] = (d11[valid] * d20[valid] - d01[valid] * d21[valid]) / denom[valid]
    w[valid] = (d00[valid] * d21[valid] - d01[valid] * d20[valid]) / denom[valid]
    u = 1.0 - v - w

    inside = (u > 0) & (v > 0) & (w > 0)
    containing = np.where(inside)[0]
    return containing[0] if containing.size > 0 else -1


def find_triangles_for_points(x, y, t, points):
    """Vectorized convenience wrapper for find_triangle over multiple points."""
    return np.array([find_triangle(x, y, t, p) for p in points])


def mask_flipped_triangles(x0, y0, x1, y1, t, n=0, max_iter=100):
    """Recursively mask points that induce flipped triangles.
      1. Compute signed areas before (x0,y0) and after (x1,y1) displacement.
      2. Identify triangles whose orientation sign changed (flipped triangles).
      3. Remove (mask) the most probable offending point:
         a) If a vertex lies strictly inside another triangle (self-intersection), mask it.
         b) Otherwise choose the vertex participating in the largest number of flipped triangles.
      4. Recompute until no flipped triangles remain or iteration cap reached.

    Args:
        x0, y0 (ndarray): Original coordinates.
        x1, y1 (ndarray): Displaced coordinates (modified in-place conceptually via copies here).
        t (ndarray): Triangle indices (n_triangles x 3).
        n (int): Current recursion depth.
        max_iter (int): Safety cap on recursion.
    Returns:
        tuple: (x1_masked, y1_masked, t_masked) with offending points set to NaN and
               triangles referencing masked points removed.
    """
    if n >= max_iter:
        return x1, y1, t

    a0 = get_area(x0, y0, t)
    a1 = get_area(x1, y1, t)
    flipped = np.sign(a0) != np.sign(a1)
    if not np.any(flipped):
        return x1, y1, t

    # Focus on smallest-magnitude flipped triangle for robust culprit detection.
    min_area_idx = np.where(flipped)[0][np.argmin(np.abs(a1[flipped]))]
    neg_pts_idx = t[min_area_idx].flatten()
    potential_pts = np.column_stack([x1[neg_pts_idx], y1[neg_pts_idx]])
    tri_i = find_triangles_for_points(x1, y1, t, potential_pts)
    bad_idx = neg_pts_idx[tri_i > 0]

    if bad_idx.size > 0:
        # A vertex intrudes inside another triangle -> mask intruding vertices.
        x1[bad_idx] = np.nan
        y1[bad_idx] = np.nan
    else:
        # Heuristic: vertex involved in most flipped triangles is culprit.
        neighbor_tris = [np.where(np.any(t == v, axis=1))[0] for v in neg_pts_idx]
        flip_counts = [np.sum(flipped[nt]) for nt in neighbor_tris]
        culprit = neg_pts_idx[np.argmax(flip_counts)]
        x1[culprit] = np.nan
        y1[culprit] = np.nan

    # Drop triangles referencing any masked vertex.
    t = t[np.all(np.isfinite(x1[t]), axis=1)]
    return mask_flipped_triangles(x0, y0, x1, y1, t, n + 1, max_iter)


def interpolate_vectors(x0, y0, x1m, y1m, tri, min_neighbours=3):
    """Interpolate masked (NaN) displaced positions using neighboring displacements.

    For each masked point, gather neighbors via Triangulation edges, require a minimum
    number of valid neighbors, and apply median displacement (robust to outliers).

    Args:
        x0, y0 (ndarray): Original coordinates.
        x1m, y1m (ndarray): Masked displaced coordinates (NaNs to fill).
        tri (Triangulation): Triangulation providing .edges.
        min_neighbours (int): Minimum valid neighbors to interpolate.
    Returns:
        tuple: (x1_interp, y1_interp, info) where info holds boolean arrays:
               was_interpolated, high_confidence, neighbor_count.
    """
    x1i = x1m.copy()
    y1i = y1m.copy()
    bad_points = np.nonzero(np.isnan(x1m))[0]

    info = {
        'was_interpolated': np.zeros(len(x1m), dtype=bool),
        'high_confidence': np.zeros(len(x1m), dtype=bool),
        'neighbor_count': np.zeros(len(x1m), dtype=int)
    }

    for bp in bad_points:
        neighbors = np.unique(tri.edges[np.any(tri.edges == bp, axis=1)])
        neighbors = neighbors[neighbors != bp]
        valid_mask = np.isfinite(x1m[neighbors]) & np.isfinite(y1m[neighbors])
        valid_neighbors = neighbors[valid_mask]
        info['neighbor_count'][bp] = len(valid_neighbors)
        if len(valid_neighbors) < min_neighbours:
            continue
        u_med = np.nanmedian(x1m[valid_neighbors] - x0[valid_neighbors])
        v_med = np.nanmedian(y1m[valid_neighbors] - y0[valid_neighbors])
        x1i[bp] = x0[bp] + u_med
        y1i[bp] = y0[bp] + v_med
        info['was_interpolated'][bp] = True
        info['high_confidence'][bp] = len(valid_neighbors) >= min_neighbours

    logger.debug(
        f"Interpolated {np.sum(info['was_interpolated'])}/{len(bad_points)} masked points (flipped triangle removal)."
    )
    return x1i, y1i, info


def filter_and_interpolate_flipped_triangles(x0, y0, x1_raw, y1_raw):
    """Mask and interpolate displacements that induce flipped triangles.

    Workflow:
        1. Identify and mask points causing flipped triangles.
        2. Interpolate masked points using neighbor median displacements.
        3. Re-run masking to ensure interpolation did not reintroduce flips.

    Args:
        x0, y0 (ndarray): Original coordinates.
        x1_raw, y1_raw (ndarray): Displaced coordinates prior to filtering.
    Returns:
        tuple: (x1_final, y1_final, was_interpolated_mask)
            x1_final, y1_final: Cleaned displaced coordinates (NaNs for unresolved points).
            was_interpolated_mask (ndarray[bool]): Points successfully interpolated and retained.
    """
    if len(x0) < 3:
        logger.debug(f"Skipping flipped triangle filtering: only {len(x0)} points.")
        return x1_raw, y1_raw, np.zeros(len(x0), dtype=bool)

    tri = Triangulation(x0, y0)
    x1m, y1m, t1 = mask_flipped_triangles(x0, y0, x1_raw.copy(), y1_raw.copy(), tri.triangles.copy())
    x1i, y1i, info = interpolate_vectors(x0, y0, x1m, y1m, tri, min_neighbours=5)
    x1_final, y1_final, _ = mask_flipped_triangles(x0, y0, x1i, y1i, t1)
    was_interpolated_mask = info['was_interpolated'] & np.isfinite(x1_final)
    return x1_final, y1_final, was_interpolated_mask
