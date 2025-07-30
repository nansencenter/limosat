

# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

"""
functions for filtering and interpolating vectors based on triangulation and deformation
"""

import numpy as np
import geopandas as gpd
from matplotlib.tri import Triangulation
from .utils import logger
import cartopy.crs as ccrs
from scipy.spatial import Delaunay


def jacobian(x0, y0, x1, y1, x2, y2):
    """ Calculates the Jacobian determinant for a triangle defined by three points.
    Args:
        x0, y0: Numpy arrays with coordinates of the first point.
        x1, y1: Numpy arrays with coordinates of the second point.
        x2, y2: Numpy arrays with coordinates of the third point.
    Returns:
        Numpy array with the Jacobian determinant, which is twice the area of the triangle.
    """
    return (x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)

def get_area(x, y, t):
    """Calculates the area of triangles defined by vertices in x and y at indices in t.
    Args:
        x: Numpy array with x-coordinates of vertices.
        y: Numpy array with y-coordinates of vertices.
        t: Numpy array with indices of triangles (shape: n_triangles x 3).
    Returns:
        Numpy array with the area of each triangle.
    Note:
        The area is negative for flipped triangles.
    """
    return .5*jacobian(x[t][:,0], y[t][:,0], x[t][:,1], y[t][:,1], x[t][:,2], y[t][:,2])

def find_triangle(x, y, t, point):
    """ Finds the first triangle containing a given point.
    Args:
        x: Numpy array with x-coordinates of vertices.
        y: Numpy array with y-coordinates of vertices.
        t: Numpy array with indices of triangles (shape: n_triangles x 3).
        point: A tuple or list with the x and y coordinates of the point to check.
    Returns:
        The index of the first triangle containing the point, or -1 if no triangle contains it.
    """
    # Reshape to get triangle vertices as (n_triangles, 3, 2)
    vertices = np.dstack([x[t], y[t]])

    # Calculate vectors
    v0 = vertices[:, 1] - vertices[:, 0]
    v1 = vertices[:, 2] - vertices[:, 0]

    # Broadcast the point to calculate v2 for all triangles at once
    point_array = np.array(point)
    v2 = point_array - vertices[:, 0]

    # Compute dot products
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)

    # Calculate denominators
    denom = d00 * d11 - d01 * d01

    # Avoid division by zero
    valid_denom = denom != 0

    # Initialize barycentric coordinates
    v = np.full_like(denom, np.inf)
    w = np.full_like(denom, np.inf)

    # Calculate only where denominator is valid
    v[valid_denom] = (d11[valid_denom] * d20[valid_denom] - d01[valid_denom] * d21[valid_denom]) / denom[valid_denom]
    w[valid_denom] = (d00[valid_denom] * d21[valid_denom] - d01[valid_denom] * d20[valid_denom]) / denom[valid_denom]
    u = 1.0 - v - w

    # Point is inside triangle if all barycentric coordinates are positive
    inside = (u > 0) & (v > 0) & (w > 0)
    # tol = 1e-8
    # inside = (u >= -tol) & (v >= -tol) & (w >= -tol) 

    # Return the index of the first triangle containing the point, or -1 if none
    containing_triangles = np.where(inside)[0]
    return containing_triangles[0] if len(containing_triangles) > 0 else -1

def find_triangles_for_points(x, y, t, points):
    """ Finds triangles containing each point in a list of points.
    Args:
        x: Numpy array with x-coordinates of vertices.
        y: Numpy array with y-coordinates of vertices.
        t: Numpy array with indices of triangles (shape: n_triangles x 3).
        points: A list or array of points, where each point is a tuple or list with x and y coordinates.
    Returns:
        Numpy array with the index of the first triangle containing each point, or -1 if no triangle contains it.
    """
    return np.array([find_triangle(x, y, t, point) for point in points])

def mask_flipped_triangles(x0, y0, x1, y1, t, n=0, max_iter=100):
    """ Recursively masks bad points in x1, y1 with NaN. Bad points are located within other triangles.
    Args:
        x1: Numpy array with x-coordinates of the second set of points.
        y1: Numpy array with y-coordinates of the second set of points.
        t: Numpy array with indices of triangles (shape: n_triangles x 3).
        n: Current recursion depth, used to limit iterations.
        max_iter: Maximum number of iterations to prevent infinite recursion.
    Returns:
        Tuple of masked x1, y1, and t arrays.
    """
    if n >= max_iter:
        return x1, y1, t

    a0 = get_area(x0, y0, t)
    a1 = get_area(x1, y1, t)
    flipped = np.sign(a0) != np.sign(a1)

    if not np.any(flipped):
        return x1, y1, t

    min_area_idx = np.where(flipped)[0][np.argmin(np.abs(a1[flipped]))]
    # indices of nodes forming the triangle with the smallest area
    neg_pts_idx = t[min_area_idx].flatten()
    # coordinates of nodes of the triangle with the smallest area
    potential_trouble_points = np.column_stack([x1[neg_pts_idx], y1[neg_pts_idx]])
    tri_i = find_triangles_for_points(x1, y1, t, potential_trouble_points)
    # index of point that is inside a triangle
    bad_vector_idx = neg_pts_idx[tri_i > 0]
    if bad_vector_idx.size > 0:
        # mask bad second coordinate with nan
        x1[bad_vector_idx] = np.nan
        y1[bad_vector_idx] = np.nan
        # remove triangles containing bad points
        t = t[np.all(np.isfinite(x1[t]), axis=1)]
    else:
        # if no bad points found, it means the error is likely confined to one of the
        # vertices of the flipped triangle itself. We need to decide which of the three
        # vertices is the most likely culprit.
        
        # for each of the three vertices, count how many of its neighboring triangles are also flipped. 
        # The vertex that is part of the highest number of flipped triangles is the most likely source of the error.
        
        # Find all triangles neighboring each vertex of the problematic triangle
        neighbor_triangles = [np.where(np.any(t == i, axis=1))[0] for i in neg_pts_idx]
        
        # Count how many of these neighboring triangles are flipped
        flip_counts = [np.sum(flipped[neighbors]) for neighbors in neighbor_triangles]
        
        # Identify the vertex with the highest number of associated flipped triangles
        most_likely_culprit_idx = neg_pts_idx[np.argmax(flip_counts)]
        
        # Mask this point and remove all triangles associated with it
        x1[most_likely_culprit_idx] = np.nan
        y1[most_likely_culprit_idx] = np.nan
        t = t[np.all(np.isfinite(x1[t]), axis=1)]
    return mask_flipped_triangles(x0, y0, x1, y1, t, n+1, max_iter)

def interpolate_vectors(x0, y0, x1m, y1m, tri, min_neighbours=3):
    """ Interpolates bad vectors in x1m, y1m using the average of neighbouring points.
    Args:
        x0: Numpy array with x-coordinates of the first set of points.
        y0: Numpy array with y-coordinates of the first set of points.
        x1m: Numpy array with x-coordinates of the second set of points (may contain NaN).
        y1m: Numpy array with y-coordinates of the second set of points (may contain NaN).
        tri: Delaunay triangulation object containing edges.
        min_neighbours: Minimum number of valid neighbors required for interpolation.
    Returns:
        Tuple of interpolated x1i, y1i arrays and an interpolation_info dict.
    """
    x1i = x1m.copy()
    y1i = y1m.copy()
    bad_points = np.nonzero(np.isnan(x1m))[0]
    
    # Track interpolation information
    interpolation_info = {
        'was_interpolated': np.zeros(len(x1m), dtype=bool),
        'high_confidence': np.zeros(len(x1m), dtype=bool),
        'neighbor_count': np.zeros(len(x1m), dtype=int)
    }
        
    for bad_point in bad_points:
        # Find neighbors through triangulation edges
        neighbors = np.unique(tri.edges[np.any(tri.edges == bad_point, axis=1)])
        neighbors = neighbors[neighbors != bad_point]
        
        # Keep only neighbors with finite displacement
        valid = np.isfinite(x1m[neighbors]) & np.isfinite(y1m[neighbors])
        valid_neighbors = neighbors[valid]
        
        interpolation_info['neighbor_count'][bad_point] = len(valid_neighbors)
        
        if len(valid_neighbors) < min_neighbours:
            # Leave as NaN - not enough reliable neighbors
            continue
        
        # Calculate average displacement from neighbors
        u_avg = np.nanmedian(x1m[valid_neighbors] - x0[valid_neighbors])
        v_avg = np.nanmedian(y1m[valid_neighbors] - y0[valid_neighbors])
        
        # Apply interpolation
        x1i[bad_point] = x0[bad_point] + u_avg
        y1i[bad_point] = y0[bad_point] + v_avg
        
        # Mark as interpolated and set confidence
        interpolation_info['was_interpolated'][bad_point] = True
        interpolation_info['high_confidence'][bad_point] = len(valid_neighbors) >= min_neighbours
    
    successful_interpolations = np.sum(interpolation_info['was_interpolated'])
    logger.debug(f"Interpolated {successful_interpolations}/{len(bad_points)} points that were removed due to triangle inversions.")
    
    return x1i, y1i, interpolation_info

def filter_and_interpolate_flipped_triangles(x0, y0, x1_raw, y1_raw):
    """
    Masks vectors causing triangle inversions and interpolates their positions.
    This function now re-runs the masking process after interpolation to ensure
    that the interpolation itself does not introduce new invalid geometries.

    Returns:
        tuple: (x1_interpolated, y1_interpolated, was_interpolated_mask)
        The arrays contain NaNs for points that could not be interpolated.
        The mask identifies which points were successfully interpolated.
    """
    if len(x0) < 3:
        logger.debug(f"Skipping triangle filtering: only {len(x0)} points provided.")
        was_interpolated_mask = np.zeros(len(x0), dtype=bool)
        return x1_raw, y1_raw, was_interpolated_mask

    tri = Triangulation(x0, y0)
    
    # 1. Initial masking of vectors causing flips
    x1m, y1m, t1 = mask_flipped_triangles(x0, y0, x1_raw.copy(), y1_raw.copy(), tri.triangles.copy())
    
    # 2. Interpolate the positions of the masked vectors
    x1i, y1i, confidence = interpolate_vectors(x0, y0, x1m, y1m, tri, min_neighbours=5)
    
    # 3. Re-mask after interpolation to clean up any new flipped triangles
    # This is crucial because the interpolation, being a simple average of neighbors,
    # can itself create new geometric inconsistencies.
    x1_final, y1_final, _ = mask_flipped_triangles(x0, y0, x1i, y1i, t1)
    
    # The final mask should reflect points that were originally interpolated
    # and survived the second masking pass.
    was_interpolated_mask = confidence['was_interpolated'] & ~np.isnan(x1_final)

    return x1_final, y1_final, was_interpolated_mask
