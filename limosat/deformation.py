"""Pair-field deformation products in explicit SI units.

Deformation here is computed one image pair at a time, so triangles never cross a pair boundary.
Sentinel-1 cuts each acquisition (datatake) into frames seconds apart, and frames of one source
datatake matched to one target datatake share the same interval; computing per image pair leaves
a gap about one point spacing wide along every frame boundary. For maps or products, group by
source/target datatake pair and triangulate each group together (see ``by_datatake`` in the
paper repository, ``src/eloftr_paper/deformation.py``, and its ``analysis/deformation-seams``).
This module is used only by the older SQLite pipeline (``limosat/stages.py``); the Parquet chain
does not produce deformation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, QhullError

from .models import DisplacementField


@dataclass(frozen=True)
class DeformationCell:
    pair_id: str
    triangle_index: int
    centroid_x_m: float
    centroid_y_m: float
    source_area_m2: float
    divergence_s_1: float
    shear_s_1: float
    total_deformation_s_1: float
    vorticity_s_1: float
    crs_epsg: int = 3413


def deformation_from_field(
    field: DisplacementField, maximum_triangle_edge_m: float
) -> tuple[DeformationCell, ...]:
    """Fit displacement gradients on local, orientation-preserving triangles."""
    elapsed_seconds = (field.target_time_utc - field.source_time_utc).total_seconds()
    if elapsed_seconds <= 0:
        raise ValueError("deformation interval must be positive")
    valid = field.available & np.isfinite(field.displacement_m).all(axis=1)
    if valid.sum() < 3:
        return ()
    source = field.source_xy_m[valid]
    displacement = field.displacement_m[valid]
    target = source + displacement
    try:
        triangles = Delaunay(source).simplices
    except QhullError:
        return ()
    cells = []
    for triangle_index, vertices in enumerate(triangles):
        source_triangle = source[vertices]
        target_triangle = target[vertices]
        edges = np.stack(
            (
                source_triangle[1] - source_triangle[0],
                source_triangle[2] - source_triangle[1],
                source_triangle[0] - source_triangle[2],
            )
        )
        if np.linalg.norm(edges, axis=1).max() > maximum_triangle_edge_m:
            continue
        source_twice_area = _twice_area(source_triangle)
        target_twice_area = _twice_area(target_triangle)
        if source_twice_area * target_twice_area <= 0:
            continue
        design = np.column_stack((source_triangle, np.ones(3)))
        gradient = np.linalg.solve(design, displacement[vertices])
        du_dx, dv_dx = gradient[0]
        du_dy, dv_dy = gradient[1]
        strain_xx = du_dx / elapsed_seconds
        strain_yy = dv_dy / elapsed_seconds
        strain_xy = 0.5 * (du_dy + dv_dx) / elapsed_seconds
        divergence = strain_xx + strain_yy
        shear = np.sqrt((strain_xx - strain_yy) ** 2 + 4 * strain_xy**2)
        cells.append(
            DeformationCell(
                pair_id=field.pair_id,
                triangle_index=triangle_index,
                centroid_x_m=float(source_triangle[:, 0].mean()),
                centroid_y_m=float(source_triangle[:, 1].mean()),
                source_area_m2=abs(float(source_twice_area)) / 2.0,
                divergence_s_1=float(divergence),
                shear_s_1=float(shear),
                total_deformation_s_1=float(np.hypot(divergence, shear)),
                vorticity_s_1=float((dv_dx - du_dy) / elapsed_seconds),
            )
        )
    return tuple(cells)


def _twice_area(triangle: np.ndarray) -> float:
    first, second = triangle[1] - triangle[0], triangle[2] - triangle[0]
    return float(first[0] * second[1] - first[1] * second[0])
