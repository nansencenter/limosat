"""Matcher-neutral spatial priors for dense sea-ice drift pairs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry.base import BaseGeometry

from .imagery import north_up_patch
from .trajectory import sample_field
from .types import DriftField


@dataclass(frozen=True)
class CoarseTranslation:
    displacement_m: tuple[float, float]
    response: float
    overlap_fraction: float
    pixel_size_m: float
    pixels: int


def preceding_field_shifts(
    source_centers_m: np.ndarray,
    mode: str,
    previous_field: DriftField | None,
    previous_elapsed_days: float | None,
    current_elapsed_days: float,
    minimum_nodes: int,
    grid_spacing_m: float,
    initial_displacement_m: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target-tile shifts using no information later than the source image."""
    source_centers_m = np.asarray(source_centers_m, dtype=np.float64)
    shifts = np.zeros_like(source_centers_m)
    sources = np.full(len(source_centers_m), "same_center_fallback", dtype=object)
    if previous_field is None or previous_elapsed_days is None:
        if initial_displacement_m is not None:
            shifts[:] = initial_displacement_m
            sources[:] = "coarse_phase_translation"
        return shifts, sources
    if mode == "same_center" or previous_elapsed_days <= 0:
        return shifts, sources

    valid = previous_field.available & np.isfinite(
        previous_field.displacement_m
    ).all(axis=1)
    if valid.sum() < minimum_nodes:
        return shifts, sources
    scale = current_elapsed_days / previous_elapsed_days
    previous_displacement = previous_field.displacement_m[valid]
    shifts[:] = np.median(previous_displacement, axis=0) * scale
    sources[:] = "preceding_global_velocity"
    if mode not in {"sequential", "sequential_local"}:
        return shifts, sources

    current_coordinate_field = DriftField(
        grid_row=previous_field.grid_row[valid],
        grid_column=previous_field.grid_column[valid],
        source_xy_m=previous_field.source_xy_m[valid] + previous_displacement,
        displacement_m=previous_displacement * scale,
        available=np.ones(valid.sum(), dtype=bool),
        selected_matches=previous_field.selected_matches[valid],
        candidate_matches=previous_field.candidate_matches[valid],
        support_radius_m=previous_field.support_radius_m[valid],
        maximum_residual_m=previous_field.maximum_residual_m[valid],
    )
    sampled = sample_field(
        current_coordinate_field,
        source_centers_m,
        grid_spacing_m * 1.6,
    )
    shifts[sampled.available] = sampled.displacement_m[sampled.available]
    sources[sampled.available] = "preceding_local_velocity"
    return shifts, sources


def coarse_phase_translation(
    source_path: str,
    target_path: str,
    domain: BaseGeometry,
    maximum_displacement_m: float,
    analysis_epsg: int,
    transform_grid_spacing_px: int,
    preferred_pixel_size_m: float = 1000.0,
    maximum_pixels: int = 1024,
) -> CoarseTranslation:
    """Estimate a sequence-start translation on a coarse projected image pair."""
    if domain.is_empty:
        raise ValueError("coarse translation domain cannot be empty")
    minx, miny, maxx, maxy = domain.bounds
    required_extent_m = max(maxx - minx, maxy - miny) + 2 * maximum_displacement_m
    pixel_size_m = max(
        preferred_pixel_size_m, required_extent_m / maximum_pixels
    )
    pixels = int(math.ceil(required_extent_m / pixel_size_m / 32.0) * 32)
    pixels = max(32, min(pixels, maximum_pixels))
    center = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)
    source, source_valid = north_up_patch(
        source_path,
        center,
        pixels,
        pixel_size_m,
        analysis_epsg,
        transform_grid_spacing_px,
    )
    target, target_valid = north_up_patch(
        target_path,
        center,
        pixels,
        pixel_size_m,
        analysis_epsg,
        transform_grid_spacing_px,
    )
    overlap = source_valid & target_valid
    if overlap.sum() < 64:
        raise ValueError("coarse translation has insufficient valid overlap")
    standardized = []
    for image in (source, target):
        values = image[overlap].astype(np.float32)
        normalized = np.zeros_like(image, dtype=np.float32)
        normalized[overlap] = (
            values - values.mean()
        ) / max(float(values.std()), 1.0e-6)
        standardized.append(normalized)
    window = cv2.createHanningWindow((pixels, pixels), cv2.CV_32F)
    shift_px, response = cv2.phaseCorrelate(
        standardized[0], standardized[1], window
    )
    displacement = np.array(
        [shift_px[0] * pixel_size_m, -shift_px[1] * pixel_size_m]
    )
    magnitude = float(np.linalg.norm(displacement))
    if magnitude > maximum_displacement_m and magnitude > 0:
        displacement *= maximum_displacement_m / magnitude
    return CoarseTranslation(
        displacement_m=(float(displacement[0]), float(displacement[1])),
        response=float(response),
        overlap_fraction=float(overlap.mean()),
        pixel_size_m=float(pixel_size_m),
        pixels=pixels,
    )
