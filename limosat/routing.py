"""Causal routing priors and targeted recovery geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from shapely import MultiPoint
from shapely.geometry.base import BaseGeometry

from .config import FieldConfig, MatcherConfig, RoutingConfig
from .field import sample_field
from .imagery import north_up_patch
from .models import DisplacementField


@dataclass(frozen=True)
class CoarseTranslation:
    displacement_m: tuple[float, float]
    response: float
    overlap_fraction: float


class CoarseTranslationUnavailable(ValueError):
    """The image pair has too little valid support for phase correlation."""


def tile_shifts(
    centers_xy_m: np.ndarray,
    routing: RoutingConfig,
    field_config: FieldConfig,
    previous_field: DisplacementField | None,
    previous_elapsed_seconds: float | None,
    current_elapsed_seconds: float,
    initial_displacement_m: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(centers_xy_m, dtype=np.float64)
    shifts = np.zeros_like(centers)
    sources = np.full(len(centers), "same_center", dtype=object)
    if previous_field is None or previous_elapsed_seconds is None:
        if initial_displacement_m is not None:
            shifts[:] = initial_displacement_m
            sources[:] = "phase_correlation"
        return shifts, sources
    if routing.mode == "same_center" or previous_elapsed_seconds <= 0:
        return shifts, sources
    valid = previous_field.available
    if valid.sum() < field_config.minimum_agreeing_matches:
        return shifts, sources
    scale = current_elapsed_seconds / previous_elapsed_seconds
    shifts[:] = np.median(previous_field.displacement_m[valid], axis=0) * scale
    sources[:] = "preceding_global"
    if routing.mode == "sequential_local":
        current_coordinates = previous_field.source_xy_m + previous_field.displacement_m
        values = dict(previous_field.__dict__)
        values.update(
            source_xy_m=current_coordinates,
            displacement_m=previous_field.displacement_m * scale,
        )
        current_field = DisplacementField(**values)
        sampled = sample_field(current_field, centers, field_config.maximum_triangle_edge_m)
        shifts[sampled.available] = sampled.displacement_m[sampled.available]
        sources[sampled.available] = "preceding_local"
    return shifts, sources


def coarse_phase_translation(
    source_path: str,
    target_path: str,
    domain: BaseGeometry,
    matcher: MatcherConfig,
    elapsed_seconds: float,
) -> CoarseTranslation:
    if domain.is_empty:
        raise CoarseTranslationUnavailable("coarse translation domain cannot be empty")
    minx, miny, maxx, maxy = domain.bounds
    maximum_displacement_m = matcher.maximum_displacement_m(elapsed_seconds)
    required_extent = max(maxx - minx, maxy - miny) + 2 * maximum_displacement_m
    pixel_size = max(1_000.0, required_extent / 1_024)
    pixels = max(32, min(1_024, int(math.ceil(required_extent / pixel_size / 32) * 32)))
    center = ((minx + maxx) / 2, (miny + maxy) / 2)
    source, source_valid = north_up_patch(source_path, center, pixels, pixel_size, transform_grid_spacing_px=matcher.transform_grid_spacing_px)
    target, target_valid = north_up_patch(target_path, center, pixels, pixel_size, transform_grid_spacing_px=matcher.transform_grid_spacing_px)
    overlap = source_valid & target_valid
    if overlap.sum() < 64:
        raise CoarseTranslationUnavailable(
            "coarse translation has insufficient valid overlap"
        )
    normalized = []
    for image in (source, target):
        values = image[overlap].astype(np.float32)
        output = np.zeros_like(image, dtype=np.float32)
        output[overlap] = (values - values.mean()) / max(float(values.std()), 1.0e-6)
        normalized.append(output)
    shift, response = cv2.phaseCorrelate(
        normalized[0], normalized[1], cv2.createHanningWindow((pixels, pixels), cv2.CV_32F)
    )
    displacement = np.array([shift[0] * pixel_size, -shift[1] * pixel_size])
    magnitude = np.linalg.norm(displacement)
    if magnitude > maximum_displacement_m:
        displacement *= maximum_displacement_m / magnitude
    return CoarseTranslation(tuple(displacement), float(response), float(overlap.mean()))


def targeted_domain(
    positions_xy_m: np.ndarray, buffer_m: float, overlap: BaseGeometry
) -> BaseGeometry:
    points = np.asarray(positions_xy_m, dtype=np.float64)
    if not len(points):
        return overlap.intersection(MultiPoint([]))
    return MultiPoint(points).buffer(buffer_m).intersection(overlap)


def residual_edge_correction(
    source_xy_m: np.ndarray,
    target_xy_m: np.ndarray,
    target_px: np.ndarray,
    routed_shift_m: np.ndarray,
    matcher: MatcherConfig,
) -> np.ndarray | None:
    """Return a rerouting correction only when residual and edge pressure align."""
    if len(source_xy_m) < 8:
        return None
    residual = np.median(target_xy_m - source_xy_m - routed_shift_m, axis=0)
    slack = (matcher.tile_margin_px - matcher.endpoint_support_radius_px) * matcher.pixel_size_m
    if slack <= 0:
        return None
    edge = matcher.tile_margin_px
    aligned = []
    if residual[0] < -slack:
        aligned.append(np.mean(target_px[:, 0] < edge) >= 0.25)
    elif residual[0] > slack:
        aligned.append(np.mean(target_px[:, 0] >= matcher.tile_size_px - edge) >= 0.25)
    if residual[1] > slack:
        aligned.append(np.mean(target_px[:, 1] < edge) >= 0.25)
    elif residual[1] < -slack:
        aligned.append(np.mean(target_px[:, 1] >= matcher.tile_size_px - edge) >= 0.25)
    return residual if aligned and any(aligned) else None
