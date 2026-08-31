"""Tiled EfficientLoFTR pair processing."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from shapely import box
from shapely.geometry.base import BaseGeometry

from .catalog import ImagePair
from .config import RunConfig
from .efficientloftr import (
    source_core_mask,
    speed_limit_mask,
    valid_endpoints,
    valid_support,
)
from .field import estimate_field, reject_folds
from .imagery import north_up_patch, projected_coordinates, projected_footprint
from .models import DisplacementField, MotionMatches, PairResult
from .routing import (
    coarse_phase_translation,
    residual_edge_correction,
    targeted_domain,
    tile_shifts,
)


class TileMatcher(Protocol):
    def match(self, source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class TileRegion:
    tile_id: int
    row: int
    column: int
    center_xy_m: tuple[float, float]
    core: BaseGeometry


class PairProcessor:
    def __init__(self, config: RunConfig, matcher: TileMatcher) -> None:
        self.config = config
        self.matcher = matcher

    def process(
        self,
        pair: ImagePair,
        previous_field: DisplacementField | None = None,
        previous_elapsed_seconds: float | None = None,
        targeted_positions_xy_m: np.ndarray | None = None,
    ) -> PairResult:
        started = time.perf_counter()
        overlap = self._overlap(pair)
        domain = overlap
        if targeted_positions_xy_m is not None:
            domain = targeted_domain(
                targeted_positions_xy_m,
                self.config.routing.targeted_selection_buffer_m,
                overlap,
            )
        regions = tile_layout(domain, self.config)
        centers = np.asarray([item.center_xy_m for item in regions], dtype=np.float64)
        initial = None
        if (
            previous_field is None
            and self.config.routing.initial == "phase_correlation"
            and not domain.is_empty
        ):
            initial = coarse_phase_translation(
                str(pair.source.path),
                str(pair.target.path),
                domain,
                self.config.matcher,
                pair.elapsed_seconds,
            ).displacement_m
        shifts, _routing_sources = tile_shifts(
            centers,
            self.config.routing,
            self.config.field,
            previous_field,
            previous_elapsed_seconds,
            pair.elapsed_seconds,
            initial,
        )
        sampling_seconds = 0.0
        matching_seconds = 0.0
        matcher_calls = 0
        batches: list[MotionMatches] = []
        for region, shift in zip(regions, shifts, strict=True):
            sampled_at = time.perf_counter()
            prepared = self._sample_pair(pair, region.center_xy_m, shift)
            sampling_seconds += time.perf_counter() - sampled_at
            if prepared is None:
                continue
            source, target, source_valid, target_valid = prepared
            matched_at = time.perf_counter()
            batch, target_px = self._match_tile(
                pair, region, shift, source, target, source_valid, target_valid
            )
            matching_seconds += time.perf_counter() - matched_at
            matcher_calls += 1
            if (
                self.config.routing.residual_edge_recovery
                and len(batch)
                and (correction := residual_edge_correction(
                    batch.source_xy_m,
                    batch.target_xy_m,
                    target_px,
                    shift,
                    self.config.matcher,
                )) is not None
            ):
                corrected_shift = shift + correction
                sampled_at = time.perf_counter()
                recovered = self._sample_pair(pair, region.center_xy_m, corrected_shift)
                sampling_seconds += time.perf_counter() - sampled_at
                if recovered is not None:
                    matched_at = time.perf_counter()
                    candidate, _ = self._match_tile(
                        pair, region, corrected_shift, *recovered
                    )
                    matching_seconds += time.perf_counter() - matched_at
                    matcher_calls += 1
                    if len(candidate) > len(batch):
                        batch = candidate
            if len(batch):
                batches.append(batch)
        matches = _combine(batches)
        field_at = time.perf_counter()
        field = estimate_field(matches, pair, domain, self.config.field)
        field, rejected = reject_folds(field, self.config.field.maximum_triangle_edge_m)
        field_seconds = time.perf_counter() - field_at
        return PairResult(
            matches,
            field,
            rejected,
            {
                "sampling": sampling_seconds,
                "matching": matching_seconds,
                "field": field_seconds,
                "total": time.perf_counter() - started,
            },
            matcher_calls,
        )

    def _overlap(self, pair: ImagePair) -> BaseGeometry:
        source = (
            pair.source.footprint
            if pair.source.footprint is not None
            else projected_footprint(pair.source.path)
        )
        target = (
            pair.target.footprint
            if pair.target.footprint is not None
            else projected_footprint(pair.target.path)
        )
        overlap = source.intersection(target).buffer(0)
        if overlap.is_empty:
            raise ValueError(f"pair {pair.pair_id} has no projected overlap")
        return overlap

    def _sample_pair(self, pair: ImagePair, center, shift):
        settings = self.config.matcher
        source, source_valid = north_up_patch(
            pair.source.path,
            center,
            settings.tile_size_px,
            settings.pixel_size_m,
            self.config.analysis_epsg,
            settings.transform_grid_spacing_px,
        )
        target_center = tuple(np.asarray(center) + np.asarray(shift))
        target, target_valid = north_up_patch(
            pair.target.path,
            target_center,
            settings.tile_size_px,
            settings.pixel_size_m,
            self.config.analysis_epsg,
            settings.transform_grid_spacing_px,
        )
        source_valid = valid_support(source_valid, settings.endpoint_support_radius_px)
        target_valid = valid_support(target_valid, settings.endpoint_support_radius_px)
        core = np.zeros_like(source_valid)
        margin = settings.tile_margin_px
        core[margin : settings.tile_size_px - margin, margin : settings.tile_size_px - margin] = True
        if not np.any(source_valid & core) or not target_valid.any():
            return None
        return source, target, source_valid, target_valid

    def _match_tile(self, pair, region, shift, source, target, source_valid, target_valid):
        settings = self.config.matcher
        source_px, target_px, score = self.matcher.match(source, target)
        keep = (
            source_core_mask(source_px, settings.tile_size_px, settings.tile_margin_px)
            & valid_endpoints(source_px, source_valid)
            & valid_endpoints(target_px, target_valid)
        )
        source_px, target_px, score = source_px[keep], target_px[keep], score[keep]
        target_center = tuple(np.asarray(region.center_xy_m) + np.asarray(shift))
        source_xy = projected_coordinates(source_px, region.center_xy_m, settings.tile_size_px, settings.pixel_size_m)
        target_xy = projected_coordinates(target_px, target_center, settings.tile_size_px, settings.pixel_size_m)
        keep = speed_limit_mask(
            source_xy,
            target_xy,
            pair.elapsed_seconds,
            settings.maximum_speed_m_per_day,
        )
        batch = MotionMatches(
            source_xy[keep],
            target_xy[keep],
            score[keep],
            np.full(keep.sum(), region.tile_id),
            np.full(keep.sum(), region.tile_id),
        )
        return batch, target_px[keep]


def tile_layout(domain: BaseGeometry, config: RunConfig) -> tuple[TileRegion, ...]:
    if domain.is_empty:
        return ()
    core_size = config.matcher.tile_core_size_m
    origin = config.matcher.tile_grid_origin_m
    minx, miny, maxx, maxy = domain.bounds
    columns = range(math.floor((minx - origin) / core_size), math.ceil((maxx - origin) / core_size))
    rows = range(math.floor((miny - origin) / core_size), math.ceil((maxy - origin) / core_size))
    regions = []
    for row in rows:
        for column in columns:
            x0, y0 = origin + column * core_size, origin + row * core_size
            core = box(x0, y0, x0 + core_size, y0 + core_size)
            if not core.intersects(domain):
                continue
            regions.append(
                TileRegion(
                    len(regions),
                    row,
                    column,
                    (x0 + core_size / 2, y0 + core_size / 2),
                    core,
                )
            )
    return tuple(regions)


def _combine(batches: list[MotionMatches]) -> MotionMatches:
    if not batches:
        return MotionMatches.empty()
    return MotionMatches(
        np.vstack([item.source_xy_m for item in batches]),
        np.vstack([item.target_xy_m for item in batches]),
        np.concatenate([item.score for item in batches]),
        np.concatenate([item.source_tile for item in batches]),
        np.concatenate([item.target_tile for item in batches]),
    )
