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
    CoarseTranslationUnavailable,
    coarse_phase_translation,
    residual_edge_correction,
    targeted_domain,
    tile_shifts,
)
from .tile_gates import (
    SicFileIndex,
    load_sic_field,
    sic_file_sha256,
    tile_open_water_evidence,
    valid_tile_overlap_gate,
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


@dataclass(frozen=True)
class _HypothesisResult:
    matches: MotionMatches
    field: DisplacementField
    fold_rejected_indices: np.ndarray
    sampling_seconds: float
    matching_seconds: float
    field_seconds: float
    matcher_calls: int
    gate_counts: dict[str, int]


class PairProcessor:
    def __init__(
        self,
        config: RunConfig,
        matcher: TileMatcher,
        sic_index: SicFileIndex | None = None,
    ) -> None:
        self.config = config
        self.matcher = matcher
        self.sic_index = sic_index
        if self.sic_index is None and config.open_water.enabled:
            self.sic_index = SicFileIndex(config.open_water.sic_root)

    def process(
        self,
        pair: ImagePair,
        previous_field: DisplacementField | None = None,
        previous_elapsed_seconds: float | None = None,
        targeted_positions_xy_m: np.ndarray | None = None,
    ) -> PairResult:
        started = time.perf_counter()
        overlap = self._overlap(pair)
        reachable_domain = self._motion_reachable_domain(pair)
        domain = reachable_domain
        if targeted_positions_xy_m is not None:
            domain = targeted_domain(
                targeted_positions_xy_m,
                self.config.routing.targeted_selection_buffer_m,
                domain,
            )
        regions = tile_layout(domain, self.config)
        centers = np.asarray([item.center_xy_m for item in regions], dtype=np.float64)
        diagnostics: dict[str, int | float | str | None] = {
            "planned_tiles": len(regions),
            "direct_overlap_area_m2": float(overlap.area),
            "motion_reachable_area_m2": float(reachable_domain.area),
            "processing_domain_area_m2": float(domain.area),
            "estimated_field_grid_points": int(
                round(reachable_domain.area / self.config.field.grid_spacing_m**2)
            ),
            "phase_correlation_status": "not_requested",
            "skipped_open_water_both_dates": 0,
            "skipped_no_source_core_support": 0,
            "skipped_no_target_support": 0,
            "skipped_no_physics_reachable_valid_overlap": 0,
        }
        ancillary_inputs: dict[str, str] = {}
        source_sic = target_sic = None
        if self.sic_index is not None:
            source_path = self.sic_index.resolve(
                pair.source.time_utc, self.config.open_water.maximum_age_days
            )
            target_path = self.sic_index.resolve(
                pair.target.time_utc, self.config.open_water.maximum_age_days
            )
            if source_path is not None:
                source_sic = load_sic_field(source_path)
                ancillary_inputs[str(source_path)] = sic_file_sha256(source_path)
            if target_path is not None:
                target_sic = load_sic_field(target_path)
                ancillary_inputs[str(target_path)] = sic_file_sha256(target_path)
            diagnostics["source_sic_status"] = (
                "loaded" if source_path is not None else "missing_or_stale"
            )
            diagnostics["target_sic_status"] = (
                "loaded" if target_path is not None else "missing_or_stale"
            )
        else:
            diagnostics["source_sic_status"] = "disabled"
            diagnostics["target_sic_status"] = "disabled"
        hypotheses: list[tuple[str, tuple[float, float] | None]] = [
            ("same_center", None)
        ]
        if (
            previous_field is None
            and self.config.routing.initial == "phase_correlation"
            and not domain.is_empty
        ):
            try:
                coarse = coarse_phase_translation(
                    str(pair.source.path),
                    str(pair.target.path),
                    overlap,
                    self.config.matcher,
                    pair.elapsed_seconds,
                )
                hypotheses = [("phase_correlation", coarse.displacement_m)]
                diagnostics.update(
                    phase_correlation_status="used",
                    phase_correlation_response=coarse.response,
                    phase_correlation_overlap_fraction=coarse.overlap_fraction,
                )
                if (
                    coarse.response
                    < self.config.routing.phase_correlation_minimum_response
                ):
                    hypotheses.append(("same_center", None))
                    diagnostics["phase_correlation_status"] = (
                        "low_response_compared"
                    )
            except CoarseTranslationUnavailable:
                diagnostics["phase_correlation_status"] = "same_center_fallback"
                if self.config.routing.phase_correlation_failure == "error":
                    raise
        evaluated = [
            (
                name,
                self._process_hypothesis(
                    pair,
                    domain,
                    regions,
                    centers,
                    initial,
                    previous_field,
                    previous_elapsed_seconds,
                    source_sic,
                    target_sic,
                ),
            )
            for name, initial in hypotheses
        ]
        selected_name, selected = max(
            evaluated,
            key=lambda item: (
                _hypothesis_quality(item[1]),
                item[0] == "phase_correlation",
            ),
        )
        diagnostics["routing_hypotheses_evaluated"] = len(evaluated)
        diagnostics["selected_routing_hypothesis"] = selected_name
        for key, value in selected.gate_counts.items():
            diagnostics[key] = value
            if len(evaluated) > 1:
                diagnostics[f"{key}_all_hypotheses"] = sum(
                    item.gate_counts[key] for _name, item in evaluated
                )
        if len(evaluated) > 1:
            for name, item in evaluated:
                diagnostics[f"{name}_available_node_count"] = int(
                    item.field.available.sum()
                )
                diagnostics[f"{name}_match_count"] = len(item.matches)
            diagnostics["phase_correlation_status"] = (
                f"low_response_{selected_name}_selected"
            )
        return PairResult(
            selected.matches,
            selected.field,
            selected.fold_rejected_indices,
            {
                "sampling": sum(item.sampling_seconds for _name, item in evaluated),
                "matching": sum(item.matching_seconds for _name, item in evaluated),
                "field": sum(item.field_seconds for _name, item in evaluated),
                "total": time.perf_counter() - started,
            },
            sum(item.matcher_calls for _name, item in evaluated),
            diagnostics,
            ancillary_inputs,
        )

    def _process_hypothesis(
        self,
        pair: ImagePair,
        domain: BaseGeometry,
        regions: tuple[TileRegion, ...],
        centers: np.ndarray,
        initial: tuple[float, float] | None,
        previous_field: DisplacementField | None,
        previous_elapsed_seconds: float | None,
        source_sic,
        target_sic,
    ) -> _HypothesisResult:
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
        gate_counts = {
            "skipped_open_water_both_dates": 0,
            "skipped_no_source_core_support": 0,
            "skipped_no_target_support": 0,
            "skipped_no_physics_reachable_valid_overlap": 0,
        }
        for region, shift in zip(regions, shifts, strict=True):
            target_center = tuple(np.asarray(region.center_xy_m) + np.asarray(shift))
            if self._both_dates_open_water(
                source_sic, target_sic, region.center_xy_m, target_center
            ):
                gate_counts["skipped_open_water_both_dates"] += 1
                continue
            sampled_at = time.perf_counter()
            prepared, skip_reason = self._sample_pair(pair, region.center_xy_m, shift)
            sampling_seconds += time.perf_counter() - sampled_at
            if prepared is None:
                gate_counts[f"skipped_{skip_reason}"] += 1
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
                recovered, _skip_reason = self._sample_pair(
                    pair, region.center_xy_m, corrected_shift
                )
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
        return _HypothesisResult(
            matches,
            field,
            rejected,
            sampling_seconds,
            matching_seconds,
            field_seconds,
            matcher_calls,
            gate_counts,
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

    def _motion_reachable_domain(self, pair: ImagePair) -> BaseGeometry:
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
        maximum = self.config.matcher.maximum_displacement_m(pair.elapsed_seconds)
        return source.intersection(target.buffer(maximum)).buffer(0)

    def _both_dates_open_water(
        self, source_sic, target_sic, source_center, target_center
    ) -> bool:
        settings = self.config.open_water
        if not settings.enabled:
            return False
        extent = self.config.matcher.tile_size_px * self.config.matcher.pixel_size_m
        source = tile_open_water_evidence(
            source_sic,
            source_center,
            extent,
            self.config.analysis_epsg,
            settings.threshold_percent,
            settings.samples_per_axis,
        )
        target = tile_open_water_evidence(
            target_sic,
            target_center,
            extent,
            self.config.analysis_epsg,
            settings.threshold_percent,
            settings.samples_per_axis,
        )
        return source.confidently_open and target.confidently_open

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
        target_center = tuple(np.asarray(center) + np.asarray(shift))
        gate = valid_tile_overlap_gate(
            source_valid & core,
            target_valid,
            center,
            target_center,
            settings.pixel_size_m,
            settings.maximum_displacement_m(pair.elapsed_seconds),
        )
        if gate.skip:
            return None, gate.reason
        return (source, target, source_valid, target_valid), None

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


def _hypothesis_quality(result: _HypothesisResult) -> tuple:
    """Rank truth-free pair outcomes after the normal field and fold gates."""
    available = result.field.available
    residuals = result.field.maximum_residual_m[available]
    finite_residuals = residuals[np.isfinite(residuals)]
    median_residual = (
        float(np.median(finite_residuals)) if len(finite_residuals) else math.inf
    )
    represented_tiles = len(np.unique(result.matches.source_tile))
    return (
        int(available.sum()),
        represented_tiles,
        int(result.field.selected_matches[available].sum()),
        -len(result.fold_rejected_indices),
        -median_residual,
        len(result.matches),
    )
