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
    def match(
        self, source: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class TileRegion:
    tile_id: int
    row: int
    column: int
    center_xy_m: tuple[float, float]
    core: BaseGeometry


@dataclass(frozen=True)
class _PreparedTile:
    region: TileRegion
    shift: np.ndarray
    source: np.ndarray
    target: np.ndarray
    source_valid: np.ndarray
    target_valid: np.ndarray


@dataclass(frozen=True)
class _HypothesisResult:
    matches: MotionMatches
    field: DisplacementField
    fold_rejected_indices: np.ndarray
    sampling_seconds: float
    matching_seconds: float
    field_seconds: float
    matcher_calls: int
    matcher_tiles: int
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
        diagnostics["matcher_tile_batch_size"] = self.config.matcher.tile_batch_size
        diagnostics["matcher_tile_evaluations"] = sum(
            item.matcher_tiles for _name, item in evaluated
        )
        diagnostics["matcher_execution"] = getattr(
            self.matcher,
            "execution_mode",
            (
                "batch"
                if callable(getattr(self.matcher, "match_batch", None))
                else "eager"
            ),
        )
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
        matcher_tiles = 0
        pending_tiles: list[_PreparedTile] = []
        matched_tiles = []
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
            pending_tiles.append(_PreparedTile(region, shift, *prepared))
            if len(pending_tiles) == self.config.matcher.tile_batch_size:
                results, elapsed, calls = self._match_prepared_tiles(
                    pair, pending_tiles
                )
                matched_tiles.extend(
                    (item.region, item.shift, result)
                    for item, result in zip(pending_tiles, results, strict=True)
                )
                matching_seconds += elapsed
                matcher_calls += calls
                matcher_tiles += len(pending_tiles)
                pending_tiles = []
        if pending_tiles:
            results, elapsed, calls = self._match_prepared_tiles(pair, pending_tiles)
            matched_tiles.extend(
                (item.region, item.shift, result)
                for item, result in zip(pending_tiles, results, strict=True)
            )
            matching_seconds += elapsed
            matcher_calls += calls
            matcher_tiles += len(pending_tiles)

        recovery_tiles: list[tuple[int, _PreparedTile]] = []
        for index, (region, shift, (batch, target_px)) in enumerate(matched_tiles):
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
                    recovery_tiles.append(
                        (
                            index,
                            _PreparedTile(region, corrected_shift, *recovered),
                        )
                    )
                    if len(recovery_tiles) == self.config.matcher.tile_batch_size:
                        elapsed, calls = self._apply_recovery_batch(
                            pair, matched_tiles, recovery_tiles
                        )
                        matching_seconds += elapsed
                        matcher_calls += calls
                        matcher_tiles += len(recovery_tiles)
                        recovery_tiles = []
        if recovery_tiles:
            elapsed, calls = self._apply_recovery_batch(
                pair, matched_tiles, recovery_tiles
            )
            matching_seconds += elapsed
            matcher_calls += calls
            matcher_tiles += len(recovery_tiles)
        batches = [
            batch
            for _region, _shift, (batch, _target_px) in matched_tiles
            if len(batch)
        ]
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
            matcher_tiles,
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

    def _match_prepared_tiles(self, pair, prepared_tiles):
        if not prepared_tiles:
            return [], 0.0, 0
        batch_size = self.config.matcher.tile_batch_size
        match_batch = getattr(self.matcher, "match_batch", None)
        results = []
        matcher_calls = 0
        started = time.perf_counter()
        for offset in range(0, len(prepared_tiles), batch_size):
            tiles = prepared_tiles[offset : offset + batch_size]
            if callable(match_batch):
                raw_matches = match_batch(
                    tuple(item.source for item in tiles),
                    tuple(item.target for item in tiles),
                )
                matcher_calls += 1
            else:
                raw_matches = [
                    self.matcher.match(item.source, item.target) for item in tiles
                ]
                matcher_calls += len(tiles)
            if len(raw_matches) != len(tiles):
                raise ValueError("matcher returned a different number of tile results")
            results.extend(
                self._filter_tile_matches(pair, tile, raw)
                for tile, raw in zip(tiles, raw_matches, strict=True)
            )
        return results, time.perf_counter() - started, matcher_calls

    def _apply_recovery_batch(self, pair, matched_tiles, recovery_tiles):
        recovered_results, elapsed, calls = self._match_prepared_tiles(
            pair, [item for _index, item in recovery_tiles]
        )
        for (index, _prepared), candidate in zip(
            recovery_tiles, recovered_results, strict=True
        ):
            region, shift, existing = matched_tiles[index]
            if len(candidate[0]) > len(existing[0]):
                matched_tiles[index] = (region, shift, candidate)
        return elapsed, calls

    def _filter_tile_matches(self, pair, tile, raw_matches):
        settings = self.config.matcher
        source_px, target_px, score = raw_matches
        keep = (
            source_core_mask(source_px, settings.tile_size_px, settings.tile_margin_px)
            & valid_endpoints(source_px, tile.source_valid)
            & valid_endpoints(target_px, tile.target_valid)
        )
        source_px, target_px, score = source_px[keep], target_px[keep], score[keep]
        target_center = tuple(
            np.asarray(tile.region.center_xy_m) + np.asarray(tile.shift)
        )
        source_xy = projected_coordinates(
            source_px,
            tile.region.center_xy_m,
            settings.tile_size_px,
            settings.pixel_size_m,
        )
        target_xy = projected_coordinates(
            target_px,
            target_center,
            settings.tile_size_px,
            settings.pixel_size_m,
        )
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
            np.full(keep.sum(), tile.region.tile_id),
            np.full(keep.sum(), tile.region.tile_id),
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
