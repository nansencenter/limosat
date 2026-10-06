"""Tiled EfficientLoFTR pair processing."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field as dataclass_field, replace
from typing import Protocol

import cv2
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
from .field import estimate_field, flipped_indices, reject_folds
from .imagery import (
    ProjectedPatchCache,
    north_up_patch,
    projected_coordinates,
    projected_footprint,
)
from .models import DisplacementField, MotionMatches, PairResult
from .routing import (
    CoarseTranslationUnavailable,
    coarse_match_shift,
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
    grid: int = 0


@dataclass(frozen=True)
class _PreparedTile:
    region: TileRegion
    shift: np.ndarray
    source: np.ndarray
    target: np.ndarray
    source_valid: np.ndarray
    target_valid: np.ndarray
    masked: bool = False


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


@dataclass(frozen=True)
class _CoarseRoutingResult:
    shifts: np.ndarray
    sampling_seconds: float
    matching_seconds: float
    matcher_calls: int
    matcher_tiles: int
    counts: dict[str, int]
    matches: MotionMatches = dataclass_field(default_factory=MotionMatches.empty)


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
        self._patch_cache = (
            ProjectedPatchCache(
                config.analysis_epsg, config.matcher.transform_grid_spacing_px
            )
            if config.matcher.tile_sampling == "pair_cache"
            else None
        )

    def process(
        self,
        pair: ImagePair,
        previous_field: DisplacementField | None = None,
        previous_elapsed_seconds: float | None = None,
        targeted_positions_xy_m: np.ndarray | None = None,
    ) -> PairResult:
        started = time.perf_counter()
        if self._patch_cache is not None:
            self._patch_cache.clear()
        try:
            return self._process(
                pair, started, previous_field, previous_elapsed_seconds,
                targeted_positions_xy_m,
            )
        finally:
            if self._patch_cache is not None:
                self._patch_cache.clear()

    def _process(
        self,
        pair: ImagePair,
        started: float,
        previous_field: DisplacementField | None,
        previous_elapsed_seconds: float | None,
        targeted_positions_xy_m: np.ndarray | None,
    ) -> PairResult:
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
        matched_tiles = []
        gate_counts = {
            "skipped_open_water_both_dates": 0,
            "skipped_no_source_core_support": 0,
            "skipped_no_target_support": 0,
            "skipped_no_physics_reachable_valid_overlap": 0,
        }
        if self.config.routing.coarse_matching:
            refined = self._refine_tile_shifts(
                pair, regions, shifts, source_sic, target_sic
            )
            shifts = refined.shifts
            sampling_seconds += refined.sampling_seconds
            matching_seconds += refined.matching_seconds
            matcher_calls += refined.matcher_calls
            matcher_tiles += refined.matcher_tiles
            gate_counts.update(refined.counts)
        region_shifts = iter(zip(regions, shifts, strict=True))

        def prepare_batch():
            pending: list[_PreparedTile] = []
            elapsed = 0.0
            skipped = {key: 0 for key in gate_counts}
            for region, shift in region_shifts:
                shift = self._place(region.center_xy_m, shift)
                target_center = tuple(
                    np.asarray(region.center_xy_m) + np.asarray(shift)
                )
                if self._both_dates_open_water(
                    source_sic,
                    target_sic,
                    region.center_xy_m,
                    target_center,
                ):
                    skipped["skipped_open_water_both_dates"] += 1
                    continue
                sampled_at = time.perf_counter()
                prepared, skip_reason = self._sample_pair(
                    pair, region.center_xy_m, shift
                )
                elapsed += time.perf_counter() - sampled_at
                if prepared is None:
                    skipped[f"skipped_{skip_reason}"] += 1
                    continue
                pending.append(_PreparedTile(region, shift, *prepared))
                if (
                    0.0
                    < prepared[3].mean()
                    < self.config.matcher.masked_rematch_target_valid_fraction
                ):
                    pending.append(_masked_tile(
                        pending[-1], self.config.matcher.endpoint_support_radius_px
                    ))
                if len(pending) >= self.config.matcher.tile_batch_size:
                    break
            return pending, elapsed, skipped

        with ThreadPoolExecutor(max_workers=1) as executor:
            prepared = executor.submit(prepare_batch)
            while True:
                pending_tiles, elapsed, skipped = prepared.result()
                sampling_seconds += elapsed
                for key, count in skipped.items():
                    gate_counts[key] += count
                if not pending_tiles:
                    break
                prepared = executor.submit(prepare_batch)
                results, elapsed, calls = self._match_prepared_tiles(
                    pair, pending_tiles
                )
                matched_tiles.extend(
                    (item.region, item.shift, result, item.masked)
                    for item, result in zip(pending_tiles, results, strict=True)
                )
                matching_seconds += elapsed
                matcher_calls += calls
                matcher_tiles += len(pending_tiles)

        recovery_tiles: list[tuple[int, _PreparedTile]] = []
        for index, (region, shift, (batch, target_px), masked) in enumerate(
            matched_tiles
        ):
            if (
                self.config.routing.residual_edge_recovery
                and not masked
                and len(batch)
                and (correction := residual_edge_correction(
                    batch.source_xy_m,
                    batch.target_xy_m,
                    target_px,
                    shift,
                    self.config.matcher,
                )) is not None
            ):
                corrected_shift = self._place(region.center_xy_m, shift + correction)
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
            for _region, _shift, (batch, _target_px), _masked in matched_tiles
            if len(batch)
        ]
        matches = _combine(batches)
        field_at = time.perf_counter()
        edge = self.config.field.maximum_triangle_edge_m
        if self.config.matcher.layered_layout:
            matches, layout_counts = _deduplicate(
                matches, self.config.matcher.layout_dedup_m
            )
            gate_counts.update(layout_counts)
            gate_counts["layout_secondary_tiles"] = sum(
                1 for region, *_rest, masked in matched_tiles
                if region.grid and not masked
            )
            gate_counts["layout_masked_rematches"] = sum(
                1 for *_rest, masked in matched_tiles if masked
            )
            field, rejected = reject_folds(
                estimate_field(matches, pair, domain, self.config.field), edge
            )
            if self.config.matcher.layout_field_merge == "primary_first":
                primary = _combine([
                    batch
                    for region, _shift, (batch, _target_px), masked in matched_tiles
                    if len(batch) and not region.grid and not masked
                ])
                primary_field, primary_rejected = reject_folds(
                    estimate_field(primary, pair, domain, self.config.field), edge
                )
                gate_counts["layout_primary_nodes"] = int(primary_field.available.sum())
                field, merge_rejected = _merge_primary_first(
                    primary_field, field, edge
                )
                rejected = np.union1d(primary_rejected, merge_rejected)
            gate_counts["layout_nodes"] = int(field.available.sum())
        else:
            field = estimate_field(matches, pair, domain, self.config.field)
            field, rejected = reject_folds(field, edge)
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

    def _refine_tile_shifts(
        self,
        pair: ImagePair,
        regions: tuple[TileRegion, ...],
        shifts: np.ndarray,
        source_sic,
        target_sic,
    ) -> _CoarseRoutingResult:
        """Run one bounded coarse pass with the same model before fine sampling.

        Coarse windows follow the primary grid only. Tiles of the other grids take
        their shift from the pooled coarse matches around their own centre, so
        extra grids add no coarse matching.
        """
        refine = (
            self._refine_tile_shifts_shared
            if self.config.routing.coarse_window_stride > 1
            else self._refine_tile_shifts_per_tile
        )
        primary = np.array([not region.grid for region in regions], dtype=bool)
        if primary.all():
            return refine(pair, regions, shifts, source_sic, target_sic)
        indices = np.flatnonzero(primary)
        result = refine(
            pair, tuple(regions[index] for index in indices), shifts[indices],
            source_sic, target_sic,
        )
        refined = shifts.copy()
        refined[indices] = result.shifts
        counts = dict(result.counts)
        counts["coarse_secondary_refined"] = 0
        counts["coarse_secondary_fallback"] = 0
        settings = self.config.routing
        maximum = self.config.matcher.maximum_displacement_m(pair.elapsed_seconds)
        for index in np.flatnonzero(~primary):
            shift, _reason = coarse_match_shift(
                result.matches,
                regions[index].center_xy_m,
                settings.coarse_support_radius_m,
                settings.coarse_minimum_matches,
                maximum,
            ) if len(result.matches) else (None, "insufficient_support")
            if shift is None:
                counts["coarse_secondary_fallback"] += 1
            else:
                refined[index] = shift
                counts["coarse_secondary_refined"] += 1
        return _CoarseRoutingResult(
            refined, result.sampling_seconds, result.matching_seconds,
            result.matcher_calls, result.matcher_tiles, counts, result.matches,
        )

    def _refine_tile_shifts_per_tile(
        self,
        pair: ImagePair,
        regions: tuple[TileRegion, ...],
        shifts: np.ndarray,
        source_sic,
        target_sic,
    ) -> _CoarseRoutingResult:
        """Centre one coarse window on each fine tile."""
        settings = self.config.routing
        coarse_config = replace(
            self.config,
            matcher=replace(
                self.config.matcher, pixel_size_m=settings.coarse_pixel_size_m
            ),
        )
        coarse = PairProcessor(coarse_config, self.matcher, self.sic_index)
        refined = shifts.copy()
        counts = {
            "coarse_tiles_refined": 0,
            "coarse_fallback_insufficient_support": 0,
            "coarse_fallback_invalid_shift": 0,
            "coarse_fallback_no_source_core_support": 0,
            "coarse_fallback_no_target_support": 0,
            "coarse_fallback_no_physics_reachable_valid_overlap": 0,
            "coarse_skipped_open_water_both_dates": 0,
        }
        sampling_seconds = matching_seconds = 0.0
        calls = evaluations = 0
        pending: list[_PreparedTile] = []
        indices: list[int] = []
        collected: list[MotionMatches] = []

        def match_pending():
            nonlocal matching_seconds, calls, evaluations
            results, elapsed, batch_calls = coarse._match_prepared_tiles(pair, pending)
            matching_seconds += elapsed
            calls += batch_calls
            evaluations += len(pending)
            for index, (matches, _target_px) in zip(indices, results, strict=True):
                collected.append(matches)
                shift, reason = coarse_match_shift(
                    matches,
                    regions[index].center_xy_m,
                    settings.coarse_support_radius_m,
                    settings.coarse_minimum_matches,
                    self.config.matcher.maximum_displacement_m(pair.elapsed_seconds),
                )
                if shift is not None:
                    refined[index] = shift
                    counts["coarse_tiles_refined"] += 1
                else:
                    counts[f"coarse_fallback_{reason}"] += 1
            pending.clear()
            indices.clear()

        for index, (region, shift) in enumerate(zip(regions, shifts, strict=True)):
            shift = coarse._place(region.center_xy_m, shift)
            target_center = tuple(np.asarray(region.center_xy_m) + shift)
            if coarse._both_dates_open_water(
                source_sic, target_sic, region.center_xy_m, target_center
            ):
                counts["coarse_skipped_open_water_both_dates"] += 1
                continue
            started = time.perf_counter()
            sampled, reason = coarse._sample_pair(pair, region.center_xy_m, shift)
            sampling_seconds += time.perf_counter() - started
            if sampled is None:
                counts[f"coarse_fallback_{reason}"] += 1
                continue
            pending.append(_PreparedTile(region, shift, *sampled))
            indices.append(index)
            if len(pending) == self.config.matcher.tile_batch_size:
                match_pending()
        if pending:
            match_pending()
        counts["coarse_matcher_tile_evaluations"] = evaluations
        return _CoarseRoutingResult(
            refined, sampling_seconds, matching_seconds, calls, evaluations, counts,
            _combine([item for item in collected if len(item)]),
        )

    def _refine_tile_shifts_shared(
        self,
        pair: ImagePair,
        regions: tuple[TileRegion, ...],
        shifts: np.ndarray,
        source_sic,
        target_sic,
    ) -> _CoarseRoutingResult:
        """Share each coarse window among the fine tiles nearest its centre.

        Windows sit on every ``coarse_window_stride``-th fine tile of the fixed
        tile grid. Each fine tile uses only the window nearest to it, so its
        support circle stays inside that window's core (stride 3 offsets a
        tile centre by at most one tile per axis). The window's target crop is
        placed with the median prior shift of its member tiles. A member without
        local support in a window that did measure motion is retried with its
        own centred window, which tolerates a larger shift error.
        """
        settings = self.config.routing
        stride = settings.coarse_window_stride
        core_size = self.config.matcher.tile_core_size_m
        origin = self.config.matcher.tile_grid_origin_m
        half = stride // 2
        coarse_config = replace(
            self.config,
            matcher=replace(
                self.config.matcher, pixel_size_m=settings.coarse_pixel_size_m
            ),
        )
        coarse = PairProcessor(coarse_config, self.matcher, self.sic_index)
        refined = shifts.copy()
        counts = {
            "coarse_tiles_refined": 0,
            "coarse_fallback_insufficient_support": 0,
            "coarse_fallback_invalid_shift": 0,
            "coarse_fallback_no_source_core_support": 0,
            "coarse_fallback_no_target_support": 0,
            "coarse_fallback_no_physics_reachable_valid_overlap": 0,
            "coarse_skipped_open_water_both_dates": 0,
        }
        members: dict[tuple[int, int], list[int]] = {}
        for index, region in enumerate(regions):
            key = (
                math.floor((region.row + half) / stride) * stride,
                math.floor((region.column + half) / stride) * stride,
            )
            members.setdefault(key, []).append(index)
        sampling_seconds = matching_seconds = 0.0
        calls = evaluations = 0
        pending: list[_PreparedTile] = []
        pending_members: list[list[int]] = []
        retry: list[int] = []
        collected: list[MotionMatches] = []
        maximum = self.config.matcher.maximum_displacement_m(pair.elapsed_seconds)

        def match_pending():
            nonlocal matching_seconds, calls, evaluations
            results, elapsed, batch_calls = coarse._match_prepared_tiles(pair, pending)
            matching_seconds += elapsed
            calls += batch_calls
            evaluations += len(pending)
            for indices, (matches, _target_px) in zip(
                pending_members, results, strict=True
            ):
                collected.append(matches)
                for index in indices:
                    shift, reason = coarse_match_shift(
                        matches,
                        regions[index].center_xy_m,
                        settings.coarse_support_radius_m,
                        settings.coarse_minimum_matches,
                        maximum,
                    )
                    if shift is not None:
                        refined[index] = shift
                        counts["coarse_tiles_refined"] += 1
                    elif len(matches) >= settings.coarse_minimum_matches:
                        # The window measured motion elsewhere; this tile may sit
                        # beyond its shift tolerance, so give it its own window.
                        retry.append(index)
                    else:
                        counts[f"coarse_fallback_{reason}"] += 1
            pending.clear()
            pending_members.clear()

        for (row, column), indices in sorted(members.items()):
            center = (
                origin + (column + 0.5) * core_size,
                origin + (row + 0.5) * core_size,
            )
            shift = coarse._place(center, np.median(shifts[indices], axis=0))
            target_center = tuple(np.asarray(center) + shift)
            if coarse._both_dates_open_water(
                source_sic, target_sic, center, target_center
            ):
                counts["coarse_skipped_open_water_both_dates"] += len(indices)
                continue
            started = time.perf_counter()
            sampled, reason = coarse._sample_pair(pair, center, shift)
            sampling_seconds += time.perf_counter() - started
            if sampled is None:
                counts[f"coarse_fallback_{reason}"] += len(indices)
                continue
            window = TileRegion(-1, row, column, center, box(
                center[0] - core_size / 2, center[1] - core_size / 2,
                center[0] + core_size / 2, center[1] + core_size / 2,
            ))
            pending.append(_PreparedTile(window, shift, *sampled))
            pending_members.append(indices)
            if len(pending) == self.config.matcher.tile_batch_size:
                match_pending()
        if pending:
            match_pending()
        counts["coarse_shared_windows"] = len(members)
        counts["coarse_per_tile_retries"] = len(retry)
        if retry:
            single = self._refine_tile_shifts_per_tile(
                pair,
                tuple(regions[index] for index in retry),
                shifts[retry],
                source_sic,
                target_sic,
            )
            refined[retry] = single.shifts
            collected.append(single.matches)
            sampling_seconds += single.sampling_seconds
            matching_seconds += single.matching_seconds
            calls += single.matcher_calls
            evaluations += single.matcher_tiles
            for key, value in single.counts.items():
                if key != "coarse_matcher_tile_evaluations":
                    counts[key] += value
        counts["coarse_matcher_tile_evaluations"] = evaluations
        return _CoarseRoutingResult(
            refined, sampling_seconds, matching_seconds, calls, evaluations, counts,
            _combine([item for item in collected if len(item)]),
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

    def _place(self, center, shift) -> np.ndarray:
        """Snap the target window to the sampling lattice when tiles are cached."""
        shift = np.asarray(shift, dtype=np.float64)
        if self._patch_cache is None:
            return shift
        size = self.config.matcher.pixel_size_m
        return np.round((np.asarray(center) + shift) / size) * size - np.asarray(center)

    def _patch(self, path, center):
        settings = self.config.matcher
        if self._patch_cache is not None:
            return self._patch_cache.patch(
                path, center, settings.tile_size_px, settings.pixel_size_m
            )
        return north_up_patch(
            path,
            center,
            settings.tile_size_px,
            settings.pixel_size_m,
            self.config.analysis_epsg,
            settings.transform_grid_spacing_px,
        )

    def _sample_pair(self, pair: ImagePair, center, shift):
        """Sample source and target windows; callers pass a ``_place``d shift."""
        settings = self.config.matcher
        source, source_valid = self._patch(pair.source.path, center)
        target_center = tuple(np.asarray(center) + np.asarray(shift))
        target, target_valid = self._patch(pair.target.path, target_center)
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
            region, shift, existing, masked = matched_tiles[index]
            if len(candidate[0]) > len(existing[0]):
                matched_tiles[index] = (region, shift, candidate, masked)
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
    minx, miny, maxx, maxy = domain.bounds
    regions = []
    for grid, (offset_x, offset_y) in enumerate(config.matcher.tile_grid_offsets):
        origin_x = config.matcher.tile_grid_origin_m + offset_x * core_size
        origin_y = config.matcher.tile_grid_origin_m + offset_y * core_size
        columns = range(
            math.floor((minx - origin_x) / core_size),
            math.ceil((maxx - origin_x) / core_size),
        )
        rows = range(
            math.floor((miny - origin_y) / core_size),
            math.ceil((maxy - origin_y) / core_size),
        )
        for row in rows:
            for column in columns:
                x0, y0 = origin_x + column * core_size, origin_y + row * core_size
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
                        grid,
                    )
                )
    return tuple(regions)


def _masked_tile(tile: _PreparedTile, support_radius_px: int) -> _PreparedTile:
    """Blank the source where the aligned target window has no valid support.

    A textured source facing a mostly invalid target suppresses confident
    matches; showing both images only the shared strip restores them. The
    target support was eroded by ``support_radius_px``; dilating it back by the
    same radius keeps source texture up to the target's valid edge.
    """
    kernel = np.ones((2 * support_radius_px + 1,) * 2, np.uint8)
    support = cv2.dilate(tile.target_valid.astype(np.uint8), kernel).astype(bool)
    source = tile.source.copy()
    source[~(tile.source_valid & support)] = 0
    target = tile.target.copy()
    target[~tile.target_valid] = 0
    return replace(tile, source=source, target=target, masked=True)


def _deduplicate(
    matches: MotionMatches, bin_m: float
) -> tuple[MotionMatches, dict[str, int | float]]:
    """Keep the highest-score match per source bin and report co-located agreement.

    Overlapping tiles measure the same ice independently, so the displacement
    difference between co-located matches is a truth-free consistency signal.
    """
    counts: dict[str, int | float] = {
        "layout_dedup_removed": 0,
        "layout_colocated_bins": 0,
        "layout_colocated_difference_median_m": 0.0,
        "layout_colocated_difference_gt_1km_share": 0.0,
    }
    if not len(matches):
        return matches, counts
    bins = np.floor(matches.source_xy_m / bin_m).astype(np.int64)
    order = np.lexsort((-matches.score, bins[:, 1], bins[:, 0]))
    sorted_bins = bins[order]
    first = np.r_[True, np.any(sorted_bins[1:] != sorted_bins[:-1], axis=1)]
    leader = order[np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))]
    duplicate = ~first
    if duplicate.any():
        displacement = matches.displacement_m
        difference = np.linalg.norm(
            displacement[order[duplicate]] - displacement[leader[duplicate]], axis=1
        )
        counts.update(
            layout_dedup_removed=int(duplicate.sum()),
            layout_colocated_bins=int(len(np.unique(leader[duplicate]))),
            layout_colocated_difference_median_m=float(np.median(difference)),
            layout_colocated_difference_gt_1km_share=float(np.mean(difference > 1_000.0)),
        )
    keep = np.sort(order[first])
    return (
        MotionMatches(
            matches.source_xy_m[keep],
            matches.target_xy_m[keep],
            matches.score[keep],
            matches.source_tile[keep],
            matches.target_tile[keep],
        ),
        counts,
    )


def _merge_primary_first(
    primary: DisplacementField, combined: DisplacementField, maximum_triangle_edge_m: float
) -> tuple[DisplacementField, np.ndarray]:
    """Keep every primary node; fill primary gaps from the combined field.

    Folds created by added nodes remove added nodes first, so the overlap can
    only add measurements to the primary-grid field.
    """
    added = combined.available & ~primary.available
    values = dict(primary.__dict__)
    for name in (
        "displacement_m", "selected_matches", "candidate_matches",
        "support_radius_m", "maximum_residual_m",
    ):
        mine, other = getattr(primary, name), getattr(combined, name)
        mask = added[:, None] if mine.ndim == 2 else added
        values[name] = np.where(mask, other, mine)
    values["available"] = primary.available | added
    merged = DisplacementField(**values)
    available = merged.available.copy()
    rejected: list[np.ndarray] = []
    while True:
        selected = flipped_indices(merged.with_available(available), maximum_triangle_edge_m)
        if not len(selected):
            break
        drop = selected[added[selected]]
        selected = drop if len(drop) else selected
        available[selected] = False
        rejected.append(selected)
    indices = np.unique(np.concatenate(rejected)) if rejected else np.empty(0, dtype=int)
    return merged.with_available(available), indices


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
