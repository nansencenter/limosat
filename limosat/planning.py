"""Deterministic global candidate and primary image-pair planning."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
from shapely.geometry.base import BaseGeometry

from .catalog import ImageCatalogue, ImagePair, ImageRecord
from .config import RoutingConfig, RunConfig, _pop_default_planning
from .field import regular_grid
from .imagery import projected_footprint


@dataclass(frozen=True)
class PlannedPair:
    pair: ImagePair
    ordinal: int
    selection: Literal["candidate", "primary"]
    overlap_fraction: float | None
    overlap_area_m2: float | None
    skipped_images: int
    planning_component_id: str


@dataclass(frozen=True)
class CandidatePlan:
    pairs: tuple[PlannedPair, ...]
    exclusion_counts: Mapping[str, int]


def run_plan(config: RunConfig, catalogue: ImageCatalogue) -> CandidatePlan:
    """The run's candidate plan, cached under ``output_directory/plan-cache``.

    Planning a pan-Arctic week takes about four minutes and every stage (and
    every matcher process) needs the same plan. The cache key covers every
    catalogue record, the routing settings and the planning code, so any change
    to them plans afresh.
    """
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in (package / "planning.py", package / "catalog.py"):
        digest.update(path.read_bytes())
    records = [
        [image.image_id, str(image.path), image.time_utc.isoformat(), image.component_id,
         image.footprint.wkb_hex if image.footprint is not None else None,
         image.platform, image.absolute_orbit]
        for image in catalogue.chronological()
    ]
    routing = asdict(config.routing)
    if not config.routing.source_coverage_completion:
        # Plans made before the option existed keep their cache key.
        routing.pop("source_coverage_completion")
        routing.pop("source_coverage_minimum_cells")
    _pop_default_planning(routing, config.routing)
    key = [records, routing, config.matcher.maximum_speed_m_per_day, config.analysis_epsg]
    cell_filter = None
    if config.routing.plan_surface_cells:
        from .surface import surface_gate

        package_surface = package / "surface.py"
        digest.update(package_surface.read_bytes())
        key.append(asdict(config.surface))
        for name in ("land_raster",):
            value = getattr(config.surface, name)
            if value:
                key.append(hashlib.sha256(Path(value).read_bytes()).hexdigest())
        cell_filter = _surface_cell_filter(surface_gate(config), config.routing.planning_grid_spacing_m)
    digest.update(json.dumps(key, sort_keys=True, default=str).encode("utf-8"))
    path = Path(config.output_directory) / "plan-cache" / f"{digest.hexdigest()}.pickle"
    if path.is_file():
        with path.open("rb") as stream:
            return pickle.load(stream)
    plan = build_candidate_plan(
        catalogue,
        config.routing,
        grid_spacing_m=config.routing.planning_grid_spacing_m,
        maximum_speed_m_per_day=config.matcher.maximum_speed_m_per_day,
        cell_filter=cell_filter,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    with temporary.open("wb") as stream:
        pickle.dump(plan, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    return plan


def plan_candidate_pairs(
    catalogue: ImageCatalogue,
    config: RoutingConfig,
    grid_spacing_m: float = 4_000.0,
    maximum_speed_m_per_day: float = 43_200.0,
) -> tuple[PlannedPair, ...]:
    """Plan overlap-qualified candidates and a deterministic primary subset."""
    return build_candidate_plan(
        catalogue,
        config,
        grid_spacing_m=grid_spacing_m,
        maximum_speed_m_per_day=maximum_speed_m_per_day,
    ).pairs


def build_candidate_plan(
    catalogue: ImageCatalogue,
    config: RoutingConfig,
    grid_spacing_m: float = 4_000.0,
    maximum_speed_m_per_day: float = 43_200.0,
    cell_filter=None,
) -> CandidatePlan:
    """Plan image pairs and retain deterministic exclusion diagnostics.

    ``cell_filter(source_image, cells) -> cells`` drops planning cells (for example land
    and open water) before primary selection; ``None`` keeps them all.
    """
    next_observation = config.primary_selection == "next_observation"
    if grid_spacing_m <= 0:
        raise ValueError("grid_spacing_m must be positive")
    if maximum_speed_m_per_day <= 0:
        raise ValueError("maximum_speed_m_per_day must be positive")
    images = catalogue.chronological()
    if config.require_orbit_metadata:
        missing = [
            image.image_id
            for image in images
            if image.platform is None or image.absolute_orbit is None
        ]
        if missing:
            sample = ", ".join(missing[:5])
            raise ValueError(
                "production pair planning requires platform and absolute orbit "
                f"metadata; missing for {len(missing)} images ({sample})"
            )
    minimum_seconds = config.candidate_minimum_elapsed_hours * 3_600.0
    maximum_seconds = config.candidate_maximum_elapsed_hours * 3_600.0
    footprints = {image.image_id: _footprint(image) for image in images}
    candidates: list[
        tuple[
            ImagePair,
            float | None,
            float | None,
            int,
            frozenset[tuple[int, int]] | None,
        ]
    ] = []
    exclusions = {
        "below_minimum_elapsed": 0,
        "above_maximum_elapsed": 0,
        "same_acquisition_pass": 0,
        "below_minimum_overlap": 0,
        "no_planning_cells": 0,
        "not_in_pair_allowlist": 0,
    }
    for source_index, source in enumerate(images):
        for target_index, target in enumerate(
            images[source_index + 1 :], start=source_index + 1
        ):
            elapsed = (target.time_utc - source.time_utc).total_seconds()
            if elapsed > maximum_seconds:
                exclusions["above_maximum_elapsed"] += len(images) - target_index
                break
            if elapsed < minimum_seconds:
                exclusions["below_minimum_elapsed"] += 1
                continue
            pair = ImagePair(source, target)
            if (
                config.exclude_same_acquisition_pass
                and source.same_acquisition_pass(target)
            ):
                exclusions["same_acquisition_pass"] += 1
                continue
            if config.candidate_pair_ids and pair.pair_id not in config.candidate_pair_ids:
                exclusions["not_in_pair_allowlist"] += 1
                continue
            source_footprint = footprints[source.image_id]
            target_footprint = footprints[target.image_id]
            if source_footprint is None or target_footprint is None:
                candidates.append(
                    (pair, None, None, target_index - source_index - 1, None)
                )
                continue
            overlap_area = _overlap_area(source_footprint, target_footprint)
            overlap = _overlap_fraction(
                source_footprint, target_footprint, overlap_area
            )
            if (
                overlap >= config.candidate_minimum_overlap_fraction
                and overlap_area >= config.candidate_minimum_overlap_area_m2
            ):
                cells = (
                    _imaged_cells(source_footprint, target_footprint, grid_spacing_m)
                    if next_observation else
                    _planning_cells(
                        source_footprint,
                        target_footprint,
                        pair.elapsed_seconds,
                        grid_spacing_m,
                        maximum_speed_m_per_day,
                    )
                )
                if cell_filter is not None:
                    cells = frozenset(cell_filter(source, cells))
                if not cells:
                    exclusions["no_planning_cells"] += 1
                    continue
                candidates.append(
                    (
                        pair,
                        overlap,
                        overlap_area,
                        target_index - source_index - 1,
                        cells,
                    )
                )
            else:
                exclusions["below_minimum_overlap"] += 1

    primary_ids: set[str] = set()
    primary_pairs_before_target_maximum = 0
    primary_pairs_excluded_by_target_maximum = 0
    targets_affected_by_primary_pair_maximum = 0
    by_target: dict[
        str,
        list[
            tuple[
                ImagePair,
                float | None,
                float | None,
                int,
                frozenset[tuple[int, int]] | None,
            ]
        ],
    ] = {}
    for candidate in candidates:
        by_target.setdefault(candidate[0].target.image_id, []).append(candidate)
    next_observation_counts: dict[str, int] = {}
    if config.candidate_pair_ids:
        primary_ids.update(pair.pair_id for pair, *_rest in candidates)
        primary_pairs_before_target_maximum = len(primary_ids)
    elif next_observation:
        next_observation_counts = _next_observation_selection(
            candidates, primary_ids,
            config.next_observation_minimum_cells,
            config.next_observation_backup_after_hours,
        )
    else:
        for target_candidates in by_target.values():
            selected, unrestricted_count = _coverage_primary_selection(
                target_candidates,
                config.primary_maximum_pairs_per_target,
            )
            primary_ids.update(selected)
            primary_pairs_before_target_maximum += unrestricted_count
            excluded = unrestricted_count - len(selected)
            primary_pairs_excluded_by_target_maximum += excluded
            targets_affected_by_primary_pair_maximum += int(excluded > 0)

    added_for_source_coverage = 0
    if config.source_coverage_completion and not config.candidate_pair_ids:
        added_for_source_coverage = _complete_source_coverage(
            candidates, primary_ids, footprints, grid_spacing_m,
            config.source_coverage_minimum_cells,
        )

    ordered = sorted(
        candidates,
        key=lambda item: (
            item[0].target.time_utc,
            item[0].target.image_id,
            item[0].source.time_utc,
            item[0].source.image_id,
        ),
    )
    pairs = tuple(
        PlannedPair(
            pair=pair,
            ordinal=ordinal,
            selection="primary" if pair.pair_id in primary_ids else "candidate",
            overlap_fraction=overlap,
            overlap_area_m2=overlap_area,
            skipped_images=skipped_images,
            planning_component_id=pair.source.component_id,
        )
        for ordinal, (
            pair,
            overlap,
            overlap_area,
            skipped_images,
            _cells,
        ) in enumerate(ordered)
    )
    recovery_pool = [
        item
        for item in pairs
        if item.selection == "candidate" and item.skipped_images > 0
    ]
    within_recovery_horizon = [
        item
        for item in recovery_pool
        if item.pair.elapsed_seconds
        <= config.maximum_recovery_elapsed_hours * 3_600.0
    ]
    selected_recovery = recovery_candidates(
        pairs,
        maximum_elapsed_hours=config.maximum_recovery_elapsed_hours,
    )
    counts = {
        **exclusions,
        "accepted_candidate_pairs": len(pairs),
        **next_observation_counts,
        "primary_pairs_selected_for_cell_coverage": len(primary_ids) - added_for_source_coverage,
        "primary_pairs_added_for_source_coverage": added_for_source_coverage,
        "primary_pairs_before_target_maximum": (
            primary_pairs_before_target_maximum
        ),
        "primary_pairs_excluded_by_target_maximum": (
            primary_pairs_excluded_by_target_maximum
        ),
        "targets_affected_by_primary_pair_maximum": (
            targets_affected_by_primary_pair_maximum
        ),
        "candidate_planning_cell_assignments": sum(
            len(cells) for *_prefix, cells in candidates if cells is not None
        ),
        "target_planning_cells": sum(
            len(
                set().union(
                    *(cells for *_prefix, cells in target if cells is not None)
                )
            )
            for target in by_target.values()
            if any(cells is not None for *_prefix, cells in target)
        ),
        "unselected_recovery_candidates": len(recovery_pool),
        "recovery_candidates_outside_elapsed_horizon": (
            len(recovery_pool) - len(within_recovery_horizon)
        ),
        "recovery_candidates_within_elapsed_horizon": len(
            within_recovery_horizon
        ),
        "recovery_candidates_selected_by_elapsed_horizon": len(
            selected_recovery
        ),
    }
    return CandidatePlan(pairs, counts)


def recovery_candidates(
    planned: tuple[PlannedPair, ...], maximum_elapsed_hours: float = 96.0,
) -> tuple[PlannedPair, ...]:
    """Select recent recovery candidates inside an elapsed-time horizon."""
    if maximum_elapsed_hours <= 0:
        raise ValueError("maximum_elapsed_hours must be positive")
    maximum_elapsed_seconds = maximum_elapsed_hours * 3_600.0
    by_target: dict[str, list[PlannedPair]] = {}
    for item in planned:
        if (
            item.selection == "candidate"
            and item.skipped_images > 0
            and item.pair.elapsed_seconds <= maximum_elapsed_seconds
        ):
            by_target.setdefault(item.pair.target.image_id, []).append(item)
    selected = []
    for target_id in sorted(by_target):
        ranked = sorted(
            by_target[target_id],
            key=lambda item: (
                -item.pair.source.time_utc.timestamp(),
                -(item.overlap_fraction or 0.0),
                item.pair.pair_id,
            ),
        )
        selected.extend(ranked)
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                item.pair.target.time_utc,
                item.pair.target.image_id,
                -item.pair.source.time_utc.timestamp(),
                item.pair.source.image_id,
            ),
        )
    )


def select_overlap_probe(
    planned: tuple[PlannedPair, ...],
    bins: tuple[tuple[float, float], ...],
    maximum_per_bin: int,
) -> dict[str, tuple[PlannedPair, ...]]:
    """Select a bounded deterministic sample of image pairs per overlap bin."""
    if maximum_per_bin < 1:
        raise ValueError("maximum_per_bin must be positive")
    selected: dict[str, tuple[PlannedPair, ...]] = {}
    for lower, upper in bins:
        if not 0 <= lower < upper <= 1:
            raise ValueError("overlap bins must satisfy 0 <= lower < upper <= 1")
        label = f"{lower:.3f}-{upper:.3f}"
        if label in selected:
            raise ValueError(f"duplicate overlap bin: {label}")
        candidates = sorted(
            (
                item
                for item in planned
                if item.overlap_fraction is not None
                and lower <= item.overlap_fraction
                and (
                    item.overlap_fraction < upper
                    or (upper == 1.0 and item.overlap_fraction <= upper)
                )
            ),
            key=lambda item: (
                item.pair.target.time_utc,
                item.pair.target.image_id,
                item.overlap_fraction,
                item.pair.pair_id,
            ),
        )
        if len(candidates) <= maximum_per_bin:
            sample = candidates
        else:
            indices = np.linspace(
                0, len(candidates) - 1, maximum_per_bin, dtype=int
            )
            sample = [candidates[index] for index in indices]
        selected[label] = tuple(sample)
    return selected


def _footprint(image: ImageRecord) -> BaseGeometry | None:
    if image.footprint is not None:
        return image.footprint
    try:
        return projected_footprint(image.path)
    except (OSError, ValueError):
        return None


def _overlap_area(source: BaseGeometry, target: BaseGeometry) -> float:
    if not source.intersects(target):
        return 0.0
    return float(source.intersection(target).area)


def _overlap_fraction(
    source: BaseGeometry, target: BaseGeometry, overlap_area: float | None = None
) -> float:
    denominator = min(float(source.area), float(target.area))
    if denominator <= 0:
        return 0.0
    numerator = _overlap_area(source, target) if overlap_area is None else overlap_area
    return numerator / denominator


def _planning_cells(
    source: BaseGeometry,
    target: BaseGeometry,
    elapsed_seconds: float,
    grid_spacing_m: float,
    maximum_speed_m_per_day: float,
) -> frozenset[tuple[int, int]]:
    maximum_displacement_m = (
        maximum_speed_m_per_day * elapsed_seconds / 86_400.0
    )
    domain = source.intersection(target.buffer(maximum_displacement_m)).buffer(0)
    _rows, _columns, coordinates = regular_grid(domain, grid_spacing_m)
    return frozenset(
        (int(round(x / grid_spacing_m)), int(round(y / grid_spacing_m)))
        for x, y in coordinates
    )


def _imaged_cells(source: BaseGeometry, target: BaseGeometry, grid_spacing_m: float):
    """Planning cells of the source that the target also images (no drift allowance)."""
    _rows, _columns, coordinates = regular_grid(source.intersection(target).buffer(0), grid_spacing_m)
    return frozenset(
        (int(round(x / grid_spacing_m)), int(round(y / grid_spacing_m)))
        for x, y in coordinates
    )


def _surface_cell_filter(gate, grid_spacing_m: float):
    """``(source_image, cells) -> cells`` keeping cells the gate does not exclude at the
    source time; one evaluation per source image and cell, cached."""
    cache: dict[str, dict] = {}

    def keep(source: ImageRecord, cells):
        if gate is None or not cells:
            return cells
        known = cache.setdefault(source.image_id, {})
        new = [cell for cell in cells if cell not in known]
        if new:
            xy = np.asarray(new, dtype=np.float64) * grid_spacing_m
            excluded = gate.excluded(xy, source.time_utc)
            known.update(zip(new, (not value for value in excluded)))
        if len(cache) > 64:
            cache.pop(next(iter(cache)))
        return {cell for cell in cells if known[cell]}

    return keep


def _next_observation_selection(
    candidates, primary_ids: set[str], minimum_cells: int, backup_after_hours: float | None
) -> dict[str, int]:
    """Keep, per source image, the pairs to the next observation of its cells.

    Candidates carry the cells both images image (after the cell filter). For each source,
    its candidate pairs are visited in target-time order; a pair is kept when it is the
    earliest (or, for a backup, the second) observation for at least ``minimum_cells`` of
    the source's cells. Cells first observed by a kept pair longer than
    ``backup_after_hours`` also wait for a backup. Returns diagnostic counts; ``primary_ids``
    is updated in place.
    """
    by_source: dict[str, list] = {}
    for item in candidates:
        if item[4] is not None:
            by_source.setdefault(item[0].source.image_id, []).append(item)
    kept = backups = assigned = imaged_total = unassigned = 0
    backup_seconds = None if backup_after_hours is None else backup_after_hours * 3_600.0
    for source_id in sorted(by_source):
        items = sorted(by_source[source_id], key=lambda item: (item[0].target.time_utc, item[0].pair_id))
        need = set().union(*(item[4] for item in items))
        waiting_backup: set = set()
        for item in items:
            pair, cells = item[0], item[4]
            first = cells & need
            second = cells & waiting_backup
            if len(first) + len(second) < minimum_cells:
                continue
            primary_ids.add(pair.pair_id)
            kept += 1
            backups += int(len(first) < minimum_cells)
            assigned += len(first) + len(second)
            imaged_total += len(cells)
            need -= cells
            waiting_backup -= cells
            if backup_seconds is not None and pair.elapsed_seconds > backup_seconds:
                waiting_backup |= first
        unassigned += len(need)
    return {
        "next_observation_pairs": kept,
        "next_observation_backup_pairs": backups,
        "next_observation_assigned_cells": assigned,
        "next_observation_kept_pair_cells": imaged_total,
        "next_observation_cells_without_pair": unassigned,
    }


def _complete_source_coverage(
    candidates, primary_ids: set[str], footprints, grid_spacing_m: float, minimum_cells: int
) -> int:
    """Add primary pairs so that each source image's ice imaged by a later candidate
    target has a selected pair forward.

    Coverage is judged on the cells both images actually image, not on the drift-buffered
    cells the per-target selection uses: a buffered cell may claim ice the target never
    imaged. Per source, repeatedly add the candidate covering most uncovered cells
    (ties: the shorter interval, then pair ID) until fewer than ``minimum_cells`` would be
    added. Returns the number of pairs added; ``primary_ids`` is updated in place.
    """
    by_source: dict[str, list] = {}
    for item in candidates:
        if item[4] is not None:
            by_source.setdefault(item[0].source.image_id, []).append(item[0])
    added = 0
    for source_id in sorted(by_source):
        pairs = by_source[source_id]
        imaged = {
            pair.pair_id: _imaged_cells(
                footprints[pair.source.image_id], footprints[pair.target.image_id], grid_spacing_m
            )
            for pair in pairs
        }
        covered: set = set()
        for pair in pairs:
            if pair.pair_id in primary_ids:
                covered |= imaged[pair.pair_id]
        need = set().union(*imaged.values()) - covered
        remaining = [pair for pair in pairs if pair.pair_id not in primary_ids]
        while need and remaining:
            best = max(
                remaining,
                key=lambda pair: (len(imaged[pair.pair_id] & need), -pair.elapsed_seconds, pair.pair_id),
            )
            if len(imaged[best.pair_id] & need) < minimum_cells:
                break
            primary_ids.add(best.pair_id)
            need -= imaged[best.pair_id]
            remaining.remove(best)
            added += 1
    return added


def _coverage_primary_selection(
    target_candidates, maximum_pairs: int | None = None
) -> tuple[set[str], int]:
    """Keep each pair that is the most recent option for at least one cell."""
    known = [item for item in target_candidates if item[4] is not None]
    if not known:
        most_recent = max(item[0].source.time_utc for item in target_candidates)
        ranked = sorted(
            (
                item
                for item in target_candidates
                if item[0].source.time_utc == most_recent
            ),
            key=lambda item: (
                -(item[2] or 0.0),
                item[0].pair_id,
            ),
        )
        unrestricted_count = len(ranked)
        selected = ranked if maximum_pairs is None else ranked[:maximum_pairs]
        return {item[0].pair_id for item in selected}, unrestricted_count
    latest_by_cell: dict[tuple[int, int], object] = {}
    for pair, _overlap, _area, _skipped, cells in known:
        assert cells is not None
        for cell in cells:
            previous = latest_by_cell.get(cell)
            if previous is None or pair.source.time_utc > previous:
                latest_by_cell[cell] = pair.source.time_utc
    contributions = [
        (
            item,
            frozenset(
                cell
                for cell in item[4]
                if latest_by_cell[cell] == item[0].source.time_utc
            ),
        )
        for item in known
    ]
    contributions = [item for item in contributions if item[1]]
    if maximum_pairs is None:
        return (
            {item[0][0].pair_id for item in contributions},
            len(contributions),
        )

    selected: set[str] = set()
    covered: set[tuple[int, int]] = set()
    remaining = contributions
    while remaining and len(selected) < maximum_pairs:
        ranked = sorted(
            remaining,
            key=lambda value: (
                -len(value[1] - covered),
                -len(value[1]),
                -value[0][0].source.time_utc.timestamp(),
                -(value[0][2] or 0.0),
                value[0][0].pair_id,
            ),
        )
        item, cells = ranked[0]
        selected.add(item[0].pair_id)
        covered.update(cells)
        remaining = [
            value
            for value in remaining
            if value[0][0].pair_id not in selected
        ]
    return selected, len(contributions)
