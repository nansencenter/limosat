"""Deterministic global candidate and primary image-pair planning."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from .catalog import ImageCatalogue, ImagePair, ImageRecord
from .config import RoutingConfig, RunConfig
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
    key = [records, asdict(config.routing), config.analysis_epsg]
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
        cell_filter = _surface_cell_filter(surface_gate(config))
    digest.update(json.dumps(key, sort_keys=True, default=str).encode("utf-8"))
    path = Path(config.output_directory) / "plan-cache" / f"{digest.hexdigest()}.pickle"
    if path.is_file():
        with path.open("rb") as stream:
            return pickle.load(stream)
    plan = build_candidate_plan(
        catalogue,
        config.routing,
        grid_spacing_m=config.routing.planning_grid_spacing_m,
        cell_filter=cell_filter,
        workers=_planning_workers(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    with temporary.open("wb") as stream:
        pickle.dump(plan, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    return plan


def _planning_workers() -> int:
    """Processes for planning: LIMOSAT_PLANNING_WORKERS, else the CPUs this process may use."""
    value = os.environ.get("LIMOSAT_PLANNING_WORKERS")
    if value:
        return max(1, int(value))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - macOS
        return max(1, os.cpu_count() or 1)


def plan_candidate_pairs(
    catalogue: ImageCatalogue,
    config: RoutingConfig,
    grid_spacing_m: float = 4_000.0,
) -> tuple[PlannedPair, ...]:
    """Plan overlap-qualified candidates and a deterministic primary subset."""
    return build_candidate_plan(catalogue, config, grid_spacing_m=grid_spacing_m).pairs


def build_candidate_plan(
    catalogue: ImageCatalogue,
    config: RoutingConfig,
    grid_spacing_m: float = 4_000.0,
    cell_filter=None,
    workers: int = 1,
) -> CandidatePlan:
    """Plan image pairs and retain deterministic exclusion diagnostics.

    Candidates are image pairs within the elapsed-time window whose footprints overlap
    enough. Primary pairs are chosen by next observation on the planning cells both images
    image. ``cell_filter(source_image, xy_m) -> keep`` drops planning cells (for example land
    and open water) first; ``None`` keeps them all.

    Planning runs one source image at a time: its cells are a small array on the planning
    grid, each candidate target is a boolean mask over them, and the selection needs nothing
    from other sources. Memory therefore stays at one source's worth whatever the period,
    and sources are planned in parallel with ``workers`` processes (fork).
    """
    if grid_spacing_m <= 0:
        raise ValueError("grid_spacing_m must be positive")
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
    global _PLANNING
    indices = range(len(images))
    parallel = workers > 1 and len(images) > 1
    if parallel:
        import multiprocessing

        context = multiprocessing.get_context("fork")
        with context.Pool(workers) as pool:          # footprints read the scenes' GCPs
            footprints = pool.map(_footprint, images, chunksize=16)
    else:
        footprints = [_footprint(image) for image in images]
    _PLANNING = {
        "images": images,
        "footprints": footprints,
        "config": config,
        "grid_spacing_m": grid_spacing_m,
        "cell_filter": cell_filter,
    }
    try:
        if parallel:
            with context.Pool(workers) as pool:
                results = pool.map(_plan_source, indices, chunksize=16)
        else:
            results = [_plan_source(index) for index in indices]
    finally:
        _PLANNING = {}

    exclusions: dict[str, int] = {}
    selection: dict[str, int] = {}
    candidates = []
    for source_candidates, source_exclusions, source_selection in results:
        candidates.extend(source_candidates)
        for name, value in source_exclusions.items():
            exclusions[name] = exclusions.get(name, 0) + value
        for name, value in source_selection.items():
            selection[name] = selection.get(name, 0) + value
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
            selection="primary" if primary else "candidate",
            overlap_fraction=overlap,
            overlap_area_m2=overlap_area,
            skipped_images=skipped_images,
            planning_component_id=pair.source.component_id,
        )
        for ordinal, (pair, overlap, overlap_area, skipped_images, primary) in enumerate(ordered)
    )
    counts = {**exclusions, "accepted_candidate_pairs": len(pairs), **selection}
    return CandidatePlan(pairs, counts)


_PLANNING: dict = {}
_EXCLUSIONS = (
    "below_minimum_elapsed", "above_maximum_elapsed", "same_acquisition_pass",
    "below_minimum_overlap", "no_planning_cells", "not_in_pair_allowlist",
)


def _plan_source(source_index: int):
    """Candidates from one source image and the next-observation selection among them.

    Returns ``([(pair, overlap, overlap_area, skipped_images, primary)], exclusions,
    selection counts)``.
    """
    images = _PLANNING["images"]
    footprints = _PLANNING["footprints"]
    config: RoutingConfig = _PLANNING["config"]
    spacing = _PLANNING["grid_spacing_m"]
    cell_filter = _PLANNING["cell_filter"]
    source = images[source_index]
    source_footprint = footprints[source_index]
    exclusions = dict.fromkeys(_EXCLUSIONS, 0)
    minimum_seconds = config.candidate_minimum_elapsed_hours * 3_600.0
    maximum_seconds = config.candidate_maximum_elapsed_hours * 3_600.0
    grid = None          # planning-grid points of the source and the cells the filter keeps
    found = []           # (pair, overlap, overlap_area, skipped, imaged mask or None)
    for target_index in range(source_index + 1, len(images)):
        target = images[target_index]
        elapsed = (target.time_utc - source.time_utc).total_seconds()
        if elapsed > maximum_seconds:
            exclusions["above_maximum_elapsed"] += len(images) - target_index
            break
        if elapsed < minimum_seconds:
            exclusions["below_minimum_elapsed"] += 1
            continue
        pair = ImagePair(source, target)
        if config.exclude_same_acquisition_pass and source.same_acquisition_pass(target):
            exclusions["same_acquisition_pass"] += 1
            continue
        if config.candidate_pair_ids and pair.pair_id not in config.candidate_pair_ids:
            exclusions["not_in_pair_allowlist"] += 1
            continue
        target_footprint = footprints[target_index]
        if source_footprint is None or target_footprint is None:
            found.append((pair, None, None, target_index - source_index - 1, None, None))
            continue
        overlap_area = _overlap_area(source_footprint, target_footprint)
        overlap = _overlap_fraction(source_footprint, target_footprint, overlap_area)
        if not (
            overlap >= config.candidate_minimum_overlap_fraction
            and overlap_area >= config.candidate_minimum_overlap_area_m2
        ):
            exclusions["below_minimum_overlap"] += 1
            continue
        if grid is None:
            grid = _source_grid(source, source_footprint, spacing, cell_filter)
        imaged = _imaged_mask(source_footprint, target_footprint, grid)
        if not imaged.any():
            exclusions["no_planning_cells"] += 1
            continue
        edge = None
        if config.next_observation_edge_drift_m_per_day is not None:
            margin = config.next_observation_edge_drift_m_per_day * elapsed / 86_400.0
            edge = imaged & ~_imaged_mask(source_footprint, target_footprint.buffer(-margin), grid)
        found.append((pair, overlap, overlap_area, target_index - source_index - 1, imaged, edge))

    if config.candidate_pair_ids:
        primary = {item[0].pair_id for item in found}
        selection = {}
    else:
        primary, selection = _next_observation_selection(
            found, config.next_observation_minimum_cells, config.next_observation_backup_after_hours,
        )
    return (
        [(pair, overlap, area, skipped, pair.pair_id in primary) for pair, overlap, area, skipped, _mask, _edge in found],
        exclusions,
        selection,
    )


def _source_grid(source: ImageRecord, footprint: BaseGeometry, spacing: float, cell_filter):
    """Planning-grid points (multiples of ``spacing``) over the source footprint's bounds,
    with the mask of points the cell filter keeps."""
    minx, miny, maxx, maxy = footprint.bounds
    xs = np.arange(math.ceil(minx / spacing) * spacing, maxx, spacing)
    ys = np.arange(math.ceil(miny / spacing) * spacing, maxy, spacing)
    gx, gy = (values.ravel() for values in np.meshgrid(xs, ys))
    keep = np.ones(len(gx), dtype=bool)
    if cell_filter is not None and len(gx):
        inside = shapely.intersects_xy(footprint, gx, gy)
        if inside.any():
            keep[inside] = np.asarray(cell_filter(source, np.column_stack((gx[inside], gy[inside]))), dtype=bool)
    return gx, gy, keep


def _imaged_mask(source: BaseGeometry, target: BaseGeometry, grid) -> np.ndarray:
    """Planning cells of the source that the target also images (no drift allowance), as a
    mask over the source grid: the grid points inside the intersection of both footprints."""
    gx, gy, keep = grid
    domain = source.intersection(target).buffer(0)
    if domain.is_empty:
        return np.zeros(len(gx), dtype=bool)
    return shapely.intersects_xy(domain, gx, gy) & keep


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


def _surface_cell_filter(gate):
    """``(source_image, xy_m) -> keep``: cells the surface gate does not exclude at the
    source time (land, point open water)."""
    def keep(source: ImageRecord, xy):
        if gate is None:
            return np.ones(len(xy), dtype=bool)
        return ~np.asarray(gate.excluded(xy, source.time_utc), dtype=bool)

    return keep


def _next_observation_selection(found, minimum_cells: int, backup_after_hours: float | None):
    """The pairs from one source image to the next observation of its cells.

    ``found`` holds the source's candidates with their imaged-cell masks over a common grid.
    Candidates are visited in target-time order; a pair is kept when it is the earliest (or,
    for a backup, the second) observation for at least ``minimum_cells`` of the source's
    cells. Cells first observed by a kept pair longer than ``backup_after_hours`` also wait
    for a backup. Returns the kept pair IDs and diagnostic counts.
    """
    items = sorted((item for item in found if item[4] is not None), key=lambda item: (item[0].target.time_utc, item[0].pair_id))
    kept_ids: set[str] = set()
    counts = {"next_observation_pairs": 0, "next_observation_backup_pairs": 0, "next_observation_assigned_cells": 0,
              "next_observation_kept_pair_cells": 0, "next_observation_cells_without_pair": 0}
    if not items:
        return kept_ids, counts
    backup_seconds = None if backup_after_hours is None else backup_after_hours * 3_600.0
    need = np.logical_or.reduce([item[4] for item in items])
    waiting = np.zeros_like(need)
    for pair, _overlap, _area, _skipped, cells, edge in items:
        first = cells & need
        second = cells & waiting
        n_first, n_second = int(first.sum()), int(second.sum())
        if n_first + n_second < minimum_cells:
            continue
        kept_ids.add(pair.pair_id)
        counts["next_observation_pairs"] += 1
        counts["next_observation_backup_pairs"] += int(n_first < minimum_cells)
        counts["next_observation_assigned_cells"] += n_first + n_second
        counts["next_observation_kept_pair_cells"] += int(cells.sum())
        need &= ~cells
        waiting &= ~cells
        if backup_seconds is not None and pair.elapsed_seconds > backup_seconds:
            waiting |= first
        elif edge is not None:
            waiting |= first & edge          # may drift out of the target before it is acquired
    counts["next_observation_cells_without_pair"] = int(need.sum())
    return kept_ids, counts
