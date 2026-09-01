"""Deterministic global candidate and primary image-pair planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from shapely.geometry.base import BaseGeometry

from .catalog import ImageCatalogue, ImagePair, ImageRecord
from .config import RoutingConfig
from .imagery import projected_footprint


@dataclass(frozen=True)
class PlannedPair:
    pair: ImagePair
    ordinal: int
    selection: Literal["candidate", "primary"]
    overlap_fraction: float | None
    planning_component_id: str


def plan_candidate_pairs(
    catalogue: ImageCatalogue, config: RoutingConfig
) -> tuple[PlannedPair, ...]:
    """Plan overlap-qualified candidates and a deterministic primary subset."""
    images = catalogue.chronological()
    minimum_seconds = config.candidate_minimum_elapsed_hours * 3_600.0
    maximum_seconds = config.candidate_maximum_elapsed_hours * 3_600.0
    footprints = {image.image_id: _footprint(image) for image in images}
    candidates: list[tuple[ImagePair, float | None]] = []
    for source_index, source in enumerate(images):
        for target in images[source_index + 1 :]:
            elapsed = (target.time_utc - source.time_utc).total_seconds()
            if elapsed > maximum_seconds:
                break
            if elapsed < minimum_seconds:
                continue
            source_footprint = footprints[source.image_id]
            target_footprint = footprints[target.image_id]
            if source_footprint is None or target_footprint is None:
                candidates.append((ImagePair(source, target), None))
                continue
            overlap = _overlap_fraction(source_footprint, target_footprint)
            if overlap >= config.candidate_minimum_overlap_fraction:
                candidates.append((ImagePair(source, target), overlap))

    primary_ids: set[str] = set()
    by_target: dict[str, list[tuple[ImagePair, float | None]]] = {}
    for candidate in candidates:
        by_target.setdefault(candidate[0].target.image_id, []).append(candidate)
    for target_candidates in by_target.values():
        most_recent = max(pair.source.time_utc for pair, _overlap in target_candidates)
        primary_ids.update(
            pair.pair_id
            for pair, _overlap in target_candidates
            if pair.source.time_utc == most_recent
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
    return tuple(
        PlannedPair(
            pair=pair,
            ordinal=ordinal,
            selection="primary" if pair.pair_id in primary_ids else "candidate",
            overlap_fraction=overlap,
            planning_component_id=pair.source.component_id,
        )
        for ordinal, (pair, overlap) in enumerate(ordered)
    )


def recovery_candidates(
    planned: tuple[PlannedPair, ...], maximum_per_target: int
) -> tuple[PlannedPair, ...]:
    """Select a bounded, recent-first recovery set from unselected candidates."""
    if maximum_per_target <= 0:
        return ()
    by_target: dict[str, list[PlannedPair]] = {}
    for item in planned:
        if item.selection == "candidate":
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
        selected.extend(ranked[:maximum_per_target])
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


def _footprint(image: ImageRecord) -> BaseGeometry | None:
    if image.footprint is not None:
        return image.footprint
    try:
        return projected_footprint(image.path)
    except (OSError, ValueError):
        return None


def _overlap_fraction(source: BaseGeometry, target: BaseGeometry) -> float:
    if not source.intersects(target):
        return 0.0
    denominator = min(float(source.area), float(target.area))
    if denominator <= 0:
        return 0.0
    return float(source.intersection(target).area) / denominator
