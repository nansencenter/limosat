"""Deterministic, outcome-blind selection of additional measured image pairs."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PairOption:
    pair_id: str
    elapsed_hours: float

    def __post_init__(self) -> None:
        if (
            not self.pair_id
            or not math.isfinite(self.elapsed_hours)
            or self.elapsed_hours <= 0
        ):
            raise ValueError("pair option needs an identity and positive elapsed hours")


def choose_pair_batch(
    options_by_trajectory: Mapping[str, Sequence[PairOption]],
    *,
    resolved: frozenset[str] = frozenset(),
    processed_pairs: frozenset[str] = frozenset(),
    maximum_pairs: int,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Choose a bounded round; retry only trajectories still unresolved.

    Each unresolved trajectory nominates its shortest unprocessed option. Pairs
    with more nominations come first. Once a pair is selected, all unresolved
    trajectories eligible for it are submitted so its field can be reused.
    Reinvoke after composing the round with updated ``resolved`` and
    ``processed_pairs``; no match outcomes enter the ranking.
    """
    if maximum_pairs < 1:
        raise ValueError("maximum_pairs must be positive")
    nominations: dict[str, set[str]] = defaultdict(set)
    eligible: dict[str, set[str]] = defaultdict(set)
    elapsed: dict[str, float] = {}
    for identity, options in options_by_trajectory.items():
        if not identity or identity in resolved:
            continue
        available = []
        seen = set()
        for option in options:
            if option.pair_id in seen:
                raise ValueError(f"duplicate pair option for {identity}")
            seen.add(option.pair_id)
            if option.pair_id in processed_pairs:
                continue
            prior = elapsed.setdefault(option.pair_id, option.elapsed_hours)
            if prior != option.elapsed_hours:
                raise ValueError(f"inconsistent elapsed time for {option.pair_id}")
            available.append(option)
            eligible[option.pair_id].add(identity)
        if available:
            first = min(
                available, key=lambda value: (value.elapsed_hours, value.pair_id)
            )
            nominations[first.pair_id].add(identity)
    ranked = sorted(
        nominations,
        key=lambda pair_id: (-len(nominations[pair_id]), elapsed[pair_id], pair_id),
    )[:maximum_pairs]
    return tuple((pair_id, tuple(sorted(eligible[pair_id]))) for pair_id in ranked)


def build_pair_request_plan(
    batch: Sequence[tuple[str, Sequence[str]]],
    *,
    run_id: str,
    config_sha256: str,
    primary_composition_manifest_sha256: str,
) -> dict:
    """Bind a queue batch to the frozen primary artifact for target preparation."""
    if not run_id or not config_sha256 or not primary_composition_manifest_sha256:
        raise ValueError("request plan needs a frozen run and primary manifest")
    return {
        "pair_request_schema_version": 1,
        "run_id": run_id,
        "config_sha256": config_sha256,
        "primary_composition_manifest_sha256": primary_composition_manifest_sha256,
        "pairs": [
            {"pair_id": pair_id, "trajectory_ids": list(identities)}
            for pair_id, identities in batch
        ],
    }


def loss_targeted_pairs(
    candidates: Iterable[tuple[str, float, Sequence[str]]],
) -> frozenset[str]:
    """Select the pairs that some loss nominates as its shortest option.

    Each candidate is ``(pair_id, elapsed_seconds, trajectory_ids)``. Every
    lost identity nominates its shortest eligible pair (ties by pair id). A
    selected pair keeps all identities eligible for it, so its field is reused
    for losses that nominated a longer pair. Outcome-blind and deterministic.
    """
    codes: dict[str, int] = {}
    ordered = []
    for pair_id, elapsed, identities in candidates:
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"pair option needs positive elapsed time: {pair_id}")
        ordered.append((
            float(elapsed), pair_id,
            [codes.setdefault(identity, len(codes)) for identity in identities],
        ))
    ordered.sort(key=lambda item: (item[0], item[1]))
    nominated = bytearray(len(codes))
    selected = set()
    for _elapsed, pair_id, identities in ordered:
        fresh = [code for code in identities if not nominated[code]]
        if fresh:
            selected.add(pair_id)
            for code in fresh:
                nominated[code] = 1
    return frozenset(selected)
