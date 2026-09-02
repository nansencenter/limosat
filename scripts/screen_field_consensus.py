#!/usr/bin/env python3
"""CPU-only consensus screen using immutable completed EfficientLoFTR matches.

This is an analysis path, not a pair recomputation path.  It reads raw matches
and the existing 4 km query grids from a completed production run, then compares
field-consensus policies without loading a matcher or checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from limosat.field import reject_folds
from limosat.models import DisplacementField, MotionMatches
from limosat.store import file_sha256


DEFAULT_SOURCE = Path(
    "/Volumes/KINGSTON/arktalas-nrt/method-neutral-benchmark/"
    "efficientloftr-production/april2020-week01-primary-v2"
)


@dataclass(frozen=True)
class Tier:
    neighbour_count: int
    minimum_agreeing_matches: int
    maximum_neighbour_distance_m: float
    agreement_distance_m: float


@dataclass(frozen=True)
class PairSource:
    pair_id: str
    pair_directory: Path
    source_image_id: int
    target_image_id: int
    elapsed_hours: float
    overlap_fraction: float
    recorded_coverage: float
    recorded_field_sha256: str
    source_scene_id: str
    target_scene_id: str
    source_platform: str
    target_platform: str
    source_orbit_number: int | None
    target_orbit_number: int | None
    recorded_fold_rejected_nodes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=36)
    parser.add_argument("--query-chunk-size", type=int, default=5_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"screen output already exists: {output}")
    output.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()

    inventory = discover_pairs(source)
    selected = stratified_pairs(inventory, args.pairs)
    population_strata = _stratum_counts(inventory)
    sample_strata = _stratum_counts(selected)
    policies = policy_definitions()
    per_pair: list[dict[str, object]] = []
    accumulators = {name: _Accumulator() for name in policies}
    source_records = []

    for pair_index, pair in enumerate(selected, start=1):
        stratum = stratum_key(pair)
        sampling_weight = population_strata[stratum] / sample_strata[stratum]
        frame = pd.read_csv(pair.pair_directory / "field_4km.csv")
        query = frame[["source_x", "source_y"]].to_numpy(np.float64)
        with np.load(pair.pair_directory / "matches.npz") as values:
            matches = MotionMatches(
                values["source_xy_m"],
                values["target_xy_m"],
                values["score"],
                values["source_tile_id"],
                values["target_tile_id"],
            )
        tree = cKDTree(matches.source_xy_m)
        recorded_available = frame["available"].astype(bool).to_numpy()
        recorded_displacement = frame[
            ["proposal_dx_m", "proposal_dy_m"]
        ].to_numpy(np.float64)
        recorded_displacement[~recorded_available] = np.nan
        recorded_estimates = {
            "displacement_m": recorded_displacement,
            "available": recorded_available,
            "selected_matches": frame["selected_vectors"].to_numpy(np.int32),
            "candidate_matches": frame["candidate_count"].to_numpy(np.int32),
            "support_radius_m": frame["support_radius_m"].to_numpy(np.float64),
            "maximum_residual_m": frame[
                "maximum_vector_residual_m"
            ].to_numpy(np.float64),
        }
        recorded_baseline = _field(pair, frame, query, recorded_estimates)
        fields: dict[str, DisplacementField] = {}
        tiers_used: dict[str, np.ndarray] = {}
        fold_counts: dict[str, int] = {}
        recomputed_baseline = None
        for name, tier_factory in policies.items():
            tiers = tier_factory(pair.elapsed_hours * 3_600.0)
            estimates, tier_used = estimate_policy(
                matches,
                query,
                tiers,
                tree=tree,
                query_chunk_size=args.query_chunk_size,
            )
            raw_field = _field(pair, frame, query, estimates)
            field, folded = reject_folds(raw_field, 6_400.0)
            if name == "baseline":
                recomputed_baseline = field
                fields[name] = recorded_baseline
                tiers_used[name] = np.where(
                    recorded_baseline.available, 0, -1
                ).astype(np.int16)
                fold_counts[name] = pair.recorded_fold_rejected_nodes
            else:
                fields[name] = field
                tiers_used[name] = tier_used
                fold_counts[name] = len(folded)

        baseline = fields["baseline"]
        assert recomputed_baseline is not None
        baseline_common = recomputed_baseline.available & recorded_available
        baseline_difference = np.linalg.norm(
            recomputed_baseline.displacement_m[baseline_common]
            - recorded_displacement[baseline_common],
            axis=1,
        )
        baseline_validation = {
            "recorded_available_nodes": int(recorded_available.sum()),
            "recomputed_available_nodes": int(recomputed_baseline.available.sum()),
            "availability_mismatch_nodes": int(
                np.count_nonzero(
                    recomputed_baseline.available != recorded_available
                )
            ),
            "common_displacement_p99_difference_m": _quantile(
                baseline_difference, 0.99
            ),
            "common_displacement_max_difference_m": _quantile(
                baseline_difference, 1.0
            ),
        }
        baseline_cells = regular_cell_deformation(baseline)

        for name, field in fields.items():
            cells = regular_cell_deformation(field)
            common_keys = sorted(set(baseline_cells) & set(cells))
            base_cell_values = np.asarray(
                [baseline_cells[key] for key in common_keys], dtype=np.float64
            )
            cell_values = np.asarray(
                [cells[key] for key in common_keys], dtype=np.float64
            )
            common_nodes = field.available & baseline.available
            displacement_difference = np.linalg.norm(
                field.displacement_m[common_nodes]
                - baseline.displacement_m[common_nodes],
                axis=1,
            )
            roughness = adjacent_displacement_gradient(field)
            used = tiers_used[name]
            tier_counts = {
                str(index + 1): int(np.count_nonzero((used == index) & field.available))
                for index in range(len(policies[name](pair.elapsed_hours * 3_600.0)))
            }
            row = {
                "pair_id": pair.pair_id,
                "policy": name,
                "source_image_id": pair.source_image_id,
                "target_image_id": pair.target_image_id,
                "elapsed_hours": pair.elapsed_hours,
                "overlap_fraction": pair.overlap_fraction,
                "grid_nodes": len(field),
                "available_nodes": int(field.available.sum()),
                "coverage_fraction": float(field.available.mean()),
                "fold_rejected_nodes": fold_counts[name],
                "new_available_vs_baseline": int(
                    np.count_nonzero(field.available & ~baseline.available)
                ),
                "lost_available_vs_baseline": int(
                    np.count_nonzero(~field.available & baseline.available)
                ),
                "support_radius_p90_m": _quantile(
                    field.support_radius_m[field.available], 0.90
                ),
                "maximum_residual_p90_m": _quantile(
                    field.maximum_residual_m[field.available], 0.90
                ),
                "common_node_displacement_p95_difference_m": _quantile(
                    displacement_difference, 0.95
                ),
                "regular_cells": len(cells),
                "common_baseline_regular_cells": len(common_keys),
                "common_cell_baseline_total_deformation_p99_s_1": _quantile(
                    base_cell_values, 0.99
                ),
                "common_cell_policy_total_deformation_p99_s_1": _quantile(
                    cell_values, 0.99
                ),
                "adjacent_displacement_gradient_p99": _quantile(
                    roughness, 0.99
                ),
                "tier_counts": json.dumps(tier_counts, sort_keys=True),
            }
            per_pair.append(row)
            accumulators[name].add(
                field=field,
                baseline=baseline,
                folds=fold_counts[name],
                displacement_difference=displacement_difference,
                roughness=roughness,
                baseline_cells=base_cell_values,
                policy_cells=cell_values,
                tier_used=used,
                tier_count=len(policies[name](pair.elapsed_hours * 3_600.0)),
                sampling_weight=sampling_weight,
            )

        matches_path = pair.pair_directory / "matches.npz"
        source_records.append(
            {
                **asdict(pair),
                "pair_directory": str(pair.pair_directory),
                "matches_size_bytes": matches_path.stat().st_size,
                "matches_sha256": file_sha256(matches_path),
                "baseline_validation": baseline_validation,
            }
        )
        print(
            f"screened {pair_index}/{len(selected)} {pair.pair_id} "
            f"({len(matches):,} matches; {len(query):,} nodes)",
            flush=True,
        )

    aggregates = [
        accumulators[name].summary(name, len(selected)) for name in policies
    ]
    baseline_coverage = next(
        row["coverage_fraction"] for row in aggregates if row["policy"] == "baseline"
    )
    baseline_population_coverage = next(
        row["population_estimated_coverage_fraction"]
        for row in aggregates
        if row["policy"] == "baseline"
    )
    for row in aggregates:
        row["coverage_gain_percentage_points"] = 100.0 * (
            row["coverage_fraction"] - baseline_coverage
        )
        row["population_estimated_coverage_gain_percentage_points"] = 100.0 * (
            row["population_estimated_coverage_fraction"]
            - baseline_population_coverage
        )

    per_pair_path = output / "per-pair-consensus-screen.csv"
    aggregate_path = output / "aggregate-consensus-screen.csv"
    _write_csv(per_pair_path, per_pair)
    _write_csv(aggregate_path, aggregates)
    figure_path = output / "consensus-coverage-deformation-screen.png"
    plot_screen(aggregates, figure_path)

    source_digest = hashlib.sha256()
    for record in source_records:
        source_digest.update(record["pair_id"].encode())
        source_digest.update(record["recorded_field_sha256"].encode())
        source_digest.update(record["matches_sha256"].encode())
    report_path = output / "consensus-screen-report-v1.json"
    report = {
        "schema_version": "limosat_consensus_screen_v1",
        "label": "FIELD CONSENSUS SCREEN — no EfficientLoFTR inference",
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - clock,
        "source": {
            "root": str(source),
            "state_sqlite_sha256": file_sha256(source / "control" / "state.sqlite"),
            "week_plan_sha256": file_sha256(source / "control" / "week_plan.json"),
            "completed_pair_count": len(inventory),
            "sampled_pair_count": len(selected),
            "sampled_raw_match_and_field_set_sha256": source_digest.hexdigest(),
            "sampled_pairs": source_records,
        },
        "sampling": {
            "method": (
                "deterministic round-robin from elapsed-time, overlap, and "
                "recorded-coverage strata"
            ),
            "requested_pairs": args.pairs,
            "selected_pairs": len(selected),
            "population_stratum_counts": {
                str(key): value for key, value in sorted(population_strata.items())
            },
            "sample_stratum_counts": {
                str(key): value for key, value in sorted(sample_strata.items())
            },
        },
        "completed_inventory_diagnostics": inventory_diagnostics(inventory),
        "coordinates": {
            "crs": "EPSG:3413",
            "dtype": "float64",
            "distance_unit": "metre",
            "time_unit": "second",
        },
        "physics_screen": {
            "time_scaled_agreement_formula": (
                "clip(400 m + 0.7e-6 s^-1 * elapsed_seconds * "
                "neighbour_radius_m, 500 m, 1500 m)"
            ),
            "status": (
                "diagnostic hypothesis only; not a calibrated ice-physics "
                "acceptance rule"
            ),
            "deformation_check": (
                "total deformation on identical fully observed regular 4 km "
                "cells; no interpolation or smoothing is applied"
            ),
        },
        "policies": {
            name: [asdict(tier) for tier in factory(24.0 * 3_600.0)]
            for name, factory in policies.items()
        },
        "aggregate": aggregates,
        "outputs": {},
    }
    for name, path in {
        "per_pair_csv": per_pair_path,
        "aggregate_csv": aggregate_path,
        "figure": figure_path,
    }.items():
        report["outputs"][name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"report": str(report_path), "figure": str(figure_path)}))
    return 0


def policy_definitions():
    def fixed(k, minimum, radius, agreement):
        return lambda _elapsed: (Tier(k, minimum, radius, agreement),)

    def adaptive_conservative(_elapsed):
        return (
            Tier(8, 6, 3_000.0, 750.0),
            Tier(12, 8, 6_000.0, 1_000.0),
            Tier(16, 10, 8_000.0, 1_250.0),
        )

    def adaptive_time_scaled(elapsed):
        return tuple(
            Tier(k, minimum, radius, physics_agreement_distance(elapsed, radius))
            for k, minimum, radius in (
                (8, 6, 3_000.0),
                (12, 8, 6_000.0),
                (16, 10, 8_000.0),
            )
        )

    def adaptive_relaxed(elapsed):
        return tuple(
            Tier(k, minimum, radius, physics_agreement_distance(elapsed, radius))
            for k, minimum, radius in (
                (8, 5, 3_000.0),
                (12, 7, 6_000.0),
                (16, 8, 8_000.0),
            )
        )

    return {
        "baseline": fixed(12, 8, 6_000.0, 1_000.0),
        "minimum-7": fixed(12, 7, 6_000.0, 1_000.0),
        "minimum-6": fixed(12, 6, 6_000.0, 1_000.0),
        "agreement-750m": fixed(12, 8, 6_000.0, 750.0),
        "agreement-1500m": fixed(12, 8, 6_000.0, 1_500.0),
        "adaptive-conservative": adaptive_conservative,
        "adaptive-time-scaled": adaptive_time_scaled,
        "adaptive-relaxed": adaptive_relaxed,
    }


def physics_agreement_distance(elapsed_seconds: float, radius_m: float) -> float:
    value = 400.0 + 0.7e-6 * elapsed_seconds * radius_m
    return float(np.clip(value, 500.0, 1_500.0))


def estimate_policy(
    matches: MotionMatches,
    query_xy_m: np.ndarray,
    tiers: tuple[Tier, ...],
    *,
    tree: cKDTree | None = None,
    query_chunk_size: int = 5_000,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Apply first-supported tier selection without filling missing motion."""
    query = np.asarray(query_xy_m, dtype=np.float64)
    count = len(query)
    output = {
        "displacement_m": np.full((count, 2), np.nan, dtype=np.float64),
        "available": np.zeros(count, dtype=bool),
        "selected_matches": np.zeros(count, dtype=np.int32),
        "candidate_matches": np.zeros(count, dtype=np.int32),
        "support_radius_m": np.full(count, np.nan, dtype=np.float64),
        "maximum_residual_m": np.full(count, np.nan, dtype=np.float64),
    }
    tier_used = np.full(count, -1, dtype=np.int16)
    if not len(matches) or not count:
        return output, tier_used
    tree = cKDTree(matches.source_xy_m) if tree is None else tree
    unresolved = np.arange(count)
    for tier_index, tier in enumerate(tiers):
        tier_result = _estimate_tier(
            matches,
            query[unresolved],
            tier,
            tree,
            query_chunk_size,
        )
        if tier_index == len(tiers) - 1:
            for key in output:
                output[key][unresolved] = tier_result[key]
        accepted_local = np.flatnonzero(tier_result["available"])
        accepted = unresolved[accepted_local]
        for key in output:
            output[key][accepted] = tier_result[key][accepted_local]
        output["available"][accepted] = True
        tier_used[accepted] = tier_index
        unresolved = unresolved[~tier_result["available"]]
        if not len(unresolved):
            break
    return output, tier_used


def _estimate_tier(matches, query, tier, tree, chunk_size):
    count = len(query)
    displacement = np.full((count, 2), np.nan, dtype=np.float64)
    available = np.zeros(count, dtype=bool)
    selected = np.zeros(count, dtype=np.int32)
    candidates = np.zeros(count, dtype=np.int32)
    radius = np.full(count, np.nan, dtype=np.float64)
    residual = np.full(count, np.nan, dtype=np.float64)
    vectors = matches.displacement_m
    scores = np.maximum(matches.score, 1.0e-12)
    k = min(tier.neighbour_count, len(matches))
    for first in range(0, count, chunk_size):
        last = min(first + chunk_size, count)
        distances, indices = tree.query(
            query[first:last],
            k=k,
            distance_upper_bound=tier.maximum_neighbour_distance_m,
            workers=-1,
        )
        distances = np.asarray(distances)
        indices = np.asarray(indices)
        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        valid = np.isfinite(distances) & (indices < len(matches))
        safe_indices = indices.copy()
        safe_indices[~valid] = 0
        local_vectors = vectors[safe_indices]
        weights = scores[safe_indices] * valid
        local_candidates = valid.sum(axis=1).astype(np.int32)
        candidates[first:last] = local_candidates
        any_candidate = local_candidates > 0
        if np.any(any_candidate):
            radius[first:last][any_candidate] = np.max(
                np.where(valid[any_candidate], distances[any_candidate], -np.inf),
                axis=1,
            )
        separation = np.linalg.norm(
            local_vectors[:, :, None, :] - local_vectors[:, None, :, :], axis=3
        )
        support = np.einsum(
            "nij,nj->ni",
            separation <= tier.agreement_distance_m,
            weights,
        )
        anchors = np.argmax(support, axis=1)
        agreeing = (
            separation[np.arange(last - first), anchors]
            <= tier.agreement_distance_m
        ) & valid
        local_selected = agreeing.sum(axis=1).astype(np.int32)
        selected[first:last] = local_selected
        accepted = local_selected >= tier.minimum_agreeing_matches
        if not np.any(accepted):
            continue
        accepted_vectors = local_vectors[accepted]
        accepted_weights = weights[accepted] * agreeing[accepted]
        estimates = _batched_weighted_geometric_median(
            accepted_vectors, accepted_weights
        )
        accepted_global = np.flatnonzero(accepted) + first
        displacement[accepted_global] = estimates
        residual[accepted_global] = np.max(
            np.where(
                accepted_weights > 0,
                np.linalg.norm(accepted_vectors - estimates[:, None, :], axis=2),
                -np.inf,
            ),
            axis=1,
        )
        available[accepted_global] = True
    return {
        "displacement_m": displacement,
        "available": available,
        "selected_matches": selected,
        "candidate_matches": candidates,
        "support_radius_m": radius,
        "maximum_residual_m": residual,
    }


def _batched_weighted_geometric_median(vectors, weights):
    estimates = np.sum(vectors * weights[:, :, None], axis=1) / np.sum(
        weights, axis=1
    )[:, None]
    active = np.ones(len(estimates), dtype=bool)
    for _ in range(100):
        active_indices = np.flatnonzero(active)
        if not len(active_indices):
            break
        local_vectors = vectors[active_indices]
        local_weights = weights[active_indices]
        local_estimates = estimates[active_indices]
        distance = np.linalg.norm(
            local_vectors - local_estimates[:, None, :], axis=2
        )
        coincident = (distance <= 1.0e-3) & (local_weights > 0)
        has_coincident = coincident.any(axis=1)
        if np.any(has_coincident):
            selected_indices = active_indices[has_coincident]
            selected_weights = local_weights[has_coincident] * coincident[
                has_coincident
            ]
            estimates[selected_indices] = np.sum(
                local_vectors[has_coincident] * selected_weights[:, :, None],
                axis=1,
            ) / np.sum(selected_weights, axis=1)[:, None]
            active[selected_indices] = False
        updating = ~has_coincident
        if not np.any(updating):
            continue
        selected_indices = active_indices[updating]
        effective = local_weights[updating] / np.maximum(
            distance[updating], 1.0e-3
        )
        updated = np.sum(
            local_vectors[updating] * effective[:, :, None], axis=1
        ) / np.sum(effective, axis=1)[:, None]
        converged = (
            np.linalg.norm(updated - estimates[selected_indices], axis=1)
            <= 1.0e-3
        )
        estimates[selected_indices] = updated
        active[selected_indices[converged]] = False
    return estimates


def discover_pairs(source: Path) -> list[PairSource]:
    state = source / "control" / "state.sqlite"
    with sqlite3.connect(f"file:{state}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            """
            SELECT e.edge_id,e.source_scene_id,e.target_scene_id,
                   e.elapsed_seconds,e.overlap_fraction,
                   s.image_id,t.image_id,s.orbit_number,t.orbit_number
            FROM edges e
            JOIN scenes s ON s.scene_id=e.source_scene_id
            JOIN scenes t ON t.scene_id=e.target_scene_id
            JOIN edge_attempts a ON a.edge_id=e.edge_id
            WHERE e.role='primary' AND a.status='completed'
            ORDER BY e.edge_id
            """
        ).fetchall()
    edge_by_images = {(row[5], row[6]): row for row in rows}
    result = []
    for summary_path in sorted(source.glob("shards/*/raw/learned_output/pair_*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete":
            continue
        key = (int(summary["source_image_id"]), int(summary["target_image_id"]))
        edge = edge_by_images[key]
        result.append(
            PairSource(
                pair_id=edge[0],
                pair_directory=summary_path.parent,
                source_image_id=key[0],
                target_image_id=key[1],
                elapsed_hours=float(summary["elapsed_hours"]),
                overlap_fraction=float(edge[4]),
                recorded_coverage=float(summary["coverage_after_fold_rejection"]),
                recorded_field_sha256=str(summary["field_sha256"]),
                source_scene_id=edge[1],
                target_scene_id=edge[2],
                source_platform=edge[1].split("_", 1)[0],
                target_platform=edge[2].split("_", 1)[0],
                source_orbit_number=edge[7],
                target_orbit_number=edge[8],
                recorded_fold_rejected_nodes=int(summary["fold_rejected_nodes"]),
            )
        )
    return result


def stratified_pairs(pairs: list[PairSource], count: int) -> list[PairSource]:
    if count <= 0:
        raise ValueError("pair count must be positive")
    if count >= len(pairs):
        return pairs

    groups: dict[tuple[int, int, int], list[PairSource]] = {}
    for pair in pairs:
        key = stratum_key(pair)
        groups.setdefault(key, []).append(pair)
    for values in groups.values():
        values.sort(key=lambda pair: hashlib.sha256(pair.pair_id.encode()).hexdigest())
    selected = []
    round_index = 0
    while len(selected) < count:
        added = False
        for key in sorted(groups):
            if round_index < len(groups[key]):
                selected.append(groups[key][round_index])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        round_index += 1
    return sorted(selected, key=lambda pair: pair.pair_id)


def stratum_key(pair: PairSource) -> tuple[int, int, int]:
    return (
        int(np.searchsorted((24.0, 48.0), pair.elapsed_hours, side="right")),
        int(np.searchsorted((0.4, 0.7), pair.overlap_fraction, side="right")),
        int(np.searchsorted((0.1, 0.5), pair.recorded_coverage, side="right")),
    )


def _stratum_counts(pairs):
    counts = {}
    for pair in pairs:
        key = stratum_key(pair)
        counts[key] = counts.get(key, 0) + 1
    return counts


def inventory_diagnostics(pairs: list[PairSource]) -> dict[str, object]:
    def binned(name, boundaries, accessor):
        groups = []
        lower = boundaries[0]
        for upper in boundaries[1:]:
            values = [pair for pair in pairs if lower <= accessor(pair) < upper]
            coverage = np.asarray(
                [pair.recorded_coverage for pair in values], dtype=np.float64
            )
            groups.append(
                {
                    "lower_inclusive": lower,
                    "upper_exclusive": upper,
                    "pair_count": len(values),
                    "nonzero_pair_fraction": (
                        float(np.mean(coverage > 0)) if len(coverage) else None
                    ),
                    "coverage_mean": (
                        float(np.mean(coverage)) if len(coverage) else None
                    ),
                    "coverage_median": (
                        float(np.median(coverage)) if len(coverage) else None
                    ),
                    "coverage_p10": (
                        float(np.quantile(coverage, 0.10)) if len(coverage) else None
                    ),
                    "pairs_at_least_10_percent_coverage": int(
                        np.count_nonzero(coverage >= 0.10)
                    ),
                    "pairs_at_least_50_percent_coverage": int(
                        np.count_nonzero(coverage >= 0.50)
                    ),
                }
            )
            lower = upper
        return {"quantity": name, "bins": groups}

    same_absolute_orbit = sum(
        pair.source_orbit_number is not None
        and pair.source_orbit_number == pair.target_orbit_number
        for pair in pairs
    )
    same_pass = sum(
        pair.source_platform == pair.target_platform
        and pair.source_orbit_number is not None
        and pair.source_orbit_number == pair.target_orbit_number
        for pair in pairs
    )
    return {
        "completed_primary_pairs": len(pairs),
        "minimum_overlap_fraction": min(pair.overlap_fraction for pair in pairs),
        "minimum_elapsed_hours": min(pair.elapsed_hours for pair in pairs),
        "missing_orbit_pairs": sum(
            pair.source_orbit_number is None or pair.target_orbit_number is None
            for pair in pairs
        ),
        "same_absolute_orbit_pairs": same_absolute_orbit,
        "same_platform_absolute_orbit_pairs": same_pass,
        "orbit_exclusion_interpretation": (
            "same acquisition pass is platform plus absolute orbit; same "
            "relative orbit on a later repeat remains eligible"
        ),
        "overlap": binned(
            "overlap_fraction",
            (0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01),
            lambda pair: pair.overlap_fraction,
        ),
        "elapsed_hours": binned(
            "elapsed_hours",
            (1.0, 12.0, 24.0, 36.0, 48.0, 72.0, 96.01),
            lambda pair: pair.elapsed_hours,
        ),
    }


def _field(pair, frame, query, estimates):
    source_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return DisplacementField(
        pair_id=pair.pair_id,
        source_image_id=str(pair.source_image_id),
        target_image_id=str(pair.target_image_id),
        source_time_utc=source_time,
        target_time_utc=source_time + timedelta(hours=pair.elapsed_hours),
        grid_row=frame["grid_row"].to_numpy(np.int32),
        grid_column=frame["grid_column"].to_numpy(np.int32),
        source_xy_m=query,
        **estimates,
    )


def adjacent_displacement_gradient(field: DisplacementField) -> np.ndarray:
    lookup = {
        (int(row), int(column)): index
        for index, (row, column) in enumerate(
            zip(field.grid_row, field.grid_column, strict=True)
        )
        if field.available[index]
    }
    values = []
    for (row, column), first in lookup.items():
        for neighbour in ((row + 1, column), (row, column + 1)):
            second = lookup.get(neighbour)
            if second is None:
                continue
            separation = np.linalg.norm(
                field.source_xy_m[first] - field.source_xy_m[second]
            )
            values.append(
                np.linalg.norm(
                    field.displacement_m[first] - field.displacement_m[second]
                )
                / separation
            )
    return np.asarray(values, dtype=np.float64)


def regular_cell_deformation(field: DisplacementField) -> dict[tuple[int, int], float]:
    elapsed = (field.target_time_utc - field.source_time_utc).total_seconds()
    lookup = {
        (int(row), int(column)): index
        for index, (row, column) in enumerate(
            zip(field.grid_row, field.grid_column, strict=True)
        )
        if field.available[index]
    }
    result = {}
    for row, column in sorted(lookup):
        keys = (
            (row, column),
            (row + 1, column),
            (row, column + 1),
            (row + 1, column + 1),
        )
        if not all(key in lookup for key in keys):
            continue
        indices = [lookup[key] for key in keys]
        design = np.column_stack(
            (field.source_xy_m[indices], np.ones(len(indices)))
        )
        gradient = np.linalg.lstsq(
            design, field.displacement_m[indices], rcond=None
        )[0]
        du_dx, dv_dx = gradient[0]
        du_dy, dv_dy = gradient[1]
        divergence = (du_dx + dv_dy) / elapsed
        shear = np.sqrt(
            (du_dx - dv_dy) ** 2 + (du_dy + dv_dx) ** 2
        ) / elapsed
        result[(row, column)] = float(np.hypot(divergence, shear))
    return result


class _Accumulator:
    def __init__(self):
        self.node_count = 0
        self.available_count = 0
        self.fold_count = 0
        self.new_count = 0
        self.lost_count = 0
        self.support = []
        self.residual = []
        self.displacement_difference = []
        self.roughness = []
        self.baseline_cells = []
        self.policy_cells = []
        self.tier_counts: dict[int, int] = {}
        self.weighted_node_count = 0.0
        self.weighted_available_count = 0.0
        self.weighted_fold_count = 0.0
        self.weighted_new_count = 0.0
        self.weighted_lost_count = 0.0

    def add(
        self,
        *,
        field,
        baseline,
        folds,
        displacement_difference,
        roughness,
        baseline_cells,
        policy_cells,
        tier_used,
        tier_count,
        sampling_weight,
    ):
        self.node_count += len(field)
        self.available_count += int(field.available.sum())
        self.fold_count += folds
        self.new_count += int(np.count_nonzero(field.available & ~baseline.available))
        self.lost_count += int(np.count_nonzero(~field.available & baseline.available))
        self.weighted_node_count += sampling_weight * len(field)
        self.weighted_available_count += sampling_weight * int(field.available.sum())
        self.weighted_fold_count += sampling_weight * folds
        self.weighted_new_count += sampling_weight * int(
            np.count_nonzero(field.available & ~baseline.available)
        )
        self.weighted_lost_count += sampling_weight * int(
            np.count_nonzero(~field.available & baseline.available)
        )
        self.support.append(field.support_radius_m[field.available])
        self.residual.append(field.maximum_residual_m[field.available])
        self.displacement_difference.append(displacement_difference)
        self.roughness.append(roughness)
        self.baseline_cells.append(baseline_cells)
        self.policy_cells.append(policy_cells)
        for index in range(tier_count):
            self.tier_counts[index + 1] = self.tier_counts.get(index + 1, 0) + int(
                np.count_nonzero((tier_used == index) & field.available)
            )

    def summary(self, name, pair_count):
        baseline_cells = _concatenate(self.baseline_cells)
        policy_cells = _concatenate(self.policy_cells)
        base_p99 = _quantile(baseline_cells, 0.99)
        policy_p99 = _quantile(policy_cells, 0.99)
        return {
            "policy": name,
            "sampled_pairs": pair_count,
            "grid_nodes": self.node_count,
            "available_nodes": self.available_count,
            "coverage_fraction": self.available_count / self.node_count,
            "population_estimated_coverage_fraction": (
                self.weighted_available_count / self.weighted_node_count
            ),
            "fold_rejected_nodes": self.fold_count,
            "fold_rejected_fraction_of_available_before_rejection": (
                self.fold_count / (self.available_count + self.fold_count)
                if self.available_count + self.fold_count
                else 0.0
            ),
            "new_available_vs_baseline": self.new_count,
            "lost_available_vs_baseline": self.lost_count,
            "population_estimated_new_available_vs_baseline": self.weighted_new_count,
            "population_estimated_lost_available_vs_baseline": self.weighted_lost_count,
            "population_estimated_fold_rejected_fraction_of_available_before_rejection": (
                self.weighted_fold_count
                / (self.weighted_available_count + self.weighted_fold_count)
                if self.weighted_available_count + self.weighted_fold_count
                else 0.0
            ),
            "support_radius_p50_m": _quantile(_concatenate(self.support), 0.50),
            "support_radius_p90_m": _quantile(_concatenate(self.support), 0.90),
            "maximum_residual_p90_m": _quantile(
                _concatenate(self.residual), 0.90
            ),
            "common_node_displacement_p95_difference_m": _quantile(
                _concatenate(self.displacement_difference), 0.95
            ),
            "adjacent_displacement_gradient_p99": _quantile(
                _concatenate(self.roughness), 0.99
            ),
            "common_regular_cells": len(policy_cells),
            "common_cell_baseline_total_deformation_p99_s_1": base_p99,
            "common_cell_policy_total_deformation_p99_s_1": policy_p99,
            "common_cell_total_deformation_p99_ratio": (
                policy_p99 / base_p99 if base_p99 and np.isfinite(base_p99) else None
            ),
            "tier_counts": json.dumps(self.tier_counts, sort_keys=True),
        }


def plot_screen(rows, path):
    names = [row["policy"] for row in rows]
    labels = [name.replace("adaptive-", "adapt.\n").replace("agreement-", "agree.\n") for name in names]
    x = np.arange(len(rows))
    coverage_gain = [
        row["population_estimated_coverage_gain_percentage_points"] for row in rows
    ]
    deformation_ratio = [
        row["common_cell_total_deformation_p99_ratio"] for row in rows
    ]
    support_radius = [row["support_radius_p90_m"] / 1_000.0 for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    colors = ["#3b82f6" if name == "baseline" else "#0f766e" for name in names]
    axes[0].bar(x, coverage_gain, color=colors)
    axes[0].axhline(0, color="0.25", linewidth=0.8)
    axes[0].set_ylabel("Population-estimated gain (percentage points)")
    axes[0].set_title("Available 4 km field grid points")
    axes[1].scatter(coverage_gain, deformation_ratio, c=colors, s=58)
    axes[1].axhline(1.0, color="0.25", linewidth=0.8)
    for index, name in enumerate(names):
        axes[1].annotate(str(index + 1), (coverage_gain[index], deformation_ratio[index]), xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlabel("Population-estimated gain (percentage points)")
    axes[1].set_ylabel("Common-cell p99 deformation / baseline")
    axes[1].set_title("Tail preservation on identical cells")
    axes[2].bar(x, support_radius, color=colors)
    axes[2].set_ylabel("90th-percentile support radius (km)")
    axes[2].set_title("Spatial support of accepted field grid points")
    for axis in (axes[0], axes[2]):
        axis.set_xticks(x, [f"{i + 1}" for i in x])
    figure.suptitle(
        "CPU field-consensus screen (numbers map to policies below)", fontsize=13
    )
    legend = "   ".join(f"{i + 1} {name}" for i, name in enumerate(names))
    figure.text(0.5, -0.01, legend, ha="center", va="top", fontsize=8, wrap=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _concatenate(values):
    finite = [np.asarray(value)[np.isfinite(value)] for value in values if len(value)]
    return np.concatenate(finite) if finite else np.empty(0, dtype=np.float64)


def _quantile(values, q):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, q)) if len(finite) else None


if __name__ == "__main__":
    raise SystemExit(main())
