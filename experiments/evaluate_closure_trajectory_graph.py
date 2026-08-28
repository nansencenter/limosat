#!/usr/bin/env python3
"""Replay adjacent and skip fields with a truth-free closure-fusion policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_efficientloftr_leave_one_out import (
    deformation_comparison,
    quantiles,
)
from experiments.evaluate_learned_trajectory_graph import trajectory_summary
from experiments.run_efficientloftr_sequence import field_from_csv
from limosat.learned_drift import FieldEdge, advect_trajectory_graph
from limosat.learned_drift.trajectory import sample_field


@dataclass(frozen=True)
class EdgeInput:
    edge: FieldEdge
    role: str
    field_path: Path
    field_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjacent-sequence-dir", type=Path, required=True)
    parser.add_argument("--skip-sequence-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--reference-pair-run-dir", type=Path, action="append", default=[]
    )
    parser.add_argument("--closure-agreement-m", type=float, default=1000.0)
    parser.add_argument("--uncertainty-floor-m", type=float, default=80.0)
    parser.add_argument("--maximum-triangle-edge-m", type=float, default=6400.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(run_dir: Path, role: str) -> tuple[list[str], list[EdgeInput]]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    pairs = manifest["pairs_summary"]
    image_ids = [str(pairs[0]["source_image_id"])] + [
        str(pair["target_image_id"]) for pair in pairs
    ]
    inputs = []
    for pair in pairs:
        source = str(pair["source_image_id"])
        target = str(pair["target_image_id"])
        path = run_dir / f"pair_{source}_{target}" / "field_4km.csv"
        inputs.append(
            EdgeInput(
                FieldEdge(
                    source,
                    target,
                    float(pair["elapsed_hours"]),
                    field_from_csv(path),
                ),
                role,
                path,
                sha256(path),
            )
        )
    return image_ids, inputs


def load_graph_inputs(
    adjacent_dir: Path, skip_dirs: list[Path]
) -> tuple[list[str], list[EdgeInput]]:
    image_ids, adjacent = load_run(adjacent_dir, "adjacent")
    adjacent_keys = {
        (str(item.edge.source_image_id), str(item.edge.target_image_id))
        for item in adjacent
    }
    skip_by_key: dict[tuple[str, str], EdgeInput] = {}
    for run_dir in skip_dirs:
        _ids, candidates = load_run(run_dir, "skip")
        for item in candidates:
            key = (str(item.edge.source_image_id), str(item.edge.target_image_id))
            if key in adjacent_keys:
                continue
            if key in skip_by_key and skip_by_key[key].field_sha256 != item.field_sha256:
                raise ValueError(f"multiple non-identical skip fields for {key}")
            skip_by_key[key] = item
    return image_ids, adjacent + list(skip_by_key.values())


def _seed_rows(image_id: str, seed_xy_m: np.ndarray) -> pd.DataFrame:
    count = len(seed_xy_m)
    return pd.DataFrame(
        {
            "trajectory_id": np.arange(count, dtype=np.int64),
            "image_index": 0,
            "image_id": image_id,
            "seed_image_index": 0,
            "seed_image_id": image_id,
            "x_m": seed_xy_m[:, 0],
            "y_m": seed_xy_m[:, 1],
            "active": True,
            "trajectory_state": "seed",
            "reconnected_after_gap": False,
            "edge_source_image_id": "",
            "skipped_images": -1,
            "step_dx_m": np.nan,
            "step_dy_m": np.nan,
            "field_selected_matches": np.nan,
            "field_support_radius_m": np.nan,
            "field_maximum_residual_m": np.nan,
            "candidate_count": 0,
            "consistent_candidate_count": 0,
            "conflicting_candidate_count": 0,
            "closure_max_m": np.nan,
            "primary_to_fused_m": np.nan,
            "closure_fused": False,
        }
    )


def advect_closure_fused_graph(
    edges: list[FieldEdge],
    image_ids: list[str],
    seed_xy_m: np.ndarray,
    maximum_triangle_edge_m: float,
    closure_agreement_m: float,
    uncertainty_floor_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prefer the shortest observation and fuse locally consistent alternatives.

    Candidate endpoint weights use ``selected_matches / sigma**2``, where
    ``sigma`` is the field maximum residual with an 80 m default floor. A
    candidate farther than the existing 1 km consensus radius from the primary
    endpoint is retained in the candidate ledger but not fused.
    """
    if closure_agreement_m <= 0 or uncertainty_floor_m <= 0:
        raise ValueError("closure and uncertainty distances must be positive")
    keys = [str(value) for value in image_ids]
    index = {value: step for step, value in enumerate(keys)}
    indexed = []
    for edge in edges:
        source = index[str(edge.source_image_id)]
        target = index[str(edge.target_image_id)]
        if source >= target:
            raise ValueError("all graph edges must point forward")
        indexed.append((source, target, edge))

    seeds = np.asarray(seed_xy_m, dtype=np.float64)
    count = len(seeds)
    positions = [np.full((count, 2), np.nan) for _ in keys]
    available = [np.zeros(count, dtype=bool) for _ in keys]
    positions[0][:] = seeds
    available[0][:] = True
    trajectory_rows = [_seed_rows(keys[0], seeds)]
    candidate_rows: list[dict] = []

    for target_step in range(1, len(keys)):
        candidates = []
        for source_step, target, edge in indexed:
            if target != target_step:
                continue
            source_indices = np.flatnonzero(available[source_step])
            endpoint = np.full((count, 2), np.nan)
            candidate_available = np.zeros(count, dtype=bool)
            selected = np.full(count, np.nan)
            radius = np.full(count, np.nan)
            residual = np.full(count, np.nan)
            if len(source_indices):
                sampled = sample_field(
                    edge.field,
                    positions[source_step][source_indices],
                    maximum_triangle_edge_m,
                )
                supported = source_indices[sampled.available]
                candidate_available[supported] = True
                endpoint[supported] = (
                    positions[source_step][supported]
                    + sampled.displacement_m[sampled.available]
                )
                selected[supported] = sampled.selected_matches[sampled.available]
                radius[supported] = sampled.support_radius_m[sampled.available]
                residual[supported] = sampled.maximum_residual_m[sampled.available]
            candidates.append(
                {
                    "source_step": source_step,
                    "edge": edge,
                    "skipped": target_step - source_step - 1,
                    "available": candidate_available,
                    "endpoint": endpoint,
                    "selected": selected,
                    "radius": radius,
                    "residual": residual,
                }
            )

        target_positions = np.full((count, 2), np.nan)
        target_available = np.zeros(count, dtype=bool)
        state = np.full(count, "unreached", dtype=object)
        edge_source = np.full(count, "", dtype=object)
        skipped = np.full(count, -1, dtype=np.int32)
        step_displacement = np.full((count, 2), np.nan)
        selected = np.full(count, np.nan)
        radius = np.full(count, np.nan)
        residual = np.full(count, np.nan)
        candidate_count = np.zeros(count, dtype=np.int32)
        consistent_count = np.zeros(count, dtype=np.int32)
        conflict_count = np.zeros(count, dtype=np.int32)
        closure_max = np.full(count, np.nan)
        primary_to_fused = np.full(count, np.nan)
        fused = np.zeros(count, dtype=bool)

        for trajectory_id in range(count):
            choices = [
                candidate
                for candidate in candidates
                if candidate["available"][trajectory_id]
            ]
            if not choices:
                continue
            primary = min(
                choices,
                key=lambda item: (
                    item["skipped"],
                    -item["selected"][trajectory_id],
                ),
            )
            primary_endpoint = primary["endpoint"][trajectory_id]
            distances = np.asarray(
                [
                    np.linalg.norm(item["endpoint"][trajectory_id] - primary_endpoint)
                    for item in choices
                ]
            )
            consistent = distances <= closure_agreement_m
            accepted = [item for item, keep in zip(choices, consistent) if keep]
            weights = np.asarray(
                [
                    max(float(item["selected"][trajectory_id]), 1.0)
                    / max(
                        float(item["residual"][trajectory_id]),
                        uncertainty_floor_m,
                    )
                    ** 2
                    for item in accepted
                ]
            )
            endpoints = np.asarray(
                [item["endpoint"][trajectory_id] for item in accepted]
            )
            fused_endpoint = np.average(endpoints, axis=0, weights=weights)
            source_step = int(primary["source_step"])
            target_available[trajectory_id] = True
            target_positions[trajectory_id] = fused_endpoint
            skipped[trajectory_id] = int(primary["skipped"])
            state[trajectory_id] = (
                "observed_adjacent" if primary["skipped"] == 0 else "observed_skip_edge"
            )
            edge_source[trajectory_id] = keys[source_step]
            step_displacement[trajectory_id] = (
                fused_endpoint - positions[source_step][trajectory_id]
            )
            selected[trajectory_id] = primary["selected"][trajectory_id]
            radius[trajectory_id] = primary["radius"][trajectory_id]
            residual[trajectory_id] = primary["residual"][trajectory_id]
            candidate_count[trajectory_id] = len(choices)
            consistent_count[trajectory_id] = int(consistent.sum())
            conflict_count[trajectory_id] = int((~consistent).sum())
            if len(choices) > 1:
                closure_max[trajectory_id] = float(distances.max())
            primary_to_fused[trajectory_id] = float(
                np.linalg.norm(fused_endpoint - primary_endpoint)
            )
            fused[trajectory_id] = len(accepted) > 1

            for item, distance, keep in zip(choices, distances, consistent):
                candidate_rows.append(
                    {
                        "trajectory_id": trajectory_id,
                        "target_image_index": target_step,
                        "target_image_id": keys[target_step],
                        "source_image_id": keys[int(item["source_step"])],
                        "skipped_images": int(item["skipped"]),
                        "endpoint_x_m": item["endpoint"][trajectory_id, 0],
                        "endpoint_y_m": item["endpoint"][trajectory_id, 1],
                        "selected_matches": item["selected"][trajectory_id],
                        "support_radius_m": item["radius"][trajectory_id],
                        "maximum_residual_m": item["residual"][trajectory_id],
                        "closure_to_primary_m": float(distance),
                        "primary": item is primary,
                        "accepted_for_fusion": bool(keep),
                    }
                )

        positions[target_step] = target_positions
        available[target_step] = target_available
        reconnected = target_available & ~available[target_step - 1]
        trajectory_rows.append(
            pd.DataFrame(
                {
                    "trajectory_id": np.arange(count, dtype=np.int64),
                    "image_index": target_step,
                    "image_id": keys[target_step],
                    "seed_image_index": 0,
                    "seed_image_id": keys[0],
                    "x_m": target_positions[:, 0],
                    "y_m": target_positions[:, 1],
                    "active": target_available,
                    "trajectory_state": state,
                    "reconnected_after_gap": reconnected,
                    "edge_source_image_id": edge_source,
                    "skipped_images": skipped,
                    "step_dx_m": step_displacement[:, 0],
                    "step_dy_m": step_displacement[:, 1],
                    "field_selected_matches": selected,
                    "field_support_radius_m": radius,
                    "field_maximum_residual_m": residual,
                    "candidate_count": candidate_count,
                    "consistent_candidate_count": consistent_count,
                    "conflicting_candidate_count": conflict_count,
                    "closure_max_m": closure_max,
                    "primary_to_fused_m": primary_to_fused,
                    "closure_fused": fused,
                }
            )
        )
    return pd.concat(trajectory_rows, ignore_index=True), pd.DataFrame(candidate_rows)


def compare_graphs(
    shortest: pd.DataFrame,
    fused: pd.DataFrame,
    elapsed_hours: list[float],
) -> dict:
    report = {"by_image": {}, "deformation_by_image": {}}
    seed = shortest.loc[
        shortest.image_index.eq(0), ["trajectory_id", "x_m", "y_m"]
    ].rename(columns={"x_m": "source_x", "y_m": "source_y"})
    cumulative_hours = np.cumsum(elapsed_hours)
    for image_index in range(1, int(shortest.image_index.max()) + 1):
        first = shortest.loc[
            shortest.image_index.eq(image_index),
            ["trajectory_id", "active", "x_m", "y_m"],
        ].rename(
            columns={
                "active": "shortest_available",
                "x_m": "shortest_x_m",
                "y_m": "shortest_y_m",
            }
        )
        second = fused.loc[
            fused.image_index.eq(image_index),
            ["trajectory_id", "active", "x_m", "y_m"],
        ].rename(
            columns={
                "active": "fused_available",
                "x_m": "fused_x_m",
                "y_m": "fused_y_m",
            }
        )
        rows = seed.merge(first, on="trajectory_id").merge(second, on="trajectory_id")
        common = rows.shortest_available & rows.fused_available
        difference = np.hypot(
            rows.loc[common, "shortest_x_m"] - rows.loc[common, "fused_x_m"],
            rows.loc[common, "shortest_y_m"] - rows.loc[common, "fused_y_m"],
        )
        image_id = str(
            shortest.loc[shortest.image_index.eq(image_index), "image_id"].iloc[0]
        )
        report["by_image"][image_id] = {
            "shortest_available": int(rows.shortest_available.sum()),
            "fused_available": int(rows.fused_available.sum()),
            "common_available": int(common.sum()),
            "shortest_only": int((rows.shortest_available & ~rows.fused_available).sum()),
            "fused_only": int((~rows.shortest_available & rows.fused_available).sum()),
            "position_difference_m": quantiles(difference.to_numpy(float)),
        }
        first_vectors = rows.loc[rows.shortest_available].copy()
        first_vectors["dx_m"] = first_vectors.shortest_x_m - first_vectors.source_x
        first_vectors["dy_m"] = first_vectors.shortest_y_m - first_vectors.source_y
        second_vectors = rows.loc[rows.fused_available].copy()
        second_vectors["dx_m"] = second_vectors.fused_x_m - second_vectors.source_x
        second_vectors["dy_m"] = second_vectors.fused_y_m - second_vectors.source_y
        queries = rows.loc[common, ["source_x", "source_y"]].to_numpy(float)
        report["deformation_by_image"][image_id] = deformation_comparison(
            first_vectors[["source_x", "source_y", "dx_m", "dy_m"]],
            second_vectors[["source_x", "source_y", "dx_m", "dy_m"]],
            queries,
            float(cumulative_hours[image_index - 1]),
        )
    return report


def buoy_metrics(rows: pd.DataFrame) -> dict:
    expected = rows.truth_x_m.notna()
    available = expected & rows.active
    correct = available & rows.error_m.le(2000.0)
    errors = rows.loc[available, "error_m"].dropna().to_numpy(float)
    return {
        "expected": int(expected.sum()),
        "available": int(available.sum()),
        "correct_within_2km": int(correct.sum()),
        "median_error_m": float(np.median(errors)) if len(errors) else None,
        "p90_error_m": float(np.quantile(errors, 0.90)) if len(errors) else None,
        "maximum_error_m": float(np.max(errors)) if len(errors) else None,
    }


def evaluate_buoys(
    edges: list[FieldEdge],
    image_ids: list[str],
    references: list[Path],
    maximum_triangle_edge_m: float,
    closure_agreement_m: float,
    uncertainty_floor_m: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if len(references) != len(image_ids) - 1:
        raise ValueError("one adjacent buoy reference is required per image transition")
    reference_rows = [
        pd.read_csv(path / "buoy_results.csv", dtype={"buoy_id": str})
        for path in references
    ]
    first = reference_rows[0].drop_duplicates("buoy_id").reset_index(drop=True)
    seeds = first[["source_x", "source_y"]].to_numpy(np.float64)
    adjacent_keys = set(zip(image_ids, image_ids[1:]))
    adjacent_edges = [
        edge
        for edge in edges
        if (str(edge.source_image_id), str(edge.target_image_id)) in adjacent_keys
    ]
    adjacent_only = advect_trajectory_graph(
        adjacent_edges,
        image_ids,
        4000.0,
        seeds,
        maximum_triangle_edge_m=maximum_triangle_edge_m,
    )
    shortest = advect_trajectory_graph(
        edges,
        image_ids,
        4000.0,
        seeds,
        maximum_triangle_edge_m=maximum_triangle_edge_m,
    )
    fused, _ = advect_closure_fused_graph(
        edges,
        image_ids,
        seeds,
        maximum_triangle_edge_m,
        closure_agreement_m,
        uncertainty_floor_m,
    )
    truth = []
    for image_index, rows in enumerate(reference_rows, start=1):
        values = rows.drop_duplicates("buoy_id").copy()
        values["truth_x_m"] = values.source_x + values.truth_dx_m
        values["truth_y_m"] = values.source_y + values.truth_dy_m
        values["image_index"] = image_index
        truth.append(values[["buoy_id", "image_index", "truth_x_m", "truth_y_m"]])
    truth = pd.concat(truth, ignore_index=True)

    def attach(rows: pd.DataFrame) -> pd.DataFrame:
        result = rows.copy()
        result["buoy_id"] = first.loc[result.trajectory_id, "buoy_id"].to_numpy()
        result = result.merge(
            truth, on=["buoy_id", "image_index"], how="left", validate="many_to_one"
        )
        result["error_m"] = np.where(
            result.active & result.truth_x_m.notna(),
            np.hypot(result.x_m - result.truth_x_m, result.y_m - result.truth_y_m),
            np.nan,
        )
        return result

    adjacent_only = attach(adjacent_only)
    shortest = attach(shortest)
    fused = attach(fused)
    report = {
        "adjacent_only": buoy_metrics(adjacent_only),
        "shortest": buoy_metrics(shortest),
        "closure_fused": buoy_metrics(fused),
        "by_image": {},
    }
    for image_index, image_id in enumerate(image_ids[1:], start=1):
        report["by_image"][image_id] = {
            "adjacent_only": buoy_metrics(
                adjacent_only[adjacent_only.image_index.eq(image_index)]
            ),
            "shortest": buoy_metrics(shortest[shortest.image_index.eq(image_index)]),
            "closure_fused": buoy_metrics(fused[fused.image_index.eq(image_index)]),
        }
    return report, adjacent_only, shortest, fused


def main() -> int:
    args = parse_args()
    image_ids, inputs = load_graph_inputs(
        args.adjacent_sequence_dir, args.skip_sequence_dir
    )
    edges = [item.edge for item in inputs]
    adjacent = [item.edge for item in inputs if item.role == "adjacent"]
    elapsed_hours = [float(edge.elapsed_hours) for edge in adjacent]
    first = adjacent[0].field
    valid = first.available & np.isfinite(first.displacement_m).all(axis=1)
    seeds = np.asarray(first.source_xy_m[valid], dtype=np.float64)

    adjacent_only = advect_trajectory_graph(
        adjacent,
        image_ids,
        4000.0,
        seeds,
        maximum_triangle_edge_m=args.maximum_triangle_edge_m,
    )
    shortest = advect_trajectory_graph(
        edges,
        image_ids,
        4000.0,
        seeds,
        maximum_triangle_edge_m=args.maximum_triangle_edge_m,
    )
    fused, candidates = advect_closure_fused_graph(
        edges,
        image_ids,
        seeds,
        args.maximum_triangle_edge_m,
        args.closure_agreement_m,
        args.uncertainty_floor_m,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adjacent_only.to_csv(args.output_dir / "adjacent_only_graph.csv", index=False)
    shortest.to_csv(args.output_dir / "shortest_observed_graph.csv", index=False)
    fused.to_csv(args.output_dir / "closure_fused_graph.csv", index=False)
    candidates.to_csv(args.output_dir / "closure_candidates.csv", index=False)

    multiple = fused.candidate_count.gt(1)
    report = {
        "status": "complete",
        "truth_sealed": not bool(args.reference_pair_run_dir),
        "image_ids": image_ids,
        "policy": {
            "primary": "fewest skipped images, then most selected matches",
            "closure_agreement_m": args.closure_agreement_m,
            "uncertainty_floor_m": args.uncertainty_floor_m,
            "maximum_triangle_edge_m": args.maximum_triangle_edge_m,
            "fusion_weight": "selected_matches / max(maximum_residual_m, uncertainty_floor_m)^2",
            "conflict_action": "retain in candidate ledger; do not fuse into primary",
        },
        "edges": [
            {
                "source_image_id": str(item.edge.source_image_id),
                "target_image_id": str(item.edge.target_image_id),
                "elapsed_hours": item.edge.elapsed_hours,
                "role": item.role,
                "field_path": str(item.field_path),
                "field_sha256": item.field_sha256,
            }
            for item in inputs
        ],
        "adjacent_only_graph": trajectory_summary(adjacent_only),
        "shortest_graph": trajectory_summary(shortest),
        "closure_fused_graph": trajectory_summary(fused),
        "closure_diagnostics": {
            "multiple_candidate_rows": int(multiple.sum()),
            "fused_rows": int(fused.closure_fused.sum()),
            "conflict_rows": int(fused.conflicting_candidate_count.gt(0).sum()),
            "closure_max_m": quantiles(fused.loc[multiple, "closure_max_m"].to_numpy(float)),
            "primary_to_fused_m": quantiles(
                fused.loc[fused.closure_fused, "primary_to_fused_m"].to_numpy(float)
            ),
        },
        "shortest_vs_closure_fused": compare_graphs(
            shortest, fused, elapsed_hours
        ),
    }
    if args.reference_pair_run_dir:
        buoy_report, adjacent_buoy, shortest_buoy, fused_buoy = evaluate_buoys(
            edges,
            image_ids,
            args.reference_pair_run_dir,
            args.maximum_triangle_edge_m,
            args.closure_agreement_m,
            args.uncertainty_floor_m,
        )
        adjacent_buoy.to_csv(args.output_dir / "adjacent_only_buoys.csv", index=False)
        shortest_buoy.to_csv(args.output_dir / "shortest_buoys.csv", index=False)
        fused_buoy.to_csv(args.output_dir / "closure_fused_buoys.csv", index=False)
        report["buoys_unsealed"] = buoy_report
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
