#!/usr/bin/env python3
"""Replay a local ALIKED field policy through chained buoy-linked image pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_aliked_dense_pair import nearest_consensus_at_queries


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sequence_inputs(
    run_dirs: list[Path],
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict]]:
    matches = []
    buoy_rows = []
    manifests = []
    for run_dir in run_dirs:
        matches.append(pd.read_csv(run_dir / "matches.csv"))
        buoy_rows.append(
            pd.read_csv(run_dir / "buoy_results.csv", dtype={"buoy_id": str})
        )
        manifests.append(json.loads((run_dir / "run_manifest.json").read_text()))
    for previous, current in zip(manifests, manifests[1:]):
        if previous["target_image_filepath"] != current["source_image_filepath"]:
            raise ValueError("dense pair inputs do not form a contiguous image chain")
    return matches, buoy_rows, manifests


def common_paths(buoy_rows: list[pd.DataFrame]) -> list[str]:
    path_sets = [set(rows["continuous_trajectory_id"]) for rows in buoy_rows]
    return sorted(set.intersection(*path_sets)) if path_sets else []


def estimate_at_position(
    matches: pd.DataFrame,
    source_xy: np.ndarray,
    maximum_radius_m: float,
    candidate_count: int,
    minimum_selected_vectors: int,
    consensus_radius_m: float,
) -> pd.Series:
    query = pd.DataFrame(
        [{"source_x": float(source_xy[0]), "source_y": float(source_xy[1])}]
    )
    return nearest_consensus_at_queries(
        matches,
        query,
        maximum_radius_m=maximum_radius_m,
        candidate_count=candidate_count,
        minimum_selected_vectors=minimum_selected_vectors,
        consensus_radius_m=consensus_radius_m,
    ).iloc[0]


def replay_sequence(
    matches: list[pd.DataFrame],
    buoy_rows: list[pd.DataFrame],
    *,
    maximum_radius_m: float = 6000.0,
    candidate_count: int = 12,
    minimum_selected_vectors: int = 8,
    consensus_radius_m: float = 1000.0,
    continuity_tolerance_m: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    if len(matches) != len(buoy_rows) or not matches:
        raise ValueError("one non-empty buoy table is required for every match table")
    paths = common_paths(buoy_rows)
    indexed = [rows.set_index("continuous_trajectory_id") for rows in buoy_rows]
    path_rows = {
        path_id: [rows.loc[path_id] for rows in indexed] for path_id in paths
    }
    for path_id, rows in path_rows.items():
        if any(isinstance(row, pd.DataFrame) for row in rows):
            raise ValueError(f"duplicate transition for continuous path {path_id}")
        for previous, current in zip(rows, rows[1:]):
            previous_target = np.array(
                [
                    previous.source_x + previous.truth_dx_m,
                    previous.source_y + previous.truth_dy_m,
                ]
            )
            current_source = np.array([current.source_x, current.source_y])
            if np.linalg.norm(previous_target - current_source) > continuity_tolerance_m:
                raise ValueError(f"buoy truth is discontinuous for path {path_id}")

    records = []
    propagated_xy = {
        path_id: np.array([rows[0].source_x, rows[0].source_y], dtype=float)
        for path_id, rows in path_rows.items()
    }
    propagated_active = {path_id: True for path_id in paths}
    for step_index, pair_matches in enumerate(matches):
        truth_queries = pd.DataFrame(
            [
                {
                    "continuous_trajectory_id": path_id,
                    "source_x": rows[step_index].source_x,
                    "source_y": rows[step_index].source_y,
                }
                for path_id, rows in path_rows.items()
            ]
        )
        truth_estimates = nearest_consensus_at_queries(
            pair_matches,
            truth_queries,
            maximum_radius_m=maximum_radius_m,
            candidate_count=candidate_count,
            minimum_selected_vectors=minimum_selected_vectors,
            consensus_radius_m=consensus_radius_m,
        ).set_index("continuous_trajectory_id")
        active_paths = [path_id for path_id in paths if propagated_active[path_id]]
        propagated_queries = pd.DataFrame(
            [
                {
                    "continuous_trajectory_id": path_id,
                    "source_x": propagated_xy[path_id][0],
                    "source_y": propagated_xy[path_id][1],
                }
                for path_id in active_paths
            ]
        )
        propagated_estimates = (
            nearest_consensus_at_queries(
                pair_matches,
                propagated_queries,
                maximum_radius_m=maximum_radius_m,
                candidate_count=candidate_count,
                minimum_selected_vectors=minimum_selected_vectors,
                consensus_radius_m=consensus_radius_m,
            ).set_index("continuous_trajectory_id")
            if len(propagated_queries)
            else pd.DataFrame()
        )
        for path_id in paths:
            truth = path_rows[path_id][step_index]
            truth_source = np.array([truth.source_x, truth.source_y], dtype=float)
            truth_target = truth_source + np.array(
                [truth.truth_dx_m, truth.truth_dy_m], dtype=float
            )
            estimates: dict[str, pd.Series | None] = {
                "truth_reinitialized": truth_estimates.loc[path_id],
                "propagated": propagated_estimates.loc[path_id]
                if propagated_active[path_id]
                else None,
            }
            next_propagated_xy = None
            for mode, estimate in estimates.items():
                tracking_source = (
                    truth_source
                    if mode == "truth_reinitialized"
                    else propagated_xy[path_id]
                )
                available = estimate is not None and bool(estimate.available)
                predicted_target = (
                    tracking_source
                    + np.array(
                        [estimate.proposal_dx_m, estimate.proposal_dy_m], dtype=float
                    )
                    if available
                    else np.array([np.nan, np.nan])
                )
                records.append(
                    {
                        "transition_id": getattr(truth, "transition_id", None),
                        "continuous_trajectory_id": path_id,
                        "buoy_id": str(truth.buoy_id),
                        "mode": mode,
                        "step": step_index + 1,
                        "source_image_id": int(truth.source_image_id),
                        "target_image_id": int(truth.target_image_id),
                        "truth_source_x": truth_source[0],
                        "truth_source_y": truth_source[1],
                        "tracking_source_x": tracking_source[0],
                        "tracking_source_y": tracking_source[1],
                        "tracking_source_error_m": float(
                            np.linalg.norm(tracking_source - truth_source)
                        ),
                        "available": available,
                        "selected_vectors": int(estimate.selected_vectors)
                        if available
                        else 0,
                        "proposal_dx_m": estimate.proposal_dx_m
                        if available
                        else np.nan,
                        "proposal_dy_m": estimate.proposal_dy_m
                        if available
                        else np.nan,
                        "predicted_target_x": predicted_target[0],
                        "predicted_target_y": predicted_target[1],
                        "truth_target_x": truth_target[0],
                        "truth_target_y": truth_target[1],
                        "endpoint_error_m": float(
                            np.linalg.norm(predicted_target - truth_target)
                        )
                        if available
                        else np.nan,
                    }
                )
                if mode == "propagated" and available:
                    next_propagated_xy = predicted_target
            if next_propagated_xy is None:
                propagated_active[path_id] = False
            else:
                propagated_xy[path_id] = next_propagated_xy

    results = pd.DataFrame.from_records(records)
    summary = summarize_paths(results, len(paths), len(matches))
    return results, summary


def summarize_paths(results: pd.DataFrame, path_count: int, steps: int) -> dict:
    summary: dict[str, object] = {
        "paths": path_count,
        "steps_per_path": steps,
        "requested_mode_steps": int(path_count * steps),
    }
    for mode in ("truth_reinitialized", "propagated"):
        selected = results[results["mode"] == mode] if len(results) else results
        complete = (
            selected.groupby("continuous_trajectory_id")["available"].all()
            if len(selected)
            else pd.Series(dtype=bool)
        )
        final = selected[selected["step"] == steps]
        errors = final.loc[final["available"], "endpoint_error_m"]
        summary[mode] = {
            "available_steps": int(selected["available"].sum()) if len(selected) else 0,
            "complete_paths": int(complete.sum()),
            "final_available_paths": int(final["available"].sum()),
            "final_correct_within_2km": int((errors <= 2000.0).sum()),
            "final_median_error_m": float(errors.median()) if len(errors) else None,
            "final_p90_error_m": float(errors.quantile(0.90)) if len(errors) else None,
            "final_maximum_error_m": float(errors.max()) if len(errors) else None,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-radius-m", type=float, default=6000.0)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--minimum-selected-vectors", type=int, default=8)
    parser.add_argument("--consensus-radius-m", type=float, default=1000.0)
    args = parser.parse_args()
    if len(args.dense_run_dir) < 2:
        parser.error("at least two --dense-run-dir values are required")

    matches, buoy_rows, manifests = load_sequence_inputs(args.dense_run_dir)
    results, summary = replay_sequence(
        matches,
        buoy_rows,
        maximum_radius_m=args.maximum_radius_m,
        candidate_count=args.candidate_count,
        minimum_selected_vectors=args.minimum_selected_vectors,
        consensus_radius_m=args.consensus_radius_m,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "path_steps.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "status": "complete",
        "dense_run_dirs": [str(path) for path in args.dense_run_dir],
        "input_hashes": [
            {
                "matches": sha256(path / "matches.csv"),
                "buoy_results": sha256(path / "buoy_results.csv"),
            }
            for path in args.dense_run_dir
        ],
        "image_chain": [
            {
                "source": item["source_image_filepath"],
                "target": item["target_image_filepath"],
            }
            for item in manifests
        ],
        "policy": {
            "name": "nearest_consensus",
            "maximum_radius_m": args.maximum_radius_m,
            "candidate_count": args.candidate_count,
            "minimum_selected_vectors": args.minimum_selected_vectors,
            "consensus_radius_m": args.consensus_radius_m,
        },
        "summary": summary,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
