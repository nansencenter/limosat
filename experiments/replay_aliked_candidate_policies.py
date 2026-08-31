#!/usr/bin/env python3
"""Replay spatially local ALIKED vector-selection policies without feature extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


POLICIES = (
    "four_nearest_inverse_distance",
    "highest_confidence_within_2km",
    "consensus_within_2km",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_geometric_median(
    vectors: np.ndarray,
    weights: np.ndarray,
    maximum_iterations: int = 100,
    tolerance_m: float = 1.0e-3,
) -> np.ndarray:
    """Return a deterministic weighted geometric median using Weiszfeld updates."""
    vectors = np.asarray(vectors, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 2 or len(vectors) == 0:
        raise ValueError("vectors must have shape (n, 2) with n > 0")
    weights = np.maximum(weights, 0.0)
    if not np.any(weights):
        weights = np.ones(len(vectors), dtype=float)
    estimate = np.average(vectors, axis=0, weights=weights)
    for _ in range(maximum_iterations):
        distance = np.linalg.norm(vectors - estimate, axis=1)
        coincident = distance <= tolerance_m
        if np.any(coincident):
            return np.average(vectors[coincident], axis=0, weights=weights[coincident])
        updated = np.average(
            vectors,
            axis=0,
            weights=weights / np.maximum(distance, tolerance_m),
        )
        if np.linalg.norm(updated - estimate) <= tolerance_m:
            return updated
        estimate = updated
    return estimate


def estimate_policy(
    matches: pd.DataFrame,
    policy: str,
    tight_radius_m: float = 2000.0,
    consensus_radius_m: float = 1000.0,
) -> dict:
    """Estimate one displacement from already physics-filtered match vectors."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    valid = matches.loc[matches["physics_valid"].fillna(False)].copy()
    score_column = (
        "matcher_score" if "matcher_score" in valid else "lightglue_score"
    )
    if policy == "four_nearest_inverse_distance":
        selected = valid.nsmallest(4, "source_distance_m")
        if selected.empty:
            return {"available": False, "selected_vectors": 0}
        weights = 1.0 / np.maximum(selected["source_distance_m"].to_numpy(), 1.0)
        estimate = np.average(selected[["dx_m", "dy_m"]], axis=0, weights=weights)
    else:
        selected = valid.loc[valid["source_distance_m"] <= tight_radius_m].copy()
        if selected.empty:
            return {"available": False, "selected_vectors": 0}
        if policy == "highest_confidence_within_2km":
            selected = selected.nlargest(1, score_column)
            estimate = selected[["dx_m", "dy_m"]].to_numpy()[0]
        else:
            vectors = selected[["dx_m", "dy_m"]].to_numpy(dtype=float)
            scores = np.maximum(selected[score_column].to_numpy(dtype=float), 0.0)
            separation = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=2)
            support = (separation <= consensus_radius_m) @ np.maximum(scores, 1.0e-12)
            seed = int(np.argmax(support))
            keep = separation[seed] <= consensus_radius_m
            selected = selected.iloc[np.flatnonzero(keep)].copy()
            estimate = weighted_geometric_median(
                selected[["dx_m", "dy_m"]].to_numpy(dtype=float),
                np.maximum(
                    selected[score_column].to_numpy(dtype=float), 1.0e-12
                ),
            )
    vectors = selected[["dx_m", "dy_m"]].to_numpy(dtype=float)
    return {
        "available": True,
        "selected_vectors": int(len(selected)),
        "proposal_dx_m": float(estimate[0]),
        "proposal_dy_m": float(estimate[1]),
        "maximum_source_distance_m": float(selected["source_distance_m"].max()),
        "maximum_vector_residual_m": float(
            np.linalg.norm(vectors - estimate, axis=1).max()
        ),
    }


def recenter_matches(
    matches: pd.DataFrame,
    source_state_xy: np.ndarray,
    source_radius_m: float = 10000.0,
    maximum_speed_m_per_day: float = 30000.0,
) -> pd.DataFrame:
    """Reapply spatial physics gates around an estimated, rather than true, state."""
    recentered = matches.copy()
    source_xy = recentered[["source_x", "source_y"]].to_numpy(dtype=float)
    recentered["source_distance_m"] = np.linalg.norm(
        source_xy - np.asarray(source_state_xy, dtype=float), axis=1
    )
    recentered["physics_valid"] = (
        recentered["source_distance_m"].le(source_radius_m)
        & recentered["speed_m_per_day"].le(maximum_speed_m_per_day)
    )
    return recentered


def replay_propagated_paths(
    transitions: pd.DataFrame,
    vector_groups: dict,
    tight_radius_m: float,
    consensus_radius_m: float,
    source_radius_m: float = 10000.0,
    maximum_speed_m_per_day: float = 30000.0,
) -> tuple[pd.DataFrame, list[dict]]:
    """Propagate cached proposals while using each estimate as the next source state."""
    representative = transitions.loc[
        transitions["representative_panel"].fillna(False)
    ].sort_values(["continuous_trajectory_id", "source_image_time"])
    empty_matches = next(iter(vector_groups.values())).iloc[:0]
    records = []
    for path_id, path in representative.groupby(
        "continuous_trajectory_id", sort=True
    ):
        if len(path) < 2:
            continue
        rows = list(path.itertuples(index=False))
        if any(
            previous.target_image_id != following.source_image_id
            for previous, following in zip(rows[:-1], rows[1:])
        ):
            continue
        for policy in POLICIES:
            state = np.array([rows[0].source_x, rows[0].source_y], dtype=float)
            active = True
            for step_index, row in enumerate(rows):
                source_truth = np.array([row.source_x, row.source_y], dtype=float)
                target_truth = source_truth + np.array(
                    [row.truth_dx_m, row.truth_dy_m], dtype=float
                )
                source_state_error = float(np.linalg.norm(state - source_truth))
                proposal = {"available": False, "selected_vectors": 0}
                active_at_start = active
                if active_at_start:
                    matches = recenter_matches(
                        vector_groups.get(row.transition_id, empty_matches),
                        state,
                        source_radius_m=source_radius_m,
                        maximum_speed_m_per_day=maximum_speed_m_per_day,
                    )
                    proposal = estimate_policy(
                        matches,
                        policy,
                        tight_radius_m=tight_radius_m,
                        consensus_radius_m=consensus_radius_m,
                    )
                available = bool(proposal["available"])
                if available:
                    state = state + np.array(
                        [proposal["proposal_dx_m"], proposal["proposal_dy_m"]],
                        dtype=float,
                    )
                    error = float(np.linalg.norm(state - target_truth))
                else:
                    active = False
                    error = np.nan
                records.append(
                    {
                        "continuous_trajectory_id": path_id,
                        "transition_id": row.transition_id,
                        "step_index": step_index,
                        "policy": policy,
                        "source_state_error_m": source_state_error,
                        "active_at_start": active_at_start,
                        "available": available,
                        "error_m": error,
                        **proposal,
                    }
                )

    results = pd.DataFrame.from_records(records)
    summary = []
    for policy, policy_rows in results.groupby("policy", sort=True):
        path_groups = list(policy_rows.groupby("continuous_trajectory_id", sort=True))
        complete = [group["available"].fillna(False).all() for _, group in path_groups]
        correct = [
            bool(
                (
                    group["available"].fillna(False)
                    & group["error_m"].le(2000.0)
                ).all()
            )
            for _, group in path_groups
        ]
        complete_groups = [
            group for (_, group), is_complete in zip(path_groups, complete) if is_complete
        ]
        summary.append(
            {
                "policy": policy,
                "consecutive_paths": len(path_groups),
                "complete_paths": int(sum(complete)),
                "all_steps_within_2km": int(sum(correct)),
                "median_complete_path_maximum_step_error_m": (
                    float(np.median([group["error_m"].max() for group in complete_groups]))
                    if complete_groups
                    else None
                ),
                "median_complete_path_final_error_m": (
                    float(
                        np.median(
                            [
                                group.sort_values("step_index")["error_m"].iloc[-1]
                                for group in complete_groups
                            ]
                        )
                    )
                    if complete_groups
                    else None
                ),
                "maximum_attempted_source_state_error_m": (
                    float(
                        policy_rows.loc[
                            policy_rows["active_at_start"], "source_state_error_m"
                        ].max()
                    )
                    if policy_rows["active_at_start"].any()
                    else None
                ),
                "maximum_complete_path_source_state_error_m": (
                    float(
                        max(
                            group["source_state_error_m"].max()
                            for group in complete_groups
                        )
                    )
                    if complete_groups
                    else None
                ),
                "interpretation": (
                    "propagated state with cached truth-centred feature crops; "
                    "full crop propagation remains pending"
                ),
            }
        )
    return results, summary


def summarize(results: pd.DataFrame) -> list[dict]:
    records = []
    for (panel, policy), group in results.groupby(["panel", "policy"], sort=True):
        available = group["available"].fillna(False)
        error = group["error_m"]
        accepted_error = error.loc[available].dropna()
        record = {
            "panel": panel,
            "policy": policy,
            "expected": int(len(group)),
            "available": int(available.sum()),
            "median_error_m": float(accepted_error.median()) if len(accepted_error) else None,
            "p90_error_m": float(accepted_error.quantile(0.90)) if len(accepted_error) else None,
        }
        for threshold in (100, 250, 500, 1000, 2000):
            record[f"within_{threshold}m"] = int(
                (available & error.le(threshold)).sum()
            )
        records.append(record)
    return records


def summarize_truth_reinitialized_paths(results: pd.DataFrame) -> list[dict]:
    """Summarize consecutive one-step results without claiming propagated tracking."""
    representative = results.loc[results["panel"] == "representative"]
    records = []
    for policy, policy_rows in representative.groupby("policy", sort=True):
        paths = [
            group.sort_values("source_image_time")
            for _, group in policy_rows.groupby("continuous_trajectory_id")
            if len(group) >= 2
        ]
        complete = [bool(path["available"].fillna(False).all()) for path in paths]
        correct = [
            bool(
                (
                    path["available"].fillna(False)
                    & path["error_m"].le(2000.0)
                ).all()
            )
            for path in paths
        ]
        maximum_error = [
            float(path.loc[path["available"].fillna(False), "error_m"].max())
            for path in paths
            if path["available"].fillna(False).any()
        ]
        records.append(
            {
                "policy": policy,
                "paths_with_at_least_two_steps": len(paths),
                "complete_paths": int(sum(complete)),
                "all_steps_within_2km": int(sum(correct)),
                "median_path_maximum_step_error_m": (
                    float(np.median(maximum_error)) if maximum_error else None
                ),
                "interpretation": "truth-reinitialized one-step lower bound; not propagated state",
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tight-radius-m", type=float, default=2000.0)
    parser.add_argument("--consensus-radius-m", type=float, default=1000.0)
    args = parser.parse_args()

    vectors = pd.read_csv(args.vectors, low_memory=False)
    transitions = pd.read_csv(args.transitions, low_memory=False)
    vector_groups = {
        transition_id: group
        for transition_id, group in vectors.groupby("transition_id", sort=False)
    }
    empty_matches = vectors.iloc[:0]
    records = []
    for row in transitions.itertuples(index=False):
        transition_id = row.transition_id
        matches = vector_groups.get(transition_id, empty_matches)
        truth = np.array([row.truth_dx_m, row.truth_dy_m], dtype=float)
        panels = []
        if row.representative_panel:
            panels.append("representative")
        if row.challenge_panel:
            panels.append("challenge")
        for policy in POLICIES:
            proposal = estimate_policy(
                matches,
                policy,
                tight_radius_m=args.tight_radius_m,
                consensus_radius_m=args.consensus_radius_m,
            )
            available = bool(proposal["available"])
            error = (
                float(
                    np.linalg.norm(
                        np.array(
                            [proposal["proposal_dx_m"], proposal["proposal_dy_m"]]
                        )
                        - truth
                    )
                )
                if available
                else np.nan
            )
            for panel in panels:
                records.append(
                    {
                        "transition_id": transition_id,
                        "continuous_trajectory_id": row.continuous_trajectory_id,
                        "source_image_time": row.source_image_time,
                        "panel": panel,
                        "policy": policy,
                        "error_m": error,
                        **proposal,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame.from_records(records)
    results.to_csv(args.output_dir / "policy_results.csv", index=False)
    summary = summarize(results)
    path_summary = summarize_truth_reinitialized_paths(results)
    propagated_results, propagated_summary = replay_propagated_paths(
        transitions,
        vector_groups,
        tight_radius_m=args.tight_radius_m,
        consensus_radius_m=args.consensus_radius_m,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "path_summary.json").write_text(
        json.dumps(path_summary, indent=2) + "\n"
    )
    propagated_results.to_csv(
        args.output_dir / "propagated_path_results.csv", index=False
    )
    (args.output_dir / "propagated_path_summary.json").write_text(
        json.dumps(propagated_summary, indent=2) + "\n"
    )
    manifest = {
        "vectors": str(args.vectors.resolve()),
        "vectors_sha256": sha256(args.vectors),
        "transitions": str(args.transitions.resolve()),
        "transitions_sha256": sha256(args.transitions),
        "policies": list(POLICIES),
        "tight_radius_m": args.tight_radius_m,
        "consensus_radius_m": args.consensus_radius_m,
        "correctness_threshold_m": 2000.0,
        "path_summary_interpretation": (
            "truth-reinitialized one-step lower bound; propagated tracking remains pending"
        ),
        "propagated_path_interpretation": (
            "positions are propagated, but cached feature crops remain truth-centred"
        ),
        "rows": int(len(results)),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
