#!/usr/bin/env python3
"""Summarize B1 matcher-candidate and trajectory-stage event streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _bool_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def summarize_target(
    target_image_id: int,
    candidates: pd.DataFrame,
    stages: pd.DataFrame,
) -> dict:
    target_candidates = candidates.loc[
        candidates["target_image_id"] == target_image_id
    ]
    target_stages = stages.loc[stages["target_image_id"] == target_image_id]

    def stage(name: str) -> pd.DataFrame:
        return target_stages.loc[target_stages["stage"] == name]

    pattern = stage("pattern_matching")
    topology = stage("topology_filter")
    recheck = stage("topology_pattern_recheck")
    velocity = stage("velocity_filter")
    convergence = stage("convergence")
    return {
        "target_image_id": int(target_image_id),
        "candidate_matches": int(len(target_candidates)),
        "descriptor_pass": _bool_count(target_candidates, "descriptor_pass"),
        "motion_pass": _bool_count(target_candidates, "motion_pass"),
        "model_inliers": _bool_count(target_candidates, "model_inlier"),
        "model_accepted": _bool_count(target_candidates, "accepted"),
        "post_model": int(len(stage("post_model"))),
        "post_interpolation": int(len(stage("post_interpolation"))),
        "velocity_rejected": int(
            len(velocity) - _bool_count(velocity, "accepted")
        ),
        "convergence_rejected": int(len(convergence)),
        "pattern_evaluated": int(len(pattern)),
        "pattern_unavailable": int(
            len(pattern) - _bool_count(pattern, "pattern_available")
        ),
        "pattern_correlation_pass": _bool_count(pattern, "accepted"),
        "topology_evaluated": int(len(topology)),
        "topology_coordinate_rejected": int(
            len(topology) - _bool_count(topology, "accepted")
        ),
        "topology_recheck_evaluated": int(len(recheck)),
        "topology_recheck_rejected": int(
            len(recheck) - _bool_count(recheck, "accepted")
        ),
        "descriptor_updates": _bool_count(stage("descriptor_update"), "accepted"),
        "final_accepted": int(len(stage("final_acceptance"))),
    }


def summarize(candidates: pd.DataFrame, stages: pd.DataFrame) -> pd.DataFrame:
    target_ids = sorted(
        set(candidates["target_image_id"].dropna().astype(int))
        | set(stages["target_image_id"].dropna().astype(int))
    )
    return pd.DataFrame.from_records(
        [summarize_target(target_id, candidates, stages) for target_id in target_ids]
    )


def overall(by_image: pd.DataFrame) -> pd.DataFrame:
    totals = by_image.drop(columns="target_image_id").sum(axis=0)
    rows = [
        {"metric": metric, "value": int(value)} for metric, value in totals.items()
    ]
    candidate_count = float(totals.get("candidate_matches", 0))
    post_interpolation = float(totals.get("post_interpolation", 0))
    if candidate_count:
        rows.extend(
            [
                {
                    "metric": "model_accepted_fraction_of_candidates",
                    "value": totals.get("model_accepted", 0) / candidate_count,
                },
                {
                    "metric": "motion_rejected_fraction_of_candidates",
                    "value": (
                        totals.get("descriptor_pass", 0)
                        - totals.get("motion_pass", 0)
                    )
                    / candidate_count,
                },
            ]
        )
    if post_interpolation:
        rows.append(
            {
                "metric": "final_fraction_of_post_interpolation",
                "value": totals.get("final_accepted", 0) / post_interpolation,
            }
        )
    return pd.DataFrame.from_records(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    audit_dir = args.run_dir / "stage_audit"
    out_dir = args.out_dir or args.run_dir / "stage_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_json(audit_dir / "matcher_candidates.jsonl", lines=True)
    stages = pd.read_json(audit_dir / "trajectory_stages.jsonl", lines=True)
    by_image = summarize(candidates, stages)
    totals = overall(by_image)
    by_image.to_csv(out_dir / "stage_counts_by_image.csv", index=False)
    totals.to_csv(out_dir / "stage_counts_overall.csv", index=False)
    payload = {
        "candidate_records": int(len(candidates)),
        "trajectory_stage_records": int(len(stages)),
        "target_images": int(len(by_image)),
        "definitions": {
            "candidate_matches": "combined cross-check/Lowe candidates entering descriptor and motion filters",
            "model_accepted": "candidates returned as accepted model inliers",
            "post_interpolation": "direct model inliers plus accepted interpolation proposals before velocity/pattern checks",
            "final_accepted": "trajectories returned by _match_existing_points after all checks and descriptor recomputation",
        },
    }
    (out_dir / "summary_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(totals.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
