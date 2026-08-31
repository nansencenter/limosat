#!/usr/bin/env python3
"""Trace operational dense points near development buoys through one image pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from experiments.analyze_buoy_probe_stage_fates import _last_stage, _outcome
    from experiments.analyze_dense_field_at_buoys import (
        DEFAULT_OBSERVATIONS,
        DEFAULT_TRANSITIONS,
        load_points,
        transition_truth,
    )
except ModuleNotFoundError:  # Direct ``python experiments/<script>.py`` execution.
    from analyze_buoy_probe_stage_fates import _last_stage, _outcome
    from analyze_dense_field_at_buoys import (
        DEFAULT_OBSERVATIONS,
        DEFAULT_TRANSITIONS,
        load_points,
        transition_truth,
    )


def displacement_error(
    source_point_x: float,
    source_point_y: float,
    target_point_x: float,
    target_point_y: float,
    buoy_source_x: float,
    buoy_source_y: float,
    buoy_target_x: float,
    buoy_target_y: float,
) -> float:
    predicted_x = buoy_source_x + target_point_x - source_point_x
    predicted_y = buoy_source_y + target_point_y - source_point_y
    return float(np.hypot(predicted_x - buoy_target_x, predicted_y - buoy_target_y))


def trace_local_dense_points(
    truth: pd.DataFrame,
    source_points: pd.DataFrame,
    candidates: pd.DataFrame,
    stages: pd.DataFrame,
    target_run_image_id: int,
    local_radius_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_records = []
    buoy_records = []
    for transition in truth.itertuples(index=False):
        distances = np.hypot(
            source_points["x"] - transition.source_x,
            source_points["y"] - transition.source_y,
        )
        local_points = source_points.loc[distances <= local_radius_m].copy()
        local_points["source_distance_m"] = distances.loc[local_points.index]
        local_records = []
        for point in local_points.itertuples(index=False):
            candidate_rows = candidates.loc[
                candidates["trajectory_id"].eq(int(point.trajectory_id))
                & candidates["target_image_id"].eq(target_run_image_id)
            ].copy()
            stage_rows = stages.loc[
                stages["trajectory_id"].eq(int(point.trajectory_id))
                & stages["target_image_id"].eq(target_run_image_id)
            ]
            candidate_errors = []
            for candidate in candidate_rows.itertuples(index=False):
                candidate_errors.append(
                    displacement_error(
                        point.x,
                        point.y,
                        candidate.target_x,
                        candidate.target_y,
                        transition.source_x,
                        transition.source_y,
                        transition.target_x,
                        transition.target_y,
                    )
                )
            post = _last_stage(stage_rows, "post_interpolation")
            pattern = _last_stage(stage_rows, "pattern_matching")
            final = _last_stage(stage_rows, "final_acceptance")

            def stage_error(stage: pd.Series | None, x_name: str, y_name: str) -> float:
                if stage is None or pd.isna(stage.get(x_name)) or pd.isna(stage.get(y_name)):
                    return np.nan
                return displacement_error(
                    point.x,
                    point.y,
                    float(stage[x_name]),
                    float(stage[y_name]),
                    transition.source_x,
                    transition.source_y,
                    transition.target_x,
                    transition.target_y,
                )

            record = {
                "transition_id": transition.transition_id,
                "buoy_id": transition.buoy_id,
                "source_image_id": transition.source_image_id,
                "target_image_id": transition.target_image_id,
                "trajectory_id": int(point.trajectory_id),
                "source_point_x": float(point.x),
                "source_point_y": float(point.y),
                "source_distance_m": float(point.source_distance_m),
                "candidate_generated": not candidate_rows.empty,
                "candidate_best_displacement_error_m": min(candidate_errors)
                if candidate_errors
                else np.nan,
                "candidate_motion_pass": bool(
                    not candidate_rows.empty
                    and candidate_rows["motion_pass"].fillna(False).astype(bool).any()
                ),
                "candidate_model_accepted": bool(
                    not candidate_rows.empty
                    and candidate_rows["accepted"].fillna(False).astype(bool).any()
                ),
                "post_interpolation_present": post is not None,
                "post_interpolation_was_interpolated": bool(
                    post is not None and post.get("interpolated", False)
                ),
                "post_interpolation_displacement_error_m": stage_error(
                    post, "target_x", "target_y"
                ),
                "pattern_present": pattern is not None,
                "pattern_correlation": float(pattern.get("correlation", np.nan))
                if pattern is not None
                else np.nan,
                "pattern_displacement_error_m": stage_error(
                    pattern, "corrected_x", "corrected_y"
                ),
                "final_accepted": bool(final is not None and final.get("accepted", False)),
                "final_displacement_error_m": stage_error(final, "target_x", "target_y"),
                "outcome_stage": _outcome(candidate_rows, stage_rows),
            }
            local_records.append(record)
            point_records.append(record)

        local_frame = pd.DataFrame.from_records(local_records)
        source_count = len(local_frame)
        candidate_count = (
            int(local_frame["candidate_generated"].sum()) if source_count else 0
        )
        final_count = int(local_frame["final_accepted"].sum()) if source_count else 0
        final_errors = (
            local_frame.loc[local_frame["final_accepted"], "final_displacement_error_m"]
            if source_count
            else pd.Series(dtype=float)
        )
        if source_count == 0:
            availability_fate = "no_source_point_within_radius"
        elif final_count == 0:
            availability_fate = "source_points_present_but_none_survived"
        else:
            availability_fate = "surviving_local_vector"
        buoy_records.append(
            {
                "transition_id": transition.transition_id,
                "buoy_id": transition.buoy_id,
                "source_image_id": transition.source_image_id,
                "target_image_id": transition.target_image_id,
                "source_points_within_radius": source_count,
                "source_points_with_candidate": candidate_count,
                "surviving_local_vectors": final_count,
                "best_final_displacement_error_m": float(final_errors.min())
                if len(final_errors)
                else np.nan,
                "has_final_vector_within_2km": bool(
                    len(final_errors) and (final_errors <= 2000).any()
                ),
                "availability_fate": availability_fate,
            }
        )
    return pd.DataFrame.from_records(point_records), pd.DataFrame.from_records(buoy_records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--split", default="development")
    parser.add_argument("--source-image-id", type=int, required=True)
    parser.add_argument("--target-image-id", type=int, required=True)
    parser.add_argument("--local-radius-m", type=float, default=10000.0)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.run_dir / "run_manifest.json").read_text())
    if not manifest.get("stage_instrumentation_enabled"):
        raise ValueError("Stage instrumentation is required")
    database_url = manifest["engine_url"]
    if not database_url.startswith("sqlite:///"):
        raise ValueError(f"Unsupported database URL: {database_url}")
    database_path = Path(database_url[len("sqlite:///") :])
    table = manifest["effective_run_name"]

    observations = pd.read_csv(args.observations, low_memory=False)
    transitions = pd.read_csv(args.transitions, low_memory=False)
    truth = transition_truth(transitions, observations, args.split)
    truth = truth.loc[
        truth["source_image_id"].eq(args.source_image_id)
        & truth["target_image_id"].eq(args.target_image_id)
    ]
    image_map = pd.read_csv(args.run_dir / "image_timings.csv")
    catalog_to_run = dict(
        zip(image_map["catalog_image_id"], image_map["run_image_id"], strict=True)
    )
    source_run_id = int(catalog_to_run[args.source_image_id])
    target_run_id = int(catalog_to_run[args.target_image_id])
    points = load_points(database_path, table)
    source_points = points.loc[points["image_id"].eq(source_run_id)]
    audit_dir = args.run_dir / "stage_audit"
    candidates = pd.read_json(audit_dir / "matcher_candidates.jsonl", lines=True)
    stages = pd.read_json(audit_dir / "trajectory_stages.jsonl", lines=True)
    point_fates, buoy_summary = trace_local_dense_points(
        truth,
        source_points,
        candidates,
        stages,
        target_run_id,
        args.local_radius_m,
    )

    output_dir = args.out_dir or (
        args.run_dir
        / f"dense_near_buoy_stage_fates_{args.source_image_id}_{args.target_image_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    point_fates.to_csv(output_dir / "local_point_fates.csv", index=False)
    buoy_summary.to_csv(output_dir / "buoy_summary.csv", index=False)
    payload = {
        "split": args.split,
        "source_image_id": args.source_image_id,
        "target_image_id": args.target_image_id,
        "local_radius_m": args.local_radius_m,
        "expected_buoy_transitions": int(len(buoy_summary)),
        "source_point_instances": int(len(point_fates)),
        "buoy_availability_fates": {
            str(key): int(value)
            for key, value in buoy_summary["availability_fate"].value_counts().items()
        },
        "source_point_outcome_stages": {
            str(key): int(value)
            for key, value in point_fates["outcome_stage"].value_counts().items()
        },
        "buoys_with_vector_within_2km": int(
            buoy_summary["has_final_vector_within_2km"].sum()
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
