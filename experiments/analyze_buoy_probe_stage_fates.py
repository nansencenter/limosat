#!/usr/bin/env python3
"""Trace each one-step buoy probe through the instrumented LiMOSAT stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from experiments.analyze_operational_buoy_probe_tracks import (
        DEFAULT_OBSERVATIONS,
        DEFAULT_TRANSITIONS,
        transition_expectations,
    )
except ModuleNotFoundError:  # Direct ``python experiments/<script>.py`` execution.
    from analyze_operational_buoy_probe_tracks import (
        DEFAULT_OBSERVATIONS,
        DEFAULT_TRANSITIONS,
        transition_expectations,
    )


def _error(x: float, y: float, truth_x: float, truth_y: float) -> float:
    if pd.isna(x) or pd.isna(y):
        return np.nan
    return float(np.hypot(float(x) - truth_x, float(y) - truth_y))


def _last_stage(stages: pd.DataFrame, name: str) -> pd.Series | None:
    selected = stages.loc[stages["stage"] == name]
    if selected.empty:
        return None
    return selected.iloc[-1]


def _accepted(row: pd.Series | None) -> bool:
    return bool(row is not None and row.get("accepted", False))


def _outcome(candidates: pd.DataFrame, stages: pd.DataFrame) -> str:
    final = _last_stage(stages, "final_acceptance")
    if _accepted(final):
        return "accepted"

    recheck = _last_stage(stages, "topology_pattern_recheck")
    if recheck is not None and not _accepted(recheck):
        return "topology_pattern_recheck"
    topology = _last_stage(stages, "topology_filter")
    if topology is not None and not _accepted(topology):
        return "topology_filter"
    pattern = _last_stage(stages, "pattern_matching")
    if pattern is not None and not _accepted(pattern):
        if not bool(pattern.get("pattern_available", False)):
            return "pattern_unavailable"
        return "pattern_correlation"
    template = _last_stage(stages, "template_availability")
    if template is not None and not _accepted(template):
        return "template_unavailable"
    if not stages.loc[stages["stage"] == "convergence"].empty:
        return "convergence"
    velocity = _last_stage(stages, "velocity_filter")
    if velocity is not None and not _accepted(velocity):
        return "velocity_filter"
    if _last_stage(stages, "post_interpolation") is not None:
        return "after_interpolation_unclassified"
    if candidates.empty:
        return "no_descriptor_candidate"
    if not candidates["descriptor_pass"].fillna(False).astype(bool).any():
        return "descriptor_distance"
    if not candidates["motion_pass"].fillna(False).astype(bool).any():
        return "motion_gate"
    if not candidates["model_inlier"].fillna(False).astype(bool).any():
        return "model_estimation"
    return "before_interpolation_unclassified"


def trace_one_step_fates(
    expected: pd.DataFrame,
    linkage: pd.DataFrame,
    image_map: pd.DataFrame,
    candidates: pd.DataFrame,
    stages: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per expected transition, retaining missing probes/stages."""
    links = linkage[["probe_id", "trajectory_id"]].copy()
    links["trajectory_id"] = pd.to_numeric(links["trajectory_id"], errors="coerce")
    links = links.dropna(subset=["trajectory_id"])
    links["trajectory_id"] = links["trajectory_id"].astype(int)
    mapping = image_map[["run_image_id", "catalog_image_id"]].rename(
        columns={"run_image_id": "target_run_image_id", "catalog_image_id": "target_image_id"}
    )
    base = expected.merge(links, on="probe_id", how="left", validate="many_to_one").merge(
        mapping, on="target_image_id", how="left", validate="many_to_one"
    )

    records: list[dict] = []
    for row in base.itertuples(index=False):
        truth_x = float(row.target_x)
        truth_y = float(row.target_y)
        if pd.isna(row.trajectory_id) or pd.isna(row.target_run_image_id):
            records.append(
                {
                    **row._asdict(),
                    "source_probe_linked": False,
                    "candidate_generated": False,
                    "final_accepted": False,
                    "outcome_stage": "source_probe_unlinked",
                }
            )
            continue

        trajectory_id = int(row.trajectory_id)
        target_run_image_id = int(row.target_run_image_id)
        candidate_rows = candidates.loc[
            (candidates["trajectory_id"] == trajectory_id)
            & (candidates["target_image_id"] == target_run_image_id)
        ].copy()
        stage_rows = stages.loc[
            (stages["trajectory_id"] == trajectory_id)
            & (stages["target_image_id"] == target_run_image_id)
        ]
        if not candidate_rows.empty:
            candidate_rows["truth_error_m"] = np.hypot(
                candidate_rows["target_x"] - truth_x,
                candidate_rows["target_y"] - truth_y,
            )

        post = _last_stage(stage_rows, "post_interpolation")
        pattern = _last_stage(stage_rows, "pattern_matching")
        topology = _last_stage(stage_rows, "topology_filter")
        final = _last_stage(stage_rows, "final_acceptance")
        convergence = _last_stage(stage_rows, "convergence")
        replacement_trajectory_id = (
            int(convergence["converged_to"])
            if convergence is not None and pd.notna(convergence.get("converged_to"))
            else None
        )
        replacement_final = None
        if replacement_trajectory_id is not None:
            replacement_stages = stages.loc[
                (stages["trajectory_id"] == replacement_trajectory_id)
                & (stages["target_image_id"] == target_run_image_id)
            ]
            replacement_final = _last_stage(replacement_stages, "final_acceptance")
        direct_final_accepted = _accepted(final)
        replacement_final_accepted = _accepted(replacement_final)
        represented_final = final if direct_final_accepted else replacement_final
        accepted_candidates = candidate_rows.loc[
            candidate_rows.get("accepted", pd.Series(False, index=candidate_rows.index))
            .fillna(False)
            .astype(bool)
        ]
        reasons = sorted(
            set(candidate_rows.get("rejection_reason", pd.Series(dtype=object)).dropna())
        )
        record = {
            **row._asdict(),
            "source_probe_linked": True,
            "candidate_generated": not candidate_rows.empty,
            "candidate_count": int(len(candidate_rows)),
            "candidate_best_error_m": (
                float(candidate_rows["truth_error_m"].min())
                if not candidate_rows.empty
                else np.nan
            ),
            "candidate_within_2km": bool(
                not candidate_rows.empty
                and (candidate_rows["truth_error_m"] <= 2000).any()
            ),
            "candidate_motion_pass": bool(
                not candidate_rows.empty
                and candidate_rows["motion_pass"].fillna(False).astype(bool).any()
            ),
            "candidate_model_accepted": not accepted_candidates.empty,
            "model_accepted_best_error_m": (
                float(accepted_candidates["truth_error_m"].min())
                if not accepted_candidates.empty
                else np.nan
            ),
            "candidate_rejection_reasons": "|".join(reasons),
            "post_interpolation_present": post is not None,
            "post_interpolation_error_m": (
                _error(post.get("target_x"), post.get("target_y"), truth_x, truth_y)
                if post is not None
                else np.nan
            ),
            "post_interpolation_was_interpolated": (
                bool(post.get("interpolated", False)) if post is not None else False
            ),
            "pattern_present": pattern is not None,
            "pattern_pre_error_m": (
                _error(pattern.get("pre_pattern_x"), pattern.get("pre_pattern_y"), truth_x, truth_y)
                if pattern is not None
                else np.nan
            ),
            "pattern_corrected_error_m": (
                _error(pattern.get("corrected_x"), pattern.get("corrected_y"), truth_x, truth_y)
                if pattern is not None
                else np.nan
            ),
            "pattern_correlation": (
                float(pattern.get("correlation", np.nan)) if pattern is not None else np.nan
            ),
            "pattern_accepted": _accepted(pattern),
            "topology_present": topology is not None,
            "topology_error_m": (
                _error(topology.get("topology_x"), topology.get("topology_y"), truth_x, truth_y)
                if topology is not None
                else np.nan
            ),
            "topology_accepted": _accepted(topology),
            "final_accepted": direct_final_accepted,
            "final_error_m": (
                _error(final.get("target_x"), final.get("target_y"), truth_x, truth_y)
                if final is not None
                else np.nan
            ),
            "convergence_pruned": convergence is not None,
            "replacement_trajectory_id": replacement_trajectory_id,
            "replacement_final_accepted": replacement_final_accepted,
            "replacement_final_error_m": (
                _error(
                    replacement_final.get("target_x"),
                    replacement_final.get("target_y"),
                    truth_x,
                    truth_y,
                )
                if replacement_final is not None
                else np.nan
            ),
            "measurement_represented_final": bool(
                direct_final_accepted or replacement_final_accepted
            ),
            "measurement_represented_error_m": (
                _error(
                    represented_final.get("target_x"),
                    represented_final.get("target_y"),
                    truth_x,
                    truth_y,
                )
                if represented_final is not None
                else np.nan
            ),
            "outcome_stage": _outcome(candidate_rows, stage_rows),
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_fates(fates: pd.DataFrame) -> dict:
    def count_true(column: str) -> int:
        if column not in fates:
            return 0
        return int(fates[column].fillna(False).astype(bool).sum())

    count = len(fates)
    final_within_2km = 0
    if "final_error_m" in fates:
        final_within_2km = int(
            (
                fates["final_accepted"].fillna(False).astype(bool)
                & (fates["final_error_m"] <= 2000)
            ).sum()
        )
    represented_within_2km = 0
    if "measurement_represented_error_m" in fates:
        represented_within_2km = int(
            (
                fates["measurement_represented_final"].fillna(False).astype(bool)
                & (fates["measurement_represented_error_m"] <= 2000)
            ).sum()
        )
    return {
        "expected_transitions": int(count),
        "source_probe_linked": count_true("source_probe_linked"),
        "candidate_generated": count_true("candidate_generated"),
        "candidate_within_2km": count_true("candidate_within_2km"),
        "post_interpolation_present": count_true("post_interpolation_present"),
        "pattern_accepted": count_true("pattern_accepted"),
        "final_accepted": count_true("final_accepted"),
        "final_within_2km": final_within_2km,
        "convergence_pruned": count_true("convergence_pruned"),
        "replacement_final_accepted": count_true("replacement_final_accepted"),
        "measurement_represented_final": count_true("measurement_represented_final"),
        "measurement_represented_within_2km": represented_within_2km,
        "outcome_stage_counts": {
            str(key): int(value)
            for key, value in fates["outcome_stage"].value_counts(dropna=False).items()
        },
    }


def pattern_threshold_diagnostic(
    fates: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.20, 0.25, 0.275, 0.30, 0.325, 0.35, 0.40, 0.50),
) -> pd.DataFrame:
    """Score the observed pattern-corrected position without rerunning later stages."""
    present = fates.get("pattern_present", pd.Series(False, index=fates.index))
    present = present.fillna(False).astype(bool)
    correlation = pd.to_numeric(
        fates.get("pattern_correlation", pd.Series(np.nan, index=fates.index)),
        errors="coerce",
    )
    error = pd.to_numeric(
        fates.get("pattern_corrected_error_m", pd.Series(np.nan, index=fates.index)),
        errors="coerce",
    )
    rows = []
    for threshold in thresholds:
        accepted = present & (correlation >= threshold) & error.notna()
        accepted_errors = error.loc[accepted]
        rows.append(
            {
                "min_correlation": threshold,
                "expected_transitions": int(len(fates)),
                "pattern_positions_retained": int(accepted.sum()),
                "retained_fraction_all": float(accepted.mean()) if len(fates) else np.nan,
                "within_2km_fraction_all": float((accepted & (error <= 2000)).mean()) if len(fates) else np.nan,
                "within_2km_fraction_retained": float((accepted_errors <= 2000).mean()) if len(accepted_errors) else np.nan,
                "within_5km_fraction_retained": float((accepted_errors <= 5000).mean()) if len(accepted_errors) else np.nan,
                "catastrophic_50km_fraction_retained": float((accepted_errors > 50000).mean()) if len(accepted_errors) else np.nan,
                "median_retained_error_m": float(accepted_errors.median()) if len(accepted_errors) else np.nan,
                "p90_retained_error_m": float(accepted_errors.quantile(0.9)) if len(accepted_errors) else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--split", default="development")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir or args.run_dir / "buoy_probe_stage_fates"
    out_dir.mkdir(parents=True, exist_ok=True)
    observations = pd.read_csv(args.observations, low_memory=False)
    transitions = pd.read_csv(args.transitions, low_memory=False)
    expected = transition_expectations(transitions, observations, args.split)
    linkage = pd.read_csv(args.run_dir / "buoy_probe_linkage.csv", low_memory=False)
    image_map = pd.read_csv(args.run_dir / "image_timings.csv")
    audit_dir = args.run_dir / "stage_audit"
    candidates = pd.read_json(audit_dir / "matcher_candidates.jsonl", lines=True)
    stages = pd.read_json(audit_dir / "trajectory_stages.jsonl", lines=True)
    fates = trace_one_step_fates(expected, linkage, image_map, candidates, stages)
    summary = summarize_fates(fates)
    fates.to_csv(out_dir / "one_step_stage_fates.csv", index=False)
    pattern_threshold_diagnostic(fates).to_csv(
        out_dir / "pattern_threshold_diagnostic.csv", index=False
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
