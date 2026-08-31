#!/usr/bin/env python3
"""Paired comparison of two dense-field buoy evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = {
    "nearest_10km": ("nearest_endpoint_error_m", "nearest_source_distance_m", 10_000.0),
    "local_average_10km": ("local_average_10km_endpoint_error_m", None, None),
    "local_average_50km": ("local_average_50km_endpoint_error_m", None, None),
}
METADATA = (
    "transition_id",
    "buoy_id",
    "source_image_id",
    "target_image_id",
    "elapsed_hours",
    "cadence_band",
    "month",
)


def compare(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    correct_threshold_m: float = 2_000.0,
) -> pd.DataFrame:
    """Return one paired row per transition with explicit missing outcomes."""
    for label, frame in (("baseline", baseline), ("candidate", candidate)):
        if frame["transition_id"].duplicated().any():
            raise ValueError(f"{label} transition IDs must be unique")
    if set(baseline["transition_id"]) != set(candidate["transition_id"]):
        raise ValueError("Baseline and candidate must contain identical transitions")

    required = set(METADATA)
    for error_column, distance_column, _ in METHODS.values():
        required.add(error_column)
        if distance_column is not None:
            required.add(distance_column)
    missing = required - set(baseline.columns) | required - set(candidate.columns)
    if missing:
        raise ValueError(f"Dense evaluations lack columns: {sorted(missing)}")

    paired = baseline[list(required)].merge(
        candidate[list(required)],
        on="transition_id",
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    for column in METADATA[1:]:
        baseline_column = f"{column}_baseline"
        candidate_column = f"{column}_candidate"
        if not paired[baseline_column].equals(paired[candidate_column]):
            raise ValueError(f"Transition metadata differs for {column}")
        paired[column] = paired.pop(baseline_column)
        paired.drop(columns=candidate_column, inplace=True)

    for method, (error_column, distance_column, maximum_distance) in METHODS.items():
        for run_label in ("baseline", "candidate"):
            error = paired[f"{error_column}_{run_label}"]
            available = error.notna()
            if distance_column is not None:
                available &= (
                    paired[f"{distance_column}_{run_label}"] <= maximum_distance
                )
            paired[f"{method}_{run_label}_available"] = available
            paired[f"{method}_{run_label}_correct"] = available & (
                error <= correct_threshold_m
            )
        baseline_correct = paired[f"{method}_baseline_correct"]
        candidate_correct = paired[f"{method}_candidate_correct"]
        paired[f"{method}_correct_change"] = np.select(
            [~baseline_correct & candidate_correct, baseline_correct & ~candidate_correct],
            ["gain", "loss"],
            default="unchanged",
        )
    return paired.sort_values("transition_id", kind="stable").reset_index(drop=True)


def summarize_group(group: pd.DataFrame, stratum: str, value: str) -> list[dict]:
    rows = []
    for method in METHODS:
        baseline_available = group[f"{method}_baseline_available"]
        candidate_available = group[f"{method}_candidate_available"]
        baseline_correct = group[f"{method}_baseline_correct"]
        candidate_correct = group[f"{method}_candidate_correct"]
        change = group[f"{method}_correct_change"]
        rows.append(
            {
                "stratum": stratum,
                "stratum_value": value,
                "method": method,
                "expected": int(len(group)),
                "baseline_available": int(baseline_available.sum()),
                "candidate_available": int(candidate_available.sum()),
                "availability_delta": int(
                    candidate_available.sum() - baseline_available.sum()
                ),
                "baseline_correct": int(baseline_correct.sum()),
                "candidate_correct": int(candidate_correct.sum()),
                "correct_delta": int(candidate_correct.sum() - baseline_correct.sum()),
                "correct_gains": int((change == "gain").sum()),
                "correct_losses": int((change == "loss").sum()),
            }
        )
    return rows


def summarize(paired: pd.DataFrame) -> pd.DataFrame:
    rows = summarize_group(paired, "all", "all")
    for column in ("cadence_band", "month"):
        for value, group in paired.groupby(column, dropna=False, sort=True):
            rows.extend(summarize_group(group, column, str(value)))
    return pd.DataFrame.from_records(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--correct-threshold-m", type=float, default=2_000.0)
    args = parser.parse_args()

    paired = compare(
        pd.read_csv(args.baseline),
        pd.read_csv(args.candidate),
        correct_threshold_m=args.correct_threshold_m,
    )
    summary = summarize(paired)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.out_dir / "paired_transitions.csv", index=False)
    summary.to_csv(args.out_dir / "paired_summary.csv", index=False)
    (args.out_dir / "comparison_manifest.json").write_text(
        json.dumps(
            {
                "baseline": str(args.baseline.resolve()),
                "candidate": str(args.candidate.resolve()),
                "correct_threshold_m": args.correct_threshold_m,
                "expected_transitions": int(len(paired)),
                "missing_predictions_retained_in_denominators": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.loc[summary["stratum"] == "all"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
