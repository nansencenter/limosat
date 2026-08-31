#!/usr/bin/env python3
"""Summarize the frozen north-up ALIKED versus ORB development comparisons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_COLUMNS = ["source_image_id", "target_image_id"]


def add_method_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["aliked_average_error_m"] = result["aliked_pre_pattern_error_m"]
    result["aliked_highest_confidence_error_m"] = np.hypot(
        result["aliked_highest_confidence_dx_m"] - result["truth_dx_m"],
        result["aliked_highest_confidence_dy_m"] - result["truth_dy_m"],
    )
    result["aliked_nearest_tight_error_m"] = np.hypot(
        result["aliked_nearest_tight_dx_m"] - result["truth_dx_m"],
        result["aliked_nearest_tight_dy_m"] - result["truth_dy_m"],
    )
    availability = {
        "aliked_average": result["aliked_available"].fillna(False),
        "aliked_highest_confidence": result["aliked_available"].fillna(False),
        "aliked_nearest_tight": result["aliked_nearest_tight_available"].fillna(
            False
        ),
        "aliked_pattern": result["aliked_pm_accepted"].fillna(False),
        "orb_proposal": result["orb_available"].fillna(False)
        & result["orb_motion_pass"].fillna(False),
        "orb_pattern": result["orb_pm_accepted"].fillna(False),
    }
    error_columns = {
        "aliked_average": "aliked_average_error_m",
        "aliked_highest_confidence": "aliked_highest_confidence_error_m",
        "aliked_nearest_tight": "aliked_nearest_tight_error_m",
        "aliked_pattern": "aliked_pm_error_m",
        "orb_proposal": "orb_pre_pattern_error_m",
        "orb_pattern": "orb_pm_error_m",
    }
    for method, available in availability.items():
        result[f"{method}_available"] = available.astype(bool)
        result[f"{method}_correct"] = available & result[
            error_columns[method]
        ].le(2000.0)
    return result


def method_summary(frame: pd.DataFrame, panel: str, unit: str) -> pd.DataFrame:
    rows = []
    methods = (
        "aliked_average",
        "aliked_highest_confidence",
        "aliked_nearest_tight",
        "aliked_pattern",
        "orb_proposal",
        "orb_pattern",
    )
    for method in methods:
        available = frame[f"{method}_available"]
        correct = frame[f"{method}_correct"]
        error = frame[
            {
                "aliked_average": "aliked_average_error_m",
                "aliked_highest_confidence": "aliked_highest_confidence_error_m",
                "aliked_nearest_tight": "aliked_nearest_tight_error_m",
                "aliked_pattern": "aliked_pm_error_m",
                "orb_proposal": "orb_pre_pattern_error_m",
                "orb_pattern": "orb_pm_error_m",
            }[method]
        ]
        rows.append(
            {
                "panel": panel,
                "sampling_unit": unit,
                "method": method,
                "cases": len(frame),
                "available": int(available.sum()),
                "correct_within_2km": int(correct.sum()),
                "availability_percent": 100.0 * available.mean(),
                "correct_percent": 100.0 * correct.mean(),
                "median_error_m_when_available": error.loc[available].median(),
                "p90_error_m_when_available": error.loc[available].quantile(0.90),
            }
        )
    return pd.DataFrame(rows)


def paired_exact_p_value(gains: int, losses: int) -> float | None:
    discordant = gains + losses
    if not discordant:
        return None
    tail = sum(math.comb(discordant, value) for value in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / 2**discordant)


def clustered_bootstrap_difference(
    frame: pd.DataFrame,
    left: str,
    right: str,
    seed: int = 20260817,
    iterations: int = 20000,
) -> tuple[float, float, float]:
    by_pair = (
        frame.assign(
            difference=(
                frame[f"{left}_correct"].astype(float)
                - frame[f"{right}_correct"].astype(float)
            )
        )
        .groupby(PAIR_COLUMNS, sort=True)["difference"]
        .mean()
        .to_numpy()
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(by_pair, size=(iterations, len(by_pair)), replace=True).mean(
        axis=1
    )
    return (
        float(100.0 * by_pair.mean()),
        float(100.0 * np.quantile(draws, 0.025)),
        float(100.0 * np.quantile(draws, 0.975)),
    )


def paired_summary(
    frame: pd.DataFrame, panel: str, left: str, right: str
) -> dict:
    left_correct = frame[f"{left}_correct"]
    right_correct = frame[f"{right}_correct"]
    gains = int((left_correct & ~right_correct).sum())
    losses = int((~left_correct & right_correct).sum())
    difference, lower, upper = clustered_bootstrap_difference(frame, left, right)
    return {
        "panel": panel,
        "left": left,
        "right": right,
        "cases": int(len(frame)),
        "unique_image_pairs": int(frame.groupby(PAIR_COLUMNS).ngroups),
        "left_only_correct": gains,
        "right_only_correct": losses,
        "both_correct": int((left_correct & right_correct).sum()),
        "neither_correct": int((~left_correct & ~right_correct).sum()),
        "unweighted_case_difference_percentage_points": float(
            100.0 * (left_correct.mean() - right_correct.mean())
        ),
        "equal_image_pair_difference_percentage_points": difference,
        "cluster_bootstrap_95_percent_interval_percentage_points": [lower, upper],
        "exact_paired_p_value": paired_exact_p_value(gains, losses),
    }


def strata_summary(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["rotation_bin"] = pd.cut(
        working["absolute_native_rotation_difference_degrees"],
        [-0.01, 5.0, 20.0, 181.0],
        labels=["00_to_05deg", "05_to_20deg", "over_20deg"],
    )
    rows = []
    for stratum, column in (
        ("rotation", "rotation_bin"),
        ("cadence", "cadence_band"),
        ("month", "month"),
    ):
        for value, group in working.groupby(column, observed=True, dropna=False):
            rows.append(
                {
                    "stratum": stratum,
                    "value": str(value),
                    "cases": len(group),
                    "aliked_average_correct": int(
                        group["aliked_average_correct"].sum()
                    ),
                    "aliked_highest_confidence_correct": int(
                        group["aliked_highest_confidence_correct"].sum()
                    ),
                    "orb_proposal_correct": int(group["orb_proposal_correct"].sum()),
                    "orb_pattern_correct": int(group["orb_pattern_correct"].sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-results", type=Path, required=True)
    parser.add_argument("--pair-block-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair = add_method_outcomes(pd.read_csv(args.pair_results, low_memory=False))
    pair = pair.loc[pair["representative_panel"]].copy()
    pair_block = add_method_outcomes(
        pd.read_csv(args.pair_block_results, low_memory=False)
    )
    pair_block = pair_block.loc[pair_block["representative_panel"]].copy()

    methods = pd.concat(
        [
            method_summary(pair, "primary", "image_pair"),
            method_summary(
                pair_block, "spatial_sensitivity", "image_pair_x_200km_block"
            ),
        ],
        ignore_index=True,
    )
    methods.to_csv(args.output_dir / "method_summary.csv", index=False)
    strata_summary(pair).to_csv(args.output_dir / "primary_strata.csv", index=False)

    comparisons = []
    for panel_name, frame in (("primary", pair), ("spatial_sensitivity", pair_block)):
        for left, right in (
            ("aliked_average", "orb_proposal"),
            ("aliked_highest_confidence", "orb_proposal"),
            ("aliked_average", "orb_pattern"),
            ("aliked_highest_confidence", "orb_pattern"),
        ):
            comparisons.append(paired_summary(frame, panel_name, left, right))
    (args.output_dir / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )

    pattern_available = pair["aliked_average_available"]
    pattern_effect = {
        "proposal_available": int(pattern_available.sum()),
        "proposal_correct": int(pair["aliked_average_correct"].sum()),
        "pattern_accepted": int(pair["aliked_pattern_available"].sum()),
        "pattern_correct": int(pair["aliked_pattern_correct"].sum()),
        "correct_proposals_lost_or_made_incorrect": int(
            (pair["aliked_average_correct"] & ~pair["aliked_pattern_correct"]).sum()
        ),
        "incorrect_proposals_recovered": int(
            (~pair["aliked_average_correct"] & pair["aliked_pattern_correct"]).sum()
        ),
    }
    timing = {
        "primary_cases": len(pair),
        "target_tiles_total": int(pair["target_tiles"].sum()),
        "median_case_total_seconds": float(pair["case_total_seconds"].median()),
        "p90_case_total_seconds": float(pair["case_total_seconds"].quantile(0.90)),
        "total_case_seconds": float(pair["case_total_seconds"].sum()),
        "median_resampling_seconds": float(
            pair["north_up_resampling_seconds"].median()
        ),
        "median_feature_extraction_seconds": float(
            pair["aliked_extraction_seconds"].median()
        ),
        "median_lightglue_seconds": float(pair["aliked_matching_seconds"].median()),
    }
    summary = {
        "status": "complete",
        "correct_threshold_m": 2000.0,
        "primary_sampling": "one deterministic transition per 39 acquisition pairs",
        "spatial_sensitivity_sampling": "one deterministic transition per acquisition-pair x 200 km source block; clustered by the same 39 pairs",
        "pattern_effect": pattern_effect,
        "timing": timing,
        "comparisons": comparisons,
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
