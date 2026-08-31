#!/usr/bin/env python3
"""Consolidate full-70 graph results, including strict unseen-buoy subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from experiments.orb_multiframe_graph import summarize
except ModuleNotFoundError:  # Direct execution from the experiments directory.
    from orb_multiframe_graph import summarize  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "results/orb_multiframe_graph/full70_level1"

SPLIT_DIRECTORIES = {
    "development": "development_border128",
    "validation": "validation_border128",
    "evaluation": "evaluation_border128",
    "season_edge_evaluation": "season_edge_evaluation_border128",
}


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1"})


def expected_transition_targets(coincidences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trajectory_id, group in coincidences.groupby("experiment_trajectory_id"):
        ordered = group.sort_values(["image_time", "image_id"])
        for target in ordered.iloc[1:].itertuples(index=False):
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "image_id": int(target.image_id),
                    "month_exclusive_buoy": bool(target.month_exclusive_buoy),
                }
            )
    return pd.DataFrame.from_records(rows)


def paired_effects(
    records: pd.DataFrame,
    coincidences: pd.DataFrame,
    baseline_config: str = "greedy_rolling",
) -> pd.DataFrame:
    expected = expected_transition_targets(coincidences)
    valid = records[
        records["status"].eq("ok") & records["observation_index"].gt(0)
    ][["config", "trajectory_id", "image_id", "endpoint_error_m"]]
    tables = []
    for config in records["config"].drop_duplicates():
        method = valid[valid["config"].eq(config)].rename(
            columns={"endpoint_error_m": "method_error_m"}
        )
        baseline = valid[valid["config"].eq(baseline_config)].rename(
            columns={"endpoint_error_m": "baseline_error_m"}
        )
        joined = expected.merge(
            baseline[["trajectory_id", "image_id", "baseline_error_m"]],
            on=["trajectory_id", "image_id"],
            how="left",
        ).merge(
            method[["trajectory_id", "image_id", "method_error_m"]],
            on=["trajectory_id", "image_id"],
            how="left",
        )
        for subset, subset_rows in (
            ("all_temporal", joined),
            ("strict_month_exclusive_buoy", joined[joined["month_exclusive_buoy"]]),
        ):
            baseline_success = subset_rows["baseline_error_m"].le(2000.0)
            method_success = subset_rows["method_error_m"].le(2000.0)
            tables.append(
                {
                    "evaluation_subset": subset,
                    "config": config,
                    "baseline_config": baseline_config,
                    "expected_transitions": len(subset_rows),
                    "rescued_within_2km": int((~baseline_success & method_success).sum()),
                    "regressed_from_within_2km": int((baseline_success & ~method_success).sum()),
                    "net_within_2km_change": int(method_success.sum() - baseline_success.sum()),
                }
            )
    return pd.DataFrame.from_records(tables)


def summarize_split(
    records: pd.DataFrame,
    coincidences: pd.DataFrame,
    split: str,
    invalid_support_records: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = []
    for subset, subset_records in (
        ("all_temporal", records),
        (
            "strict_month_exclusive_buoy",
            records[
                records["trajectory_id"].isin(
                    coincidences.loc[
                        as_bool(coincidences["month_exclusive_buoy"]),
                        "experiment_trajectory_id",
                    ]
                )
            ],
        ),
    ):
        if subset_records.empty:
            continue
        table = summarize(subset_records)
        table.insert(0, "evaluation_subset", subset)
        table.insert(0, "experiment_split", split)
        table["invalid_support_observations_before_tracking"] = invalid_support_records
        tables.append(table)
    effects = paired_effects(records, coincidences)
    effects.insert(0, "experiment_split", split)
    return pd.concat(tables, ignore_index=True), effects


def write_report(path: Path, comparison: pd.DataFrame, effects: pd.DataFrame) -> None:
    view = comparison.copy()
    view["median_error_km"] = view["median_error_m"] / 1000.0
    columns = [
        "experiment_split",
        "evaluation_subset",
        "config",
        "descriptor_memory",
        "tracked_transitions",
        "eligible_transitions",
        "median_error_km",
        "within_2km_fraction_all",
        "catastrophic_50km_fraction_all",
    ]
    table = view[columns].copy()
    for column in table.select_dtypes(include=["float"]).columns:
        table[column] = table[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in table.to_numpy()],
    ]
    effect_view = effects[
        [
            "experiment_split",
            "evaluation_subset",
            "config",
            "expected_transitions",
            "rescued_within_2km",
            "regressed_from_within_2km",
            "net_within_2km_change",
        ]
    ]
    effect_lines = [
        "| " + " | ".join(effect_view.columns) + " |",
        "| " + " | ".join(["---"] * len(effect_view.columns)) + " |",
        *[
            "| " + " | ".join(map(str, row)) + " |"
            for row in effect_view.to_numpy()
        ],
    ]
    path.write_text(
        "# Full-70 Level-1 ORB graph comparison\n\n"
        "All fractions retain untracked transitions in the denominator. March is "
        "development, February is validation, January is temporal evaluation, and "
        "April is the season-edge/high-cadence evaluation. `strict_month_exclusive_buoy` "
        "also removes buoy identity overlap between months.\n\n"
        "## Absolute results\n\n"
        + "\n".join(lines)
        + "\n\n## Paired change from previous-selected-descriptor greedy baseline\n\n"
        + "\n".join(effect_lines)
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    args = parser.parse_args()
    comparisons = []
    effects = []
    for split, directory in SPLIT_DIRECTORIES.items():
        run_dir = args.results_root / directory
        records = pd.read_csv(run_dir / "trajectory_results.csv", low_memory=False)
        coincidences = pd.read_csv(run_dir / "coincidences.csv", low_memory=False)
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        comparison, paired = summarize_split(
            records,
            coincidences,
            split,
            int(manifest["invalid_mask_records"]),
        )
        comparisons.append(comparison)
        effects.append(paired)
    comparison = pd.concat(comparisons, ignore_index=True)
    paired = pd.concat(effects, ignore_index=True)
    comparison.to_csv(args.results_root / "comparison.csv", index=False)
    paired.to_csv(args.results_root / "paired_effects.csv", index=False)
    write_report(args.results_root / "report.md", comparison, paired)
    payload = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "splits": list(SPLIT_DIRECTORIES),
        "evaluation_subsets": comparison["evaluation_subset"].drop_duplicates().tolist(),
        "configs": comparison["config"].drop_duplicates().tolist(),
        "denominator_policy": "all eligible transitions, including untracked",
        "baseline_config": "greedy_rolling",
    }
    (args.results_root / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
