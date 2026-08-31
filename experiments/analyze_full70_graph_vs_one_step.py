#!/usr/bin/env python3
"""Attribute graph failures against exact-buoy one-step ORB retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESCRIPTOR_DIR = (
    ROOT / "results/buoy_descriptor_benchmark/full70_level1_border128"
)
DEFAULT_GRAPH_ROOT = ROOT / "results/orb_multiframe_graph/full70_level1"
GRAPH_RUNS = {
    "development": "development_border128",
    "validation": "validation_border128",
    "evaluation": "evaluation_border128",
    "season_edge_evaluation": "season_edge_evaluation_border128",
}


def outcome_label(one_step_error_m: float, graph_error_m: float) -> str:
    one_step_success = np.isfinite(one_step_error_m) and one_step_error_m <= 2000.0
    graph_success = np.isfinite(graph_error_m) and graph_error_m <= 2000.0
    if one_step_success and graph_success:
        return "both_within_2km"
    if one_step_success:
        return "one_step_succeeds_graph_fails"
    if graph_success:
        return "graph_rescues_one_step_failure"
    return "both_fail_2km"


def graph_transition_table(records: pd.DataFrame, config: str) -> pd.DataFrame:
    selected = records[records["config"].eq(config)].copy()
    valid_paths = selected[selected["status"].isin({"ok", "skipped"})].copy()
    valid_paths = valid_paths.sort_values(["trajectory_id", "observation_index"])
    valid_paths["previous_selected_error_m"] = valid_paths.groupby(
        "trajectory_id"
    )["endpoint_error_m"].shift(1)
    valid_paths["previous_descriptor_updated"] = (
        valid_paths.groupby("trajectory_id")["descriptor_updated"].shift(1).fillna(False)
    )
    transitions = valid_paths[valid_paths["observation_index"].gt(0)].copy()
    transitions["previous_false_update"] = (
        transitions["previous_descriptor_updated"].astype(bool)
        & transitions["previous_selected_error_m"].gt(2000.0)
    )
    return transitions[
        [
            "trajectory_id",
            "image_id",
            "endpoint_error_m",
            "status",
            "descriptor_updated",
            "previous_selected_error_m",
            "previous_descriptor_updated",
            "previous_false_update",
        ]
    ].rename(columns={"endpoint_error_m": "graph_error_m"})


def build_attribution(
    pairs: pd.DataFrame,
    retrieval: pd.DataFrame,
    graph_records: pd.DataFrame,
    configs: tuple[str, ...],
) -> pd.DataFrame:
    one_step = retrieval[
        retrieval["method"].eq("orb_geo_hamming")
        & retrieval["gate"].eq("physics_50km_day")
        & retrieval["accepted"].fillna(False)
    ][
        [
            "pair_id",
            "endpoint_error_m",
            "normalized_truth_descriptor_distance",
            "truth_descriptor_rank",
            "candidate_quantization_error_m",
            "source_local_mean",
            "target_local_mean",
            "source_local_std",
            "target_local_std",
        ]
    ].rename(columns={"endpoint_error_m": "one_step_error_m"})
    expected = pairs[
        [
            "pair_id",
            "trajectory_id",
            "target_image_id",
            "source_experiment_split",
            "source_month_exclusive_buoy",
            "dt_hours",
        ]
    ].merge(one_step, on="pair_id", how="left", validate="one_to_one")
    tables = []
    for config in configs:
        graph = graph_transition_table(graph_records, config)
        table = expected.merge(
            graph,
            left_on=["trajectory_id", "target_image_id"],
            right_on=["trajectory_id", "image_id"],
            how="left",
            validate="one_to_one",
        )
        table.insert(0, "config", config)
        table["outcome"] = [
            outcome_label(one_step, graph_error)
            for one_step, graph_error in zip(
                table["one_step_error_m"], table["graph_error_m"]
            )
        ]
        table["local_mean_absolute_change"] = (
            table["target_local_mean"] - table["source_local_mean"]
        ).abs()
        table["local_std_absolute_change"] = (
            table["target_local_std"] - table["source_local_std"]
        ).abs()
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def summarize_attribution(attribution: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strict = (
        attribution["source_month_exclusive_buoy"].astype(str).str.lower().isin({"true", "1"})
    )
    for (config, split), split_rows in attribution.groupby(
        ["config", "source_experiment_split"], sort=False
    ):
        for subset, group in (
            ("all_temporal", split_rows),
            (
                "strict_month_exclusive_buoy",
                split_rows[strict.loc[split_rows.index]],
            ),
        ):
            counts = group["outcome"].value_counts()
            rows.append(
                {
                    "config": config,
                    "experiment_split": split,
                    "evaluation_subset": subset,
                    "transitions": len(group),
                    "both_within_2km": int(counts.get("both_within_2km", 0)),
                    "one_step_succeeds_graph_fails": int(
                        counts.get("one_step_succeeds_graph_fails", 0)
                    ),
                    "graph_rescues_one_step_failure": int(
                        counts.get("graph_rescues_one_step_failure", 0)
                    ),
                    "both_fail_2km": int(counts.get("both_fail_2km", 0)),
                    "previous_false_updates": int(
                        group["previous_false_update"].fillna(False).sum()
                    ),
                    "failures_after_previous_false_update": int(
                        (
                            group["previous_false_update"].fillna(False)
                            & ~group["graph_error_m"].le(2000.0)
                        ).sum()
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-dir", type=Path, default=DEFAULT_DESCRIPTOR_DIR)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument(
        "--configs",
        default="greedy_rolling,beam_anchor,beam_confidence_update_m032",
    )
    args = parser.parse_args()
    pairs = pd.read_csv(args.descriptor_dir / "pairs.csv", low_memory=False)
    retrieval = pd.read_csv(
        args.descriptor_dir / "retrieval_results.csv", low_memory=False
    )
    graph_tables = []
    for split, directory in GRAPH_RUNS.items():
        table = pd.read_csv(
            args.graph_root / directory / "trajectory_results.csv", low_memory=False
        )
        table["experiment_split"] = split
        graph_tables.append(table)
    graph_records = pd.concat(graph_tables, ignore_index=True)
    configs = tuple(item.strip() for item in args.configs.split(",") if item.strip())
    attribution = build_attribution(pairs, retrieval, graph_records, configs)
    summary = summarize_attribution(attribution)
    attribution.to_csv(
        args.graph_root / "one_step_graph_transition_attribution.csv", index=False
    )
    summary.to_csv(
        args.graph_root / "one_step_graph_attribution_summary.csv", index=False
    )
    payload = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "transitions": int(pairs["pair_id"].nunique()),
        "configs": list(configs),
        "one_step_contract": "exact buoy ORB, geographic angle, Hamming, 50 km/day gate",
        "graph_truth_policy": (
            "Buoy truth labels outcomes after matching; it is absent from graph candidates, "
            "costs, updates, and selection."
        ),
    }
    (args.graph_root / "one_step_graph_attribution_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
