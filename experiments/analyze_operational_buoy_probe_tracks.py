#!/usr/bin/env python3
"""Score exported operational buoy-probe trajectories against frozen truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBSERVATIONS = (
    ROOT / "results/arctic_tracking_next_experiment/splits/full70_2020/observations.csv"
)
DEFAULT_TRANSITIONS = (
    ROOT / "results/arctic_tracking_next_experiment/splits/full70_2020/transitions.csv"
)


def probe_id(buoy_id: pd.Series, image_id: pd.Series) -> pd.Series:
    return buoy_id.astype(str) + "|" + image_id.astype(int).astype(str)


def attach_tracked_positions(
    expected: pd.DataFrame,
    linkage: pd.DataFrame,
    tracks: pd.DataFrame,
    image_map: pd.DataFrame,
) -> pd.DataFrame:
    links = linkage[["probe_id", "trajectory_id"]].copy()
    links["trajectory_id"] = pd.to_numeric(links["trajectory_id"], errors="coerce")
    links = links.dropna(subset=["trajectory_id"])
    links["trajectory_id"] = links["trajectory_id"].astype(int)
    if links["probe_id"].duplicated().any():
        raise ValueError("Probe linkage must be unique by probe_id")

    mapping = image_map[["run_image_id", "catalog_image_id"]].copy()
    mapping["run_image_id"] = mapping["run_image_id"].astype(int)
    mapping["catalog_image_id"] = mapping["catalog_image_id"].astype(int)
    target_run = mapping.rename(
        columns={
            "run_image_id": "target_run_image_id",
            "catalog_image_id": "target_image_id",
        }
    )

    tracked = tracks.copy()
    tracked["trajectory_id"] = tracked["trajectory_id"].astype(int)
    tracked["image_id"] = tracked["image_id"].astype(int)
    if tracked.duplicated(["trajectory_id", "image_id"]).any():
        raise ValueError("Exported trajectory rows must be unique by trajectory/image")
    tracked = tracked.rename(
        columns={
            "trajectory_id": "tracking_trajectory_id",
            "image_id": "target_run_image_id",
            "interpolated": "tracked_interpolated",
            "corr": "tracked_correlation",
        }
    )

    result = expected.merge(
        links.rename(columns={"trajectory_id": "tracking_trajectory_id"}),
        on="probe_id",
        how="left",
        validate="many_to_one",
    ).merge(
        target_run,
        on="target_image_id",
        how="left",
        validate="many_to_one",
    ).merge(
        tracked[
            [
                "tracking_trajectory_id",
                "target_run_image_id",
                "tracked_x",
                "tracked_y",
                "tracked_interpolated",
                "tracked_correlation",
            ]
        ],
        on=["tracking_trajectory_id", "target_run_image_id"],
        how="left",
        validate="many_to_one",
    )
    result["source_probe_linked"] = result["tracking_trajectory_id"].notna()
    result["tracked"] = result[["tracked_x", "tracked_y"]].notna().all(axis=1)
    result["endpoint_error_m"] = np.hypot(
        result["tracked_x"] - result["target_x"],
        result["tracked_y"] - result["target_y"],
    )
    return result


def metric_summary(frame: pd.DataFrame, label: str) -> dict:
    tracked = frame["tracked"].fillna(False).astype(bool)
    errors = frame.loc[tracked, "endpoint_error_m"]
    count = len(frame)
    row = {
        "evaluation": label,
        "expected": int(count),
        "source_probe_linked": int(frame["source_probe_linked"].sum()),
        "tracked": int(tracked.sum()),
        "tracked_fraction": float(tracked.mean()) if count else np.nan,
        "median_tracked_error_m": float(errors.median()) if len(errors) else np.nan,
        "p90_tracked_error_m": float(errors.quantile(0.9)) if len(errors) else np.nan,
        "within_500m_fraction_all": float((tracked & (frame["endpoint_error_m"] <= 500)).mean()) if count else np.nan,
        "within_1km_fraction_all": float((tracked & (frame["endpoint_error_m"] <= 1000)).mean()) if count else np.nan,
        "within_2km_fraction_all": float((tracked & (frame["endpoint_error_m"] <= 2000)).mean()) if count else np.nan,
        "within_5km_fraction_all": float((tracked & (frame["endpoint_error_m"] <= 5000)).mean()) if count else np.nan,
        "catastrophic_50km_fraction_all": float((tracked & (frame["endpoint_error_m"] > 50000)).mean()) if count else np.nan,
    }
    if len(errors):
        row["within_2km_fraction_tracked"] = float((errors <= 2000).mean())
    else:
        row["within_2km_fraction_tracked"] = np.nan
    return row


def transition_expectations(
    transitions: pd.DataFrame,
    observations: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    truth = observations.loc[
        observations["within_dataset_split"] == split,
        ["buoy_id", "image_id", "x", "y"],
    ].rename(
        columns={"image_id": "target_image_id", "x": "target_x", "y": "target_y"}
    )
    selected = transitions.loc[
        transitions["within_dataset_split"] == split
    ].copy()
    selected["probe_id"] = probe_id(selected["buoy_id"], selected["source_image_id"])
    return selected.merge(
        truth,
        on=["buoy_id", "target_image_id"],
        how="left",
        validate="many_to_one",
    )


def sequential_expectations(observations: pd.DataFrame, split: str) -> pd.DataFrame:
    selected = observations.loc[
        (observations["within_dataset_split"] == split)
        & observations["usable_experiment_trajectory"].fillna(False).astype(bool)
    ].copy()
    selected["image_time"] = pd.to_datetime(selected["image_time"], utc=True)
    rows = []
    for trajectory_id, group in selected.groupby("experiment_trajectory_id", sort=True):
        ordered = group.sort_values(["image_time", "image_id"], kind="stable")
        if len(ordered) < 2:
            continue
        source = ordered.iloc[0]
        source_probe_id = f"{source.buoy_id}|{int(source.image_id)}"
        for step, target in enumerate(ordered.iloc[1:].itertuples(index=False), start=1):
            rows.append(
                {
                    "experiment_trajectory_id": trajectory_id,
                    "buoy_id": source.buoy_id,
                    "probe_id": source_probe_id,
                    "source_image_id": int(source.image_id),
                    "target_image_id": int(target.image_id),
                    "target_x": float(target.x),
                    "target_y": float(target.y),
                    "target_step": step,
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
    out_dir = args.out_dir or args.run_dir / "buoy_probe_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    observations = pd.read_csv(args.observations, low_memory=False)
    transitions = pd.read_csv(args.transitions, low_memory=False)
    linkage = pd.read_csv(args.run_dir / "buoy_probe_linkage.csv", low_memory=False)
    tracks = pd.read_csv(args.run_dir / "buoy_probe_trajectories.csv", low_memory=False)
    image_map = pd.read_csv(args.run_dir / "image_timings.csv")

    one_step_expected = transition_expectations(transitions, observations, args.split)
    one_step = attach_tracked_positions(one_step_expected, linkage, tracks, image_map)
    sequential_expected = sequential_expectations(observations, args.split)
    sequential = attach_tracked_positions(
        sequential_expected, linkage, tracks, image_map
    )
    one_step.to_csv(out_dir / "one_step_results.csv", index=False)
    sequential.to_csv(out_dir / "sequential_results.csv", index=False)

    path_summary = sequential.groupby("experiment_trajectory_id", sort=True).agg(
        expected_steps=("target_step", "count"),
        tracked_steps=("tracked", "sum"),
        final_error_m=("endpoint_error_m", "last"),
        maximum_error_m=("endpoint_error_m", "max"),
    )
    path_summary["complete"] = (
        path_summary["tracked_steps"] == path_summary["expected_steps"]
    )
    path_summary.to_csv(out_dir / "sequential_path_summary.csv")

    summaries = pd.DataFrame.from_records(
        [
            metric_summary(one_step, "all_one_step_transitions"),
            metric_summary(sequential, "first_seed_all_later_observations"),
        ]
    )
    summaries.to_csv(out_dir / "summary.csv", index=False)
    payload = {
        "split": args.split,
        "probe_seed_method": json.loads(
            (args.run_dir / "run_manifest.json").read_text()
        ).get("probe_seed_method"),
        "one_step_expected": int(len(one_step)),
        "sequential_expected": int(len(sequential)),
        "sequential_paths": int(len(path_summary)),
        "complete_sequential_paths": int(path_summary["complete"].sum()),
        "missing_predictions_retained_in_denominators": True,
    }
    (out_dir / "evaluation_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(summaries.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
