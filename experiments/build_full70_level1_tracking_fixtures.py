#!/usr/bin/env python3
"""Build leakage-labelled tracking fixtures from the full-70 Level-1 QC table.

The output keeps official IABP Level-1 positions as truth, excludes observations
that failed QC or lack a processed SAR image, and makes temporal segmentation
explicit. Two trajectory identifiers are provided:

* ``continuous_trajectory_id`` only breaks at gaps over the physics horizon.
* ``experiment_trajectory_id`` also breaks at month-based experiment splits.

The latter is the safe default for descriptor/update-policy comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results/iabp_s1_stratified_coverage"

MONTH_SPLITS = {
    "2020-01": "evaluation",
    "2020-02": "validation",
    "2020-03": "development",
    "2020-04": "season_edge_evaluation",
}

TRANSITION_COLUMNS = [
    "transition_id",
    "experiment_trajectory_id",
    "continuous_trajectory_id",
    "buoy_id",
    "experiment_split",
    "month",
    "source_image_id",
    "source_image_time",
    "source_image_filepath",
    "target_image_id",
    "target_image_time",
    "target_image_filepath",
    "elapsed_hours",
    "cadence_band",
    "truth_dx_m",
    "truth_dy_m",
    "truth_distance_m",
    "truth_speed_km_per_day",
    "source_sic_regime",
    "target_sic_regime",
    "source_spatial_block",
    "target_spatial_block",
    "month_exclusive_buoy",
    "skipped_selected_images_between",
]


def cadence_band(hours: float) -> str:
    if hours <= 3:
        return "00_to_03h"
    if hours <= 12:
        return "03_to_12h"
    if hours <= 30:
        return "12_to_30h"
    if hours <= 72:
        return "30_to_72h"
    return "over_72h"


def availability_status(row: pd.Series) -> str:
    if not bool(row["truth_ready_after_level1"]):
        return str(row["level1_final_status"])
    if not bool(row["standard_vae_available"]):
        return "hold_standard_vae_scene_unavailable"
    return "ready_level1_truth_and_standard_vae"


def assign_segment_ids(
    observations: pd.DataFrame,
    maximum_gap_hours: float,
) -> pd.DataFrame:
    """Assign continuous and split-safe trajectory identifiers."""
    result = observations.sort_values(
        ["buoy_id", "image_time", "image_id"]
    ).copy()
    continuous_ids: dict[int, str] = {}
    experiment_ids: dict[int, str] = {}
    for buoy_id, group in result.groupby("buoy_id", sort=True):
        continuous_number = 0
        experiment_number = 0
        previous_time: pd.Timestamp | None = None
        previous_split: str | None = None
        for index, row in group.iterrows():
            current_time = pd.Timestamp(row["image_time"])
            gap_hours = (
                np.inf
                if previous_time is None
                else (current_time - previous_time).total_seconds() / 3600.0
            )
            new_continuous = previous_time is None or gap_hours > maximum_gap_hours
            if new_continuous:
                continuous_number += 1
            if (
                previous_time is None
                or gap_hours > maximum_gap_hours
                or str(row["experiment_split"]) != previous_split
            ):
                experiment_number += 1
            continuous_ids[index] = (
                f"buoy_{buoy_id}_continuous_{continuous_number:02d}"
            )
            experiment_ids[index] = (
                f"buoy_{buoy_id}_{row['experiment_split']}_{experiment_number:02d}"
            )
            previous_time = current_time
            previous_split = str(row["experiment_split"])
    result["continuous_trajectory_id"] = pd.Series(continuous_ids)
    result["experiment_trajectory_id"] = pd.Series(experiment_ids)
    return result


def add_trajectory_sizes(observations: pd.DataFrame) -> pd.DataFrame:
    result = observations.copy()
    for column in ("continuous_trajectory_id", "experiment_trajectory_id"):
        size_column = column.replace("_id", "_observations")
        result[size_column] = result.groupby(column)[column].transform("size")
    result["usable_experiment_trajectory"] = (
        result["experiment_trajectory_observations"] >= 2
    )
    return result


def build_transitions(
    observations: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    audit_times = {
        buoy_id: pd.DatetimeIndex(group["image_time"].sort_values())
        for buoy_id, group in audit.groupby("buoy_id", sort=False)
    }
    usable = observations[observations["usable_experiment_trajectory"]]
    for trajectory_id, group in usable.groupby(
        "experiment_trajectory_id", sort=True
    ):
        rows = list(group.sort_values(["image_time", "image_id"]).itertuples())
        for source, target in zip(rows[:-1], rows[1:]):
            hours = (target.image_time - source.image_time).total_seconds() / 3600.0
            dx = float(target.x - source.x)
            dy = float(target.y - source.y)
            distance = float(np.hypot(dx, dy))
            all_times = audit_times[str(source.buoy_id)]
            skipped_selected_images = int(
                np.count_nonzero(
                    (all_times > source.image_time) & (all_times < target.image_time)
                )
            )
            records.append(
                {
                    "transition_id": (
                        f"{trajectory_id}_{int(source.image_id)}_{int(target.image_id)}"
                    ),
                    "experiment_trajectory_id": trajectory_id,
                    "continuous_trajectory_id": source.continuous_trajectory_id,
                    "buoy_id": source.buoy_id,
                    "experiment_split": source.experiment_split,
                    "month": source.month,
                    "source_image_id": int(source.image_id),
                    "source_image_time": source.image_time,
                    "source_image_filepath": source.image_filepath,
                    "target_image_id": int(target.image_id),
                    "target_image_time": target.image_time,
                    "target_image_filepath": target.image_filepath,
                    "elapsed_hours": hours,
                    "cadence_band": cadence_band(hours),
                    "truth_dx_m": dx,
                    "truth_dy_m": dy,
                    "truth_distance_m": distance,
                    "truth_speed_km_per_day": distance / 1000.0 / (hours / 24.0),
                    "source_sic_regime": source.sic_regime,
                    "target_sic_regime": target.sic_regime,
                    "source_spatial_block": source.spatial_block,
                    "target_spatial_block": target.spatial_block,
                    "month_exclusive_buoy": bool(source.month_exclusive_buoy),
                    "skipped_selected_images_between": skipped_selected_images,
                }
            )
    return pd.DataFrame.from_records(records, columns=TRANSITION_COLUMNS)


def build_outputs(
    frame_qc: pd.DataFrame,
    maximum_gap_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    audit = frame_qc.copy()
    audit["buoy_id"] = audit["buoy_id"].astype("string").str.replace(
        r"\.0$", "", regex=True
    )
    audit["image_time"] = pd.to_datetime(audit["image_time"], utc=True)
    audit["month"] = audit["image_time"].dt.strftime("%Y-%m")
    audit["experiment_split"] = audit["month"].map(MONTH_SPLITS).fillna(
        "outside_defined_split"
    )
    audit["standard_vae_available"] = audit["standard_vae_output_path"].map(
        lambda value: Path(str(value)).is_file()
    )
    audit["tracking_fixture_status"] = audit.apply(availability_status, axis=1)
    audit["eligible_tracking_observation"] = audit["tracking_fixture_status"].eq(
        "ready_level1_truth_and_standard_vae"
    )

    observations = audit[audit["eligible_tracking_observation"]].copy()
    observations["catalog_x_3413"] = observations["x"]
    observations["catalog_y_3413"] = observations["y"]
    observations["x"] = observations["level1_x_3413"]
    observations["y"] = observations["level1_y_3413"]
    observations["image_filepath"] = observations["standard_vae_output_path"]
    observations["image_filename"] = observations["resolved_product_name"].map(
        lambda value: f"{value}.tiff"
    )
    observations["truth_source"] = "official_iabp_level1_linear_interpolation"
    months_per_buoy = observations.groupby("buoy_id")["month"].transform("nunique")
    observations["month_exclusive_buoy"] = months_per_buoy.eq(1)
    observations = add_trajectory_sizes(
        assign_segment_ids(observations, maximum_gap_hours)
    )
    transitions = build_transitions(observations, audit)

    observation_strata = (
        observations.groupby(
            [
                "experiment_split",
                "month",
                "sic_regime",
                "month_exclusive_buoy",
                "usable_experiment_trajectory",
            ],
            dropna=False,
        )
        .agg(
            observations=("image_id", "size"),
            buoys=("buoy_id", "nunique"),
            images=("image_id", "nunique"),
            experiment_trajectories=("experiment_trajectory_id", "nunique"),
            spatial_blocks=("spatial_block", "nunique"),
        )
        .reset_index()
    )
    transition_strata = (
        transitions.groupby(
            [
                "experiment_split",
                "month",
                "cadence_band",
                "source_sic_regime",
                "month_exclusive_buoy",
            ],
            dropna=False,
        )
        .agg(
            transitions=("transition_id", "size"),
            buoys=("buoy_id", "nunique"),
            trajectories=("experiment_trajectory_id", "nunique"),
            source_images=("source_image_id", "nunique"),
            spatial_blocks=("source_spatial_block", "nunique"),
        )
        .reset_index()
    )
    strata = pd.concat(
        {
            "observations": observation_strata,
            "transitions": transition_strata,
        },
        names=["table", "row"],
    ).reset_index(level=[0, 1])

    summary = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "input_linked_observations": len(audit),
        "level1_truth_ready_observations": int(audit["truth_ready_after_level1"].sum()),
        "eligible_observations_with_standard_vae": len(observations),
        "eligible_buoys": int(observations["buoy_id"].nunique()),
        "eligible_images": int(observations["image_id"].nunique()),
        "continuous_trajectories": int(
            observations["continuous_trajectory_id"].nunique()
        ),
        "usable_split_safe_trajectories": int(
            observations.loc[
                observations["usable_experiment_trajectory"],
                "experiment_trajectory_id",
            ].nunique()
        ),
        "split_safe_transitions": len(transitions),
        "tracking_fixture_status_counts": audit[
            "tracking_fixture_status"
        ].value_counts().to_dict(),
        "experiment_split_policy": {
            split: month for month, split in MONTH_SPLITS.items()
        },
        "strict_generalization_subset": (
            "month_exclusive_buoy=true removes buoy-identity overlap between temporal splits; "
            "report it beside the full temporal split because it is much smaller."
        ),
        "segmentation": (
            f"continuous paths break above {maximum_gap_hours:g} hours; experiment paths "
            "also break at month/split boundaries"
        ),
        "truth_policy": (
            "Official IABP Level-1 positions may seed the first observation and score all "
            "observations. Future positions must not enter candidates, graph costs, or "
            "descriptor-memory updates."
        ),
        "units": {
            "coordinates": "EPSG:3413 metres",
            "time": "UTC",
            "speed": "kilometres per day",
        },
    }
    return audit, observations, transitions, {"summary": summary, "strata": strata}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--maximum-gap-hours", type=float, default=72.0)
    args = parser.parse_args()

    frame_qc = pd.read_csv(
        args.results_dir / "full70_iabp_level1_frame_qc.csv",
        dtype={"buoy_id": "string"},
        low_memory=False,
    )
    audit, observations, transitions, products = build_outputs(
        frame_qc, args.maximum_gap_hours
    )
    audit.to_csv(
        args.results_dir / "full70_level1_observation_eligibility.csv", index=False
    )
    observations.to_csv(
        args.results_dir / "full70_level1_tracking_observations.csv", index=False
    )
    transitions.to_csv(
        args.results_dir / "full70_level1_tracking_transitions.csv", index=False
    )
    products["strata"].to_csv(
        args.results_dir / "full70_level1_tracking_strata.csv", index=False
    )
    (args.results_dir / "full70_level1_tracking_summary.json").write_text(
        json.dumps(products["summary"], indent=2) + "\n"
    )
    print(json.dumps(products["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
