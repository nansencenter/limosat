#!/usr/bin/env python3
"""Build run-ready buoy transitions from observations and a frozen pair plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

try:
    from experiments.analyze_buoy_array_deformation import (
        deformation_components,
        fit_displacement_gradient,
    )
except ModuleNotFoundError:  # Direct ``python experiments/<script>.py`` execution.
    from analyze_buoy_array_deformation import (
        deformation_components,
        fit_displacement_gradient,
    )


OBSERVATION_COLUMNS = {
    "buoy_id",
    "image_id",
    "image_filename",
    "image_time",
    "x",
    "y",
    "analysis_crs",
}
IMAGE_MAP_COLUMNS = {
    "fixture_image_id",
    "operational_image_id",
    "image_time",
    "image_filename",
    "kingston_filepath",
}
PAIR_PLAN_COLUMNS = {
    "source_fixture_image_id",
    "target_fixture_image_id",
    "within_dataset_split",
    "may_tune",
    "report_primary",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_boolean(values: pd.Series, column: str) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    if (~normalized.isin({"true", "false", "1", "0"})).any():
        raise ValueError(f"{column} must contain only true/false or 1/0")
    return normalized.isin({"true", "1"})


def validate_pair_plan(plan: pd.DataFrame) -> pd.DataFrame:
    missing = PAIR_PLAN_COLUMNS.difference(plan.columns)
    if missing:
        raise ValueError(f"pair plan lacks columns: {sorted(missing)}")
    result = plan.copy()
    result["may_tune"] = as_boolean(result["may_tune"], "may_tune")
    result["report_primary"] = as_boolean(
        result["report_primary"], "report_primary"
    )
    if result.duplicated(
        ["source_fixture_image_id", "target_fixture_image_id"]
    ).any():
        raise ValueError("pair plan image pairs must be unique")
    if (result["source_fixture_image_id"] == result["target_fixture_image_id"]).any():
        raise ValueError("source and target images must differ")
    tuning_images = set(
        result.loc[result["may_tune"], "source_fixture_image_id"]
    ) | set(result.loc[result["may_tune"], "target_fixture_image_id"])
    primary_images = set(
        result.loc[result["report_primary"], "source_fixture_image_id"]
    ) | set(result.loc[result["report_primary"], "target_fixture_image_id"])
    overlap = tuning_images.intersection(primary_images)
    if overlap:
        raise ValueError(
            f"tuning and primary evaluation share fixture images: {sorted(overlap)}"
        )
    return result


def map_observations(
    observations: pd.DataFrame, image_map: pd.DataFrame
) -> pd.DataFrame:
    missing_observations = OBSERVATION_COLUMNS.difference(observations.columns)
    missing_map = IMAGE_MAP_COLUMNS.difference(image_map.columns)
    if missing_observations:
        raise ValueError(
            f"observations lack columns: {sorted(missing_observations)}"
        )
    if missing_map:
        raise ValueError(f"image map lacks columns: {sorted(missing_map)}")
    if image_map["fixture_image_id"].duplicated().any():
        raise ValueError("image map fixture IDs must be unique")
    if image_map["operational_image_id"].duplicated().any():
        raise ValueError("image map operational IDs must be unique")

    selected = observations.loc[
        observations["image_id"].isin(image_map["fixture_image_id"])
    ].copy()
    selected["buoy_id"] = selected["buoy_id"].astype(str)
    if not selected["analysis_crs"].eq("EPSG:3413").all():
        raise ValueError("observations must use EPSG:3413")
    selected["image_time"] = pd.to_datetime(selected["image_time"], utc=True)
    mapping = image_map.copy()
    mapping["image_time"] = pd.to_datetime(mapping["image_time"], utc=True)
    mapped = selected.merge(
        mapping[
            [
                "fixture_image_id",
                "operational_image_id",
                "image_time",
                "image_filename",
                "kingston_filepath",
            ]
        ],
        left_on="image_id",
        right_on="fixture_image_id",
        how="inner",
        suffixes=("_observation", "_operational"),
        validate="many_to_one",
    )
    if not mapped["image_filename_observation"].eq(
        mapped["image_filename_operational"]
    ).all():
        raise ValueError("observation and operational image filenames differ")
    time_offset_seconds = (
        mapped["image_time_observation"] - mapped["image_time_operational"]
    ).dt.total_seconds().abs()
    if time_offset_seconds.gt(1e-6).any():
        raise ValueError("observation and operational image times differ")
    if mapped.duplicated(["buoy_id", "fixture_image_id"]).any():
        raise ValueError("buoy observations must be unique per fixture image")

    mapped["image_exists"] = mapped["kingston_filepath"].map(
        lambda value: Path(value).is_file()
    )
    mapped = mapped.drop(
        columns=["image_id", "image_filepath"], errors="ignore"
    ).rename(
        columns={
            "operational_image_id": "image_id",
            "image_time_operational": "image_time",
            "image_filename_operational": "image_filename",
            "kingston_filepath": "image_filepath",
        }
    )[
        [
            "buoy_id",
            "fixture_image_id",
            "image_id",
            "image_time",
            "image_filename",
            "image_filepath",
            "image_exists",
            "x",
            "y",
            "analysis_crs",
        ]
    ].sort_values(["image_time", "buoy_id"], kind="stable")
    return mapped


def build_transitions(
    observations: pd.DataFrame, pair_plan: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for pair in pair_plan.itertuples(index=False):
        source = observations.loc[
            observations["fixture_image_id"].eq(pair.source_fixture_image_id)
        ].copy()
        target = observations.loc[
            observations["fixture_image_id"].eq(pair.target_fixture_image_id)
        ].copy()
        if source.empty or target.empty:
            raise ValueError(
                "pair plan references a fixture image without observations"
            )
        paired = source.merge(
            target,
            on="buoy_id",
            suffixes=("_source", "_target"),
            validate="one_to_one",
        )
        if paired.empty:
            raise ValueError("planned image pair has no common buoys")
        elapsed_hours = (
            paired["image_time_target"] - paired["image_time_source"]
        ).dt.total_seconds() / 3600.0
        if not elapsed_hours.gt(0).all() or elapsed_hours.nunique() != 1:
            raise ValueError("pair elapsed time must be unique and positive")
        for row, elapsed in zip(
            paired.itertuples(index=False), elapsed_hours, strict=True
        ):
            truth_dx = row.x_target - row.x_source
            truth_dy = row.y_target - row.y_source
            rows.append(
                {
                    "transition_id": (
                        f"nice_{row.buoy_id}_{pair.within_dataset_split}_"
                        f"{row.image_id_source}_{row.image_id_target}"
                    ),
                    "buoy_id": row.buoy_id,
                    "within_dataset_split": pair.within_dataset_split,
                    "may_tune": bool(pair.may_tune),
                    "report_primary": bool(pair.report_primary),
                    "source_fixture_image_id": int(row.fixture_image_id_source),
                    "target_fixture_image_id": int(row.fixture_image_id_target),
                    "source_image_id": int(row.image_id_source),
                    "target_image_id": int(row.image_id_target),
                    "source_image_time": row.image_time_source,
                    "target_image_time": row.image_time_target,
                    "source_image_filepath": row.image_filepath_source,
                    "target_image_filepath": row.image_filepath_target,
                    "elapsed_hours": float(elapsed),
                    "source_x": float(row.x_source),
                    "source_y": float(row.y_source),
                    "target_x": float(row.x_target),
                    "target_y": float(row.y_target),
                    "truth_dx_m": float(truth_dx),
                    "truth_dy_m": float(truth_dy),
                    "truth_distance_m": float(np.hypot(truth_dx, truth_dy)),
                    "truth_speed_km_per_day": float(
                        np.hypot(truth_dx, truth_dy) / 1000.0 * 24.0 / elapsed
                    ),
                    "analysis_crs": "EPSG:3413",
                }
            )
    result = pd.DataFrame.from_records(rows)
    if result["transition_id"].duplicated().any():
        raise ValueError("generated transition IDs must be unique")
    return result.sort_values(
        ["source_image_time", "target_image_time", "buoy_id"], kind="stable"
    ).reset_index(drop=True)


def split_observations(
    observations: pd.DataFrame, pair_plan: pd.DataFrame
) -> pd.DataFrame:
    """Expand images by pair role for tools that filter observations by split."""
    rows = []
    for pair in pair_plan.itertuples(index=False):
        selected = observations.loc[
            observations["fixture_image_id"].isin(
                [pair.source_fixture_image_id, pair.target_fixture_image_id]
            )
        ].copy()
        selected["within_dataset_split"] = pair.within_dataset_split
        rows.append(selected)
    expanded = pd.concat(rows, ignore_index=True).drop_duplicates(
        ["within_dataset_split", "buoy_id", "image_id"]
    )
    filename_parts = expanded["image_filename"].map(
        lambda value: Path(value).stem.split("_")
    )
    if filename_parts.map(len).lt(7).any():
        raise ValueError("cannot derive acquisition pass from image filename")
    expanded["acquisition_pass_id"] = [
        f"{parts[0]}_orbit_{parts[6]}" for parts in filename_parts
    ]
    return expanded.sort_values(
        ["within_dataset_split", "image_time", "buoy_id"], kind="stable"
    ).reset_index(drop=True)


def pair_summary(transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = [
        "source_fixture_image_id",
        "target_fixture_image_id",
        "source_image_id",
        "target_image_id",
        "within_dataset_split",
        "may_tune",
        "report_primary",
    ]
    for keys, group in transitions.groupby(columns, sort=True, dropna=False):
        record = dict(zip(columns, keys, strict=True))
        source = group[["source_x", "source_y"]].to_numpy(dtype=float)
        truth = group[["truth_dx_m", "truth_dy_m"]].to_numpy(dtype=float)
        distances = np.linalg.norm(
            source[:, None, :] - source[None, :, :], axis=2
        )
        try:
            area_km2 = float(ConvexHull(source).volume / 1e6)
        except QhullError:
            area_km2 = np.nan
        fit = fit_displacement_gradient(source, truth)
        record.update(
            {
                "common_buoys": int(len(group)),
                "elapsed_hours": float(group["elapsed_hours"].iloc[0]),
                "array_convex_hull_area_km2": area_km2,
                "array_diameter_km": float(distances.max() / 1000.0),
                "truth_affine_residual_median_m": fit["residual_median_m"]
                if fit
                else np.nan,
                "truth_affine_residual_maximum_m": fit["residual_maximum_m"]
                if fit
                else np.nan,
            }
        )
        if fit:
            components = deformation_components(
                fit["gradient"], float(group["elapsed_hours"].iloc[0]) / 24.0
            )
            record.update({f"truth_{name}": value for name, value in components.items()})
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--image-map", type=Path, required=True)
    parser.add_argument("--pair-plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    source_observations = pd.read_csv(args.observations, dtype={"buoy_id": str})
    image_map = pd.read_csv(args.image_map)
    plan = validate_pair_plan(pd.read_csv(args.pair_plan))
    observations = map_observations(source_observations, image_map)
    transitions = build_transitions(observations, plan)
    expanded_observations = split_observations(observations, plan)
    summary = pair_summary(transitions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(args.out_dir / "observations.csv", index=False)
    expanded_observations.to_csv(
        args.out_dir / "split_observations.csv", index=False
    )
    transitions.to_csv(args.out_dir / "transitions.csv", index=False)
    summary.to_csv(args.out_dir / "pair_summary.csv", index=False)
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "observations": str(args.observations.resolve()),
                "observations_sha256": sha256(args.observations),
                "image_map": str(args.image_map.resolve()),
                "image_map_sha256": sha256(args.image_map),
                "pair_plan": str(args.pair_plan.resolve()),
                "pair_plan_sha256": sha256(args.pair_plan),
                "analysis_crs": "EPSG:3413",
                "coordinate_units": "metres",
                "fixture_observations": int(len(observations)),
                "split_observations": int(len(expanded_observations)),
                "transitions": int(len(transitions)),
                "staged_images": int(
                    observations.groupby("image_id")["image_exists"].first().sum()
                ),
                "required_images": int(observations["image_id"].nunique()),
                "tuning_primary_image_overlap": [],
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
