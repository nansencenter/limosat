#!/usr/bin/env python3
"""Evaluate an unchanged dense LiMOSAT field at buoy transitions.

No buoy keypoint is inserted.  For every truth transition, trajectories that
exist in both Sentinel-1 images provide displacement vectors near the buoy's
source position.  This separates operational spatial sampling from the
supplied-point descriptor diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError
from shapely import from_wkt, get_x, get_y

try:
    from experiments.analyze_operational_buoy_probe_tracks import (
        DEFAULT_OBSERVATIONS,
        DEFAULT_TRANSITIONS,
    )
except ModuleNotFoundError:  # Direct ``python experiments/<script>.py`` execution.
    from analyze_operational_buoy_probe_tracks import (
        DEFAULT_OBSERVATIONS,
        DEFAULT_TRANSITIONS,
    )


def transition_truth(
    transitions: pd.DataFrame, observations: pd.DataFrame, split: str
) -> pd.DataFrame:
    """Attach source/target EPSG:3413 truth and acquisition metadata."""
    selected = transitions.loc[
        transitions["within_dataset_split"] == split
    ].copy()
    selected_observations = observations.loc[
        observations["within_dataset_split"] == split
    ].copy()
    coordinate_columns = {"source_x", "source_y", "target_x", "target_y"}
    if coordinate_columns.issubset(selected.columns):
        required_observations = {"buoy_id", "image_id", "acquisition_pass_id"}
        missing = required_observations.difference(selected_observations.columns)
        if missing:
            raise ValueError(
                "observations lack acquisition metadata: " f"{sorted(missing)}"
            )
        source_pass = selected_observations[
            ["buoy_id", "image_id", "acquisition_pass_id"]
        ].rename(
            columns={
                "image_id": "source_image_id",
                "acquisition_pass_id": "source_acquisition_pass_id",
            }
        )
        target_pass = selected_observations[
            ["buoy_id", "image_id", "acquisition_pass_id"]
        ].rename(
            columns={
                "image_id": "target_image_id",
                "acquisition_pass_id": "target_acquisition_pass_id",
            }
        )
        result = selected.merge(
            source_pass,
            on=["buoy_id", "source_image_id"],
            how="left",
            validate="many_to_one",
        ).merge(
            target_pass,
            on=["buoy_id", "target_image_id"],
            how="left",
            validate="many_to_one",
        )
        if {"truth_dx_m", "truth_dy_m"}.issubset(result.columns):
            coordinate_truth = result[["target_x", "target_y"]].to_numpy(
                dtype=float
            ) - result[["source_x", "source_y"]].to_numpy(dtype=float)
            declared_truth = result[["truth_dx_m", "truth_dy_m"]].to_numpy(
                dtype=float
            )
            if not np.allclose(coordinate_truth, declared_truth, atol=1e-6, rtol=0):
                raise ValueError("transition truth vectors disagree with coordinates")
        result["same_acquisition_pass"] = (
            result["source_acquisition_pass_id"].notna()
            & result["source_acquisition_pass_id"].eq(
                result["target_acquisition_pass_id"]
            )
        )
        return result

    source = selected_observations[
        ["buoy_id", "image_id", "x", "y", "acquisition_pass_id"]
    ].rename(
        columns={
            "image_id": "source_image_id",
            "x": "source_x",
            "y": "source_y",
            "acquisition_pass_id": "source_acquisition_pass_id",
        }
    )
    target = selected_observations[
        ["buoy_id", "image_id", "x", "y", "acquisition_pass_id"]
    ].rename(
        columns={
            "image_id": "target_image_id",
            "x": "target_x",
            "y": "target_y",
            "acquisition_pass_id": "target_acquisition_pass_id",
        }
    )
    result = selected.merge(
        source, on=["buoy_id", "source_image_id"], how="left", validate="many_to_one"
    ).merge(
        target, on=["buoy_id", "target_image_id"], how="left", validate="many_to_one"
    )
    result["same_acquisition_pass"] = (
        result["source_acquisition_pass_id"].notna()
        & result["source_acquisition_pass_id"].eq(
            result["target_acquisition_pass_id"]
        )
    )
    return result


def paired_positions(
    points: pd.DataFrame, source_run_image_id: int, target_run_image_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return source/target coordinates for trajectories present in both images."""
    source = points.loc[
        points["image_id"] == source_run_image_id,
        ["trajectory_id", "x", "y"],
    ]
    target = points.loc[
        points["image_id"] == target_run_image_id,
        ["trajectory_id", "x", "y"],
    ]
    paired = source.merge(
        target, on="trajectory_id", how="inner", suffixes=("_source", "_target"),
        validate="one_to_one",
    )
    return (
        paired[["x_source", "y_source"]].to_numpy(dtype=float),
        paired[["x_target", "y_target"]].to_numpy(dtype=float),
    )


def estimate_local_displacement(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    query_xy: np.ndarray,
    inverse_distance_neighbours: int = 4,
) -> dict:
    """Estimate the target from nearest, inverse-distance, and triangular fields."""
    result = {
        "paired_trajectory_count": int(len(source_xy)),
        "nearest_source_distance_m": np.nan,
        "nearest_target_x": np.nan,
        "nearest_target_y": np.nan,
        "inverse_distance_neighbour_count": 0,
        "inverse_distance_max_source_distance_m": np.nan,
        "inverse_distance_vector_spread_m": np.nan,
        "inverse_distance_target_x": np.nan,
        "inverse_distance_target_y": np.nan,
        "triangle_max_source_distance_m": np.nan,
        "triangle_target_x": np.nan,
        "triangle_target_y": np.nan,
    }
    if len(source_xy) == 0:
        return result

    distances = np.linalg.norm(source_xy - query_xy, axis=1)
    order = np.argsort(distances, kind="stable")
    nearest = int(order[0])
    vectors = target_xy - source_xy
    nearest_prediction = query_xy + vectors[nearest]
    result.update(
        {
            "nearest_source_distance_m": float(distances[nearest]),
            "nearest_target_x": float(nearest_prediction[0]),
            "nearest_target_y": float(nearest_prediction[1]),
        }
    )

    count = min(inverse_distance_neighbours, len(source_xy))
    selected = order[:count]
    selected_distances = distances[selected]
    if selected_distances[0] <= np.finfo(float).eps:
        weights = np.zeros(count, dtype=float)
        weights[0] = 1.0
    else:
        weights = 1.0 / np.maximum(selected_distances, 1.0)
        weights /= weights.sum()
    estimated_vector = np.sum(vectors[selected] * weights[:, None], axis=0)
    inverse_distance_prediction = query_xy + estimated_vector
    result.update(
        {
            "inverse_distance_neighbour_count": int(count),
            "inverse_distance_max_source_distance_m": float(selected_distances[-1]),
            "inverse_distance_vector_spread_m": float(
                np.median(np.linalg.norm(vectors[selected] - estimated_vector, axis=1))
            ),
            "inverse_distance_target_x": float(inverse_distance_prediction[0]),
            "inverse_distance_target_y": float(inverse_distance_prediction[1]),
        }
    )

    if len(source_xy) < 3:
        return result
    unique_source, unique_indices = np.unique(source_xy, axis=0, return_index=True)
    if len(unique_source) < 3:
        return result
    unique_target = target_xy[unique_indices]
    try:
        triangulation = Delaunay(unique_source)
    except QhullError:
        return result
    simplex_index = int(triangulation.find_simplex(query_xy))
    if simplex_index < 0:
        return result
    transform = triangulation.transform[simplex_index]
    first_weights = transform[:2].dot(query_xy - transform[2])
    weights = np.append(first_weights, 1.0 - first_weights.sum())
    vertices = triangulation.simplices[simplex_index]
    triangle_prediction = weights @ unique_target[vertices]
    result.update(
        {
            "triangle_max_source_distance_m": float(
                np.linalg.norm(unique_source[vertices] - query_xy, axis=1).max()
            ),
            "triangle_target_x": float(triangle_prediction[0]),
            "triangle_target_y": float(triangle_prediction[1]),
        }
    )
    return result


def estimate_local_average_within_radius(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    query_xy: np.ndarray,
    maximum_source_distance_m: float,
    maximum_neighbours: int = 4,
) -> dict:
    """Average up to ``maximum_neighbours`` displacement vectors inside a radius."""
    empty = {
        "neighbour_count": 0,
        "maximum_neighbour_distance_m": np.nan,
        "vector_spread_m": np.nan,
        "target_x": np.nan,
        "target_y": np.nan,
    }
    if len(source_xy) == 0:
        return empty
    distances = np.linalg.norm(source_xy - query_xy, axis=1)
    order = np.argsort(distances, kind="stable")
    selected = order[distances[order] <= maximum_source_distance_m][
        :maximum_neighbours
    ]
    if len(selected) == 0:
        return empty
    weights = 1.0 / np.maximum(distances[selected], 1.0)
    weights /= weights.sum()
    vectors = target_xy[selected] - source_xy[selected]
    estimated_vector = np.sum(vectors * weights[:, None], axis=0)
    prediction = query_xy + estimated_vector
    return {
        "neighbour_count": int(len(selected)),
        "maximum_neighbour_distance_m": float(distances[selected].max()),
        "vector_spread_m": float(
            np.median(np.linalg.norm(vectors - estimated_vector, axis=1))
        ),
        "target_x": float(prediction[0]),
        "target_y": float(prediction[1]),
    }


def evaluate_dense_field(
    truth: pd.DataFrame,
    image_map: pd.DataFrame,
    points: pd.DataFrame,
) -> pd.DataFrame:
    catalog_to_run = dict(
        zip(image_map["catalog_image_id"], image_map["run_image_id"], strict=True)
    )
    pair_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    source_cache: dict[int, np.ndarray] = {}
    rows = []
    for transition in truth.itertuples(index=False):
        source_run = int(catalog_to_run[int(transition.source_image_id)])
        target_run = int(catalog_to_run[int(transition.target_image_id)])
        pair_key = (source_run, target_run)
        if pair_key not in pair_cache:
            pair_cache[pair_key] = paired_positions(points, source_run, target_run)
        if source_run not in source_cache:
            source_cache[source_run] = points.loc[
                points["image_id"] == source_run, ["x", "y"]
            ].to_numpy(dtype=float)
        source_xy, target_xy = pair_cache[pair_key]
        all_source_xy = source_cache[source_run]
        query_xy = np.array(
            [transition.source_x, transition.source_y], dtype=float
        )
        estimate = estimate_local_displacement(
            source_xy,
            target_xy,
            query_xy,
        )
        record = {**transition._asdict(), **estimate}
        all_source_distances = np.linalg.norm(all_source_xy - query_xy, axis=1)
        paired_source_distances = np.linalg.norm(source_xy - query_xy, axis=1)
        record["source_point_nearest_distance_m"] = (
            float(all_source_distances.min()) if len(all_source_distances) else np.nan
        )
        for method in ("nearest", "inverse_distance", "triangle"):
            record[f"{method}_endpoint_error_m"] = float(
                np.hypot(
                    record[f"{method}_target_x"] - transition.target_x,
                    record[f"{method}_target_y"] - transition.target_y,
                )
            )
        for radius_m in (5000, 10000, 20000, 50000):
            prefix = f"local_average_{radius_m // 1000}km"
            local_average = estimate_local_average_within_radius(
                source_xy,
                target_xy,
                query_xy,
                maximum_source_distance_m=radius_m,
            )
            record[f"source_points_within_{radius_m // 1000}km"] = int(
                np.sum(all_source_distances <= radius_m)
            )
            record[f"paired_trajectories_within_{radius_m // 1000}km"] = int(
                np.sum(paired_source_distances <= radius_m)
            )
            record.update(
                {f"{prefix}_{name}": value for name, value in local_average.items()}
            )
            record[f"{prefix}_endpoint_error_m"] = float(
                np.hypot(
                    record[f"{prefix}_target_x"] - transition.target_x,
                    record[f"{prefix}_target_y"] - transition.target_y,
                )
            )
        source_near = record["source_points_within_10km"]
        paired_near = record["paired_trajectories_within_10km"]
        if source_near == 0:
            record["local_10km_availability_fate"] = "no_source_point_within_10km"
        elif paired_near == 0:
            record["local_10km_availability_fate"] = (
                "source_points_present_but_none_survived"
            )
        else:
            record["local_10km_availability_fate"] = "surviving_local_vector"
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def metric_row(
    results: pd.DataFrame,
    method: str,
    maximum_source_distance_m: float,
    stratum: str = "all",
) -> dict:
    distance_column = {
        "nearest": "nearest_source_distance_m",
        "inverse_distance": "inverse_distance_max_source_distance_m",
        "triangle": "triangle_max_source_distance_m",
    }[method]
    error_column = f"{method}_endpoint_error_m"
    available = (
        results[distance_column].notna()
        & results[error_column].notna()
        & (results[distance_column] <= maximum_source_distance_m)
    )
    errors = results.loc[available, error_column]
    count = len(results)
    return {
        "stratum": stratum,
        "method": method,
        "maximum_source_distance_m": float(maximum_source_distance_m),
        "expected": int(count),
        "available": int(available.sum()),
        "available_fraction": float(available.mean()) if count else np.nan,
        "median_available_error_m": float(errors.median()) if len(errors) else np.nan,
        "p90_available_error_m": float(errors.quantile(0.9)) if len(errors) else np.nan,
        "within_2km_fraction_all": float((available & (results[error_column] <= 2000)).mean())
        if count
        else np.nan,
        "within_2km_fraction_available": float((errors <= 2000).mean())
        if len(errors)
        else np.nan,
        "catastrophic_50km_fraction_all": float(
            (available & (results[error_column] > 50000)).mean()
        )
        if count
        else np.nan,
    }


def local_average_metric_row(
    results: pd.DataFrame,
    maximum_source_distance_m: float,
    stratum: str = "all",
) -> dict:
    prefix = f"local_average_{int(maximum_source_distance_m) // 1000}km"
    error_column = f"{prefix}_endpoint_error_m"
    available = results[error_column].notna()
    errors = results.loc[available, error_column]
    count = len(results)
    return {
        "stratum": stratum,
        "method": "local_average_up_to_4",
        "maximum_source_distance_m": float(maximum_source_distance_m),
        "expected": int(count),
        "available": int(available.sum()),
        "available_fraction": float(available.mean()) if count else np.nan,
        "median_available_error_m": float(errors.median()) if len(errors) else np.nan,
        "p90_available_error_m": float(errors.quantile(0.9)) if len(errors) else np.nan,
        "within_2km_fraction_all": float(
            (available & (results[error_column] <= 2000)).mean()
        )
        if count
        else np.nan,
        "within_2km_fraction_available": float((errors <= 2000).mean())
        if len(errors)
        else np.nan,
        "catastrophic_50km_fraction_all": float(
            (available & (results[error_column] > 50000)).mean()
        )
        if count
        else np.nan,
    }


def load_points(database_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        points = pd.read_sql_query(
            f'SELECT image_id, trajectory_id, geometry FROM "{table}"', connection
        )
    geometry = from_wkt(points["geometry"].to_numpy())
    points["x"] = get_x(geometry)
    points["y"] = get_y(geometry)
    if points.duplicated(["trajectory_id", "image_id"]).any():
        raise ValueError("Persisted points must be unique by trajectory/image")
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--split", default="development")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.run_dir / "run_manifest.json").read_text())
    if manifest.get("mode") not in {"dense", "dense_operational"} or int(
        manifest.get("buoy_probes_requested", 0)
    ) != 0:
        raise ValueError("Dense-field evaluation requires a run without injected buoy probes")
    database_url = manifest["engine_url"]
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError(f"Unsupported database URL: {database_url}")
    database_path = Path(database_url[len(prefix) :])
    table = manifest["effective_run_name"]
    output_dir = args.out_dir or args.run_dir / "dense_field_buoy_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = pd.read_csv(args.observations, low_memory=False)
    transitions = pd.read_csv(args.transitions, low_memory=False)
    image_map = pd.read_csv(args.run_dir / "image_timings.csv")
    all_truth = transition_truth(transitions, observations, args.split)
    available_catalog_images = set(image_map["catalog_image_id"].astype(int))
    truth = all_truth.loc[
        all_truth["source_image_id"].isin(available_catalog_images)
        & all_truth["target_image_id"].isin(available_catalog_images)
    ].copy()
    points = load_points(database_path, table)
    results = evaluate_dense_field(truth, image_map, points)
    results.to_csv(output_dir / "transition_results.csv", index=False)

    thresholds = (5000.0, 10000.0, 20000.0, 50000.0)
    summary_rows = [
        metric_row(results, method, threshold)
        for method in ("nearest", "inverse_distance", "triangle")
        for threshold in thresholds
    ]
    summary_rows.extend(
        local_average_metric_row(results, threshold) for threshold in thresholds
    )
    for same_pass, group in results.groupby("same_acquisition_pass", sort=True):
        label = "same_acquisition_pass" if same_pass else "different_acquisition_pass"
        for method in ("nearest", "inverse_distance", "triangle"):
            summary_rows.append(metric_row(group, method, 10000.0, label))
        summary_rows.append(local_average_metric_row(group, 10000.0, label))
    summary = pd.DataFrame.from_records(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "run_name": table,
                "database_path": str(database_path),
                "expected_transitions": int(len(results)),
                "split_transitions_before_run_image_filter": int(len(all_truth)),
                "coordinate_reference_system": "EPSG:3413",
                "coordinate_units": "metres",
                "buoy_points_inserted": False,
                "missing_predictions_retained_in_denominators": True,
                "inverse_distance_neighbours": 4,
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
