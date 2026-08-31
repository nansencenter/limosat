#!/usr/bin/env python3
"""Evaluate displacement and deformation against a colocated buoy array.

Input rows are long-form estimates: one buoy, image pair, and method per row.
Coordinates and displacements are metres in EPSG:3413; elapsed time is hours.
Unavailable estimates remain in all expected-count denominators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay, QhullError


PAIR_COLUMNS = ["source_image_id", "target_image_id"]
GROUP_COLUMNS = [*PAIR_COLUMNS, "method"]
REQUIRED_COLUMNS = {
    *GROUP_COLUMNS,
    "buoy_id",
    "source_x",
    "source_y",
    "truth_dx_m",
    "truth_dy_m",
    "estimated_dx_m",
    "estimated_dy_m",
    "available",
    "elapsed_hours",
    "analysis_crs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    unknown = ~normalized.isin({"true", "false", "1", "0"})
    if unknown.any():
        raise ValueError("available must contain only true/false or 1/0")
    return normalized.isin({"true", "1"})


def validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"input lacks required columns: {sorted(missing)}")
    result = frame.copy()
    result["buoy_id"] = result["buoy_id"].astype(str)
    result["method"] = result["method"].astype(str)
    result["available"] = _boolean(result["available"])
    if not result["analysis_crs"].eq("EPSG:3413").all():
        raise ValueError("all coordinates must use analysis_crs EPSG:3413")

    truth_columns = [
        "source_x",
        "source_y",
        "truth_dx_m",
        "truth_dy_m",
        "elapsed_hours",
    ]
    if not np.isfinite(result[truth_columns].to_numpy(dtype=float)).all():
        raise ValueError("source, truth, and elapsed values must be finite")
    if not result["elapsed_hours"].gt(0).all():
        raise ValueError("elapsed_hours must be positive")
    estimates = result[["estimated_dx_m", "estimated_dy_m"]].to_numpy(dtype=float)
    if not np.isfinite(estimates[result["available"].to_numpy()]).all():
        raise ValueError("available estimates must be finite")
    if result.duplicated([*GROUP_COLUMNS, "buoy_id"]).any():
        raise ValueError("buoy IDs must be unique within each pair and method")
    elapsed_counts = result.groupby(GROUP_COLUMNS, sort=False)[
        "elapsed_hours"
    ].nunique()
    if elapsed_counts.gt(1).any():
        raise ValueError("elapsed_hours must be unique within each pair and method")

    identity_groups = result.groupby([*PAIR_COLUMNS, "buoy_id"], sort=False)
    numeric_tolerances = {
        "source_x": 1e-6,
        "source_y": 1e-6,
        "truth_dx_m": 1e-6,
        "truth_dy_m": 1e-6,
        "elapsed_hours": 1e-9,
    }
    inconsistent = False
    for column, tolerance in numeric_tolerances.items():
        span = identity_groups[column].max() - identity_groups[column].min()
        inconsistent |= bool(span.gt(tolerance).any())
    crs_count = identity_groups["analysis_crs"].nunique(dropna=False)
    inconsistent |= bool(crs_count.gt(1).any())
    if inconsistent:
        raise ValueError("truth metadata differs between methods")
    return result


def fit_displacement_gradient(
    source_xy: np.ndarray, displacement_xy: np.ndarray
) -> dict | None:
    """Fit u(x)=A(x-centre)+b with coordinates centred for stability."""
    if len(source_xy) < 3:
        return None
    centre = source_xy.mean(axis=0)
    design = np.column_stack((source_xy - centre, np.ones(len(source_xy))))
    if np.linalg.matrix_rank(design) < 3:
        return None
    coefficients = np.linalg.lstsq(design, displacement_xy, rcond=None)[0]
    predicted = design @ coefficients
    residual = np.linalg.norm(predicted - displacement_xy, axis=1)
    return {
        "gradient": coefficients[:2].T,
        "displacement_at_centre": coefficients[2],
        "residual_median_m": float(np.median(residual)),
        "residual_maximum_m": float(np.max(residual)),
    }


def deformation_components(
    displacement_gradient: np.ndarray, elapsed_days: float
) -> dict[str, float]:
    velocity_gradient = displacement_gradient / elapsed_days
    return {
        "divergence_per_day": float(np.trace(velocity_gradient)),
        "shear_per_day": float(
            np.hypot(
                velocity_gradient[0, 0] - velocity_gradient[1, 1],
                velocity_gradient[0, 1] + velocity_gradient[1, 0],
            )
        ),
        "vorticity_per_day": float(
            velocity_gradient[1, 0] - velocity_gradient[0, 1]
        ),
        "area_ratio": float(np.linalg.det(np.eye(2) + displacement_gradient)),
    }


def _array_geometry(source_xy: np.ndarray) -> tuple[float, float]:
    distances = np.linalg.norm(
        source_xy[:, None, :] - source_xy[None, :, :], axis=2
    )
    diameter_m = float(distances.max())
    try:
        area_m2 = float(ConvexHull(source_xy).volume)
    except QhullError:
        area_m2 = np.nan
    return area_m2, diameter_m


def pairwise_records(group: pd.DataFrame) -> pd.DataFrame:
    rows = []
    elapsed_days = float(group["elapsed_hours"].iloc[0]) / 24.0
    ordered = group.sort_values("buoy_id", kind="stable").reset_index(drop=True)
    for first, second in combinations(range(len(ordered)), 2):
        left = ordered.iloc[first]
        right = ordered.iloc[second]
        baseline = np.array(
            [right.source_x - left.source_x, right.source_y - left.source_y],
            dtype=float,
        )
        baseline_m = float(np.linalg.norm(baseline))
        if baseline_m <= 0:
            raise ValueError("buoy source positions must be distinct")
        direction = baseline / baseline_m
        transverse = np.array([-direction[1], direction[0]])
        truth_relative = np.array(
            [
                right.truth_dx_m - left.truth_dx_m,
                right.truth_dy_m - left.truth_dy_m,
            ],
            dtype=float,
        )
        available = bool(left.available and right.available)
        record = {
            **{column: left[column] for column in GROUP_COLUMNS},
            "first_buoy_id": left.buoy_id,
            "second_buoy_id": right.buoy_id,
            "baseline_m": baseline_m,
            "available": available,
            "truth_relative_dx_m": float(truth_relative[0]),
            "truth_relative_dy_m": float(truth_relative[1]),
            "estimated_relative_dx_m": np.nan,
            "estimated_relative_dy_m": np.nan,
            "relative_displacement_error_m": np.nan,
            "longitudinal_strain_rate_error_per_day": np.nan,
            "transverse_strain_rate_error_per_day": np.nan,
        }
        if available:
            estimated_relative = np.array(
                [
                    right.estimated_dx_m - left.estimated_dx_m,
                    right.estimated_dy_m - left.estimated_dy_m,
                ],
                dtype=float,
            )
            error = estimated_relative - truth_relative
            record.update(
                {
                    "estimated_relative_dx_m": float(estimated_relative[0]),
                    "estimated_relative_dy_m": float(estimated_relative[1]),
                    "relative_displacement_error_m": float(np.linalg.norm(error)),
                    "longitudinal_strain_rate_error_per_day": float(
                        np.dot(error, direction) / baseline_m / elapsed_days
                    ),
                    "transverse_strain_rate_error_per_day": float(
                        np.dot(error, transverse) / baseline_m / elapsed_days
                    ),
                }
            )
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def triangle_records(group: pd.DataFrame) -> pd.DataFrame:
    source = group[["source_x", "source_y"]].to_numpy(dtype=float)
    if len(source) < 3:
        return pd.DataFrame()
    try:
        simplices = Delaunay(source).simplices
    except QhullError:
        return pd.DataFrame()
    truth_target = source + group[["truth_dx_m", "truth_dy_m"]].to_numpy(dtype=float)
    estimate_target = source + group[
        ["estimated_dx_m", "estimated_dy_m"]
    ].to_numpy(dtype=float)
    available = group["available"].to_numpy(dtype=bool)

    def twice_area(points: np.ndarray) -> float:
        first = points[1] - points[0]
        second = points[2] - points[0]
        return float(first[0] * second[1] - first[1] * second[0])

    rows = []
    for triangle_id, vertices in enumerate(simplices):
        source_area = twice_area(source[vertices])
        truth_area = twice_area(truth_target[vertices])
        triangle_available = bool(available[vertices].all())
        estimate_area = (
            twice_area(estimate_target[vertices]) if triangle_available else np.nan
        )
        rows.append(
            {
                **{column: group.iloc[0][column] for column in GROUP_COLUMNS},
                "triangle_id": triangle_id,
                "buoy_ids": "|".join(group.iloc[vertices]["buoy_id"].astype(str)),
                "available": triangle_available,
                "truth_area_ratio": truth_area / source_area,
                "estimated_area_ratio": estimate_area / source_area,
                "area_ratio_error": abs(estimate_area - truth_area) / abs(source_area)
                if triangle_available
                else np.nan,
                "orientation_correct": bool(estimate_area * truth_area > 0)
                if triangle_available
                else False,
            }
        )
    return pd.DataFrame.from_records(rows)


def _affine_comparison(group: pd.DataFrame) -> dict:
    selected = group.loc[group["available"]]
    source = selected[["source_x", "source_y"]].to_numpy(dtype=float)
    truth = selected[["truth_dx_m", "truth_dy_m"]].to_numpy(dtype=float)
    estimate = selected[["estimated_dx_m", "estimated_dy_m"]].to_numpy(dtype=float)
    truth_fit = fit_displacement_gradient(source, truth)
    estimate_fit = fit_displacement_gradient(source, estimate)
    if truth_fit is None or estimate_fit is None:
        return {"affine_available": False, "affine_points": int(len(selected))}
    elapsed_days = float(group["elapsed_hours"].iloc[0]) / 24.0
    truth_components = deformation_components(truth_fit["gradient"], elapsed_days)
    estimate_components = deformation_components(
        estimate_fit["gradient"], elapsed_days
    )
    result = {
        "affine_available": True,
        "affine_points": int(len(selected)),
        "gradient_frobenius_error": float(
            np.linalg.norm(estimate_fit["gradient"] - truth_fit["gradient"])
        ),
        "truth_fit_residual_median_m": truth_fit["residual_median_m"],
        "estimated_fit_residual_median_m": estimate_fit["residual_median_m"],
    }
    for label, fit in (("truth", truth_fit), ("estimated", estimate_fit)):
        for row in range(2):
            for column in range(2):
                result[f"{label}_gradient_{row}{column}"] = float(
                    fit["gradient"][row, column]
                )
    for component in truth_components:
        result[f"truth_{component}"] = truth_components[component]
        result[f"estimated_{component}"] = estimate_components[component]
        result[f"{component}_error"] = (
            estimate_components[component] - truth_components[component]
        )
    return result


def leave_one_out_records(group: pd.DataFrame) -> pd.DataFrame:
    selected = group.loc[group["available"]].reset_index(drop=True)
    if len(selected) < 4:
        return pd.DataFrame()
    rows = []
    for omitted in range(len(selected)):
        retained = selected.drop(index=omitted)
        comparison = _affine_comparison(retained)
        rows.append(
            {
                **{column: selected.iloc[0][column] for column in GROUP_COLUMNS},
                "omitted_buoy_id": selected.iloc[omitted]["buoy_id"],
                **comparison,
            }
        )
    return pd.DataFrame.from_records(rows)


def bootstrap_summary(
    group: pd.DataFrame, replicates: int, random_seed: int
) -> dict:
    selected = group.loc[group["available"]].reset_index(drop=True)
    result = {
        "bootstrap_replicates_requested": int(replicates),
        "bootstrap_replicates_valid": 0,
    }
    if replicates <= 0 or len(selected) < 3:
        return result
    pair_key = "|".join(str(selected.iloc[0][column]) for column in PAIR_COLUMNS)
    pair_offset = int(hashlib.sha256(pair_key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(random_seed + pair_offset)
    metrics = {
        "gradient_frobenius_error": [],
        "absolute_divergence_error_per_day": [],
        "absolute_shear_error_per_day": [],
    }
    for _ in range(replicates):
        indices = rng.integers(0, len(selected), size=len(selected))
        comparison = _affine_comparison(selected.iloc[indices])
        if not comparison["affine_available"]:
            continue
        metrics["gradient_frobenius_error"].append(
            comparison["gradient_frobenius_error"]
        )
        metrics["absolute_divergence_error_per_day"].append(
            abs(comparison["divergence_per_day_error"])
        )
        metrics["absolute_shear_error_per_day"].append(
            abs(comparison["shear_per_day_error"])
        )
    result["bootstrap_replicates_valid"] = len(
        metrics["gradient_frobenius_error"]
    )
    for metric, values in metrics.items():
        if values:
            result[f"{metric}_p025"] = float(np.quantile(values, 0.025))
            result[f"{metric}_median"] = float(np.median(values))
            result[f"{metric}_p975"] = float(np.quantile(values, 0.975))
    return result


def group_summary(
    group: pd.DataFrame,
    pairs: pd.DataFrame,
    triangles: pd.DataFrame,
    correct_threshold_m: float,
) -> dict:
    available = group["available"].to_numpy(dtype=bool)
    truth = group[["truth_dx_m", "truth_dy_m"]].to_numpy(dtype=float)
    estimate = group[["estimated_dx_m", "estimated_dy_m"]].to_numpy(dtype=float)
    error_vectors = estimate[available] - truth[available]
    errors = np.linalg.norm(error_vectors, axis=1)
    area_m2, diameter_m = _array_geometry(
        group[["source_x", "source_y"]].to_numpy(dtype=float)
    )
    paired_available = pairs["available"] if len(pairs) else pd.Series(dtype=bool)
    pair_errors = (
        pairs.loc[paired_available, "relative_displacement_error_m"].to_numpy()
        if len(pairs)
        else np.array([])
    )
    triangle_available = (
        triangles["available"] if len(triangles) else pd.Series(dtype=bool)
    )
    triangle_errors = (
        triangles.loc[triangle_available, "area_ratio_error"].to_numpy()
        if len(triangles)
        else np.array([])
    )
    result = {
        **{column: group.iloc[0][column] for column in GROUP_COLUMNS},
        "elapsed_hours": float(group["elapsed_hours"].iloc[0]),
        "analysis_crs": "EPSG:3413",
        "expected_buoys": int(len(group)),
        "available_buoys": int(available.sum()),
        "coverage_fraction": float(available.mean()),
        "correct_within_threshold": int((errors <= correct_threshold_m).sum()),
        "correct_fraction_of_expected": float(
            (errors <= correct_threshold_m).sum() / len(group)
        ),
        "median_endpoint_error_m": float(np.median(errors)) if len(errors) else np.nan,
        "p90_endpoint_error_m": float(np.quantile(errors, 0.90))
        if len(errors)
        else np.nan,
        "maximum_endpoint_error_m": float(np.max(errors)) if len(errors) else np.nan,
        "vector_bias_dx_m": float(error_vectors[:, 0].mean())
        if len(errors)
        else np.nan,
        "vector_bias_dy_m": float(error_vectors[:, 1].mean())
        if len(errors)
        else np.nan,
        "array_convex_hull_area_km2": area_m2 / 1e6,
        "array_diameter_km": diameter_m / 1e3,
        "expected_buoy_pairs": int(len(pairs)),
        "available_buoy_pairs": int(paired_available.sum()),
        "buoy_pair_coverage_fraction": float(paired_available.mean())
        if len(pairs)
        else np.nan,
        "median_relative_displacement_error_m": float(np.median(pair_errors))
        if len(pair_errors)
        else np.nan,
        "p90_relative_displacement_error_m": float(np.quantile(pair_errors, 0.90))
        if len(pair_errors)
        else np.nan,
        "expected_triangles": int(len(triangles)),
        "available_triangles": int(triangle_available.sum()),
        "triangle_coverage_fraction": float(triangle_available.mean())
        if len(triangles)
        else np.nan,
        "incorrect_triangle_orientations": int(
            (~triangles.loc[triangle_available, "orientation_correct"]).sum()
        )
        if len(triangles)
        else 0,
        "median_triangle_area_ratio_error": float(np.median(triangle_errors))
        if len(triangle_errors)
        else np.nan,
    }
    result.update(_affine_comparison(group))
    return result


def analyze(
    frame: pd.DataFrame,
    correct_threshold_m: float = 2_000.0,
    bootstrap_replicates: int = 1_000,
    random_seed: int = 20260818,
) -> dict[str, pd.DataFrame]:
    validated = validate_input(frame)
    summaries = []
    pair_frames = []
    triangle_frames = []
    leave_one_out_frames = []
    bootstrap_rows = []
    for _, group in validated.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        group = group.sort_values("buoy_id", kind="stable").reset_index(drop=True)
        pairs = pairwise_records(group)
        triangles = triangle_records(group)
        leave_one_out = leave_one_out_records(group)
        summary = group_summary(group, pairs, triangles, correct_threshold_m)
        bootstrap = bootstrap_summary(group, bootstrap_replicates, random_seed)
        summaries.append(summary)
        bootstrap_rows.append(
            {**{column: group.iloc[0][column] for column in GROUP_COLUMNS}, **bootstrap}
        )
        if len(pairs):
            pair_frames.append(pairs)
        if len(triangles):
            triangle_frames.append(triangles)
        if len(leave_one_out):
            leave_one_out_frames.append(leave_one_out)

    def combined(frames: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return {
        "pair_summary": pd.DataFrame.from_records(summaries),
        "pairwise_relative_displacement": combined(pair_frames),
        "triangles": combined(triangle_frames),
        "leave_one_buoy_out": combined(leave_one_out_frames),
        "bootstrap_summary": pd.DataFrame.from_records(bootstrap_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--correct-threshold-m", type=float, default=2_000.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--random-seed", type=int, default=20260818)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=[],
        help="Additional source file to hash in the run manifest; repeatable.",
    )
    args = parser.parse_args()

    outputs = analyze(
        pd.read_csv(args.input, dtype={"buoy_id": str}),
        correct_threshold_m=args.correct_threshold_m,
        bootstrap_replicates=args.bootstrap_replicates,
        random_seed=args.random_seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, output in outputs.items():
        output.to_csv(args.out_dir / f"{name}.csv", index=False)
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "input_sha256": sha256(args.input),
                "source_files": [
                    {
                        "path": str(path.resolve()),
                        "sha256": sha256(path),
                    }
                    for path in args.source
                ],
                "correct_threshold_m": args.correct_threshold_m,
                "bootstrap_replicates": args.bootstrap_replicates,
                "random_seed": args.random_seed,
                "coordinate_units": "metres",
                "analysis_crs": "EPSG:3413",
                "elapsed_time_units": "hours",
                "missing_predictions_retained_in_denominators": True,
                "affine_truth_and_estimate_use_identical_available_buoys": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(outputs["pair_summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
