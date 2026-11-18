#!/usr/bin/env python3
"""Gate an EfficientLoFTR batch candidate on final fields and trajectories.

This evaluator deliberately ignores raw-match parity. It compares the products
that enter the scientific workflow: availability, 4 km displacement fields,
adjacent-vector roughness, final trajectories, and optional truth errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_LIMITS = {
    "field_availability_difference_fraction": 0.002,
    "field_vector_p95_m": 40.0,
    "field_vector_p99_m": 100.0,
    "field_vector_maximum_m": 1_000.0,
    "adjacent_jump_p99_relative_increase": 0.05,
    "trajectory_p95_m": 50.0,
    "trajectory_p99_m": 150.0,
    "trajectory_maximum_m": 1_000.0,
    "truth_median_error_increase_m": 10.0,
    "truth_p90_error_increase_m": 10.0,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "count": 0,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    return {
        "count": int(len(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "maximum": float(np.max(finite)),
    }


def boolean_values(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy(bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).to_numpy() != 0
    return values.astype(str).str.lower().isin(("true", "1")).to_numpy(bool)


def ordered_frames(reference_path: Path, candidate_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.read_csv(reference_path)
    candidate = pd.read_csv(candidate_path)
    if "grid_id" not in reference or "grid_id" not in candidate:
        raise ValueError("field and trajectory CSVs must contain grid_id")
    if not np.array_equal(
        reference["grid_id"].to_numpy(str), candidate["grid_id"].to_numpy(str)
    ):
        raise ValueError("reference and candidate grid order differs")
    return reference, candidate


def nearest_distance(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return exact planar nearest distances without adding SciPy as a dependency."""
    points = np.asarray(points, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if not len(points) or not len(targets):
        return np.full(len(points), np.nan, dtype=np.float64)
    result = np.full(len(points), np.inf, dtype=np.float64)
    for start in range(0, len(targets), 4096):
        chunk = targets[start : start + 4096]
        squared = ((points[:, None, :] - chunk[None, :, :]) ** 2).sum(axis=2)
        result = np.minimum(result, np.sqrt(squared.min(axis=1)))
    return result


def availability_boundary(frame: pd.DataFrame, available: np.ndarray) -> np.ndarray:
    coordinates = frame[["grid_row", "grid_column"]].to_numpy(np.int64)
    lookup = {tuple(value): index for index, value in enumerate(coordinates)}
    boundary = np.zeros(len(frame), dtype=bool)
    for index, (row, column) in enumerate(coordinates):
        if not available[index]:
            continue
        for neighbor in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            other = lookup.get(neighbor)
            if other is None or not available[other]:
                boundary[index] = True
                break
    return boundary


def adjacent_jumps(frame: pd.DataFrame, available: np.ndarray) -> np.ndarray:
    coordinates = frame[["grid_row", "grid_column"]].to_numpy(np.int64)
    displacement = frame[["proposal_dx_m", "proposal_dy_m"]].to_numpy(np.float64)
    lookup = {tuple(value): index for index, value in enumerate(coordinates)}
    jumps = []
    for index, (row, column) in enumerate(coordinates):
        if not available[index]:
            continue
        for neighbor in ((row + 1, column), (row, column + 1)):
            other = lookup.get(neighbor)
            if other is not None and available[other]:
                jumps.append(float(np.linalg.norm(displacement[index] - displacement[other])))
    return np.asarray(jumps, dtype=np.float64)


def field_comparison(
    reference_path: Path,
    candidate_path: Path,
    *,
    limits: dict[str, float],
    outlier_threshold_m: float,
) -> dict[str, Any]:
    reference, candidate = ordered_frames(reference_path, candidate_path)
    required = {
        "grid_row", "grid_column", "source_x", "source_y", "available",
        "proposal_dx_m", "proposal_dy_m",
    }
    for name, frame in (("reference", reference), ("candidate", candidate)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} field lacks columns: {sorted(missing)}")

    reference_available = boolean_values(reference["available"])
    candidate_available = boolean_values(candidate["available"])
    common = reference_available & candidate_available
    changed = reference_available != candidate_available
    vector_error = np.linalg.norm(
        candidate.loc[common, ["proposal_dx_m", "proposal_dy_m"]].to_numpy(float)
        - reference.loc[common, ["proposal_dx_m", "proposal_dy_m"]].to_numpy(float),
        axis=1,
    )
    vector_summary = distribution(vector_error)
    common_indices = np.flatnonzero(common)
    outlier_indices = common_indices[vector_error >= outlier_threshold_m]
    xy = reference[["source_x", "source_y"]].to_numpy(float)
    boundary = availability_boundary(reference, reference_available)
    changed_distance = nearest_distance(xy[outlier_indices], xy[changed])
    boundary_distance = nearest_distance(xy[outlier_indices], xy[boundary])
    errors_by_index = dict(zip(common_indices, vector_error, strict=True))
    outliers = []
    optional = (
        "selected_vectors", "candidate_count", "support_radius_m",
        "maximum_vector_residual_m",
    )
    for position, index in enumerate(outlier_indices):
        record = {
            "grid_id": str(reference.iloc[index]["grid_id"]),
            "source_x_m": float(xy[index, 0]),
            "source_y_m": float(xy[index, 1]),
            "vector_error_m": float(errors_by_index[index]),
            "reference_dx_m": float(reference.iloc[index]["proposal_dx_m"]),
            "reference_dy_m": float(reference.iloc[index]["proposal_dy_m"]),
            "candidate_dx_m": float(candidate.iloc[index]["proposal_dx_m"]),
            "candidate_dy_m": float(candidate.iloc[index]["proposal_dy_m"]),
            "distance_to_reference_boundary_m": (
                float(boundary_distance[position]) if np.isfinite(boundary_distance[position]) else None
            ),
            "distance_to_availability_change_m": (
                float(changed_distance[position]) if np.isfinite(changed_distance[position]) else None
            ),
        }
        for column in optional:
            if column in reference:
                record[f"reference_{column}"] = (
                    float(reference.iloc[index][column])
                    if pd.notna(reference.iloc[index][column])
                    else None
                )
            if column in candidate:
                record[f"candidate_{column}"] = (
                    float(candidate.iloc[index][column])
                    if pd.notna(candidate.iloc[index][column])
                    else None
                )
        outliers.append(record)
    outliers.sort(key=lambda item: item["vector_error_m"], reverse=True)

    reference_jump = distribution(adjacent_jumps(reference, common))
    candidate_jump = distribution(adjacent_jumps(candidate, common))
    reference_p99 = float(reference_jump["p99"] or 0.0)
    candidate_p99 = float(candidate_jump["p99"] or 0.0)
    jump_relative_increase = (
        candidate_p99 / reference_p99 - 1.0 if reference_p99 else None
    )
    difference_fraction = float(changed.mean())
    checks = {
        "availability_difference": difference_fraction
        <= limits["field_availability_difference_fraction"],
        "vector_p95": bool(vector_summary["count"])
        and float(vector_summary["p95"]) <= limits["field_vector_p95_m"],
        "vector_p99": bool(vector_summary["count"])
        and float(vector_summary["p99"]) <= limits["field_vector_p99_m"],
        "vector_maximum": bool(vector_summary["count"])
        and float(vector_summary["maximum"]) <= limits["field_vector_maximum_m"],
        "adjacent_jump_p99": (
            jump_relative_increase <= limits["adjacent_jump_p99_relative_increase"]
            if jump_relative_increase is not None
            else candidate_p99 == 0.0
        ),
    }
    topology: dict[str, Any] = {"checked": False}
    reference_summary_path = reference_path.parent / "summary.json"
    candidate_summary_path = candidate_path.parent / "summary.json"
    if reference_summary_path.exists() and candidate_summary_path.exists():
        reference_topology = json.loads(reference_summary_path.read_text()).get(
            "topology_after_rejection", {}
        )
        candidate_topology = json.loads(candidate_summary_path.read_text()).get(
            "topology_after_rejection", {}
        )
        if "flipped_triangles" in reference_topology and "flipped_triangles" in candidate_topology:
            no_new_folds = int(candidate_topology["flipped_triangles"]) <= int(
                reference_topology["flipped_triangles"]
            )
            checks["no_new_flipped_triangles"] = no_new_folds
            topology = {
                "checked": True,
                "reference": reference_topology,
                "candidate": candidate_topology,
                "no_new_flipped_triangles": no_new_folds,
            }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "grid_nodes": int(len(reference)),
        "reference_available": int(reference_available.sum()),
        "candidate_available": int(candidate_available.sum()),
        "reference_only_available": int((reference_available & ~candidate_available).sum()),
        "candidate_only_available": int((candidate_available & ~reference_available).sum()),
        "availability_symmetric_difference": int(changed.sum()),
        "availability_symmetric_difference_fraction": difference_fraction,
        "common_vector_error_m": vector_summary,
        "outlier_threshold_m": outlier_threshold_m,
        "outlier_count": len(outliers),
        "outliers": outliers,
        "common_adjacent_jump_m": {
            "reference": reference_jump,
            "candidate": candidate_jump,
            "p99_relative_increase": jump_relative_increase,
        },
        "topology_after_rejection": topology,
        "reference_sha256": file_sha256(reference_path),
        "candidate_sha256": file_sha256(candidate_path),
    }


def trajectory_comparison(
    reference_path: Path,
    candidate_path: Path,
    *,
    limits: dict[str, float],
) -> dict[str, Any]:
    reference, candidate = ordered_frames(reference_path, candidate_path)
    required = {
        "pair1_valid", "pair2_valid", "complete", "pair2_target_x_m", "pair2_target_y_m",
    }
    for name, frame in (("reference", reference), ("candidate", candidate)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} trajectory lacks columns: {sorted(missing)}")
    reference_complete = boolean_values(reference["complete"])
    candidate_complete = boolean_values(candidate["complete"])
    common = reference_complete & candidate_complete
    error = np.linalg.norm(
        candidate.loc[common, ["pair2_target_x_m", "pair2_target_y_m"]].to_numpy(float)
        - reference.loc[common, ["pair2_target_x_m", "pair2_target_y_m"]].to_numpy(float),
        axis=1,
    )
    summary = distribution(error)
    reference_only = reference_complete & ~candidate_complete
    candidate_only = candidate_complete & ~reference_complete
    checks = {
        "no_complete_trajectory_loss": not reference_only.any(),
        "terminal_p95": bool(summary["count"])
        and float(summary["p95"]) <= limits["trajectory_p95_m"],
        "terminal_p99": bool(summary["count"])
        and float(summary["p99"]) <= limits["trajectory_p99_m"],
        "terminal_maximum": bool(summary["count"])
        and float(summary["maximum"]) <= limits["trajectory_maximum_m"],
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "denominator": int(len(reference)),
        "reference_complete": int(reference_complete.sum()),
        "candidate_complete": int(candidate_complete.sum()),
        "reference_only_complete": int(reference_only.sum()),
        "candidate_only_complete": int(candidate_only.sum()),
        "complete_symmetric_difference": int((reference_complete != candidate_complete).sum()),
        "common_terminal_error_m": summary,
        "reference_sha256": file_sha256(reference_path),
        "candidate_sha256": file_sha256(candidate_path),
    }


def truth_comparison(
    reference_path: Path,
    candidate_path: Path,
    *,
    limits: dict[str, float],
) -> dict[str, Any]:
    reference = pd.read_csv(reference_path)
    candidate = pd.read_csv(candidate_path)
    key_columns = [column for column in ("buoy_id", "image_id", "stage") if column in reference]
    if not key_columns or any(column not in candidate for column in key_columns):
        raise ValueError("truth CSVs need shared buoy_id plus image_id or stage keys")
    required = {"available", "error_m"}
    for name, frame in (("reference", reference), ("candidate", candidate)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} truth table lacks columns: {sorted(missing)}")
    reference = reference.set_index(key_columns).sort_index()
    candidate = candidate.set_index(key_columns).sort_index()
    if not reference.index.equals(candidate.index):
        raise ValueError("reference and candidate truth rows differ")
    reference_available = boolean_values(reference["available"])
    candidate_available = boolean_values(candidate["available"])
    common = reference_available & candidate_available
    reference_error = distribution(reference.loc[common, "error_m"].to_numpy(float))
    candidate_error = distribution(candidate.loc[common, "error_m"].to_numpy(float))
    median_increase = float(candidate_error["median"] or 0.0) - float(reference_error["median"] or 0.0)
    p90_increase = float(candidate_error["p90"] or 0.0) - float(reference_error["p90"] or 0.0)
    checks = {
        "no_truth_availability_loss": not (reference_available & ~candidate_available).any(),
        "median_error": median_increase <= limits["truth_median_error_increase_m"],
        "p90_error": p90_increase <= limits["truth_p90_error_increase_m"],
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "common_available": int(common.sum()),
        "reference_error_m": reference_error,
        "candidate_error_m": candidate_error,
        "median_error_increase_m": median_increase,
        "p90_error_increase_m": p90_increase,
        "reference_sha256": file_sha256(reference_path),
        "candidate_sha256": file_sha256(candidate_path),
    }


def evaluate(
    baseline_fields: Path,
    candidate_fields: Path,
    pairs: list[str],
    baseline_trajectory: Path,
    candidate_trajectory: Path,
    *,
    limits: dict[str, float] | None = None,
    outlier_threshold_m: float = 250.0,
    baseline_truth: Path | None = None,
    candidate_truth: Path | None = None,
) -> dict[str, Any]:
    resolved_limits = {**DEFAULT_LIMITS, **(limits or {})}
    fields = {
        pair: field_comparison(
            baseline_fields / pair / "field_4km.csv",
            candidate_fields / pair / "field_4km.csv",
            limits=resolved_limits,
            outlier_threshold_m=outlier_threshold_m,
        )
        for pair in pairs
    }
    trajectory = trajectory_comparison(
        baseline_trajectory, candidate_trajectory, limits=resolved_limits
    )
    truth = None
    if (baseline_truth is None) != (candidate_truth is None):
        raise ValueError("both truth tables are required when truth checking is enabled")
    if baseline_truth is not None and candidate_truth is not None:
        truth = truth_comparison(
            baseline_truth, candidate_truth, limits=resolved_limits
        )
    checks = {
        "fields": all(result["status"] == "pass" for result in fields.values()),
        "trajectories": trajectory["status"] == "pass",
        "truth": truth is None or truth["status"] == "pass",
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "limits": resolved_limits,
        "fields": fields,
        "trajectories": trajectory,
        "truth": truth,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-fields", type=Path, required=True)
    parser.add_argument("--candidate-fields", type=Path, required=True)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--baseline-trajectory", type=Path, required=True)
    parser.add_argument("--candidate-trajectory", type=Path, required=True)
    parser.add_argument("--baseline-truth", type=Path)
    parser.add_argument("--candidate-truth", type=Path)
    parser.add_argument("--outlier-threshold-m", type=float, default=250.0)
    parser.add_argument("--limits-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limits_json:
        document = json.loads(args.limits_json.read_text())
        limits = document.get("limits", document)
    else:
        limits = None
    result = evaluate(
        args.baseline_fields,
        args.candidate_fields,
        args.pairs,
        args.baseline_trajectory,
        args.candidate_trajectory,
        limits=limits,
        outlier_threshold_m=args.outlier_threshold_m,
        baseline_truth=args.baseline_truth,
        candidate_truth=args.candidate_truth,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
