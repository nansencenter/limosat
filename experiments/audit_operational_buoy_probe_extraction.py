#!/usr/bin/env python3
"""Audit every gate in operational ORB buoy-probe extraction.

This is a development-fold diagnostic. It reproduces
``KeypointDetector.keypoint_from_point`` and separates failures caused by the
local detector, the configured response threshold, the hardcoded 300 m
centre-distance gate, and descriptor computation. It does not alter LiMOSAT.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_operational_baseline import (  # noqa: E402
    DEFAULT_CONFIG,
    load_yaml,
    make_orb,
    resolve_repo_path,
    sha256_file,
    validate_frozen_config,
)


DEFAULT_OUTPUT = (
    ROOT
    / "results/arctic_tracking_next_experiment/operational_buoy_probe_audit/development"
)
COUNTERFACTUAL_RESPONSE_THRESHOLDS = (0.0, 0.0001, 0.0005, 0.001)
PRODUCTION_MAX_CENTER_DISTANCE_M = 300.0


def _candidate_map_xy(img, col: float, row: float) -> tuple[float, float]:
    x, y = img.transform_points(
        [float(col)], [float(row)], DstToSrc=0, dst_srs=img.srs
    )
    return float(x[0]), float(y[0])


def nearest_after_response_threshold(
    keypoints: list[cv2.KeyPoint],
    threshold: float,
    center_x: float,
    center_y: float,
) -> cv2.KeyPoint | None:
    eligible = [keypoint for keypoint in keypoints if keypoint.response >= threshold]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda keypoint: (keypoint.pt[0] - center_x) ** 2
        + (keypoint.pt[1] - center_y) ** 2,
    )


def classify_observation(
    detector,
    img,
    model,
    row: Any,
    response_threshold: float,
    octave: int,
) -> dict[str, Any]:
    image_data = img[1]
    image_height, image_width = image_data.shape
    col, pixel_row = detector.get_pixel_coords(img, float(row.x), float(row.y))
    if col is None or pixel_row is None:
        return {
            "failure_stage": "coordinate_transform_failed",
            "production_keypoint_expected": False,
        }

    patch_size = int(model.getPatchSize())
    window_size = max(32, patch_size + 16)
    patch_half = window_size // 2
    r0 = max(0, int(round(pixel_row)) - patch_half)
    r1 = min(
        image_height,
        int(round(pixel_row)) + patch_half + (window_size % 2),
    )
    c0 = max(0, int(round(col)) - patch_half)
    c1 = min(
        image_width,
        int(round(col)) + patch_half + (window_size % 2),
    )
    base = {
        "production_col_rounded": int(col),
        "production_row_rounded": int(pixel_row),
        "detection_window_size_pixels": window_size,
        "detection_window_height": r1 - r0,
        "detection_window_width": c1 - c0,
    }
    if (r1 - r0) < window_size * 0.8 or (c1 - c0) < window_size * 0.8:
        return {
            **base,
            "failure_stage": "truncated_detection_window",
            "production_keypoint_expected": False,
        }

    patch = image_data[r0:r1, c0:c1]
    raw_keypoints = list(model.detect(patch, None) or [])
    base["raw_detection_count"] = len(raw_keypoints)
    responses = np.asarray([kp.response for kp in raw_keypoints], dtype=float)
    base["maximum_raw_response"] = float(responses.max()) if len(responses) else math.nan
    base["median_raw_response"] = float(np.median(responses)) if len(responses) else math.nan
    base["configured_response_passing_count"] = int(
        np.sum(responses >= response_threshold)
    )
    if not raw_keypoints:
        return {
            **base,
            "failure_stage": "no_local_detection",
            "production_keypoint_expected": False,
        }

    center_x = patch.shape[1] / 2.0
    center_y = patch.shape[0] / 2.0
    center_col = c0 + center_x
    center_row = r0 + center_y
    center_map_x, center_map_y = _candidate_map_xy(img, center_col, center_row)
    base["detection_center_buoy_offset_m"] = float(
        np.hypot(center_map_x - float(row.x), center_map_y - float(row.y))
    )

    for threshold in COUNTERFACTUAL_RESPONSE_THRESHOLDS:
        label = f"threshold_{threshold:.4f}".replace(".", "p")
        selected = nearest_after_response_threshold(
            raw_keypoints, threshold, center_x, center_y
        )
        if selected is None:
            base[f"{label}_candidate_available"] = False
            base[f"{label}_center_distance_m"] = math.nan
            base[f"{label}_passes_300m_gate"] = False
            continue
        selected_col = c0 + float(selected.pt[0])
        selected_row = r0 + float(selected.pt[1])
        selected_x, selected_y = _candidate_map_xy(img, selected_col, selected_row)
        center_distance = float(
            np.hypot(selected_x - center_map_x, selected_y - center_map_y)
        )
        base[f"{label}_candidate_available"] = True
        base[f"{label}_center_distance_m"] = center_distance
        base[f"{label}_passes_300m_gate"] = (
            center_distance <= PRODUCTION_MAX_CENTER_DISTANCE_M
        )

    selected = nearest_after_response_threshold(
        raw_keypoints, response_threshold, center_x, center_y
    )
    if selected is None:
        return {
            **base,
            "failure_stage": "no_candidate_above_response_threshold",
            "production_keypoint_expected": False,
        }

    selected_col = c0 + float(selected.pt[0])
    selected_row = r0 + float(selected.pt[1])
    selected_x, selected_y = _candidate_map_xy(img, selected_col, selected_row)
    center_distance = float(
        np.hypot(selected_x - center_map_x, selected_y - center_map_y)
    )
    base.update(
        {
            "selected_col": selected_col,
            "selected_row": selected_row,
            "selected_response": float(selected.response),
            "selected_center_distance_m": center_distance,
            "selected_buoy_offset_m": float(
                np.hypot(selected_x - float(row.x), selected_y - float(row.y))
            ),
        }
    )
    descriptor_half_patch = int(float(selected.size) / 2.0)
    descriptor_in_bounds = (
        descriptor_half_patch <= selected_col < image_width - descriptor_half_patch
        and descriptor_half_patch <= selected_row < image_height - descriptor_half_patch
    )
    base["descriptor_footprint_in_bounds"] = bool(descriptor_in_bounds)
    if not descriptor_in_bounds:
        return {
            **base,
            "failure_stage": "descriptor_footprint_out_of_bounds",
            "production_keypoint_expected": False,
        }
    if center_distance > PRODUCTION_MAX_CENTER_DISTANCE_M:
        return {
            **base,
            "failure_stage": "selected_feature_beyond_300m_gate",
            "production_keypoint_expected": False,
        }
    return {
        **base,
        "failure_stage": "production_keypoint_returned",
        "production_keypoint_expected": True,
    }


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    summary_rows = [{"metric": "observations", "value": float(len(results))}]
    for column in (
        "production_keypoint_returned",
        "production_descriptor_available",
        "exact_supplied_descriptor_available",
    ):
        summary_rows.append(
            {
                "metric": f"{column}_fraction",
                "value": float(results[column].fillna(False).astype(bool).mean()),
            }
        )
    for threshold in COUNTERFACTUAL_RESPONSE_THRESHOLDS:
        label = f"threshold_{threshold:.4f}".replace(".", "p")
        summary_rows.append(
            {
                "metric": f"{label}_passes_300m_gate_fraction",
                "value": float(
                    results[f"{label}_passes_300m_gate"]
                    .fillna(False)
                    .astype(bool)
                    .mean()
                ),
            }
        )
    for reason, count in results["failure_stage"].value_counts(dropna=False).items():
        summary_rows.append(
            {"metric": f"failure_stage_count::{reason}", "value": float(count)}
        )
    return pd.DataFrame.from_records(summary_rows)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    from limosat.image import Image
    from limosat.keypoint_detector import KeypointDetector

    started = time.perf_counter()
    config = load_yaml(args.config)
    validate_frozen_config(config)
    observations_path = resolve_repo_path(
        args.observations or config["paths"]["tracking_observations"]
    )
    observations = pd.read_csv(observations_path, low_memory=False)
    observations = observations.loc[
        observations["within_dataset_split"] == args.split
    ].copy()
    observations["probe_id"] = (
        observations["buoy_id"].astype(str)
        + "|"
        + observations["image_id"].astype(str)
    )
    if observations["probe_id"].duplicated().any():
        raise ValueError("Expected unique buoy_id/image_id probe rows")
    if args.max_observations is not None:
        observations = observations.iloc[: args.max_observations].copy()

    response_threshold = float(config["image_processor_params"]["response_threshold"])
    octave = int(config["image_processor_params"]["octave"])
    model = make_orb(config)
    detector = KeypointDetector(model)
    records: list[dict[str, Any]] = []

    for image_path, group in observations.groupby("image_filepath", sort=True):
        img = Image(image_path)
        point_frame = gpd.GeoDataFrame(
            group.copy(),
            geometry=[Point(float(x), float(y)) for x, y in zip(group["x"], group["y"])],
            crs="EPSG:3413",
        ).set_index("probe_id", drop=False)
        production_keypoints = detector.keypoint_from_point(
            point_frame,
            octave=octave,
            img=img,
            response_threshold=response_threshold,
        )
        production_by_probe = {
            str(tag): keypoint for keypoint, tag in production_keypoints
        }
        _, _, descriptor_tags = detector.compute_descriptors(
            production_keypoints, img, polarisation=1, normalize=False
        )
        descriptor_probe_ids = set(map(str, descriptor_tags or []))

        exact_keypoints = []
        for row in group.itertuples(index=False):
            exact_col, exact_row = img.transform_points(
                [float(row.x)], [float(row.y)], DstToSrc=1, dst_srs=img.srs
            )
            exact_keypoints.append(
                (
                    cv2.KeyPoint(
                        float(exact_col[0]),
                        float(exact_row[0]),
                        size=31.0,
                        angle=float(img.angle),
                        octave=octave,
                    ),
                    str(row.probe_id),
                )
            )
        _, _, exact_descriptor_tags = detector.compute_descriptors(
            exact_keypoints, img, polarisation=1, normalize=False
        )
        exact_descriptor_probe_ids = set(map(str, exact_descriptor_tags or []))

        for row in group.itertuples(index=False):
            classified = classify_observation(
                detector, img, model, row, response_threshold, octave
            )
            production_keypoint = production_by_probe.get(str(row.probe_id))
            production_returned = production_keypoint is not None
            if production_returned != bool(classified["production_keypoint_expected"]):
                raise RuntimeError(
                    f"Reproduction disagrees with production for {row.probe_id}"
                )
            if production_returned:
                expected_pixel = np.asarray(
                    [classified["selected_col"], classified["selected_row"]], dtype=float
                )
                pixel_difference = float(
                    np.linalg.norm(np.asarray(production_keypoint.pt) - expected_pixel)
                )
                if pixel_difference > 1.0e-3:
                    raise RuntimeError(
                        f"Selected pixel disagrees for {row.probe_id}: {pixel_difference}"
                    )
            else:
                pixel_difference = math.nan
            descriptor_available = str(row.probe_id) in descriptor_probe_ids
            failure_stage = classified["failure_stage"]
            if production_returned and not descriptor_available:
                failure_stage = "production_descriptor_compute_failed"
            records.append(
                {
                    **classified,
                    "probe_id": row.probe_id,
                    "buoy_id": row.buoy_id,
                    "image_id": int(row.image_id),
                    "image_time": row.image_time,
                    "image_filepath": image_path,
                    "within_dataset_split": row.within_dataset_split,
                    "production_keypoint_returned": production_returned,
                    "production_descriptor_available": descriptor_available,
                    "exact_supplied_descriptor_available": str(row.probe_id)
                    in exact_descriptor_probe_ids,
                    "production_reproduction_pixel_difference": pixel_difference,
                    "failure_stage": failure_stage,
                }
            )

    results = pd.DataFrame.from_records(records).sort_values(
        ["image_time", "buoy_id"], kind="stable"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "probe_extraction_records.csv", index=False)

    summary = summarize_results(results)
    summary.to_csv(args.out_dir / "summary.csv", index=False)

    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "split": args.split,
        "observations": int(len(results)),
        "images": int(results["image_id"].nunique()),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "observations_path": str(observations_path),
        "observations_sha256": sha256_file(observations_path),
        "configured_response_threshold": response_threshold,
        "hardcoded_max_center_distance_m": PRODUCTION_MAX_CENTER_DISTANCE_M,
        "counterfactual_response_thresholds": list(COUNTERFACTUAL_RESPONSE_THRESHOLDS),
        "elapsed_seconds": time.perf_counter() - started,
        "method_scores_used_for_selection": False,
        "interpretation": (
            "Development-fold mechanism audit only. Counterfactual thresholds isolate "
            "gate loss and are not deployment choices."
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"manifest": manifest, "summary": summary.to_dict("records")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--split", default="development")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-observations", type=int)
    return parser.parse_args()


def main() -> int:
    payload = audit(parse_args())
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
