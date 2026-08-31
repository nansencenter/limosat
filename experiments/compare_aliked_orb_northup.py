#!/usr/bin/env python3
"""Fair development comparison of north-up ALIKED and supplied-point ORB.

The representative panel contains one deterministic transition from every
eligible development image-pair cluster. Rare ORB failures form a separate
challenge panel and never change the representative aggregate. Target truth is
used only after proposal generation to score errors and audit tile coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import kornia
import numpy as np
import pandas as pd
import torch
from kornia.feature import ALIKED, LightGlueMatcher, laf_from_center_scale_ori
from nansat import NSR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.buoy_descriptor_benchmark import image_object, read_scene
from experiments.replay_aliked_candidate_policies import estimate_policy


DEFAULT_TRANSITIONS = (
    ROOT / "results/arctic_tracking_next_experiment/splits/full70_2020/transitions.csv"
)
DEFAULT_OBSERVATIONS = (
    ROOT / "results/arctic_tracking_next_experiment/splits/full70_2020/observations.csv"
)
DEFAULT_ORB_RUN = Path(
    "/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/"
    "operational_baseline/runs/"
    "limosat_dense_30000__b1_full70_dev_exact_instrumented_persistfix_20260817"
)
ANALYSIS_EPSG = 3413


def stable_rank(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def select_cases(
    transitions: pd.DataFrame,
    orb_fates: pd.DataFrame,
    seed: str,
    maximum_elapsed_hours: float,
    maximum_speed_km_per_day: float,
    include_spatial_blocks: bool = False,
) -> pd.DataFrame:
    """Select one case per image-pair cluster plus an explicit challenge panel."""
    eligible = transitions.loc[
        transitions["within_dataset_split"].eq("development")
        & transitions["elapsed_hours"].le(maximum_elapsed_hours)
        & transitions["truth_speed_km_per_day"].le(maximum_speed_km_per_day)
    ].copy()
    eligible = eligible.merge(
        orb_fates[
            [
                "transition_id",
                "trajectory_id",
                "target_run_image_id",
                "outcome_stage",
                "candidate_count",
                "pattern_accepted",
                "measurement_represented_error_m",
            ]
        ],
        on="transition_id",
        how="left",
        validate="one_to_one",
    )
    eligible["selection_rank"] = eligible["transition_id"].map(
        lambda value: stable_rank(str(value), seed)
    )
    pair_columns = ["source_image_id", "target_image_id"]
    if include_spatial_blocks:
        pair_columns.append("source_spatial_block")
    representative = (
        eligible.sort_values(pair_columns + ["selection_rank"])
        .groupby(pair_columns, as_index=False, sort=True)
        .head(1)
    )
    rare_outcomes = {
        "no_descriptor_candidate",
        "motion_gate",
        "pattern_correlation",
    }
    challenge = eligible.loc[eligible["outcome_stage"].isin(rare_outcomes)]
    selected = pd.concat([representative, challenge], ignore_index=True).drop_duplicates(
        "transition_id"
    )
    representative_ids = set(representative["transition_id"])
    challenge_ids = set(challenge["transition_id"])
    selected["representative_panel"] = selected["transition_id"].isin(
        representative_ids
    )
    selected["challenge_panel"] = selected["transition_id"].isin(challenge_ids)
    return selected.sort_values(
        ["source_image_id", "target_image_id", "selection_rank"]
    ).reset_index(drop=True)


def attach_source_coordinates(
    cases: pd.DataFrame, observations: pd.DataFrame
) -> pd.DataFrame:
    """Attach the buoy position in EPSG:3413 at each source acquisition."""
    source = observations[["buoy_id", "image_id", "x", "y", "analysis_crs"]].rename(
        columns={
            "image_id": "source_image_id",
            "x": "source_x",
            "y": "source_y",
            "analysis_crs": "source_analysis_crs",
        }
    )
    if source.duplicated(["buoy_id", "source_image_id"]).any():
        raise ValueError("Source observations are not unique by buoy and image")
    merged = cases.merge(
        source,
        on=["buoy_id", "source_image_id"],
        how="left",
        validate="many_to_one",
    )
    if merged[["source_x", "source_y"]].isna().any(axis=None):
        missing = merged.loc[
            merged[["source_x", "source_y"]].isna().any(axis=1), "transition_id"
        ].tolist()
        raise ValueError(f"Missing source coordinates for transitions: {missing}")
    if not merged["source_analysis_crs"].astype(str).str.contains("3413").all():
        raise ValueError("Expected all buoy source coordinates in EPSG:3413")
    return merged


def round_up(value: float, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def tile_pixels(
    half_extent_m: float,
    pixel_size_m: float,
    minimum_pixels: int = 512,
    multiple: int = 32,
) -> int:
    requested = 2.0 * half_extent_m / pixel_size_m + 1.0
    return max(minimum_pixels, round_up(requested, multiple))


def north_up_patch(
    image_path: str,
    center_x: float,
    center_y: float,
    pixels: int,
    pixel_size_m: float,
    transform_grid_spacing_pixels: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a projected, north-up VAE tile and its validity mask."""
    image, mask = read_scene(image_path)
    coarse_pixels = int(
        math.ceil((pixels - 1) / transform_grid_spacing_pixels) + 1
    )
    offsets = np.linspace(
        -(pixels - 1) / 2.0 * pixel_size_m,
        (pixels - 1) / 2.0 * pixel_size_m,
        coarse_pixels,
        dtype=np.float64,
    )
    projected_x, projected_y_offset = np.meshgrid(center_x + offsets, offsets)
    projected_y = center_y - projected_y_offset
    columns, rows = image_object(image_path, ANALYSIS_EPSG).transform_points(
        projected_x.ravel(),
        projected_y.ravel(),
        DstToSrc=1,
        dst_srs=NSR(ANALYSIS_EPSG),
    )
    coarse_x = np.asarray(columns, dtype=np.float32).reshape(
        coarse_pixels, coarse_pixels
    )
    coarse_y = np.asarray(rows, dtype=np.float32).reshape(
        coarse_pixels, coarse_pixels
    )
    sample_x = interpolate_transform_grid(coarse_x, pixels)
    sample_y = interpolate_transform_grid(coarse_y, pixels)
    finite = np.isfinite(sample_x) & np.isfinite(sample_y)
    safe_x = np.where(finite, sample_x, -1).astype(np.float32)
    safe_y = np.where(finite, sample_y, -1).astype(np.float32)
    patch = cv2.remap(
        image,
        safe_x,
        safe_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    in_bounds = (
        finite
        & (sample_x >= 0)
        & (sample_x <= image.shape[1] - 1)
        & (sample_y >= 0)
        & (sample_y <= image.shape[0] - 1)
    )
    if mask is None:
        valid = in_bounds
    else:
        sampled_mask = cv2.remap(
            mask,
            safe_x,
            safe_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=2,
        )
        valid = in_bounds & (sampled_mask < 2)
    patch = patch.astype(np.uint8, copy=False)
    patch[~valid] = 0
    return patch, valid, sample_x, sample_y


def interpolate_transform_grid(coarse: np.ndarray, pixels: int) -> np.ndarray:
    """Bilinearly interpolate a geolocation grid without OpenCV map quantization."""
    coarse_positions = np.linspace(0.0, pixels - 1.0, coarse.shape[0])
    full_positions = np.arange(pixels, dtype=float)
    horizontal = np.empty((coarse.shape[0], pixels), dtype=np.float64)
    for row in range(coarse.shape[0]):
        horizontal[row] = np.interp(full_positions, coarse_positions, coarse[row])
    full = np.empty((pixels, pixels), dtype=np.float32)
    for column in range(pixels):
        full[:, column] = np.interp(
            full_positions, coarse_positions, horizontal[:, column]
        )
    return full


def sampling_grid_approximation_error(
    image_path: str,
    center_x: float,
    center_y: float,
    pixels: int,
    pixel_size_m: float,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
) -> dict[str, float | int]:
    """Compare the coarse interpolated transform with exact GDAL transforms."""
    indices = np.unique(np.rint(np.linspace(0, pixels - 1, 7)).astype(int))
    columns, rows = np.meshgrid(indices, indices)
    offsets_x = (columns.ravel() - (pixels - 1) / 2.0) * pixel_size_m
    offsets_y = (rows.ravel() - (pixels - 1) / 2.0) * pixel_size_m
    exact_x, exact_y = image_object(image_path, ANALYSIS_EPSG).transform_points(
        center_x + offsets_x,
        center_y - offsets_y,
        DstToSrc=1,
        dst_srs=NSR(ANALYSIS_EPSG),
    )
    approximate_x = sample_x[rows.ravel(), columns.ravel()]
    approximate_y = sample_y[rows.ravel(), columns.ravel()]
    differences = np.hypot(
        approximate_x - np.asarray(exact_x),
        approximate_y - np.asarray(exact_y),
    )
    finite = differences[np.isfinite(differences)]
    return {
        "points": int(len(differences)),
        "finite_points": int(len(finite)),
        "median_native_pixel_error": float(np.median(finite)),
        "maximum_native_pixel_error": float(np.max(finite)),
    }


def tile_origins(pixels: int, tile_pixels: int, overlap_pixels: int) -> list[int]:
    if tile_pixels <= overlap_pixels:
        raise ValueError("Target tile size must exceed overlap")
    if pixels <= tile_pixels:
        return [0]
    count = int(math.ceil((pixels - tile_pixels) / (tile_pixels - overlap_pixels)) + 1)
    return sorted(
        set(np.rint(np.linspace(0, pixels - tile_pixels, count)).astype(int).tolist())
    )


def retain_best_match_per_source(
    source_indices: np.ndarray, scores: np.ndarray
) -> np.ndarray:
    """Prevent overlapping target tiles from overweighting one source feature."""
    if not len(source_indices):
        return np.empty(0, dtype=int)
    order = np.argsort(-scores, kind="stable")
    _, first = np.unique(source_indices[order], return_index=True)
    return np.sort(order[first])


def projected_coordinates(
    keypoints: torch.Tensor,
    center_x: float,
    center_y: float,
    pixels: int,
    pixel_size_m: float,
) -> np.ndarray:
    values = keypoints.detach().cpu().numpy()
    center = (pixels - 1) / 2.0
    return np.column_stack(
        (
            center_x + (values[:, 0] - center) * pixel_size_m,
            center_y - (values[:, 1] - center) * pixel_size_m,
        )
    )


def valid_feature_subset(
    features,
    valid_mask: np.ndarray,
    maximum_features: int,
    support_radius_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Remove invalid-support detections and retain strongest valid features."""
    kernel_size = support_radius_pixels * 2 + 1
    support = cv2.erode(
        valid_mask.astype(np.uint8),
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    rounded = torch.round(features.keypoints).to(torch.long)
    columns = rounded[:, 0].clamp(0, support.shape[1] - 1).cpu().numpy()
    rows = rounded[:, 1].clamp(0, support.shape[0] - 1).cpu().numpy()
    keep = np.flatnonzero(support[rows, columns])
    excluded = int(len(features.keypoints) - len(keep))
    if len(keep) > maximum_features:
        keep_tensor = torch.as_tensor(keep, device=features.keypoint_scores.device)
        scores = features.keypoint_scores[keep_tensor]
        strongest = torch.topk(scores, maximum_features, sorted=True).indices
        keep_tensor = keep_tensor[strongest]
    else:
        keep_tensor = torch.as_tensor(keep, device=features.keypoint_scores.device)
    return (
        features.keypoints[keep_tensor],
        features.descriptors[keep_tensor],
        features.keypoint_scores[keep_tensor],
        excluded,
    )


def inverse_distance_local_proposal(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    scores: np.ndarray,
    source_truth: np.ndarray,
    elapsed_days: float,
    source_radius_m: float,
    maximum_speed_m_per_day: float,
    neighbours: int,
) -> dict[str, float | int | bool]:
    """Apply the frozen up-to-four local-vector estimator to learned matches."""
    if len(source_xy) == 0:
        return {"available": False, "local_match_count": 0}
    displacement = target_xy - source_xy
    source_distance = np.linalg.norm(source_xy - source_truth, axis=1)
    speed = np.linalg.norm(displacement, axis=1) / elapsed_days
    local = (source_distance <= source_radius_m) & (
        speed <= maximum_speed_m_per_day
    )
    candidates = np.flatnonzero(local)
    if not len(candidates):
        return {"available": False, "local_match_count": 0}
    ordered = candidates[np.argsort(source_distance[candidates])[:neighbours]]
    distances = source_distance[ordered]
    weights = 1.0 / np.maximum(distances, 1.0)
    estimated = np.average(displacement[ordered], axis=0, weights=weights)
    highest_confidence = candidates[np.argmax(scores[candidates])]
    nearest_tight = candidates[source_distance[candidates] <= 2000.0]
    nearest_tight_index = (
        int(nearest_tight[np.argmin(source_distance[nearest_tight])])
        if len(nearest_tight)
        else None
    )
    return {
        "available": True,
        "local_match_count": int(len(candidates)),
        "neighbour_count": int(len(ordered)),
        "maximum_neighbour_distance_m": float(distances.max()),
        "vector_spread_m": float(
            np.max(np.linalg.norm(displacement[ordered] - estimated, axis=1))
        ),
        "proposal_dx_m": float(estimated[0]),
        "proposal_dy_m": float(estimated[1]),
        "highest_confidence_dx_m": float(displacement[highest_confidence, 0]),
        "highest_confidence_dy_m": float(displacement[highest_confidence, 1]),
        "highest_confidence_source_distance_m": float(
            source_distance[highest_confidence]
        ),
        "nearest_tight_available": nearest_tight_index is not None,
        "nearest_tight_dx_m": float(displacement[nearest_tight_index, 0])
        if nearest_tight_index is not None
        else math.nan,
        "nearest_tight_dy_m": float(displacement[nearest_tight_index, 1])
        if nearest_tight_index is not None
        else math.nan,
    }


def pattern_refine(
    source_path: str,
    target_path: str,
    source_xy: np.ndarray,
    proposal_xy: np.ndarray,
    template_half_size: int,
    search_border: int,
    subpixel_method: str = "none",
    template_sampling: str = "integer",
) -> dict[str, float | bool]:
    """Run the operational template rotations around one proposed endpoint."""
    if subpixel_method not in {
        "none",
        "aligned_integer",
        "quadratic",
        "continuous",
    }:
        raise ValueError(
            "subpixel_method must be 'none', 'aligned_integer', 'quadratic', "
            "or 'continuous'"
        )
    if template_sampling not in {"integer", "bilinear"}:
        raise ValueError("template_sampling must be 'integer' or 'bilinear'")
    source_image, source_mask = read_scene(source_path)
    target_image, target_mask = read_scene(target_path)
    source_object = image_object(source_path, ANALYSIS_EPSG)
    target_object = image_object(target_path, ANALYSIS_EPSG)
    source_col, source_row = source_object.transform_points(
        [source_xy[0]], [source_xy[1]], DstToSrc=1, dst_srs=NSR(ANALYSIS_EPSG)
    )
    target_col, target_row = target_object.transform_points(
        [proposal_xy[0]], [proposal_xy[1]], DstToSrc=1, dst_srs=NSR(ANALYSIS_EPSG)
    )
    source_col_float = float(source_col[0])
    source_row_float = float(source_row[0])
    target_col_float = float(target_col[0])
    target_row_float = float(target_row[0])
    sc, sr = int(source_col_float), int(source_row_float)
    tc, tr = int(target_col[0]), int(target_row[0])
    hs = template_half_size
    if not (
        sr - hs >= 0
        and sr + hs + 1 <= source_image.shape[0]
        and sc - hs >= 0
        and sc + hs + 1 <= source_image.shape[1]
    ):
        return {"available": False, "correlation": -1.0}
    if template_sampling == "integer":
        template = source_image[sr - hs : sr + hs + 1, sc - hs : sc + hs + 1]
    else:
        template = cv2.getRectSubPix(
            source_image,
            (2 * hs + 1, 2 * hs + 1),
            (source_col_float, source_row_float),
        )
    if source_mask is not None and np.any(
        source_mask[sr - hs : sr + hs + 1, sc - hs : sc + hs + 1] >= 2
    ):
        return {"available": False, "correlation": -1.0}
    if np.var(template) < 1.0e-9:
        return {"available": False, "correlation": -1.0}
    r0, r1 = tr - hs - search_border, tr + hs + 1 + search_border
    c0, c1 = tc - hs - search_border, tc + hs + 1 + search_border
    if not (r0 >= 0 and c0 >= 0 and r1 <= target_image.shape[0] and c1 <= target_image.shape[1]):
        return {"available": False, "correlation": -1.0}
    search = target_image[r0:r1, c0:c1]
    search_valid_fraction = 1.0
    if target_mask is not None:
        search_valid_fraction = float(np.mean(target_mask[r0:r1, c0:c1] < 2))
    angle_difference = source_object.angle - target_object.angle
    best_correlation = -1.0
    best_offset = (0, 0)
    best_location = None
    best_response = None
    best_rotated = None
    best_rotation_mask = None
    for angle_offset in (0, -15, 15, -30, 30):
        if best_correlation >= 0.65 and angle_offset != 0:
            break
        rotation = angle_difference + angle_offset
        if abs(rotation) < 1.0e-3:
            rotated = template
            rotation_mask = np.ones_like(rotated, dtype=bool)
            response = cv2.matchTemplate(search, rotated, cv2.TM_CCOEFF_NORMED)
        else:
            matrix = cv2.getRotationMatrix2D((hs, hs), rotation, 1.0)
            rotated = cv2.warpAffine(template, matrix, template.shape[::-1])
            rotation_mask = (rotated > 0).astype(np.uint8)
            if rotation_mask.sum() < 0.5 * rotated.size:
                continue
            response = cv2.matchTemplate(
                search, rotated, cv2.TM_CCOEFF_NORMED, mask=rotation_mask
            )
        _, correlation, _, location = cv2.minMaxLoc(response)
        if correlation > best_correlation:
            best_correlation = float(correlation)
            best_offset = (
                int((location[0] + hs) - (hs + search_border)),
                int((location[1] + hs) - (hs + search_border)),
            )
            best_location = location
            best_response = response
            best_rotated = rotated
            best_rotation_mask = rotation_mask.astype(bool)
        if angle_offset == 0 and best_correlation >= 0.65:
            break
    delta_col = 0.0
    delta_row = 0.0
    refinement_status = "legacy_integer"
    if subpixel_method == "quadratic" and best_location is not None:
        from limosat.processing import refine_correlation_peak_quadratic

        delta_col, delta_row, refinement_status = (
            refine_correlation_peak_quadratic(best_response, best_location)
        )
        corrected_col = float(tc + best_offset[0] + delta_col)
        corrected_row = float(tr + best_offset[1] + delta_row)
    elif subpixel_method == "continuous" and best_location is not None:
        from scipy.optimize import minimize

        reference_template = np.asarray(best_rotated, dtype=np.float64)
        reference_mask = np.asarray(best_rotation_mask, dtype=bool)
        reference_values = reference_template[reference_mask]
        reference_values = reference_values - reference_values.mean()
        reference_norm = float(np.linalg.norm(reference_values))

        def negative_ncc(offset):
            patch = cv2.getRectSubPix(
                target_image,
                reference_template.shape[::-1],
                (
                    float(tc + best_offset[0] + offset[0]),
                    float(tr + best_offset[1] + offset[1]),
                ),
                patchType=cv2.CV_32F,
            )
            values = np.asarray(patch, dtype=np.float64)[reference_mask]
            values = values - values.mean()
            denominator = reference_norm * float(np.linalg.norm(values))
            if denominator <= 0 or not np.isfinite(denominator):
                return 1.0
            return -float(np.dot(reference_values, values) / denominator)

        continuous = minimize(
            negative_ncc,
            np.zeros(2, dtype=float),
            method="Powell",
            bounds=((-1.0, 1.0), (-1.0, 1.0)),
            options={"xtol": 1.0e-4, "ftol": 1.0e-8, "maxiter": 50},
        )
        if continuous.success and np.isfinite(continuous.x).all():
            delta_col, delta_row = map(float, continuous.x)
            refinement_status = "continuous_reference"
        else:
            refinement_status = "continuous_fallback"
        corrected_col = float(tc + best_offset[0] + delta_col)
        corrected_row = float(tr + best_offset[1] + delta_row)
    elif subpixel_method == "aligned_integer":
        refinement_status = "aligned_integer"
        corrected_col = float(tc + best_offset[0])
        corrected_row = float(tr + best_offset[1])
    else:
        corrected_col = float(target_col_float + best_offset[0])
        corrected_row = float(target_row_float + best_offset[1])
    corrected_x, corrected_y = target_object.transform_points(
        [corrected_col],
        [corrected_row],
        DstToSrc=0,
        dst_srs=NSR(ANALYSIS_EPSG),
    )
    return {
        "available": True,
        "correlation": best_correlation,
        "accepted": bool(best_correlation >= 0.30),
        "corrected_x": float(corrected_x[0]),
        "corrected_y": float(corrected_y[0]),
        "correction_pixels": float(np.hypot(*best_offset)),
        "subpixel_col": delta_col,
        "subpixel_row": delta_row,
        "subpixel_status": refinement_status,
        "target_search_valid_fraction": search_valid_fraction,
    }


def error_m(x: float, y: float, truth: np.ndarray) -> float:
    return float(np.hypot(x - truth[0], y - truth[1]))


def load_orb_candidates(path: Path) -> pd.DataFrame:
    candidates = pd.read_json(path, lines=True)
    counts = candidates.groupby(["trajectory_id", "target_image_id"]).size()
    if int(counts.max()) > 1:
        raise ValueError("Expected at most one operational ORB candidate per probe/image")
    return candidates


def wrapped_angle_difference_degrees(source_angle: float, target_angle: float) -> float:
    return float((source_angle - target_angle + 180.0) % 360.0 - 180.0)


def correctness_rate(frame: pd.DataFrame, available: pd.Series, error: pd.Series) -> dict:
    accepted = available.fillna(False).astype(bool)
    correct = accepted & error.le(2000.0)
    accepted_errors = error.loc[accepted & error.notna()]
    return {
        "available_count": int(accepted.sum()),
        "correct_within_2km_count": int(correct.sum()),
        "availability_percent_of_panel": float(100.0 * accepted.mean())
        if len(frame)
        else None,
        "correct_within_2km_percent_of_panel": float(100.0 * correct.mean())
        if len(frame)
        else None,
        "median_error_m_when_available": float(accepted_errors.median())
        if len(accepted_errors)
        else None,
        "p90_error_m_when_available": float(accepted_errors.quantile(0.90))
        if len(accepted_errors)
        else None,
    }


def summarize_panel(frame: pd.DataFrame) -> dict:
    def values(name: str, default) -> pd.Series:
        if name in frame:
            return frame[name]
        return pd.Series(default, index=frame.index)

    aliked_available = values("aliked_available", False)
    aliked_pm_accepted = values("aliked_pm_accepted", False).fillna(False)
    orb_available = values("orb_available", False)
    orb_motion = values("orb_motion_pass", False).fillna(False)
    orb_pm_accepted = values("orb_pm_accepted", False).fillna(False)
    return {
        "cases": int(len(frame)),
        "aliked_proposal": correctness_rate(
            frame, aliked_available, values("aliked_pre_pattern_error_m", math.nan)
        ),
        "aliked_after_pattern_matching": correctness_rate(
            frame, aliked_pm_accepted, values("aliked_pm_error_m", math.nan)
        ),
        "orb_proposal": correctness_rate(
            frame,
            orb_available & orb_motion,
            values("orb_pre_pattern_error_m", math.nan),
        ),
        "orb_after_pattern_matching": correctness_rate(
            frame, orb_pm_accepted, values("orb_pm_error_m", math.nan)
        ),
        "target_inside_physics_tile_count": int(
            frame["target_inside_physics_tile"].sum()
        ),
        "median_absolute_native_rotation_difference_degrees": float(
            values("absolute_native_rotation_difference_degrees", math.nan).median()
        )
        if len(frame)
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--orb-run", type=Path, default=DEFAULT_ORB_RUN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--selection-seed", default="aliked_northup_v1_20260817")
    parser.add_argument("--pixel-size-m", type=float, default=80.0)
    parser.add_argument("--source-radius-m", type=float, default=10000.0)
    parser.add_argument("--context-margin-m", type=float, default=5000.0)
    parser.add_argument("--maximum-speed-m-per-day", type=float, default=30000.0)
    parser.add_argument("--maximum-elapsed-hours", type=float, default=30.0)
    parser.add_argument("--source-features", type=int, default=2048)
    parser.add_argument("--target-features-per-tile", type=int, default=2048)
    parser.add_argument("--target-tile-pixels", type=int, default=512)
    parser.add_argument("--target-tile-overlap-pixels", type=int, default=64)
    parser.add_argument("--support-radius-pixels", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument(
        "--representative-unit",
        choices=("image_pair", "image_pair_spatial_block"),
        default="image_pair",
    )
    parser.add_argument(
        "--propagated-consensus-paths",
        action="store_true",
        help=(
            "Restrict to consecutive representative paths and centre each crop "
            "on the previous consensus endpoint."
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transitions = pd.read_csv(args.transitions, dtype={"buoy_id": str}, low_memory=False)
    observations = pd.read_csv(
        args.observations, dtype={"buoy_id": str}, low_memory=False
    )
    fates_path = args.orb_run / "buoy_probe_stage_fates/one_step_stage_fates.csv"
    orb_fates = pd.read_csv(fates_path, low_memory=False)
    cases = select_cases(
        transitions,
        orb_fates,
        args.selection_seed,
        args.maximum_elapsed_hours,
        args.maximum_speed_m_per_day / 1000.0,
        include_spatial_blocks=args.representative_unit == "image_pair_spatial_block",
    )
    cases = attach_source_coordinates(cases, observations)
    if args.propagated_consensus_paths:
        cases = cases.loc[cases["representative_panel"]].copy()
        path_size = cases.groupby("continuous_trajectory_id")[
            "transition_id"
        ].transform("size")
        cases = cases.loc[path_size >= 2].sort_values(
            ["continuous_trajectory_id", "source_image_time"]
        )
    if args.limit_cases is not None:
        cases = cases.head(args.limit_cases).copy()
    cases.to_csv(args.output_dir / "frozen_cases.csv", index=False)

    candidates = load_orb_candidates(
        args.orb_run / "stage_audit/matcher_candidates.jsonl"
    )
    device = torch.device(args.device)
    torch.manual_seed(20260817)
    torch.hub.set_dir(str(args.model_cache / "hub"))
    aliked = ALIKED.from_pretrained(
        model_name="aliked-n16",
        max_num_keypoints=max(args.source_features, args.target_features_per_tile),
        detection_threshold=0.2,
        device=device,
    ).eval()
    lightglue = LightGlueMatcher("aliked").to(device).eval()

    one_channel_check = None
    self_match_check = None
    transform_grid_check = None
    records = []
    match_records = []
    propagated_states = {}
    for case in cases.itertuples(index=False):
        started = time.perf_counter()
        source_truth = np.array([case.source_x, case.source_y], dtype=float)
        target_truth = source_truth + np.array(
            [case.truth_dx_m, case.truth_dy_m], dtype=float
        )
        if args.propagated_consensus_paths:
            source_tracking = propagated_states.setdefault(
                case.continuous_trajectory_id, source_truth.copy()
            ).copy()
        else:
            source_tracking = source_truth
        elapsed_days = float(case.elapsed_hours) / 24.0
        source_half_extent = args.source_radius_m + args.context_margin_m
        target_half_extent = (
            elapsed_days * args.maximum_speed_m_per_day
            + args.source_radius_m
            + args.context_margin_m
        )
        source_pixels = tile_pixels(source_half_extent, args.pixel_size_m)
        target_pixels = tile_pixels(target_half_extent, args.pixel_size_m)
        source_patch, source_valid, source_sample_x, source_sample_y = north_up_patch(
            case.source_image_filepath,
            source_tracking[0],
            source_tracking[1],
            source_pixels,
            args.pixel_size_m,
        )
        target_patch, target_valid, target_sample_x, target_sample_y = north_up_patch(
            case.target_image_filepath,
            source_tracking[0],
            source_tracking[1],
            target_pixels,
            args.pixel_size_m,
        )
        resampling_seconds = time.perf_counter() - started
        if transform_grid_check is None:
            transform_grid_check = {
                "source": sampling_grid_approximation_error(
                    case.source_image_filepath,
                    source_tracking[0],
                    source_tracking[1],
                    source_pixels,
                    args.pixel_size_m,
                    source_sample_x,
                    source_sample_y,
                ),
                "target": sampling_grid_approximation_error(
                    case.target_image_filepath,
                    source_tracking[0],
                    source_tracking[1],
                    target_pixels,
                    args.pixel_size_m,
                    target_sample_x,
                    target_sample_y,
                ),
            }
        del source_sample_x, source_sample_y, target_sample_x, target_sample_y
        source_native_angle = float(
            image_object(case.source_image_filepath, ANALYSIS_EPSG).angle
        )
        target_native_angle = float(
            image_object(case.target_image_filepath, ANALYSIS_EPSG).angle
        )
        native_angle_difference = wrapped_angle_difference_degrees(
            source_native_angle, target_native_angle
        )
        source_tensor = (
            torch.from_numpy(source_patch.copy()).to(device=device, dtype=torch.float32)[
                None, None
            ]
            / 255.0
        )
        target_tensor = (
            torch.from_numpy(target_patch.copy()).to(device=device, dtype=torch.float32)[
                None, None
            ]
            / 255.0
        )
        if one_channel_check is None:
            with torch.inference_mode():
                gray_features = aliked(source_tensor)[0]
                rgb_features = aliked(source_tensor.repeat(1, 3, 1, 1))[0]
            one_channel_check = {
                "gray_keypoints": int(len(gray_features.keypoints)),
                "rgb_keypoints": int(len(rgb_features.keypoints)),
                "keypoints_equal": bool(
                    torch.equal(gray_features.keypoints, rgb_features.keypoints)
                ),
                "descriptors_equal": bool(
                    torch.equal(gray_features.descriptors, rgb_features.descriptors)
                ),
                "scores_equal": bool(
                    torch.equal(gray_features.keypoint_scores, rgb_features.keypoint_scores)
                ),
            }
            if not all(
                one_channel_check[key]
                for key in ("keypoints_equal", "descriptors_equal", "scores_equal")
            ):
                raise RuntimeError("One-channel and repeated-RGB ALIKED inputs differ")
        feature_started = time.perf_counter()
        with torch.inference_mode():
            source_features = aliked(source_tensor)[0]
        source_keypoints, source_descriptors, _, source_invalid_excluded = (
            valid_feature_subset(
                source_features,
                source_valid,
                args.source_features,
                args.support_radius_pixels,
            )
        )
        source_laf = laf_from_center_scale_ori(source_keypoints[None])
        extraction_seconds = time.perf_counter() - feature_started
        if self_match_check is None:
            with torch.inference_mode():
                self_scores, self_indexes = lightglue(
                    source_descriptors,
                    source_descriptors,
                    source_laf,
                    source_laf,
                    hw1=(source_pixels, source_pixels),
                    hw2=(source_pixels, source_pixels),
                )
            self_match_check = {
                "keypoints": int(len(source_keypoints)),
                "matches": int(len(self_indexes)),
                "same_index_matches": int(
                    (self_indexes[:, 0] == self_indexes[:, 1]).sum().item()
                ),
            }
        source_projected = projected_coordinates(
            source_keypoints,
            source_tracking[0],
            source_tracking[1],
            source_pixels,
            args.pixel_size_m,
        )
        target_origins = [
            (row, column)
            for row in tile_origins(
                target_pixels,
                args.target_tile_pixels,
                args.target_tile_overlap_pixels,
            )
            for column in tile_origins(
                target_pixels,
                args.target_tile_pixels,
                args.target_tile_overlap_pixels,
            )
        ]
        all_source_indices = []
        all_target_projected = []
        all_match_scores = []
        target_features_before_mask = 0
        target_feature_count = 0
        target_invalid_excluded = 0
        raw_tile_matches = 0
        matching_seconds = 0.0
        for row_start, column_start in target_origins:
            row_stop = min(row_start + args.target_tile_pixels, target_pixels)
            column_stop = min(column_start + args.target_tile_pixels, target_pixels)
            tile_tensor = target_tensor[
                :, :, row_start:row_stop, column_start:column_stop
            ]
            tile_valid = target_valid[row_start:row_stop, column_start:column_stop]
            tile_feature_started = time.perf_counter()
            with torch.inference_mode():
                tile_features = aliked(tile_tensor)[0]
            extraction_seconds += time.perf_counter() - tile_feature_started
            target_features_before_mask += int(len(tile_features.keypoints))
            tile_keypoints, tile_descriptors, _, tile_invalid = valid_feature_subset(
                tile_features,
                tile_valid,
                args.target_features_per_tile,
                args.support_radius_pixels,
            )
            target_feature_count += int(len(tile_keypoints))
            target_invalid_excluded += tile_invalid
            if len(source_keypoints) < 2 or len(tile_keypoints) < 2:
                continue
            tile_laf = laf_from_center_scale_ori(tile_keypoints[None])
            matching_started = time.perf_counter()
            with torch.inference_mode():
                tile_scores, tile_indexes = lightglue(
                    source_descriptors,
                    tile_descriptors,
                    source_laf,
                    tile_laf,
                    hw1=(source_pixels, source_pixels),
                    hw2=(row_stop - row_start, column_stop - column_start),
                )
            matching_seconds += time.perf_counter() - matching_started
            indexes = tile_indexes.detach().cpu().numpy()
            if not len(indexes):
                continue
            raw_tile_matches += int(len(indexes))
            full_target_keypoints = tile_keypoints.clone()
            full_target_keypoints[:, 0] += column_start
            full_target_keypoints[:, 1] += row_start
            tile_projected = projected_coordinates(
                full_target_keypoints,
                source_tracking[0],
                source_tracking[1],
                target_pixels,
                args.pixel_size_m,
            )
            all_source_indices.append(indexes[:, 0])
            all_target_projected.append(tile_projected[indexes[:, 1]])
            all_match_scores.append(tile_scores.detach().cpu().numpy().reshape(-1))
        if all_source_indices:
            source_indices = np.concatenate(all_source_indices)
            matched_target = np.concatenate(all_target_projected)
            scores = np.concatenate(all_match_scores)
            keep = retain_best_match_per_source(source_indices, scores)
            source_indices = source_indices[keep]
            matched_target = matched_target[keep]
            scores = scores[keep]
            matched_source = source_projected[source_indices]
        else:
            matched_source = source_projected[:0]
            matched_target = source_projected[:0]
            scores = np.empty(0, dtype=float)
        if len(matched_source):
            displacement = matched_target - matched_source
            source_distance = np.linalg.norm(
                matched_source - source_tracking, axis=1
            )
            speed_m_per_day = np.linalg.norm(displacement, axis=1) / elapsed_days
            physics_valid = (
                (source_distance <= args.source_radius_m)
                & (speed_m_per_day <= args.maximum_speed_m_per_day)
            )
            for match_index in range(len(matched_source)):
                match_records.append(
                    {
                        "transition_id": case.transition_id,
                        "match_index": match_index,
                        "source_x": float(matched_source[match_index, 0]),
                        "source_y": float(matched_source[match_index, 1]),
                        "target_x": float(matched_target[match_index, 0]),
                        "target_y": float(matched_target[match_index, 1]),
                        "dx_m": float(displacement[match_index, 0]),
                        "dy_m": float(displacement[match_index, 1]),
                        "source_distance_m": float(source_distance[match_index]),
                        "speed_m_per_day": float(speed_m_per_day[match_index]),
                        "lightglue_score": float(scores[match_index]),
                        "physics_valid": bool(physics_valid[match_index]),
                    }
                )
        else:
            displacement = np.empty((0, 2), dtype=float)
            source_distance = np.empty(0, dtype=float)
            speed_m_per_day = np.empty(0, dtype=float)
            physics_valid = np.empty(0, dtype=bool)
        if args.propagated_consensus_paths:
            proposal = estimate_policy(
                pd.DataFrame(
                    {
                        "physics_valid": physics_valid,
                        "source_distance_m": source_distance,
                        "dx_m": displacement[:, 0],
                        "dy_m": displacement[:, 1],
                        "lightglue_score": scores,
                    }
                ),
                "consensus_within_2km",
                tight_radius_m=2000.0,
                consensus_radius_m=1000.0,
            )
        else:
            proposal = inverse_distance_local_proposal(
                matched_source,
                matched_target,
                scores,
                source_truth,
                elapsed_days,
                args.source_radius_m,
                args.maximum_speed_m_per_day,
                neighbours=4,
            )
        aliked_pm = {"available": False, "correlation": -1.0}
        if proposal["available"]:
            aliked_xy = source_tracking + np.array(
                [proposal["proposal_dx_m"], proposal["proposal_dy_m"]]
            )
            proposal["pre_pattern_error_m"] = error_m(
                aliked_xy[0], aliked_xy[1], target_truth
            )
            aliked_pm = pattern_refine(
                case.source_image_filepath,
                case.target_image_filepath,
                source_tracking,
                aliked_xy,
                template_half_size=16,
                search_border=48,
            )
            if aliked_pm.get("available"):
                aliked_pm["error_m"] = error_m(
                    aliked_pm["corrected_x"],
                    aliked_pm["corrected_y"],
                    target_truth,
                )
        orb_rows = candidates.loc[
            candidates["trajectory_id"].eq(int(case.trajectory_id))
            & candidates["target_image_id"].eq(int(case.target_run_image_id))
        ]
        orb_proposal = {"available": False}
        orb_pm = {"available": False, "correlation": -1.0}
        if len(orb_rows) == 1 and not args.propagated_consensus_paths:
            orb_row = orb_rows.iloc[0]
            orb_proposal = {
                "available": True,
                "motion_pass": bool(orb_row.motion_pass),
                "descriptor_distance": float(orb_row.descriptor_distance),
                "pre_pattern_error_m": error_m(
                    float(orb_row.target_x), float(orb_row.target_y), target_truth
                ),
            }
            if orb_proposal["motion_pass"]:
                orb_pm = pattern_refine(
                    case.source_image_filepath,
                    case.target_image_filepath,
                    source_truth,
                    np.array([orb_row.target_x, orb_row.target_y], dtype=float),
                    template_half_size=16,
                    search_border=48,
                )
                if orb_pm.get("available"):
                    orb_pm["error_m"] = error_m(
                        orb_pm["corrected_x"], orb_pm["corrected_y"], target_truth
                    )
        if args.propagated_consensus_paths and proposal["available"]:
            propagated_states[case.continuous_trajectory_id] = aliked_xy
        target_offset = target_truth - source_tracking
        target_inside_tile = bool(
            np.all(np.abs(target_offset) <= (target_pixels - 1) / 2 * args.pixel_size_m)
        )
        records.append(
            {
                **case._asdict(),
                "tracking_source_x": float(source_tracking[0]),
                "tracking_source_y": float(source_tracking[1]),
                "tracking_source_error_m": float(
                    np.linalg.norm(source_tracking - source_truth)
                ),
                "source_pixels": source_pixels,
                "target_pixels": target_pixels,
                "target_inside_physics_tile": target_inside_tile,
                "source_native_angle_degrees": source_native_angle,
                "target_native_angle_degrees": target_native_angle,
                "native_rotation_difference_degrees": native_angle_difference,
                "absolute_native_rotation_difference_degrees": abs(
                    native_angle_difference
                ),
                "source_valid_fraction": float(source_valid.mean()),
                "target_valid_fraction": float(target_valid.mean()),
                "source_features_before_mask": int(len(source_features.keypoints)),
                "target_tiles": int(len(target_origins)),
                "target_features_before_mask": target_features_before_mask,
                "source_features": int(len(source_keypoints)),
                "target_features": target_feature_count,
                "source_invalid_features_excluded": source_invalid_excluded,
                "target_invalid_features_excluded": target_invalid_excluded,
                "aliked_raw_tile_matches": raw_tile_matches,
                "aliked_unique_source_matches": int(len(scores)),
                "aliked_extraction_seconds": extraction_seconds,
                "aliked_matching_seconds": matching_seconds,
                "north_up_resampling_seconds": resampling_seconds,
                "case_total_seconds": time.perf_counter() - started,
                **{f"aliked_{key}": value for key, value in proposal.items()},
                **{f"aliked_pm_{key}": value for key, value in aliked_pm.items()},
                **{f"orb_{key}": value for key, value in orb_proposal.items()},
                **{f"orb_pm_{key}": value for key, value in orb_pm.items()},
            }
        )
    results = pd.DataFrame.from_records(records)
    results.to_csv(args.output_dir / "transition_results.csv", index=False)
    pd.DataFrame.from_records(match_records).to_csv(
        args.output_dir / "aliked_match_vectors.csv", index=False
    )
    representative = results.loc[results["representative_panel"]]
    challenge = results.loc[results["challenge_panel"]]
    summary = {
        "representative_panel": summarize_panel(representative),
        "challenge_panel": summarize_panel(challenge),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    manifest = {
        "status": "complete",
        "diagnostic_case_limit": args.limit_cases,
        "selection_seed": args.selection_seed,
        "representative_cases": int(results.representative_panel.sum()),
        "challenge_cases": int(results.challenge_panel.sum()),
        "total_unique_cases": int(len(results)),
        "selection_unit": args.representative_unit,
        "propagated_consensus_paths": args.propagated_consensus_paths,
        "challenge_outcomes": [
            "no_descriptor_candidate",
            "motion_gate",
            "pattern_correlation",
        ],
        "eligibility": {
            "within_dataset_split": "development",
            "maximum_elapsed_hours": args.maximum_elapsed_hours,
            "maximum_truth_speed_km_per_day": args.maximum_speed_m_per_day / 1000.0,
        },
        "input": "one-channel standard VAE; ALIKED broadcasts internally to RGB",
        "one_channel_equivalence": one_channel_check,
        "self_match_check": self_match_check,
        "projection": "north-up EPSG:3413",
        "transform_grid_spacing_pixels": 32,
        "transform_grid_approximation_check": transform_grid_check,
        "pixel_size_m": args.pixel_size_m,
        "source_radius_m": args.source_radius_m,
        "context_margin_m": args.context_margin_m,
        "target_extent_rule": "30 km/day * elapsed time + source radius + context margin",
        "source_features": args.source_features,
        "target_features_per_tile": args.target_features_per_tile,
        "target_tile_pixels": args.target_tile_pixels,
        "target_tile_overlap_pixels": args.target_tile_overlap_pixels,
        "target_feature_budget_rule": "constant per-tile feature density; best match retained per source feature across overlapping tiles",
        "invalid_support_erosion_pixels": args.support_radius_pixels,
        "proposal": "inverse-distance average of up to four nearest local physics-valid vectors",
        "pattern_matching": {
            "template_half_size_pixels": 16,
            "search_border_pixels": 48,
            "minimum_correlation": 0.30,
            "rotation_offsets_degrees": [0, -15, 15, -30, 30],
        },
        "device": str(device),
        "software": {
            "torch": torch.__version__,
            "kornia": kornia.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "summary": summary,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
