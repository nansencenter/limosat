#!/usr/bin/env python3
"""Validate tiled ALIKED displacement/deformation on known real-image warps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import shapely
import torch
from kornia.feature import ALIKED

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_aliked_orb_northup import (
    north_up_patch,
    valid_feature_subset,
)
from experiments.run_aliked_dense_pair import (
    adaptive_consensus_at_queries,
    match_tiles,
    nearest_consensus_at_queries,
    projected_footprint,
    topology_summary,
)
from experiments.aliked_matchers import build_aliked_matcher


SCENARIOS = {
    "rigid_fractional": {
        "deformation_gradient": [[1.0, 0.0], [0.0, 1.0]],
        "translation_m": [1224.0, -608.0],
    },
    "affine_divergence_shear": {
        "deformation_gradient": [[1.010, 0.006], [0.004, 0.992]],
        "translation_m": [960.0, -720.0],
    },
    "piecewise_lead_opening": {
        "left_translation_m": [800.0, -400.0],
        "right_translation_m": [1400.0, -400.0],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def warp_image(
    source: np.ndarray,
    source_valid: np.ndarray,
    deformation_gradient: np.ndarray,
    translation_m: np.ndarray,
    pixel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a target whose forward map is q = F p + translation."""
    height, width = source.shape
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    target = np.column_stack((columns.ravel(), rows.ravel()))
    inverse = np.linalg.inv(deformation_gradient)
    translation_pixels = translation_m / pixel_size_m
    sample = (
        (target - center - translation_pixels) @ inverse.T + center
    ).astype(np.float32)
    map_x = sample[:, 0].reshape(height, width)
    map_y = sample[:, 1].reshape(height, width)
    warped = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.remap(
        source_valid.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    return warped, valid


def warp_piecewise_translation(
    source: np.ndarray,
    source_valid: np.ndarray,
    left_translation_m: np.ndarray,
    right_translation_m: np.ndarray,
    pixel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Open a vertical lead between two independently translated ice plates."""
    height, width = source.shape
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    center_column = (width - 1) / 2.0
    left_pixels = left_translation_m / pixel_size_m
    right_pixels = right_translation_m / pixel_size_m
    left_target_edge = center_column + left_pixels[0]
    right_target_edge = center_column + right_pixels[0]
    use_left = columns < left_target_edge
    use_right = columns >= right_target_edge
    map_x = np.full_like(columns, -1.0)
    map_y = np.full_like(rows, -1.0)
    map_x[use_left] = columns[use_left] - left_pixels[0]
    map_y[use_left] = rows[use_left] - left_pixels[1]
    map_x[use_right] = columns[use_right] - right_pixels[0]
    map_y[use_right] = rows[use_right] - right_pixels[1]
    warped = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.remap(
        source_valid.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    valid &= use_left | use_right
    return warped, valid


def array_tiles(
    image: np.ndarray,
    valid: np.ndarray,
    model,
    device: torch.device,
    pixel_size_m: float,
    tile_pixels: int,
    margin_pixels: int,
    features_per_tile: int,
    support_radius_pixels: int,
) -> tuple[list[dict], pd.DataFrame]:
    core_pixels = tile_pixels - 2 * margin_pixels
    rows = range(0, image.shape[0], core_pixels)
    columns = range(0, image.shape[1], core_pixels)
    tiles = []
    audits = []
    tile_id = 0
    for core_row in rows:
        for core_column in columns:
            patch_row = core_row - margin_pixels
            patch_column = core_column - margin_pixels
            patch = np.zeros((tile_pixels, tile_pixels), dtype=image.dtype)
            patch_valid = np.zeros((tile_pixels, tile_pixels), dtype=bool)
            source_r0 = max(0, patch_row)
            source_c0 = max(0, patch_column)
            source_r1 = min(image.shape[0], patch_row + tile_pixels)
            source_c1 = min(image.shape[1], patch_column + tile_pixels)
            target_r0 = source_r0 - patch_row
            target_c0 = source_c0 - patch_column
            target_r1 = target_r0 + source_r1 - source_r0
            target_c1 = target_c0 + source_c1 - source_c0
            patch[target_r0:target_r1, target_c0:target_c1] = image[
                source_r0:source_r1, source_c0:source_c1
            ]
            patch_valid[target_r0:target_r1, target_c0:target_c1] = valid[
                source_r0:source_r1, source_c0:source_c1
            ]
            tensor = (
                torch.from_numpy(patch.copy()).to(
                    device=device, dtype=torch.float32
                )[None, None]
                / 255.0
            )
            started = time.perf_counter()
            with torch.inference_mode():
                raw = model(tensor)[0]
            keypoints, descriptors, scores, invalid_excluded = valid_feature_subset(
                raw,
                patch_valid,
                features_per_tile,
                support_radius_pixels,
            )
            core_width = min(core_pixels, image.shape[1] - core_column)
            core_height = min(core_pixels, image.shape[0] - core_row)
            inside_core = (
                (keypoints[:, 0] >= margin_pixels)
                & (keypoints[:, 0] < margin_pixels + core_width)
                & (keypoints[:, 1] >= margin_pixels)
                & (keypoints[:, 1] < margin_pixels + core_height)
            )
            keypoints = keypoints[inside_core].detach().cpu()
            descriptors = descriptors[inside_core].detach().cpu()
            scores = scores[inside_core].detach().cpu()
            values = keypoints.numpy()
            xy = np.column_stack(
                (
                    (patch_column + values[:, 0]) * pixel_size_m,
                    (patch_row + values[:, 1]) * pixel_size_m,
                )
            )
            core = shapely.box(
                core_column * pixel_size_m,
                core_row * pixel_size_m,
                (core_column + core_width) * pixel_size_m,
                (core_row + core_height) * pixel_size_m,
            )
            tiles.append(
                {
                    "tile_id": tile_id,
                    "row": core_row // core_pixels,
                    "column": core_column // core_pixels,
                    "center_x": (core_column + core_width / 2.0) * pixel_size_m,
                    "center_y": (core_row + core_height / 2.0) * pixel_size_m,
                    "core": core,
                    "keypoints": keypoints,
                    "descriptors": descriptors,
                    "scores": scores,
                    "xy": xy,
                }
            )
            audits.append(
                {
                    "tile_id": tile_id,
                    "raw_features": int(len(raw.keypoints)),
                    "retained_core_features": int(len(keypoints)),
                    "invalid_support_excluded": invalid_excluded,
                    "seconds": time.perf_counter() - started,
                }
            )
            tile_id += 1
    return tiles, pd.DataFrame(audits)


def queries_for_shape(
    shape: tuple[int, int], pixel_size_m: float, spacing_m: float, margin_m: float
) -> pd.DataFrame:
    xs = np.arange(margin_m, shape[1] * pixel_size_m - margin_m, spacing_m)
    ys = np.arange(margin_m, shape[0] * pixel_size_m - margin_m, spacing_m)
    x_grid, y_grid = np.meshgrid(xs, ys)
    return pd.DataFrame(
        {
            "grid_row": np.repeat(np.arange(len(ys)), len(xs)),
            "grid_column": np.tile(np.arange(len(xs)), len(ys)),
            "source_x": x_grid.ravel(),
            "source_y": y_grid.ravel(),
        }
    )


def truth_at_queries(
    queries: pd.DataFrame,
    shape: tuple[int, int],
    pixel_size_m: float,
    deformation_gradient: np.ndarray,
    translation_m: np.ndarray,
) -> np.ndarray:
    center = np.array(
        [
            (shape[1] - 1) / 2.0 * pixel_size_m,
            (shape[0] - 1) / 2.0 * pixel_size_m,
        ]
    )
    source = queries[["source_x", "source_y"]].to_numpy(dtype=float)
    target = (source - center) @ deformation_gradient.T + center + translation_m
    return target - source


def piecewise_truth_at_queries(
    queries: pd.DataFrame,
    shape: tuple[int, int],
    pixel_size_m: float,
    left_translation_m: np.ndarray,
    right_translation_m: np.ndarray,
) -> np.ndarray:
    center_x = (shape[1] - 1) / 2.0 * pixel_size_m
    left = queries["source_x"].to_numpy() < center_x
    truth = np.repeat(right_translation_m[None], len(queries), axis=0)
    truth[left] = left_translation_m
    return truth


def valid_truth_queries(
    queries: pd.DataFrame,
    source_valid: np.ndarray,
    target_valid: np.ndarray,
    truth_vectors: np.ndarray,
    pixel_size_m: float,
) -> np.ndarray:
    source_columns = np.rint(queries["source_x"] / pixel_size_m).astype(int)
    source_rows = np.rint(queries["source_y"] / pixel_size_m).astype(int)
    target_columns = np.rint(
        (queries["source_x"].to_numpy() + truth_vectors[:, 0]) / pixel_size_m
    ).astype(int)
    target_rows = np.rint(
        (queries["source_y"].to_numpy() + truth_vectors[:, 1]) / pixel_size_m
    ).astype(int)
    target_inside = (
        (target_columns >= 0)
        & (target_columns < target_valid.shape[1])
        & (target_rows >= 0)
        & (target_rows < target_valid.shape[0])
    )
    valid = source_valid[source_rows, source_columns] & target_inside
    valid[target_inside] &= target_valid[
        target_rows[target_inside], target_columns[target_inside]
    ]
    return valid


def fit_affine(field: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    available = field.loc[field["available"].fillna(False)]
    source = available[["source_x", "source_y"]].to_numpy(dtype=float)
    target = source + available[["proposal_dx_m", "proposal_dy_m"]].to_numpy(
        dtype=float
    )
    design = np.column_stack((source, np.ones(len(source))))
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return coefficients[:2].T, coefficients[2]


def scenario_summary(
    field: pd.DataFrame,
    truth_vectors: np.ndarray,
    true_gradient: np.ndarray,
    shape: tuple[int, int],
    pixel_size_m: float,
    grid_spacing_m: float,
) -> dict:
    available = field["available"].fillna(False).to_numpy()
    estimate = field[["proposal_dx_m", "proposal_dy_m"]].to_numpy(dtype=float)
    errors = np.linalg.norm(estimate[available] - truth_vectors[available], axis=1)
    fitted_gradient, fitted_intercept = fit_affine(field)
    center = np.array(
        [
            (shape[1] - 1) / 2.0 * pixel_size_m,
            (shape[0] - 1) / 2.0 * pixel_size_m,
        ]
    )
    true_intercept = center - true_gradient @ center

    def deformation_components(gradient):
        velocity_gradient = gradient - np.eye(2)
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
            "area_ratio": float(np.linalg.det(gradient)),
        }

    return {
        "queries": int(len(field)),
        "available": int(available.sum()),
        "coverage_fraction": float(available.mean()),
        "median_vector_error_m": float(np.median(errors)) if len(errors) else None,
        "p90_vector_error_m": float(np.quantile(errors, 0.90)) if len(errors) else None,
        "maximum_vector_error_m": float(np.max(errors)) if len(errors) else None,
        "true_deformation_gradient": true_gradient.tolist(),
        "fitted_deformation_gradient": fitted_gradient.tolist(),
        "gradient_frobenius_error": float(
            np.linalg.norm(fitted_gradient - true_gradient)
        ),
        "fitted_translation_m": (fitted_intercept - true_intercept).tolist(),
        "true_deformation_components": deformation_components(true_gradient),
        "fitted_deformation_components": deformation_components(fitted_gradient),
        "topology": topology_summary(field, grid_spacing_m),
    }


def piecewise_summary(
    field: pd.DataFrame,
    truth_vectors: np.ndarray,
    shape: tuple[int, int],
    pixel_size_m: float,
    grid_spacing_m: float,
    left_translation_m: np.ndarray,
    right_translation_m: np.ndarray,
) -> dict:
    available = field["available"].fillna(False).to_numpy()
    estimate = field[["proposal_dx_m", "proposal_dy_m"]].to_numpy(dtype=float)
    errors = np.linalg.norm(estimate[available] - truth_vectors[available], axis=1)
    center_x = (shape[1] - 1) / 2.0 * pixel_size_m
    distance_to_lead = np.abs(field["source_x"].to_numpy() - center_x)
    near = available & (distance_to_lead <= 8000.0)
    far_left = available & (field["source_x"].to_numpy() < center_x - 8000.0)
    far_right = available & (field["source_x"].to_numpy() > center_x + 8000.0)
    near_errors = np.linalg.norm(estimate[near] - truth_vectors[near], axis=1)
    left_estimate = np.median(estimate[far_left], axis=0)
    right_estimate = np.median(estimate[far_right], axis=0)
    return {
        "queries": int(len(field)),
        "available": int(available.sum()),
        "coverage_fraction": float(available.mean()),
        "median_vector_error_m": float(np.median(errors)),
        "p90_vector_error_m": float(np.quantile(errors, 0.90)),
        "maximum_vector_error_m": float(np.max(errors)),
        "near_lead_queries": int(near.sum()),
        "near_lead_median_error_m": float(np.median(near_errors)),
        "near_lead_p90_error_m": float(np.quantile(near_errors, 0.90)),
        "true_left_translation_m": left_translation_m.tolist(),
        "true_right_translation_m": right_translation_m.tolist(),
        "estimated_far_left_translation_m": left_estimate.tolist(),
        "estimated_far_right_translation_m": right_estimate.tolist(),
        "true_opening_m": float(right_translation_m[0] - left_translation_m[0]),
        "estimated_opening_m": float(right_estimate[0] - left_estimate[0]),
        "topology": topology_summary(field, grid_spacing_m),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-results", type=Path, required=True)
    parser.add_argument("--source-image-id", type=int, default=721)
    parser.add_argument("--target-image-id", type=int, default=731)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixels", type=int, default=1024)
    parser.add_argument("--pixel-size-m", type=float, default=80.0)
    parser.add_argument("--tile-pixels", type=int, default=512)
    parser.add_argument("--tile-margin-pixels", type=int, default=16)
    parser.add_argument("--features-per-tile", type=int, default=1024)
    parser.add_argument("--grid-spacing-m", type=float, default=4000.0)
    parser.add_argument(
        "--center-method",
        choices=("footprint_centroid", "buoy_median"),
        default="footprint_centroid",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--matcher", choices=("lightglue", "mnn", "smnn"), default="lightglue"
    )
    parser.add_argument("--smnn-ratio", type=float, default=0.95)
    parser.add_argument("--lightglue-layers", type=int, default=9)
    parser.add_argument("--lightglue-depth-confidence", type=float, default=0.95)
    parser.add_argument("--lightglue-width-confidence", type=float, default=0.99)
    parser.add_argument("--lightglue-filter-threshold", type=float, default=0.10)
    parser.add_argument(
        "--lightglue-adapter", choices=("kornia", "direct"), default="kornia"
    )
    parser.add_argument("--lightglue-compile", action="store_true")
    args = parser.parse_args()

    cases = pd.read_csv(args.case_results, low_memory=False)
    pair = cases.loc[
        cases["source_image_id"].eq(args.source_image_id)
        & cases["target_image_id"].eq(args.target_image_id)
    ]
    if pair.empty:
        raise ValueError("Requested source image is absent from case results")
    if args.center_method == "footprint_centroid":
        center = projected_footprint(pair.iloc[0].source_image_filepath).centroid
        center_x, center_y = float(center.x), float(center.y)
    else:
        center_x = float(pair["source_x"].median())
        center_y = float(pair["source_y"].median())
    source, source_valid, _, _ = north_up_patch(
        pair.iloc[0].source_image_filepath,
        center_x,
        center_y,
        args.pixels,
        args.pixel_size_m,
    )
    device = torch.device(args.device)
    torch.manual_seed(20260817)
    torch.hub.set_dir(str(args.model_cache / "hub"))
    model = ALIKED.from_pretrained(
        model_name="aliked-n16",
        max_num_keypoints=args.features_per_tile,
        detection_threshold=0.2,
        device=device,
    ).eval()
    matcher = build_aliked_matcher(
        args.matcher,
        device,
        args.smnn_ratio,
        args.lightglue_layers,
        args.lightglue_depth_confidence,
        args.lightglue_width_confidence,
        args.lightglue_filter_threshold,
        args.lightglue_adapter,
        args.lightglue_compile,
    )
    source_tiles, source_audit = array_tiles(
        source,
        source_valid,
        model,
        device,
        args.pixel_size_m,
        args.tile_pixels,
        args.tile_margin_pixels,
        args.features_per_tile,
        support_radius_pixels=16,
    )
    queries = queries_for_shape(
        source.shape, args.pixel_size_m, args.grid_spacing_m, margin_m=8000.0
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_audit.assign(image="source").to_csv(
        args.output_dir / "source_tiles.csv", index=False
    )
    cv2.imwrite(str(args.output_dir / "source.png"), source)
    summaries = {}
    for name, specification in SCENARIOS.items():
        if "deformation_gradient" in specification:
            gradient = np.asarray(
                specification["deformation_gradient"], dtype=float
            )
            translation = np.asarray(specification["translation_m"], dtype=float)
            target, target_valid = warp_image(
                source, source_valid, gradient, translation, args.pixel_size_m
            )
            truth_vectors = truth_at_queries(
                queries, source.shape, args.pixel_size_m, gradient, translation
            )
        else:
            left_translation = np.asarray(
                specification["left_translation_m"], dtype=float
            )
            right_translation = np.asarray(
                specification["right_translation_m"], dtype=float
            )
            target, target_valid = warp_piecewise_translation(
                source,
                source_valid,
                left_translation,
                right_translation,
                args.pixel_size_m,
            )
            truth_vectors = piecewise_truth_at_queries(
                queries,
                source.shape,
                args.pixel_size_m,
                left_translation,
                right_translation,
            )
        target_tiles, target_audit = array_tiles(
            target,
            target_valid,
            model,
            device,
            args.pixel_size_m,
            args.tile_pixels,
            args.tile_margin_pixels,
            args.features_per_tile,
            support_radius_pixels=16,
        )
        matches, matching_audit = match_tiles(
            source_tiles,
            target_tiles,
            matcher,
            device,
            args.tile_pixels,
            maximum_displacement_m=3000.0,
            elapsed_days=1.0,
            maximum_speed_m_per_day=3000.0,
            physics_subset_matching=True,
        )
        valid_queries = valid_truth_queries(
            queries,
            source_valid,
            target_valid,
            truth_vectors,
            args.pixel_size_m,
        )
        scenario_queries = queries.loc[valid_queries].reset_index(drop=True)
        truth_vectors = truth_vectors[valid_queries]
        field = adaptive_consensus_at_queries(
            matches,
            scenario_queries,
            [2000.0, 3000.0, 4000.0, 6000.0],
            minimum_selected_vectors=8,
            consensus_radius_m=1000.0,
        )
        field["truth_dx_m"] = truth_vectors[:, 0]
        field["truth_dy_m"] = truth_vectors[:, 1]
        if "deformation_gradient" in specification:
            adaptive_summary = scenario_summary(
                field,
                truth_vectors,
                gradient,
                source.shape,
                args.pixel_size_m,
                args.grid_spacing_m,
            )
        else:
            adaptive_summary = piecewise_summary(
                field,
                truth_vectors,
                source.shape,
                args.pixel_size_m,
                args.grid_spacing_m,
                left_translation,
                right_translation,
            )
        nearest_field = nearest_consensus_at_queries(
            matches,
            scenario_queries,
            maximum_radius_m=6000.0,
            candidate_count=12,
            minimum_selected_vectors=8,
            consensus_radius_m=1000.0,
        )
        nearest_field["truth_dx_m"] = truth_vectors[:, 0]
        nearest_field["truth_dy_m"] = truth_vectors[:, 1]
        if "deformation_gradient" in specification:
            nearest_summary = scenario_summary(
                nearest_field,
                truth_vectors,
                gradient,
                source.shape,
                args.pixel_size_m,
                args.grid_spacing_m,
            )
        else:
            nearest_summary = piecewise_summary(
                nearest_field,
                truth_vectors,
                source.shape,
                args.pixel_size_m,
                args.grid_spacing_m,
                left_translation,
                right_translation,
            )
        summaries[name] = {
            "adaptive_eight": adaptive_summary,
            "nearest12_require8": nearest_summary,
        }
        field.to_csv(args.output_dir / f"field_{name}.csv", index=False)
        nearest_field.to_csv(
            args.output_dir / f"field_{name}_nearest12.csv", index=False
        )
        matches.to_csv(args.output_dir / f"matches_{name}.csv", index=False)
        target_audit.assign(image="target").to_csv(
            args.output_dir / f"target_tiles_{name}.csv", index=False
        )
        pd.DataFrame(matching_audit).to_csv(
            args.output_dir / f"matching_tiles_{name}.csv", index=False
        )
        cv2.imwrite(str(args.output_dir / f"target_{name}.png"), target)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    manifest = {
        "status": "complete",
        "source_image_filepath": pair.iloc[0].source_image_filepath,
        "source_center_epsg3413": [center_x, center_y],
        "center_method": args.center_method,
        "source_valid_fraction": float(source_valid.mean()),
        "case_results_sha256": sha256(args.case_results),
        "input": "one-channel standard VAE north-up EPSG:3413 patch",
        "pixel_size_m": args.pixel_size_m,
        "pixels": args.pixels,
        "tile_pixels": args.tile_pixels,
        "tile_margin_pixels": args.tile_margin_pixels,
        "features_per_tile": args.features_per_tile,
        "matcher": args.matcher,
        "smnn_ratio": args.smnn_ratio if args.matcher == "smnn" else None,
        "lightglue_layers": (
            args.lightglue_layers if args.matcher == "lightglue" else None
        ),
        "lightglue_depth_confidence": (
            args.lightglue_depth_confidence if args.matcher == "lightglue" else None
        ),
        "lightglue_width_confidence": (
            args.lightglue_width_confidence if args.matcher == "lightglue" else None
        ),
        "lightglue_filter_threshold": (
            args.lightglue_filter_threshold if args.matcher == "lightglue" else None
        ),
        "lightglue_adapter": (
            args.lightglue_adapter if args.matcher == "lightglue" else None
        ),
        "lightglue_compile": bool(
            args.lightglue_compile if args.matcher == "lightglue" else False
        ),
        "grid_spacing_m": args.grid_spacing_m,
        "proposals": [
            "adaptive 2/3/4/6 km support requiring eight vectors",
            "12 nearest within 6 km requiring eight-vector consensus",
        ],
        "scenarios": SCENARIOS,
        "summary": summaries,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
