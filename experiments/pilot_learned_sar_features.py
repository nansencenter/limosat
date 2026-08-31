#!/usr/bin/env python3
"""Small native detect-to-detect learned-feature pilot on development SAR pairs.

The source crop is centred on the known source buoy position.  The target crop
is centred on that same projected position, never on target truth.  This is a
diagnostic pilot, not a production tracker or a held-out score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from kornia.feature import (
    ALIKED,
    DeDoDe,
    LightGlueMatcher,
    laf_from_center_scale_ori,
)
from nansat import NSR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from limosat.image import Image


DEFAULT_OBSERVATIONS = (
    ROOT / "results/arctic_tracking_next_experiment/splits/full70_2020/observations.csv"
)
DEFAULT_CASES = (
    ("174640", 105, 222, "orb_pattern_correlation_failure"),
    ("174640", 222, 356, "orb_accepted_control"),
    ("300025060111910", 10107, 10217, "orb_topology_rejection"),
    ("300234065980590", 10107, 10217, "orb_accepted_control"),
    ("300025060111910", 721, 731, "short_sequence_step_1"),
    ("300025060111910", 731, 740, "short_sequence_step_2"),
    ("300234068125990", 721, 731, "orb_no_descriptor_candidate"),
    ("300025060111910", 10217, 10229, "orb_convergence_replacement"),
)


def dedode_dual_softmax_indices(
    descriptions_source: torch.Tensor,
    descriptions_target: torch.Tensor,
    inverse_temperature: float = 20.0,
    threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the official DeDoDe mutual dual-softmax matching rule."""
    source = functional.normalize(descriptions_source, dim=-1)
    target = functional.normalize(descriptions_target, dim=-1)
    correlation = source @ target.transpose(-1, -2) * inverse_temperature
    probability = correlation.softmax(dim=-2) * correlation.softmax(dim=-1)
    mutual = (probability == probability.max(dim=-1, keepdim=True).values) & (
        probability == probability.max(dim=-2, keepdim=True).values
    )
    indices = torch.nonzero(mutual & (probability > threshold), as_tuple=False)
    return indices, probability[indices[:, 0], indices[:, 1]]


def crop_at_projected_position(
    image: Image, x: float, y: float, crop_size: int
) -> tuple[np.ndarray, tuple[int, int]]:
    half = crop_size // 2
    columns, rows = image.transform_points(
        [x], [y], DstToSrc=1, dst_srs=NSR(3413)
    )
    column = int(round(float(columns[0])))
    row = int(round(float(rows[0])))
    data = image[1]
    if data.shape[0] < crop_size or data.shape[1] < crop_size:
        raise ValueError(
            f"Image shape {data.shape} is smaller than crop size {crop_size}"
        )
    origin = (
        int(np.clip(column - half, 0, data.shape[1] - crop_size)),
        int(np.clip(row - half, 0, data.shape[0] - crop_size)),
    )
    crop = data[origin[1] : origin[1] + crop_size, origin[0] : origin[0] + crop_size]
    if crop.shape != (crop_size, crop_size):
        raise ValueError(
            f"Crop centred at ({column}, {row}) has shape {crop.shape}, "
            f"expected {(crop_size, crop_size)}"
        )
    return np.asarray(crop, dtype=np.uint8).copy(), origin


def keypoints_to_projected(
    image: Image, keypoints: torch.Tensor, origin: tuple[int, int]
) -> np.ndarray:
    pixels = keypoints.detach().cpu().numpy()
    columns = pixels[:, 0] + origin[0]
    rows = pixels[:, 1] + origin[1]
    x, y = image.transform_points(
        columns, rows, DstToSrc=0, dst_srs=NSR(3413)
    )
    return np.column_stack((x, y))


def occupancy_fraction(keypoints: torch.Tensor, crop_size: int, cells: int = 4) -> float:
    if len(keypoints) == 0:
        return 0.0
    pixels = keypoints.detach().cpu().numpy()
    indices = np.floor(pixels / (crop_size / cells)).astype(int)
    indices = np.clip(indices, 0, cells - 1)
    occupied = np.unique(indices, axis=0)
    return float(len(occupied) / (cells * cells))


def summarize_matches(
    method: str,
    source_keypoints: torch.Tensor,
    target_keypoints: torch.Tensor,
    match_indices: torch.Tensor,
    match_scores: torch.Tensor,
    source_projected: np.ndarray,
    target_projected: np.ndarray,
    source_truth: np.ndarray,
    target_truth: np.ndarray,
    elapsed_days: float,
    crop_size: int,
    extraction_seconds: float,
    matching_seconds: float,
    maximum_speed_m_per_day: float = 30000.0,
    local_radius_m: float = 10000.0,
    tight_source_radius_m: float = 2000.0,
) -> dict:
    indices = match_indices.detach().cpu().numpy()
    score_values = match_scores.detach().cpu().numpy().reshape(-1)
    if len(indices):
        matched_source = source_projected[indices[:, 0]]
        matched_target = target_projected[indices[:, 1]]
        displacement = matched_target - matched_source
        source_distance = np.linalg.norm(matched_source - source_truth, axis=1)
        endpoint_error = np.linalg.norm(source_truth + displacement - target_truth, axis=1)
        speed = np.linalg.norm(displacement, axis=1) / elapsed_days
        physics = speed <= maximum_speed_m_per_day
        local_physics = physics & (source_distance <= local_radius_m)
        tight_physics = physics & (source_distance <= tight_source_radius_m)
    else:
        source_distance = endpoint_error = speed = np.array([], dtype=float)
        physics = local_physics = tight_physics = np.array([], dtype=bool)

    nearest_local_index = None
    if local_physics.any():
        candidates = np.flatnonzero(local_physics)
        nearest_local_index = int(candidates[np.argmin(source_distance[candidates])])
    nearest_tight_index = None
    if tight_physics.any():
        candidates = np.flatnonzero(tight_physics)
        nearest_tight_index = int(candidates[np.argmin(source_distance[candidates])])
    highest_confidence_local_index = None
    if local_physics.any():
        candidates = np.flatnonzero(local_physics)
        highest_confidence_local_index = int(
            candidates[np.argmax(score_values[candidates])]
        )
    return {
        "method": method,
        "source_keypoints": int(len(source_keypoints)),
        "target_keypoints": int(len(target_keypoints)),
        "raw_matches": int(len(indices)),
        "physics_matches": int(physics.sum()),
        "local_physics_matches": int(local_physics.sum()),
        "local_physics_within_2km": int(
            (local_physics & (endpoint_error <= 2000.0)).sum()
        ),
        "local_physics_fraction_within_2km": float(
            (endpoint_error[local_physics] <= 2000.0).mean()
        )
        if local_physics.any()
        else np.nan,
        "tight_source_physics_matches": int(tight_physics.sum()),
        "tight_source_physics_within_2km": int(
            (tight_physics & (endpoint_error <= 2000.0)).sum()
        ),
        "nearest_tight_source_distance_m": float(
            source_distance[nearest_tight_index]
        )
        if nearest_tight_index is not None
        else np.nan,
        "nearest_tight_endpoint_error_m": float(endpoint_error[nearest_tight_index])
        if nearest_tight_index is not None
        else np.nan,
        "highest_confidence_local_source_distance_m": float(
            source_distance[highest_confidence_local_index]
        )
        if highest_confidence_local_index is not None
        else np.nan,
        "highest_confidence_local_endpoint_error_m": float(
            endpoint_error[highest_confidence_local_index]
        )
        if highest_confidence_local_index is not None
        else np.nan,
        "nearest_local_source_distance_m": float(source_distance[nearest_local_index])
        if nearest_local_index is not None
        else np.nan,
        "nearest_local_endpoint_error_m": float(endpoint_error[nearest_local_index])
        if nearest_local_index is not None
        else np.nan,
        "median_local_endpoint_error_m": float(np.median(endpoint_error[local_physics]))
        if local_physics.any()
        else np.nan,
        "source_keypoint_cell_occupancy_fraction": occupancy_fraction(
            source_keypoints, crop_size
        ),
        "matched_source_cell_occupancy_fraction": occupancy_fraction(
            source_keypoints[match_indices[:, 0]]
            if len(match_indices)
            else source_keypoints[:0],
            crop_size,
        ),
        "mean_match_score": float(match_scores.mean().item())
        if len(match_scores)
        else np.nan,
        "extraction_seconds": float(extraction_seconds),
        "matching_seconds": float(matching_seconds),
    }


def observation_pair(
    observations: pd.DataFrame, buoy_id: str, source_id: int, target_id: int
) -> tuple[pd.Series, pd.Series]:
    selected = observations.loc[
        observations["within_dataset_split"].eq("development")
        & observations["buoy_id"].astype(str).eq(buoy_id)
        & observations["image_id"].isin([source_id, target_id])
    ].copy()
    if len(selected) != 2:
        raise ValueError(
            f"Expected two development observations for {buoy_id}: {source_id}->{target_id}"
        )
    source = selected.loc[selected["image_id"] == source_id].iloc[0]
    target = selected.loc[selected["image_id"] == target_id].iloc[0]
    return source, target


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--keypoints", type=int, default=2048)
    parser.add_argument(
        "--methods", choices=("both", "dedode", "aliked"), default="both"
    )
    args = parser.parse_args()
    if args.crop_size % 2:
        raise ValueError("crop-size must be even")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260817)
    # ``TORCH_HOME=<model-cache>`` stores downloads under ``<model-cache>/hub``.
    torch.hub.set_dir(str(args.model_cache / "hub"))
    observations = pd.read_csv(args.observations, low_memory=False)

    dedode = None
    if args.methods in {"both", "dedode"}:
        dedode = DeDoDe.from_pretrained(
            detector_weights="L-C4-v2",
            descriptor_weights="B-upright",
            amp_dtype=torch.float32,
        ).eval()
    aliked = lightglue_standard = lightglue_unthresholded = None
    if args.methods in {"both", "aliked"}:
        aliked = ALIKED.from_pretrained(
            model_name="aliked-n16",
            max_num_keypoints=args.keypoints,
            detection_threshold=0.2,
        ).eval()
        lightglue_standard = LightGlueMatcher("aliked").eval()
        lightglue_unthresholded = LightGlueMatcher(
            "aliked", params={"filter_threshold": 0.0}
        ).eval()

    records = []
    self_match_check = None
    for buoy_id, source_id, target_id, case_label in DEFAULT_CASES:
        source, target = observation_pair(
            observations, buoy_id, source_id, target_id
        )
        source_image = Image(source.image_filepath)
        target_image = Image(target.image_filepath)
        source_crop, source_origin = crop_at_projected_position(
            source_image, source.x, source.y, args.crop_size
        )
        target_crop, target_origin = crop_at_projected_position(
            target_image, source.x, source.y, args.crop_size
        )
        tensor = torch.from_numpy(np.stack([source_crop, target_crop])).float()
        tensor = tensor[:, None].repeat(1, 3, 1, 1) / 255.0
        elapsed_days = (
            pd.Timestamp(target.image_time) - pd.Timestamp(source.image_time)
        ).total_seconds() / 86400.0
        common = {
            "case_label": case_label,
            "buoy_id": buoy_id,
            "source_image_id": source_id,
            "target_image_id": target_id,
            "elapsed_hours": elapsed_days * 24.0,
            "truth_displacement_m": float(
                np.hypot(target.x - source.x, target.y - source.y)
            ),
            "crop_size_pixels": args.crop_size,
            "target_crop_uses_target_truth": False,
        }

        source_columns, source_rows = target_image.transform_points(
            [source.x], [source.y], DstToSrc=1, dst_srs=NSR(3413)
        )
        target_columns, target_rows = target_image.transform_points(
            [target.x], [target.y], DstToSrc=1, dst_srs=NSR(3413)
        )
        source_crop_x = source_columns[0] - target_origin[0]
        source_crop_y = source_rows[0] - target_origin[1]
        target_crop_x = target_columns[0] - target_origin[0]
        target_crop_y = target_rows[0] - target_origin[1]
        common.update(
            {
                "source_position_target_crop_x": float(source_crop_x),
                "source_position_target_crop_y": float(source_crop_y),
                "truth_target_crop_x": float(target_crop_x),
                "truth_target_crop_y": float(target_crop_y),
                "truth_target_crop_edge_margin_pixels": float(
                    min(
                        target_crop_x,
                        target_crop_y,
                        args.crop_size - 1 - target_crop_x,
                        args.crop_size - 1 - target_crop_y,
                    )
                ),
            }
        )

        if dedode is not None:
            started = time.perf_counter()
            with torch.inference_mode():
                dedode_keypoints, _, dedode_descriptions = dedode(
                    tensor, n=args.keypoints
                )
            extraction_seconds = time.perf_counter() - started
            started = time.perf_counter()
            with torch.inference_mode():
                dedode_indices, dedode_scores = dedode_dual_softmax_indices(
                    dedode_descriptions[0], dedode_descriptions[1], threshold=0.1
                )
            matching_seconds = time.perf_counter() - started
            dedode_source_projected = keypoints_to_projected(
                source_image, dedode_keypoints[0], source_origin
            )
            dedode_target_projected = keypoints_to_projected(
                target_image, dedode_keypoints[1], target_origin
            )
            records.append(
                {
                    **common,
                    **summarize_matches(
                        "dedode_v2_b_dual_softmax_0.1",
                        dedode_keypoints[0],
                        dedode_keypoints[1],
                        dedode_indices,
                        dedode_scores,
                        dedode_source_projected,
                        dedode_target_projected,
                        np.array([source.x, source.y]),
                        np.array([target.x, target.y]),
                        elapsed_days,
                        args.crop_size,
                        extraction_seconds,
                        matching_seconds,
                    ),
                }
            )

        if aliked is not None:
            started = time.perf_counter()
            with torch.inference_mode():
                aliked_features = aliked(tensor)
            extraction_seconds = time.perf_counter() - started
            lafs = [
                laf_from_center_scale_ori(features.keypoints[None])
                for features in aliked_features
            ]
            aliked_source_projected = keypoints_to_projected(
                source_image, aliked_features[0].keypoints, source_origin
            )
            aliked_target_projected = keypoints_to_projected(
                target_image, aliked_features[1].keypoints, target_origin
            )
            if self_match_check is None:
                with torch.inference_mode():
                    self_scores, self_indices = lightglue_standard(
                        aliked_features[0].descriptors,
                        aliked_features[0].descriptors,
                        lafs[0],
                        lafs[0],
                        hw1=(args.crop_size, args.crop_size),
                        hw2=(args.crop_size, args.crop_size),
                    )
                minimum_self_matches = int(0.9 * len(aliked_features[0].keypoints))
                if len(self_indices) < minimum_self_matches:
                    raise RuntimeError(
                        "ALIKED-LightGlue self-match invariant failed: "
                        f"{len(self_indices)} matches for "
                        f"{len(aliked_features[0].keypoints)} identical keypoints"
                    )
                self_match_check = {
                    "keypoints": int(len(aliked_features[0].keypoints)),
                    "standard_threshold_matches": int(len(self_indices)),
                    "same_index_matches": int(
                        (self_indices[:, 0] == self_indices[:, 1]).sum().item()
                    ),
                    "minimum_score": float(self_scores.min().item()),
                    "maximum_score": float(self_scores.max().item()),
                }
            for label, matcher in (
                ("aliked_lightglue_standard", lightglue_standard),
                ("aliked_lightglue_unthresholded", lightglue_unthresholded),
            ):
                started = time.perf_counter()
                with torch.inference_mode():
                    match_scores, match_indices = matcher(
                        aliked_features[0].descriptors,
                        aliked_features[1].descriptors,
                        lafs[0],
                        lafs[1],
                        hw1=(args.crop_size, args.crop_size),
                        hw2=(args.crop_size, args.crop_size),
                    )
                matching_seconds = time.perf_counter() - started
                records.append(
                    {
                        **common,
                        **summarize_matches(
                            label,
                            aliked_features[0].keypoints,
                            aliked_features[1].keypoints,
                            match_indices,
                            match_scores,
                            aliked_source_projected,
                            aliked_target_projected,
                            np.array([source.x, source.y]),
                            np.array([target.x, target.y]),
                            elapsed_days,
                            args.crop_size,
                            extraction_seconds,
                            matching_seconds,
                        ),
                    }
                )

    results = pd.DataFrame.from_records(records)
    results.to_csv(args.output_dir / "learned_feature_pilot_results.csv", index=False)
    weight_files = sorted(
        path
        for path in args.model_cache.rglob("*")
        if path.is_file()
        and not path.name.startswith("._")
        and (path.suffix == ".pth" or path.name.endswith("-pth"))
    )
    manifest = {
        "status": "complete",
        "split": "development",
        "cases": [list(case) for case in DEFAULT_CASES],
        "crop_size_pixels": args.crop_size,
        "keypoints_per_crop": args.keypoints,
        "methods": args.methods,
        "input": "standard VAE uint8 repeated to three channels and divided by 255",
        "aliked_loading": "ALIKED.from_pretrained",
        "aliked_lightglue_self_match_check": self_match_check,
        "target_crop_center": "source buoy EPSG:3413 position",
        "target_truth_used_for_crop": False,
        "maximum_speed_m_per_day": 30000.0,
        "local_source_radius_m": 10000.0,
        "device": "cpu",
        "model_weights": {
            str(path.relative_to(args.model_cache)): file_sha256(path)
            for path in weight_files
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
