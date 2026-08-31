#!/usr/bin/env python3
"""Build buoy-centred SAR appearance sequences and link them to tracking errors.

The archive follows each buoy at exact SAR acquisition times. It preserves raw
standard-VAE patches, validity masks, ORB descriptors at the exact buoy point,
nearest sparse XFeat descriptors, and previous/anchor change metrics. Tracking
results are joined only after extraction so buoy truth never affects matching.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nansat import NSR
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity

from buoy_descriptor_benchmark import (
    DescriptorVariant,
    annotate_coincidences,
    exact_descriptor,
    image_object,
    read_scene,
)
from xfeat_buoy_graph import nearest_feature, precompute_layers as precompute_xfeat_layers


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_ROOT = ROOT / "results/arctic_fixture_ledger/q2q98_clahe25"
DEFAULT_ORB_ROOT = ROOT / "results/orb_multiframe_graph/final_arctic_matrix"
DEFAULT_XFEAT_RUNS = {
    "2020_03": ROOT / "results/xfeat_buoy_graph/arctic_2020_03/q2q98_clahe25_max1536_top16000",
    "2020_02": ROOT / "results/xfeat_buoy_graph/arctic_2020_02/q2q98_clahe25_max1536_top16000",
}


def parse_numbers(value: str, cast) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def build_orb() -> cv2.ORB:
    return cv2.ORB_create(
        nfeatures=100,
        scaleFactor=1.25,
        nlevels=5,
        edgeThreshold=16,
        firstLevel=0,
        WTA_K=2,
        patchSize=64,
        scoreType=cv2.ORB_HARRIS_SCORE,
    )


def orb_variant() -> DescriptorVariant:
    return DescriptorVariant(
        name="orb_current_hamming",
        extractor_key="orb",
        norm="hamming",
        angle_mode="geographic",
        keypoint_size=31.0,
        octave=5,
    )


def native_patch(
    image: np.ndarray,
    mask: np.ndarray | None,
    col: float,
    row: float,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    map_x, map_y = np.meshgrid(offsets + float(col), offsets + float(row))
    patch = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    in_bounds = (
        (map_x >= 0)
        & (map_x <= image.shape[1] - 1)
        & (map_y >= 0)
        & (map_y <= image.shape[0] - 1)
    )
    if mask is None:
        valid = in_bounds
    else:
        sampled_mask = cv2.remap(
            mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=2,
        )
        valid = in_bounds & (sampled_mask < 2)
    patch = patch.astype(np.uint8, copy=False)
    patch[~valid] = 0
    return patch, valid.astype(bool)


def map_aligned_patch(
    image_path: str,
    image: np.ndarray,
    mask: np.ndarray | None,
    center_x: float,
    center_y: float,
    width_m: float,
    output_pixels: int,
    analysis_epsg: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.linspace(-width_m / 2.0, width_m / 2.0, output_pixels)
    map_x, map_y_offset = np.meshgrid(center_x + offsets, offsets)
    map_y = center_y - map_y_offset
    cols, rows = image_object(image_path, analysis_epsg).transform_points(
        map_x.ravel(),
        map_y.ravel(),
        DstToSrc=1,
        dst_srs=NSR(analysis_epsg),
    )
    sample_x = np.asarray(cols, dtype=np.float32).reshape(output_pixels, output_pixels)
    sample_y = np.asarray(rows, dtype=np.float32).reshape(output_pixels, output_pixels)
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
    return patch, valid.astype(bool)


def patch_statistics(patch: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    values = patch[valid].astype(np.float64)
    if len(values) == 0:
        return {key: math.nan for key in (
            "mean", "std", "p05", "median", "p95", "entropy_bits",
            "gradient_rms", "laplacian_std", "structure_coherence", "structure_angle_deg",
        )} | {"valid_fraction": 0.0}
    hist = np.bincount(values.astype(np.uint8), minlength=256).astype(float)
    probabilities = hist[hist > 0] / hist.sum()
    image = patch.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    valid_inner = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if np.any(valid_inner):
        gx_values = gx[valid_inner].astype(np.float64)
        gy_values = gy[valid_inner].astype(np.float64)
        jxx = float(np.mean(gx_values * gx_values))
        jyy = float(np.mean(gy_values * gy_values))
        jxy = float(np.mean(gx_values * gy_values))
        eigenvalues = np.linalg.eigvalsh(np.array([[jxx, jxy], [jxy, jyy]]))
        coherence = float((eigenvalues[1] - eigenvalues[0]) / max(eigenvalues.sum(), 1.0e-12))
        angle = float(0.5 * np.degrees(np.arctan2(2.0 * jxy, jxx - jyy)))
        gradient_rms = float(np.sqrt(np.mean(gx_values * gx_values + gy_values * gy_values)))
        laplacian = cv2.Laplacian(image, cv2.CV_32F)
        laplacian_std = float(laplacian[valid_inner].std())
    else:
        coherence = angle = gradient_rms = laplacian_std = math.nan
    return {
        "valid_fraction": float(np.mean(valid)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "entropy_bits": float(-(probabilities * np.log2(probabilities)).sum()),
        "gradient_rms": gradient_rms,
        "laplacian_std": laplacian_std,
        "structure_coherence": coherence,
        "structure_angle_deg": angle,
    }


def masked_ncc(left: np.ndarray, right: np.ndarray, overlap: np.ndarray) -> float:
    a = left[overlap].astype(np.float64)
    b = right[overlap].astype(np.float64)
    if len(a) < 16 or a.std() < 1.0e-9 or b.std() < 1.0e-9:
        return math.nan
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (a.std() * b.std()))


def normalized_mutual_information(
    left: np.ndarray,
    right: np.ndarray,
    overlap: np.ndarray,
    bins: int = 32,
) -> float:
    a = left[overlap].astype(np.float64)
    b = right[overlap].astype(np.float64)
    if len(a) < 16:
        return math.nan
    joint, _, _ = np.histogram2d(a, b, bins=bins, range=((0, 256), (0, 256)))
    joint /= max(joint.sum(), 1.0)
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    nz = joint > 0
    product = pa[:, None] * pb[None, :]
    mutual_information = float(np.sum(joint[nz] * np.log(joint[nz] / product[nz])))
    ha = float(-np.sum(pa[pa > 0] * np.log(pa[pa > 0])))
    hb = float(-np.sum(pb[pb > 0] * np.log(pb[pb > 0])))
    return mutual_information / max(math.sqrt(ha * hb), 1.0e-12)


def patch_pair_metrics(
    left: np.ndarray,
    right: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
) -> dict[str, float]:
    overlap = left_valid & right_valid
    overlap_fraction = float(np.mean(overlap))
    if np.count_nonzero(overlap) < 16:
        return {
            "overlap_fraction": overlap_fraction,
            "ncc": math.nan,
            "gradient_ncc": math.nan,
            "ssim": math.nan,
            "normalized_mutual_information": math.nan,
            "histogram_js_distance": math.nan,
            "rmse": math.nan,
            "phase_shift_x_px": math.nan,
            "phase_shift_y_px": math.nan,
            "phase_response": math.nan,
        }
    gradient_left = cv2.magnitude(
        cv2.Sobel(left.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(left.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3),
    )
    gradient_right = cv2.magnitude(
        cv2.Sobel(right.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(right.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3),
    )
    valid_inner = cv2.erode(overlap.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    left_values = left[left_valid].astype(np.float64)
    right_values = right[right_valid].astype(np.float64)
    hist_left, _ = np.histogram(left_values, bins=64, range=(0, 256), density=False)
    hist_right, _ = np.histogram(right_values, bins=64, range=(0, 256), density=False)
    fill_left = left.astype(np.float32).copy()
    fill_right = right.astype(np.float32).copy()
    fill_left[~overlap] = float(left[overlap].mean())
    fill_right[~overlap] = float(right[overlap].mean())
    ssim_value, ssim_map = structural_similarity(
        fill_left,
        fill_right,
        data_range=255.0,
        full=True,
    )
    if np.any(valid_inner):
        ssim_value = float(np.mean(ssim_map[valid_inner]))
    standardized_left = np.zeros_like(fill_left)
    standardized_right = np.zeros_like(fill_right)
    for source, target in ((left, standardized_left), (right, standardized_right)):
        values = source[overlap].astype(np.float32)
        target[overlap] = (values - values.mean()) / max(float(values.std()), 1.0e-6)
    window = cv2.createHanningWindow((left.shape[1], left.shape[0]), cv2.CV_32F)
    phase_shift, phase_response = cv2.phaseCorrelate(
        standardized_left,
        standardized_right,
        window,
    )
    delta = left[overlap].astype(np.float64) - right[overlap].astype(np.float64)
    return {
        "overlap_fraction": overlap_fraction,
        "ncc": masked_ncc(left, right, overlap),
        "gradient_ncc": masked_ncc(gradient_left, gradient_right, valid_inner),
        "ssim": ssim_value,
        "normalized_mutual_information": normalized_mutual_information(left, right, overlap),
        "histogram_js_distance": float(jensenshannon(hist_left + 1.0e-12, hist_right + 1.0e-12)),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "phase_shift_x_px": float(phase_shift[0]),
        "phase_shift_y_px": float(phase_shift[1]),
        "phase_response": float(phase_response),
    }


def hamming_normalized(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.unpackbits(np.bitwise_xor(left, right)).sum() / (left.size * 8))


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return math.nan
    return float(1.0 - np.dot(left, right) / denominator)


def load_fixture(path: Path, analysis_epsg: int) -> pd.DataFrame:
    usable, _ = load_fixture_with_exclusions(path, analysis_epsg)
    return usable


def load_fixture_with_exclusions(
    path: Path,
    analysis_epsg: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coincidences = pd.read_csv(path)
    coincidences["image_time"] = pd.to_datetime(coincidences["image_time"], utc=True)
    coincidences["buoy_id"] = coincidences.buoy_id.astype(str)
    coincidences["fixture_row_id"] = np.arange(len(coincidences), dtype=int)
    annotated = annotate_coincidences(
        coincidences,
        analysis_epsg,
        outside_scene_policy="skip",
    )
    valid = (annotated.mask_value < 2) & np.isfinite(annotated[["col", "row"]]).all(axis=1)
    outside = coincidences.loc[
        ~coincidences.fixture_row_id.isin(annotated.fixture_row_id)
    ].copy()
    outside["exclusion_reason"] = "outside_scene_footprint"
    invalid = annotated.loc[~valid].copy()
    invalid["exclusion_reason"] = np.where(
        invalid.mask_value >= 2,
        "invalid_or_land_mask",
        "nonfinite_pixel_transform",
    )
    excluded = pd.concat([outside, invalid], ignore_index=True, sort=False).drop(
        columns="fixture_row_id"
    )
    usable = annotated.loc[valid].drop(columns="fixture_row_id").copy()
    usable["buoy_id"] = usable.buoy_id.astype(str)
    usable = usable.sort_values(["buoy_id", "image_time", "image_id"]).reset_index(drop=True)
    return usable, excluded.sort_values(["buoy_id", "image_time", "image_id"]).reset_index(drop=True)


def extract_sequence(
    sequence: str,
    coincidences: pd.DataFrame,
    out_dir: Path,
    args,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    pd.DataFrame,
    float,
]:
    started = time.perf_counter()
    map_widths = args.map_widths_m
    native_sizes = args.native_sizes_px
    patch_arrays: dict[str, list[np.ndarray]] = {
        **{f"map_{int(width)}m": [] for width in map_widths},
        **{f"native_{size}px": [] for size in native_sizes},
    }
    mask_arrays: dict[str, list[np.ndarray]] = {key: [] for key in patch_arrays}
    orb_descriptors: list[np.ndarray] = []
    orb_available: list[bool] = []

    xfeat_args = SimpleNamespace(
        max_side=args.xfeat_max_side,
        top_k=args.xfeat_top_k,
        detection_threshold=args.xfeat_detection_threshold,
        device=args.device,
        analysis_epsg=args.analysis_epsg,
    )
    xfeat_layers, feature_summary, xfeat_seconds = precompute_xfeat_layers(
        coincidences,
        xfeat_args,
    )
    xfeat_descriptors: list[np.ndarray] = []
    xfeat_available: list[bool] = []
    orb = build_orb()
    variant = orb_variant()
    records = []

    for observation_index, row in enumerate(coincidences.itertuples(index=False)):
        image, mask = read_scene(row.image_filepath)
        record = row._asdict()
        record["sequence_observation_index"] = observation_index
        record["observation_id"] = f"{sequence}:{row.buoy_id}:{int(row.image_id)}"
        for width in map_widths:
            key = f"map_{int(width)}m"
            patch, valid = map_aligned_patch(
                row.image_filepath,
                image,
                mask,
                float(row.x),
                float(row.y),
                width,
                args.map_patch_pixels,
                args.analysis_epsg,
            )
            patch_arrays[key].append(patch)
            mask_arrays[key].append(valid)
            record.update({f"{key}_{name}": value for name, value in patch_statistics(patch, valid).items()})
        for size in native_sizes:
            key = f"native_{size}px"
            patch, valid = native_patch(image, mask, row.col, row.row, size)
            patch_arrays[key].append(patch)
            mask_arrays[key].append(valid)
            record.update({f"{key}_{name}": value for name, value in patch_statistics(patch, valid).items()})

        descriptor = exact_descriptor(
            orb,
            image,
            np.array([row.col, row.row]),
            variant,
            float(row.image_angle_deg),
        )
        descriptor_ok = descriptor is not None and descriptor.shape == (32,)
        orb_available.append(descriptor_ok)
        orb_descriptors.append(
            np.asarray(descriptor, dtype=np.uint8) if descriptor_ok else np.zeros(32, dtype=np.uint8)
        )
        record["orb_descriptor_available"] = descriptor_ok

        layer = xfeat_layers[row.image_filepath]
        if len(layer.graph.descriptors):
            feature_index, feature_offset = nearest_feature(
                layer,
                np.array([row.x, row.y], dtype=float),
            )
            xfeat_descriptor = layer.graph.descriptors[feature_index].astype(np.float32)
            xfeat_ok = bool(feature_offset <= args.xfeat_max_distance_m)
            xfeat_score = float(layer.scores[feature_index])
        else:
            xfeat_descriptor = np.zeros(64, dtype=np.float32)
            feature_offset = math.nan
            xfeat_score = math.nan
            xfeat_ok = False
        xfeat_descriptors.append(xfeat_descriptor)
        xfeat_available.append(xfeat_ok)
        record["xfeat_nearest_feature_distance_m"] = feature_offset
        record["xfeat_nearest_feature_score"] = xfeat_score
        record["xfeat_descriptor_within_limit"] = xfeat_ok
        records.append(record)

    stacked_patches = {key: np.stack(values).astype(np.uint8) for key, values in patch_arrays.items()}
    stacked_masks = {key: np.stack(values).astype(bool) for key, values in mask_arrays.items()}
    descriptors = {
        "orb": np.stack(orb_descriptors).astype(np.uint8),
        "orb_available": np.asarray(orb_available, dtype=bool),
        "xfeat": np.stack(xfeat_descriptors).astype(np.float32),
        "xfeat_within_limit": np.asarray(xfeat_available, dtype=bool),
    }
    observations = pd.DataFrame.from_records(records)
    transitions = build_transitions(
        observations,
        stacked_patches,
        stacked_masks,
        descriptors,
        map_widths,
    )
    feature_summary.to_csv(out_dir / "xfeat_image_features.csv", index=False)
    return (
        observations,
        transitions,
        stacked_patches,
        stacked_masks,
        descriptors,
        feature_summary,
        time.perf_counter() - started,
    )


def build_transitions(
    observations: pd.DataFrame,
    patches: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    descriptors: dict[str, np.ndarray],
    map_widths_m: tuple[float, ...],
) -> pd.DataFrame:
    rows = []
    patch_keys = sorted(patches)
    primary_key = f"map_{int(map_widths_m[len(map_widths_m) // 2])}m"
    for buoy_id, group in observations.groupby("buoy_id", sort=True):
        indexes = group.sort_values("image_time").index.to_numpy(dtype=int)
        if len(indexes) < 2:
            continue
        anchor_index = int(indexes[0])
        for path_index, (source_index, target_index) in enumerate(zip(indexes[:-1], indexes[1:]), start=1):
            source = observations.loc[source_index]
            target = observations.loc[target_index]
            dt_hours = (pd.Timestamp(target.image_time) - pd.Timestamp(source.image_time)).total_seconds() / 3600.0
            displacement = float(np.hypot(target.x - source.x, target.y - source.y))
            record = {
                "sequence": target.sequence,
                "buoy_id": buoy_id,
                "path_observation_index": path_index,
                "source_observation_id": source.observation_id,
                "target_observation_id": target.observation_id,
                "source_image_id": int(source.image_id),
                "target_image_id": int(target.image_id),
                "source_time": source.image_time,
                "target_time": target.image_time,
                "dt_hours": dt_hours,
                "true_displacement_m": displacement,
                "true_speed_m_per_day": displacement / dt_hours * 24.0,
                "image_angle_change_deg": float(target.image_angle_deg - source.image_angle_deg),
            }
            for reference_name, reference_index in (("prev", int(source_index)), ("anchor", anchor_index)):
                for key in patch_keys:
                    metrics = patch_pair_metrics(
                        patches[key][reference_index],
                        patches[key][target_index],
                        masks[key][reference_index],
                        masks[key][target_index],
                    )
                    record.update(
                        {
                            f"{key}_{reference_name}_{metric_name}": value
                            for metric_name, value in metrics.items()
                        }
                    )
                    if key.startswith("map_"):
                        width_m = float(key.removeprefix("map_").removesuffix("m"))
                        pixels = patches[key].shape[-1]
                        record[f"{key}_{reference_name}_phase_shift_m"] = float(
                            np.hypot(metrics["phase_shift_x_px"], metrics["phase_shift_y_px"])
                            * width_m
                            / max(pixels - 1, 1)
                        ) if np.isfinite(metrics["phase_shift_x_px"]) else math.nan
                if descriptors["orb_available"][reference_index] and descriptors["orb_available"][target_index]:
                    record[f"orb_{reference_name}_hamming_norm"] = hamming_normalized(
                        descriptors["orb"][reference_index],
                        descriptors["orb"][target_index],
                    )
                else:
                    record[f"orb_{reference_name}_hamming_norm"] = math.nan
                if descriptors["xfeat_within_limit"][reference_index] and descriptors["xfeat_within_limit"][target_index]:
                    record[f"xfeat_{reference_name}_cosine_distance"] = cosine_distance(
                        descriptors["xfeat"][reference_index],
                        descriptors["xfeat"][target_index],
                    )
                else:
                    record[f"xfeat_{reference_name}_cosine_distance"] = math.nan
            for statistic in (
                "valid_fraction",
                "mean",
                "std",
                "entropy_bits",
                "gradient_rms",
                "structure_coherence",
                "structure_angle_deg",
            ):
                source_value = source[f"{primary_key}_{statistic}"]
                target_value = target[f"{primary_key}_{statistic}"]
                record[f"source_{primary_key}_{statistic}"] = source_value
                record[f"target_{primary_key}_{statistic}"] = target_value
                record[f"change_{primary_key}_{statistic}"] = target_value - source_value
            record["source_xfeat_nearest_feature_distance_m"] = source.xfeat_nearest_feature_distance_m
            record["target_xfeat_nearest_feature_distance_m"] = target.xfeat_nearest_feature_distance_m
            rows.append(record)
    return pd.DataFrame.from_records(rows)


def tracking_sources(sequence: str, orb_root: Path) -> list[tuple[str, Path]]:
    sources = [("ORB", orb_root / sequence / "trajectory_results.csv")]
    xfeat = DEFAULT_XFEAT_RUNS.get(sequence)
    if xfeat is not None:
        sources.append(("XFeat", xfeat / "trajectory_results.csv"))
    return [(backend, path) for backend, path in sources if path.exists()]


def link_tracking_results(
    sequence: str,
    transitions: pd.DataFrame,
    orb_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    linked = []
    outcomes = []
    for backend, path in tracking_sources(sequence, orb_root):
        results = pd.read_csv(path)
        results["buoy_id"] = results.buoy_id.astype(str)
        for config, config_rows in results.groupby("config", sort=False):
            path_status = {}
            for buoy_id, group in config_rows.groupby("buoy_id"):
                if np.any(group.status == "seed_unavailable"):
                    status = "seed_unavailable"
                elif np.any(group.status == "graph_failed"):
                    status = "graph_failed"
                elif np.any(group.status == "skipped"):
                    status = "completed_with_skip"
                else:
                    status = "completed"
                path_status[str(buoy_id)] = status
                outcomes.append(
                    {
                        "sequence": sequence,
                        "backend": backend,
                        "config": config,
                        "buoy_id": str(buoy_id),
                        "path_status": status,
                        "path_observations": int(group.path_observations.max()),
                        "skipped_observations": int(np.sum(group.status == "skipped")),
                    }
                )
            observation_rows = config_rows[
                config_rows.status.isin(["ok", "skipped"])
                & (config_rows.observation_index > 0)
            ].copy()
            keep = [
                "buoy_id",
                "image_id",
                "status",
                "endpoint_error_m",
                "edge_cost",
                "candidate_count",
                "descriptor_updated",
            ]
            observation_rows = observation_rows[keep].rename(
                columns={
                    "image_id": "target_image_id",
                    "status": "tracking_status",
                }
            )
            base = transitions.copy()
            base["backend"] = backend
            base["config"] = config
            base = base.merge(
                observation_rows,
                on=["buoy_id", "target_image_id"],
                how="left",
            )
            base["path_status"] = base.buoy_id.map(path_status).fillna("not_attempted")
            base["tracking_status"] = base.tracking_status.fillna(base.path_status)
            base["within_2km"] = base.endpoint_error_m <= 2000.0
            base["catastrophic_50km"] = base.endpoint_error_m > 50000.0
            linked.append(base)
    return pd.concat(linked, ignore_index=True, sort=False), pd.DataFrame.from_records(outcomes)


def association_metrics(linked: pd.DataFrame, primary_key: str) -> pd.DataFrame:
    metrics = [
        "dt_hours",
        "true_speed_m_per_day",
        "orb_prev_hamming_norm",
        "orb_anchor_hamming_norm",
        "xfeat_prev_cosine_distance",
        "xfeat_anchor_cosine_distance",
        f"{primary_key}_prev_ncc",
        f"{primary_key}_anchor_ncc",
        f"{primary_key}_prev_gradient_ncc",
        f"{primary_key}_prev_ssim",
        f"{primary_key}_prev_normalized_mutual_information",
        f"{primary_key}_prev_histogram_js_distance",
        f"{primary_key}_prev_phase_shift_m",
        f"{primary_key}_prev_phase_response",
        f"target_{primary_key}_std",
        f"target_{primary_key}_entropy_bits",
        f"target_{primary_key}_gradient_rms",
        f"target_{primary_key}_structure_coherence",
        "target_xfeat_nearest_feature_distance_m",
    ]
    records = []
    valid_tracking = linked[
        (linked.tracking_status == "ok")
        & np.isfinite(linked.endpoint_error_m)
    ].copy()
    for sequence_value, sequence_rows in list(valid_tracking.groupby("sequence")) + [("ALL", valid_tracking)]:
        for (backend, config), group in sequence_rows.groupby(["backend", "config"]):
            failure = (group.endpoint_error_m > 2000.0).astype(int)
            for metric in metrics:
                if metric not in group:
                    continue
                finite = np.isfinite(group[metric]) & np.isfinite(group.endpoint_error_m)
                values = group.loc[finite, metric].astype(float)
                labels = failure.loc[finite]
                errors = group.loc[finite, "endpoint_error_m"].astype(float)
                if len(values) < 8:
                    continue
                correlation = spearmanr(values, np.log1p(errors)).statistic
                if labels.nunique() == 2:
                    raw_auc = roc_auc_score(labels, values)
                    auc = max(raw_auc, 1.0 - raw_auc)
                    direction = "higher" if raw_auc >= 0.5 else "lower"
                else:
                    auc = math.nan
                    direction = ""
                records.append(
                    {
                        "sequence": sequence_value,
                        "backend": backend,
                        "config": config,
                        "metric": metric,
                        "observations": len(values),
                        "failures_over_2km": int(labels.sum()),
                        "success_median": float(values[labels == 0].median()) if np.any(labels == 0) else math.nan,
                        "failure_median": float(values[labels == 1].median()) if np.any(labels == 1) else math.nan,
                        "failure_auc_discrimination": auc,
                        "failure_direction": direction,
                        "spearman_log_error": correlation,
                    }
                )
    return pd.DataFrame.from_records(records)


def clustered_association_intervals(
    linked: pd.DataFrame,
    primary_key: str,
    bootstrap_replicates: int = 1000,
    random_seed: int = 3413,
) -> pd.DataFrame:
    metrics = [
        "orb_anchor_hamming_norm",
        "orb_prev_hamming_norm",
        f"{primary_key}_anchor_ncc",
        f"{primary_key}_prev_ncc",
        f"{primary_key}_prev_histogram_js_distance",
    ]
    representative = linked[
        (linked.backend == "ORB")
        & (linked.config == "beam_confidence_update_m032")
        & (linked.tracking_status == "ok")
        & np.isfinite(linked.endpoint_error_m)
    ].copy()
    rng = np.random.default_rng(random_seed)
    records = []
    for sequence in ("2020_02", "2015_full15"):
        sequence_rows = representative[representative.sequence == sequence]
        for metric in metrics:
            finite = np.isfinite(sequence_rows[metric])
            data = sequence_rows.loc[
                finite, ["buoy_id", metric, "endpoint_error_m"]
            ].copy()
            data["failure"] = data.endpoint_error_m > 2000.0
            if len(data) < 8 or data.failure.nunique() != 2:
                continue
            raw_auc = roc_auc_score(data.failure, data[metric])
            direction = "higher" if raw_auc >= 0.5 else "lower"
            auc = raw_auc if direction == "higher" else 1.0 - raw_auc
            correlation = spearmanr(data[metric], np.log1p(data.endpoint_error_m)).statistic
            clusters = [group for _, group in data.groupby("buoy_id", sort=True)]
            bootstrap_auc = []
            bootstrap_correlation = []
            for _ in range(bootstrap_replicates):
                sampled = [
                    clusters[index]
                    for index in rng.integers(0, len(clusters), size=len(clusters))
                ]
                sample = pd.concat(sampled, ignore_index=True)
                if sample.failure.nunique() != 2:
                    continue
                sample_auc = roc_auc_score(sample.failure, sample[metric])
                bootstrap_auc.append(
                    sample_auc if direction == "higher" else 1.0 - sample_auc
                )
                bootstrap_correlation.append(
                    spearmanr(
                        sample[metric],
                        np.log1p(sample.endpoint_error_m),
                    ).statistic
                )
            auc_interval = np.quantile(bootstrap_auc, [0.025, 0.975])
            correlation_interval = np.quantile(
                np.asarray(bootstrap_correlation)[
                    np.isfinite(bootstrap_correlation)
                ],
                [0.025, 0.975],
            )
            records.append(
                {
                    "sequence": sequence,
                    "metric": metric,
                    "observations": len(data),
                    "unique_buoys": data.buoy_id.nunique(),
                    "failures_over_2km": int(data.failure.sum()),
                    "failure_direction": direction,
                    "failure_auc": auc,
                    "failure_auc_ci025": float(auc_interval[0]),
                    "failure_auc_ci975": float(auc_interval[1]),
                    "spearman_log_error": correlation,
                    "spearman_ci025": float(correlation_interval[0]),
                    "spearman_ci975": float(correlation_interval[1]),
                    "bootstrap_replicates_requested": bootstrap_replicates,
                    "bootstrap_replicates_valid": len(bootstrap_auc),
                    "random_seed": random_seed,
                }
            )
    return pd.DataFrame.from_records(records)


def tracking_path_summary(linked: pd.DataFrame, primary_key: str) -> pd.DataFrame:
    records = []
    for keys, group in linked.groupby(
        ["sequence", "backend", "config", "buoy_id"], sort=False
    ):
        group = group.sort_values(["target_time", "path_observation_index"])
        ok = (group.tracking_status == "ok") & np.isfinite(group.endpoint_error_m)
        successful = ok & (group.endpoint_error_m <= 2000.0)
        catastrophic = ok & (group.endpoint_error_m > 50000.0)
        first_failure = group.loc[~successful].head(1)
        valid_errors = group.loc[ok, "endpoint_error_m"]
        record = {
            "sequence": keys[0],
            "backend": keys[1],
            "config": keys[2],
            "buoy_id": keys[3],
            "path_status": group.path_status.iloc[0],
            "transitions": len(group),
            "tracked_transitions": int(ok.sum()),
            "tracking_coverage": float(ok.mean()),
            "within_2km_count": int(successful.sum()),
            "within_2km_fraction_all": float(successful.mean()),
            "within_2km_fraction_tracked": float(successful.sum() / max(ok.sum(), 1)),
            "catastrophic_50km_count": int(catastrophic.sum()),
            "median_endpoint_error_m": float(valid_errors.median()) if len(valid_errors) else math.nan,
            "maximum_endpoint_error_m": float(valid_errors.max()) if len(valid_errors) else math.nan,
            "final_endpoint_error_m": float(valid_errors.iloc[-1]) if len(valid_errors) else math.nan,
            "first_failure_target_observation_id": (
                first_failure.target_observation_id.iloc[0] if len(first_failure) else ""
            ),
            "median_orb_prev_hamming_norm": float(group.orb_prev_hamming_norm.median()),
            "maximum_orb_prev_hamming_norm": float(group.orb_prev_hamming_norm.max()),
            "median_orb_anchor_hamming_norm": float(group.orb_anchor_hamming_norm.median()),
            "maximum_orb_anchor_hamming_norm": float(group.orb_anchor_hamming_norm.max()),
            f"median_{primary_key}_prev_ncc": float(group[f"{primary_key}_prev_ncc"].median()),
            f"minimum_{primary_key}_prev_ncc": float(group[f"{primary_key}_prev_ncc"].min()),
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def paired_update_effects(linked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = [
        "sequence",
        "buoy_id",
        "source_observation_id",
        "target_observation_id",
    ]
    baseline = linked[
        (linked.backend == "ORB") & (linked.config == "beam_anchor")
    ][identity + ["tracking_status", "endpoint_error_m"]].rename(
        columns={
            "tracking_status": "anchor_tracking_status",
            "endpoint_error_m": "anchor_endpoint_error_m",
        }
    )
    updated = linked[
        (linked.backend == "ORB")
        & (linked.config == "beam_confidence_update_m032")
    ].copy()
    effects = updated.merge(baseline, on=identity, how="inner", validate="one_to_one")
    anchor_ok = (
        (effects.anchor_tracking_status == "ok")
        & np.isfinite(effects.anchor_endpoint_error_m)
    )
    update_ok = (
        (effects.tracking_status == "ok")
        & np.isfinite(effects.endpoint_error_m)
    )
    paired_ok = anchor_ok & update_ok
    effects["anchor_trackable"] = anchor_ok
    effects["update_trackable"] = update_ok
    effects["newly_trackable_with_update"] = ~anchor_ok & update_ok
    effects["lost_with_update"] = anchor_ok & ~update_ok
    effects["paired_endpoint_improvement_m"] = np.where(
        paired_ok,
        effects.anchor_endpoint_error_m - effects.endpoint_error_m,
        math.nan,
    )
    effects["anchor_within_2km"] = anchor_ok & (effects.anchor_endpoint_error_m <= 2000.0)
    effects["update_within_2km"] = update_ok & (effects.endpoint_error_m <= 2000.0)
    effects["rescued_within_2km"] = (
        paired_ok
        & (effects.anchor_endpoint_error_m > 2000.0)
        & (effects.endpoint_error_m <= 2000.0)
    )
    effects["harmed_beyond_2km"] = (
        paired_ok
        & (effects.anchor_endpoint_error_m <= 2000.0)
        & (effects.endpoint_error_m > 2000.0)
    )

    summary_rows = []
    groups = list(effects.groupby("sequence", sort=False)) + [("ALL", effects)]
    for sequence, group in groups:
        paired = group[np.isfinite(group.paired_endpoint_improvement_m)]
        summary_rows.append(
            {
                "sequence": sequence,
                "transitions": len(group),
                "anchor_trackable": int(group.anchor_trackable.sum()),
                "update_trackable": int(group.update_trackable.sum()),
                "newly_trackable_with_update": int(group.newly_trackable_with_update.sum()),
                "lost_with_update": int(group.lost_with_update.sum()),
                "paired_trackable": len(paired),
                "median_paired_improvement_m": float(paired.paired_endpoint_improvement_m.median()),
                "mean_paired_improvement_m": float(paired.paired_endpoint_improvement_m.mean()),
                "paired_win_fraction": float((paired.paired_endpoint_improvement_m > 0).mean()),
                "anchor_within_2km_fraction_all": float(group.anchor_within_2km.mean()),
                "update_within_2km_fraction_all": float(group.update_within_2km.mean()),
                "rescued_within_2km": int(group.rescued_within_2km.sum()),
                "harmed_beyond_2km": int(group.harmed_beyond_2km.sum()),
            }
        )
    return effects, pd.DataFrame.from_records(summary_rows)


def contact_sheets(
    observations: pd.DataFrame,
    transitions: pd.DataFrame,
    patches: dict[str, np.ndarray],
    linked: pd.DataFrame,
    out_dir: Path,
    primary_key: str,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    sequence = str(observations.sequence.iloc[0])
    selected_config = "greedy_rolling" if sequence == "2015_full15" else "beam_confidence_update_m032"
    tracking = linked[(linked.backend == "ORB") & (linked.config == selected_config)].copy()
    transition_lookup = transitions.set_index("target_observation_id")
    tracking_lookup = tracking.set_index("target_observation_id")
    created = 0
    for buoy_id, group in observations.groupby("buoy_id", sort=True):
        group = group.sort_values("image_time")
        if len(group) < 2:
            continue
        count = len(group)
        columns = min(4, count)
        rows = math.ceil(count / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.35 * rows), squeeze=False)
        for axis in axes.ravel():
            axis.axis("off")
        for axis, (index, observation) in zip(axes.ravel(), group.iterrows()):
            axis.imshow(patches[primary_key][index], cmap="gray", vmin=0, vmax=255)
            axis.axis("off")
            timestamp = pd.Timestamp(observation.image_time)
            title = f"{timestamp:%Y-%m-%d %H:%MZ}"
            if observation.observation_id in transition_lookup.index:
                transition = transition_lookup.loc[observation.observation_id]
                title += (
                    f"\nNCC {transition[f'{primary_key}_prev_ncc']:.2f}"
                    f" | ORB Δ {transition['orb_prev_hamming_norm']:.2f}"
                )
            if observation.observation_id in tracking_lookup.index:
                tracked = tracking_lookup.loc[observation.observation_id]
                if np.isfinite(tracked.endpoint_error_m):
                    title += f"\n{selected_config}: {tracked.endpoint_error_m / 1000:.2f} km"
                else:
                    title += f"\n{selected_config}: {tracked.tracking_status}"
            axis.set_title(title, fontsize=8)
        fig.suptitle(
            f"{sequence} buoy {buoy_id} — north-up {primary_key.replace('_', ' ')} standard-VAE patches",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{safe_name(buoy_id)}.png", dpi=150)
        plt.close(fig)
        created += 1
    return created


def summary_figure(linked: pd.DataFrame, primary_key: str, output_path: Path) -> None:
    selections = []
    for sequence, config in (
        ("2020_03", "beam_confidence_update_m032"),
        ("2020_02", "beam_confidence_update_m032"),
        ("2015_full15", "greedy_rolling"),
    ):
        subset = linked[
            (linked.sequence == sequence)
            & (linked.backend == "ORB")
            & (linked.config == config)
            & (linked.tracking_status == "ok")
        ].copy()
        selections.append(subset)
    data = pd.concat(selections, ignore_index=True)
    metrics = [
        ("orb_anchor_hamming_norm", "ORB anchor Hamming / 256"),
        ("orb_prev_hamming_norm", "ORB previous Hamming / 256"),
        (f"{primary_key}_anchor_ncc", f"{primary_key} anchor NCC"),
        (
            f"{primary_key}_prev_histogram_js_distance",
            f"{primary_key} previous histogram JS distance",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    colors = {"2020_03": "tab:blue", "2020_02": "tab:orange", "2015_full15": "tab:green"}
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        for sequence, group in data.groupby("sequence"):
            axis.scatter(
                group[metric],
                group.endpoint_error_m / 1000.0,
                s=16,
                alpha=0.65,
                label=sequence,
                color=colors[sequence],
            )
        axis.axhline(2.0, color="0.4", linestyle="--", linewidth=0.8)
        axis.set_yscale("log")
        axis.set_xlabel(label)
        axis.set_ylabel("Buoy endpoint error (km, log)")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    sequence_summaries: list[dict],
    associations: pd.DataFrame,
    clustered_intervals: pd.DataFrame,
    update_summary: pd.DataFrame,
    primary_key: str,
) -> None:
    summary = pd.DataFrame.from_records(sequence_summaries)
    columns = [
        "sequence",
        "fixture_observations",
        "observations",
        "excluded_observations",
        "paths_ge_2",
        "transitions",
        "orb_descriptor_coverage",
        "xfeat_within_5km_coverage",
        "contact_sheets",
        "elapsed_seconds",
    ]
    view = summary[columns].copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(lambda value: f"{value:.3f}")
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()],
    ]
    selected = associations[
        (associations.sequence == "ALL")
        & (associations.backend == "ORB")
        & associations.config.isin(["beam_confidence_update_m032", "greedy_rolling"])
    ].sort_values("failure_auc_discrimination", ascending=False).head(12)
    association_columns = [
        "config",
        "metric",
        "observations",
        "failure_auc_discrimination",
        "failure_direction",
        "spearman_log_error",
    ]
    association_view = selected[association_columns].copy()
    for column in ("failure_auc_discrimination", "spearman_log_error"):
        association_view[column] = association_view[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    association_table = [
        "| " + " | ".join(association_columns) + " |",
        "| " + " | ".join(["---"] * len(association_columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in association_view.to_numpy()],
    ]
    interval_columns = [
        "sequence",
        "metric",
        "observations",
        "unique_buoys",
        "failure_direction",
        "failure_auc_95ci",
        "spearman_95ci",
    ]
    interval_view = clustered_intervals.copy()
    interval_view["failure_auc_95ci"] = interval_view.apply(
        lambda row: (
            f"{row.failure_auc:.3f} [{row.failure_auc_ci025:.3f}, "
            f"{row.failure_auc_ci975:.3f}]"
        ),
        axis=1,
    )
    interval_view["spearman_95ci"] = interval_view.apply(
        lambda row: (
            f"{row.spearman_log_error:.3f} [{row.spearman_ci025:.3f}, "
            f"{row.spearman_ci975:.3f}]"
        ),
        axis=1,
    )
    interval_table = [
        "| " + " | ".join(interval_columns) + " |",
        "| " + " | ".join(["---"] * len(interval_columns)) + " |",
        *[
            "| " + " | ".join(map(str, row)) + " |"
            for row in interval_view[interval_columns].to_numpy()
        ],
    ]
    update_columns = [
        "sequence",
        "transitions",
        "anchor_trackable",
        "update_trackable",
        "newly_trackable_with_update",
        "anchor_within_2km_fraction_all",
        "update_within_2km_fraction_all",
        "rescued_within_2km",
        "harmed_beyond_2km",
    ]
    update_view = update_summary[update_columns].copy()
    for column in ("anchor_within_2km_fraction_all", "update_within_2km_fraction_all"):
        update_view[column] = update_view[column].map(lambda value: f"{value:.3f}")
    update_table = [
        "| " + " | ".join(update_columns) + " |",
        "| " + " | ".join(["---"] * len(update_columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in update_view.to_numpy()],
    ]
    path.write_text(
        "# Exact-time buoy patch and descriptor evolution archive\n\n"
        "All patches use the unchanged standard VAE band. Map-aligned patches are "
        "north-up in EPSG:3413 and centred on the exact-time buoy position; native "
        "patches retain image orientation. Invalid and land support are stored as masks.\n\n"
        + "\n".join(table)
        + f"\n\nThe primary diagnostic view is `{primary_key}`. Each transition contains "
        "previous and immutable-anchor comparisons for raw NCC, gradient NCC, SSIM, "
        "normalized mutual information, histogram Jensen-Shannon distance, RMSE, "
        "phase correlation, ORB Hamming, and XFeat cosine distance.\n\n"
        "## Strongest descriptive associations with >2 km tracking failure\n\n"
        + "\n".join(association_table)
        + "\n\nAUC values are descriptive on the existing fixtures, not trained thresholds. "
        "Metrics were computed before joining tracking errors. Path failures and "
        "skipped observations remain explicit in `path_outcomes.csv` and "
        "`tracking_linked.csv`.\n\n"
        "## Validation and holdout uncertainty\n\n"
        + "\n".join(interval_table)
        + "\n\nIntervals use 1,000 deterministic bootstrap resamples of whole buoy paths, "
        "not individual transitions. They quantify sampling uncertainty only. "
        "Exact-buoy appearance is an evaluation diagnostic and must not enter an "
        "operational matcher; a deployable gate must use candidate-location evidence "
        "and be retested on held-out sequences.\n\n"
        "## Paired fixed-anchor versus confidence-update graph results\n\n"
        + "\n".join(update_table)
        + "\n\nFractions use every eligible transition as the denominator, so an untrackable "
        "graph path is not silently removed. `update_effects.csv` retains the paired "
        "per-transition errors and appearance diagnostics.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--orb-results-root", type=Path, default=DEFAULT_ORB_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/buoy_patch_evolution/q2q98_clahe25",
    )
    parser.add_argument("--sequences", default="2020_03,2020_02,2015_full15")
    parser.add_argument("--analysis-epsg", type=int, default=3413)
    parser.add_argument("--map-widths-m", default="2500,5000,10000")
    parser.add_argument("--map-patch-pixels", type=int, default=129)
    parser.add_argument("--native-sizes-px", default="31,65,129")
    parser.add_argument("--xfeat-max-side", type=int, default=1536)
    parser.add_argument("--xfeat-top-k", type=int, default=16000)
    parser.add_argument("--xfeat-detection-threshold", type=float, default=0.05)
    parser.add_argument("--xfeat-max-distance-m", type=float, default=5000.0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--reuse-extracted",
        action="store_true",
        help="Reuse existing patches/descriptors and rebuild only result joins and summaries.",
    )
    args = parser.parse_args()
    args.sequences = tuple(item.strip() for item in args.sequences.split(",") if item.strip())
    args.map_widths_m = parse_numbers(args.map_widths_m, float)
    args.native_sizes_px = parse_numbers(args.native_sizes_px, int)
    if args.map_patch_pixels % 2 != 1 or any(size % 2 != 1 for size in args.native_sizes_px):
        parser.error("Patch pixel dimensions must be odd.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_observations = []
    all_transitions = []
    all_linked = []
    all_outcomes = []
    all_excluded = []
    all_path_summaries = []
    summaries = []
    primary_key = f"map_{int(args.map_widths_m[len(args.map_widths_m) // 2])}m"
    run_started = time.perf_counter()
    prior_summary_path = args.out_dir / "sequence_summary.csv"
    prior_summary = (
        pd.read_csv(prior_summary_path).set_index("sequence")
        if args.reuse_extracted and prior_summary_path.exists()
        else pd.DataFrame()
    )

    for sequence in args.sequences:
        sequence_dir = args.out_dir / sequence
        sequence_dir.mkdir(parents=True, exist_ok=True)
        if args.reuse_extracted:
            observations = pd.read_csv(
                sequence_dir / "observations.csv",
                parse_dates=["image_time", "nearest_observation_time"],
                dtype={"buoy_id": str},
            )
            transitions = pd.read_csv(
                sequence_dir / "transitions.csv",
                parse_dates=["source_time", "target_time"],
                dtype={"buoy_id": str},
            )
            with np.load(sequence_dir / "patches.npz") as archive:
                patches = {
                    key: archive[key]
                    for key in archive.files
                    if key != "observation_id" and not key.endswith("_valid")
                }
            excluded = pd.read_csv(
                sequence_dir / "excluded_observations.csv",
                dtype={"buoy_id": str},
            )
            fixture_observations = len(observations) + len(excluded)
            if sequence in prior_summary.index:
                elapsed = float(prior_summary.loc[sequence, "elapsed_seconds"])
                prior_sheets = int(prior_summary.loc[sequence, "contact_sheets"])
            else:
                elapsed = math.nan
                prior_sheets = len(list((sequence_dir / "contact_sheets").glob("*.png")))
        else:
            fixture = args.fixture_root / f"coincidences_{sequence}.csv"
            coincidences, excluded = load_fixture_with_exclusions(fixture, args.analysis_epsg)
            excluded["sequence"] = sequence
            excluded.to_csv(sequence_dir / "excluded_observations.csv", index=False)
            fixture_observations = len(coincidences) + len(excluded)
            (
                observations,
                transitions,
                patches,
                masks,
                descriptors,
                features,
                elapsed,
            ) = extract_sequence(sequence, coincidences, sequence_dir, args)
            observation_ids = observations.observation_id.astype(str).to_numpy(dtype="U")
            np.savez_compressed(
                sequence_dir / "patches.npz",
                observation_id=observation_ids,
                **patches,
                **{f"{key}_valid": value for key, value in masks.items()},
            )
            np.savez_compressed(
                sequence_dir / "descriptors.npz",
                observation_id=observation_ids,
                **descriptors,
            )
            observations.to_csv(sequence_dir / "observations.csv", index=False)
            transitions.to_csv(sequence_dir / "transitions.csv", index=False)
        linked, outcomes = link_tracking_results(sequence, transitions, args.orb_results_root)
        path_summary = tracking_path_summary(linked, primary_key)
        linked.to_csv(sequence_dir / "tracking_linked.csv", index=False)
        outcomes.to_csv(sequence_dir / "path_outcomes.csv", index=False)
        path_summary.to_csv(sequence_dir / "tracking_path_summary.csv", index=False)
        sheet_count = (
            prior_sheets
            if args.reuse_extracted
            else contact_sheets(
                observations,
                transitions,
                patches,
                linked,
                sequence_dir / "contact_sheets",
                primary_key,
            )
        )
        summaries.append(
            {
                "sequence": sequence,
                "fixture_observations": fixture_observations,
                "observations": len(observations),
                "excluded_observations": len(excluded),
                "paths_ge_2": int((observations.groupby("buoy_id").size() >= 2).sum()),
                "transitions": len(transitions),
                "orb_descriptor_coverage": float(observations.orb_descriptor_available.mean()),
                "xfeat_within_5km_coverage": float(observations.xfeat_descriptor_within_limit.mean()),
                "contact_sheets": sheet_count,
                "elapsed_seconds": elapsed,
            }
        )
        all_observations.append(observations)
        all_transitions.append(transitions)
        all_linked.append(linked)
        all_outcomes.append(outcomes)
        all_excluded.append(excluded)
        all_path_summaries.append(path_summary)

    observations = pd.concat(all_observations, ignore_index=True, sort=False)
    transitions = pd.concat(all_transitions, ignore_index=True, sort=False)
    linked = pd.concat(all_linked, ignore_index=True, sort=False)
    outcomes = pd.concat(all_outcomes, ignore_index=True, sort=False)
    excluded = pd.concat(all_excluded, ignore_index=True, sort=False)
    path_summaries = pd.concat(all_path_summaries, ignore_index=True, sort=False)
    associations = association_metrics(linked, primary_key)
    clustered_intervals = clustered_association_intervals(linked, primary_key)
    update_effects, update_summary = paired_update_effects(linked)
    observations.to_csv(args.out_dir / "observations_all.csv", index=False)
    transitions.to_csv(args.out_dir / "transitions_all.csv", index=False)
    linked.to_csv(args.out_dir / "tracking_linked_all.csv", index=False)
    outcomes.to_csv(args.out_dir / "path_outcomes_all.csv", index=False)
    excluded.to_csv(args.out_dir / "excluded_observations_all.csv", index=False)
    path_summaries.to_csv(args.out_dir / "tracking_path_summary_all.csv", index=False)
    update_effects.to_csv(args.out_dir / "update_effects.csv", index=False)
    update_summary.to_csv(args.out_dir / "update_effect_summary.csv", index=False)
    associations.to_csv(args.out_dir / "failure_associations.csv", index=False)
    clustered_intervals.to_csv(
        args.out_dir / "representative_association_bootstrap.csv", index=False
    )
    pd.DataFrame.from_records(summaries).to_csv(args.out_dir / "sequence_summary.csv", index=False)
    summary_figure(linked, primary_key, args.out_dir / "appearance_vs_tracking_error.png")
    write_report(
        args.out_dir / "report.md",
        summaries,
        associations,
        clustered_intervals,
        update_summary,
        primary_key,
    )
    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "fixture_root": str(args.fixture_root),
        "orb_results_root": str(args.orb_results_root),
        "sequences": args.sequences,
        "analysis_crs": f"EPSG:{args.analysis_epsg}",
        "preprocessing": "balanced_q2q98_clahe25 standard VAE band; no additional preprocessing",
        "map_patch_widths_m": args.map_widths_m,
        "map_patch_pixels": args.map_patch_pixels,
        "map_patch_axis": "north-up EPSG:3413; row zero is maximum y",
        "native_patch_sizes_px": args.native_sizes_px,
        "image_dtype": "uint8",
        "orb": {
            "dtype": "uint8",
            "dimensions": 32,
            "WTA_K": 2,
            "norm": "hamming",
            "nlevels": 5,
            "patch_size": 64,
            "keypoint_size": 31,
            "octave": 5,
            "orientation": "geographic image angle",
        },
        "xfeat": {
            "dtype": "float32",
            "dimensions": 64,
            "norm": "cosine",
            "max_side": args.xfeat_max_side,
            "top_k": args.xfeat_top_k,
            "detection_threshold": args.xfeat_detection_threshold,
            "maximum_nearest_feature_distance_m": args.xfeat_max_distance_m,
        },
        "primary_diagnostic_patch": primary_key,
        "fixture_observations": int(len(observations) + len(excluded)),
        "extracted_observations": int(len(observations)),
        "excluded_observations": int(len(excluded)),
        "analysis_run_mode": "reuse_extracted" if args.reuse_extracted else "full_extraction",
        "sequence_extraction_elapsed_seconds": {
            row["sequence"]: row["elapsed_seconds"] for row in summaries
        },
        "extraction_elapsed_seconds_total": float(
            np.nansum([row["elapsed_seconds"] for row in summaries])
        ),
        "analysis_elapsed_seconds": time.perf_counter() - run_started,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame.from_records(summaries).to_string(index=False))
    print(associations.sort_values("failure_auc_discrimination", ascending=False).head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
