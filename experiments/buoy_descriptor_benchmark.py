#!/usr/bin/env python3
"""Benchmark descriptor stability and buoy tracking on coincident SAR images.

The experiment deliberately separates appearance matching from physics:

1. Interpolate each buoy track to the exact SAR acquisition time.
2. Extract a descriptor at that known location.
3. Retrieve the descriptor in the next coincident image, first scene-wide and
   then with only a speed-scaled candidate gate.
4. Report endpoint error, truth rank, and the candidate-grid error floor.

This is an experiment harness, not a production LiMOSAT backend.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nansat import NSR
from osgeo import gdal
from pyproj import Transformer

from limosat.image import Image

gdal.UseExceptions()


@dataclass(frozen=True)
class DescriptorVariant:
    name: str
    extractor_key: str
    norm: str
    angle_mode: str = "geographic"
    preprocessing: str = "native"
    keypoint_size: float = 31.0
    octave: int = 0


@dataclass
class CandidateGrid:
    pixel_xy: np.ndarray
    map_xy: np.ndarray


DESCRIPTOR_VARIANTS = (
    DescriptorVariant(
        "orb_geo_hamming",
        "orb",
        "hamming",
        angle_mode="geographic",
        keypoint_size=31.0,
        octave=5,
    ),
    DescriptorVariant(
        "orb_geo_hamming2",
        "orb",
        "hamming2",
        angle_mode="geographic",
        keypoint_size=31.0,
        octave=5,
    ),
    DescriptorVariant(
        "orb_zero_hamming",
        "orb",
        "hamming",
        angle_mode="zero",
        keypoint_size=31.0,
        octave=5,
    ),
    DescriptorVariant(
        "brisk_geo_hamming",
        "brisk",
        "hamming",
        angle_mode="geographic",
        keypoint_size=31.0,
    ),
)


def repaired_path(value: str) -> str:
    """Resolve the historical catalog root used by the local fixture."""
    return str(value).replace(
        "/Users/seachu/arktalas/arktalas_vae",
        "/Users/seachu/results/arktalas_vae",
    )


def timestamp_seconds(values: Iterable) -> np.ndarray:
    """Convert timestamps to UTC seconds without mixing pandas resolutions."""
    series = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    return series.astype("int64").to_numpy(dtype=np.float64) / 1.0e9


def interpolate_xy_at_time(track: pd.DataFrame, when) -> tuple[float, float]:
    """Linearly interpolate projected buoy coordinates at an exact time."""
    ordered = track.copy()
    ordered["_time"] = pd.to_datetime(ordered["timestamp"], utc=True, errors="coerce")
    ordered = (
        ordered.dropna(subset=["_time", "x", "y"])
        .sort_values("_time")
        .drop_duplicates("_time")
    )
    if len(ordered) < 2:
        raise ValueError("At least two unique buoy observations are required for interpolation.")
    times = timestamp_seconds(ordered["_time"])
    target = float(timestamp_seconds([when])[0])
    if target < times[0] or target > times[-1]:
        raise ValueError("Requested time lies outside the buoy track.")
    return (
        float(np.interp(target, times, ordered["x"].to_numpy(dtype=float))),
        float(np.interp(target, times, ordered["y"].to_numpy(dtype=float))),
    )


def load_catalog(path: Path) -> pd.DataFrame:
    catalog = gpd.read_file(path).sort_values("image_id").reset_index(drop=True)
    path_col = "filepath" if "filepath" in catalog.columns else "filename"
    catalog["image_filepath"] = catalog[path_col].astype(str).map(repaired_path)
    catalog["image_filename"] = catalog["image_filepath"].map(lambda value: Path(value).name)
    catalog["image_time"] = pd.to_datetime(catalog["datetime"], utc=True)
    missing = [value for value in catalog["image_filepath"] if not Path(value).exists()]
    if missing:
        raise FileNotFoundError(f"Catalog contains {len(missing)} missing images; first: {missing[0]}")
    return catalog


def build_coincidences(
    catalog: pd.DataFrame,
    buoy_path: Path,
    max_time_difference_minutes: float,
    outside_track_policy: str = "error",
) -> pd.DataFrame:
    if outside_track_policy not in {"error", "skip"}:
        raise ValueError("outside_track_policy must be 'error' or 'skip'.")
    buoy_rows = gpd.read_file(buoy_path)
    buoy_rows["timestamp"] = pd.to_datetime(buoy_rows["timestamp"], utc=True)
    buoy_rows["image_timestamp"] = pd.to_datetime(buoy_rows["image_timestamp"], utc=True)
    catalog_names = set(catalog["image_filename"])
    matched = buoy_rows[buoy_rows["image_filename"].isin(catalog_names)].copy()
    matched["abs_time_diff_min"] = matched["time_diff_min"].abs()
    matched = matched[matched["abs_time_diff_min"] <= max_time_difference_minutes]
    matched = (
        matched.sort_values("abs_time_diff_min")
        .drop_duplicates(["BuoyID", "image_filename"])
        .reset_index(drop=True)
    )
    catalog_by_name = catalog.set_index("image_filename")
    track_rows = (
        buoy_rows[["BuoyID", "timestamp", "x", "y"]]
        .dropna()
        .groupby(["BuoyID", "timestamp"], as_index=False)[["x", "y"]]
        .mean()
    )

    records = []
    for row in matched.itertuples(index=False):
        image = catalog_by_name.loc[row.image_filename]
        image_time = pd.Timestamp(image["image_time"])
        track = track_rows[track_rows["BuoyID"] == row.BuoyID]
        try:
            x, y = interpolate_xy_at_time(track, image_time)
        except ValueError:
            if outside_track_policy == "skip":
                continue
            raise
        records.append(
            {
                "buoy_id": str(row.BuoyID),
                "image_id": int(image["image_id"]),
                "image_filename": row.image_filename,
                "image_filepath": image["image_filepath"],
                "image_time": image_time,
                "nearest_observation_time": pd.Timestamp(row.timestamp),
                "nearest_observation_offset_minutes": float(row.abs_time_diff_min),
                "x": x,
                "y": y,
            }
        )
    coincidences = pd.DataFrame.from_records(records)
    return coincidences.sort_values(["buoy_id", "image_time"]).reset_index(drop=True)


def build_antarctic_coincidences(
    scene_matches_path: Path,
    buoy_path: Path,
    image_root: Path,
) -> pd.DataFrame:
    """Build exact-time AWI buoy/SAR coincidences in EPSG:3412 metres."""
    matches = pd.read_csv(scene_matches_path)
    tracks = pd.read_csv(buoy_path)
    matches["image_time"] = pd.to_datetime(matches["image_time"], utc=True)
    tracks["timestamp"] = pd.to_datetime(tracks["time"], utc=True)
    tracks["buoy_id"] = tracks["buoy_id"].astype(str)
    matches["buoy_name"] = matches["buoy_name"].astype(str)
    if "mask_value" in matches:
        matches = matches[matches["mask_value"] < 2].copy()

    ordered_images = (
        matches[["image_filepath", "image_time"]]
        .drop_duplicates()
        .sort_values(["image_time", "image_filepath"])
        .reset_index(drop=True)
    )
    image_ids = {
        row.image_filepath: image_id
        for image_id, row in enumerate(ordered_images.itertuples(index=False))
    }
    records = []
    for row in matches.itertuples(index=False):
        track = tracks[tracks["buoy_id"] == row.buoy_name]
        if len(track) < 2:
            continue
        x, y = interpolate_xy_at_time(track, row.image_time)
        time_deltas = (track["timestamp"] - row.image_time).abs()
        nearest_index = time_deltas.idxmin()
        image_path = image_root / Path(row.image_filepath).name
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        records.append(
            {
                "buoy_id": row.buoy_name,
                "floe_id": str(row.floe_id),
                "image_id": image_ids[row.image_filepath],
                "image_filename": Path(row.image_filepath).name,
                "image_filepath": str(image_path),
                "image_time": pd.Timestamp(row.image_time),
                "nearest_observation_time": pd.Timestamp(tracks.loc[nearest_index, "timestamp"]),
                "nearest_observation_offset_minutes": float(time_deltas.loc[nearest_index].total_seconds() / 60.0),
                "x": x,
                "y": y,
            }
        )
    coincidences = pd.DataFrame.from_records(records)
    return coincidences.sort_values(["buoy_id", "image_time"]).reset_index(drop=True)


def build_pairs(coincidences: pd.DataFrame) -> pd.DataFrame:
    records = []
    path_column = (
        "experiment_trajectory_id"
        if "experiment_trajectory_id" in coincidences
        else "buoy_id"
    )
    for trajectory_id, group in coincidences.groupby(path_column, sort=True):
        rows = group.sort_values("image_time").to_dict("records")
        for source, target in zip(rows[:-1], rows[1:]):
            dt_hours = (target["image_time"] - source["image_time"]).total_seconds() / 3600.0
            if dt_hours <= 0:
                continue
            records.append(
                {
                    "pair_id": len(records),
                    "buoy_id": str(source["buoy_id"]),
                    "trajectory_id": trajectory_id,
                    "dt_hours": dt_hours,
                    **{f"source_{key}": value for key, value in source.items()},
                    **{f"target_{key}": value for key, value in target.items()},
                }
            )
    return pd.DataFrame.from_records(records)


def read_band(path: str, band: int) -> np.ndarray:
    dataset = gdal.Open(path)
    if dataset is None:
        raise FileNotFoundError(path)
    return np.asarray(dataset.GetRasterBand(band).ReadAsArray())


@lru_cache(maxsize=4)
def read_scene(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    image = read_band(path, 1)
    if image.dtype != np.uint8:
        finite = np.isfinite(image)
        lo, hi = np.nanpercentile(image[finite], [1, 99])
        image = np.clip((image - lo) * 255.0 / max(hi - lo, 1.0e-9), 0, 255).astype(np.uint8)
    else:
        image = image.copy()
    dataset = gdal.Open(path)
    mask = read_band(path, 2).astype(np.uint8) if dataset.RasterCount >= 2 else None
    if mask is not None:
        image[mask >= 2] = 0
    return image, mask


@lru_cache(maxsize=4)
def image_object(path: str, analysis_epsg: int) -> Image:
    return Image(path, srs=NSR(analysis_epsg))


@lru_cache(maxsize=16)
def image_angle(path: str, analysis_epsg: int) -> float:
    """Image rotation from projected north, valid in either polar hemisphere."""
    lons, lats = image_object(path, analysis_epsg).get_corners()
    transformer = Transformer.from_crs(4326, analysis_epsg, always_xy=True)
    x, y = transformer.transform(lons, lats)
    return float(np.degrees(np.arctan2(x[1] - x[0], y[1] - y[0])))


def map_to_pixel(path: str, x: float, y: float, analysis_epsg: int) -> tuple[float, float]:
    col, row = image_object(path, analysis_epsg).transform_points(
        [x], [y], DstToSrc=1, dst_srs=NSR(analysis_epsg)
    )
    return float(col[0]), float(row[0])


def pixels_to_map(path: str, pixel_xy: np.ndarray, analysis_epsg: int) -> np.ndarray:
    if len(pixel_xy) == 0:
        return np.empty((0, 2), dtype=float)
    x, y = image_object(path, analysis_epsg).transform_points(
        pixel_xy[:, 0], pixel_xy[:, 1], DstToSrc=0, dst_srs=NSR(analysis_epsg)
    )
    return np.column_stack((x, y)).astype(float)


def local_statistics(image: np.ndarray, col: float, row: float, size: int = 65) -> dict:
    patch = cv2.getRectSubPix(image, (size, size), (float(col), float(row)))
    hist = np.bincount(patch.ravel(), minlength=256).astype(float)
    probabilities = hist[hist > 0] / hist.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return {
        "local_mean": float(patch.mean()),
        "local_std": float(patch.std()),
        "local_entropy_bits": entropy,
    }


def annotate_coincidences(
    coincidences: pd.DataFrame,
    analysis_epsg: int,
    outside_scene_policy: str = "error",
) -> pd.DataFrame:
    if outside_scene_policy not in {"error", "skip"}:
        raise ValueError("outside_scene_policy must be 'error' or 'skip'.")
    records = []
    ordered = coincidences.copy()
    ordered["_input_order"] = np.arange(len(ordered), dtype=int)
    ordered = ordered.sort_values(["image_filepath", "_input_order"])
    for row in ordered.to_dict("records"):
        image, mask = read_scene(row["image_filepath"])
        col, pixel_row = map_to_pixel(
            row["image_filepath"], row["x"], row["y"], analysis_epsg
        )
        in_bounds = 0 <= col < image.shape[1] and 0 <= pixel_row < image.shape[0]
        mask_value = None
        if in_bounds and mask is not None:
            mask_row = int(np.clip(round(pixel_row), 0, image.shape[0] - 1))
            mask_col = int(np.clip(round(col), 0, image.shape[1] - 1))
            mask_value = int(mask[mask_row, mask_col])
        if not in_bounds:
            if outside_scene_policy == "skip":
                continue
            raise ValueError(
                f"Buoy point ({row['x']}, {row['y']}) is outside {row['image_filename']}"
            )
        records.append(
            {
                **row,
                "col": col,
                "row": pixel_row,
                "image_height": image.shape[0],
                "image_width": image.shape[1],
                "image_angle_deg": image_angle(row["image_filepath"], analysis_epsg),
                "mask_value": mask_value,
                **local_statistics(image, col, pixel_row),
            }
        )
    annotated = pd.DataFrame.from_records(records)
    if annotated.empty:
        return annotated.drop(columns=["_input_order"], errors="ignore")
    return (
        annotated.sort_values("_input_order")
        .drop(columns=["_input_order"])
        .reset_index(drop=True)
    )


def local_contrast_image(image: np.ndarray, sigma: float = 16.0) -> np.ndarray:
    values = image.astype(np.float32)
    mean = cv2.GaussianBlur(values, (0, 0), sigmaX=sigma, sigmaY=sigma)
    second = cv2.GaussianBlur(values * values, (0, 0), sigmaX=sigma, sigmaY=sigma)
    std = np.sqrt(np.maximum(second - mean * mean, 1.0))
    normalized = np.clip((values - mean) / std, -3.0, 3.0)
    return np.rint((normalized + 3.0) * (255.0 / 6.0)).astype(np.uint8)


def gradient_image(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    gx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    scale = float(np.percentile(magnitude, 99))
    return np.clip(magnitude * (255.0 / max(scale, 1.0e-6)), 0, 255).astype(np.uint8)


def build_extractor(key: str):
    if key == "orb":
        return cv2.ORB_create(
            nfeatures=100,
            scaleFactor=1.25,
            nlevels=5,
            edgeThreshold=16,
            firstLevel=0,
            patchSize=64,
            scoreType=cv2.ORB_HARRIS_SCORE,
        )
    if key == "sift":
        return cv2.SIFT_create(contrastThreshold=0.01)
    if key == "brisk":
        return cv2.BRISK_create()
    raise ValueError(f"Unknown extractor: {key}")


def make_keypoints(
    pixel_xy: np.ndarray,
    size: float,
    octave: int,
    angle: float,
) -> list[cv2.KeyPoint]:
    return [
        cv2.KeyPoint(float(col), float(row), size=size, angle=angle, octave=octave)
        for col, row in pixel_xy
    ]


def compute_descriptors(
    extractor,
    image: np.ndarray,
    pixel_xy: np.ndarray,
    variant: DescriptorVariant,
    angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    keypoint_angle = angle if variant.angle_mode == "geographic" else 0.0
    keypoints = make_keypoints(
        pixel_xy,
        size=variant.keypoint_size,
        octave=variant.octave,
        angle=keypoint_angle,
    )
    output_keypoints, descriptors = extractor.compute(image, keypoints)
    if descriptors is None or not output_keypoints:
        width = 128 if variant.extractor_key == "sift" else 32
        dtype = np.float32 if variant.norm == "l2" else np.uint8
        return np.empty((0, 2), dtype=float), np.empty((0, width), dtype=dtype)
    output_xy = np.asarray([keypoint.pt for keypoint in output_keypoints], dtype=float)
    return output_xy, np.asarray(descriptors)


HAMMING_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)
HAMMING2_LUT = np.asarray(
    [sum(((value >> shift) & 0b11) != 0 for shift in (0, 2, 4, 6)) for value in range(256)],
    dtype=np.uint8,
)


def descriptor_distances(source: np.ndarray, candidates: np.ndarray, norm: str) -> np.ndarray:
    if source.ndim != 1:
        source = np.asarray(source).reshape(-1)
    if norm == "hamming":
        return HAMMING_LUT[np.bitwise_xor(candidates, source)].sum(axis=1).astype(float)
    if norm == "hamming2":
        return HAMMING2_LUT[np.bitwise_xor(candidates, source)].sum(axis=1).astype(float)
    if norm == "l2":
        delta = candidates.astype(np.float32) - source.astype(np.float32)
        return np.sqrt(np.einsum("ij,ij->i", delta, delta, optimize=True))
    if norm == "cosine":
        candidate_norm = np.linalg.norm(candidates, axis=1)
        source_norm = np.linalg.norm(source)
        similarity = (candidates @ source) / np.maximum(candidate_norm * source_norm, 1.0e-12)
        return 1.0 - similarity
    raise ValueError(f"Unknown norm: {norm}")


def normalized_descriptor_distance(value: float, descriptor_length: int, norm: str) -> float:
    if norm == "hamming":
        return value / float(descriptor_length * 8)
    if norm == "hamming2":
        return value / float(descriptor_length * 4)
    if norm == "cosine":
        return value / 2.0
    return value / math.sqrt(float(descriptor_length))


def select_gate(map_xy: np.ndarray, source_xy: np.ndarray, radius_m: float | None) -> np.ndarray:
    finite = np.isfinite(map_xy).all(axis=1)
    if radius_m is None:
        return finite
    return finite & (np.linalg.norm(map_xy - source_xy, axis=1) <= radius_m)


def candidate_grid(path: str, stride: int, border: int, analysis_epsg: int) -> CandidateGrid:
    image, mask = read_scene(path)
    rows, cols = np.mgrid[border : image.shape[0] - border + 1 : stride, border : image.shape[1] - border + 1 : stride]
    pixel_xy = np.column_stack((cols.ravel(), rows.ravel())).astype(float)
    if mask is not None:
        valid = mask[rows.ravel(), cols.ravel()] < 2
        pixel_xy = pixel_xy[valid]
    map_xy = pixels_to_map(path, pixel_xy, analysis_epsg)
    finite = np.isfinite(map_xy).all(axis=1)
    return CandidateGrid(pixel_xy=pixel_xy[finite], map_xy=map_xy[finite])


def map_output_descriptors_to_grid(
    requested_grid: CandidateGrid,
    output_xy: np.ndarray,
    descriptors: np.ndarray,
) -> tuple[CandidateGrid, np.ndarray]:
    if len(output_xy) == len(requested_grid.pixel_xy) and np.allclose(output_xy, requested_grid.pixel_xy, atol=0.51):
        return requested_grid, descriptors
    lookup = {
        (round(float(col), 3), round(float(row), 3)): index
        for index, (col, row) in enumerate(requested_grid.pixel_xy)
    }
    indexes = [lookup.get((round(float(col), 3), round(float(row), 3))) for col, row in output_xy]
    keep = np.asarray([index is not None for index in indexes], dtype=bool)
    matched_indexes = np.asarray([index for index in indexes if index is not None], dtype=int)
    return (
        CandidateGrid(
            pixel_xy=requested_grid.pixel_xy[matched_indexes],
            map_xy=requested_grid.map_xy[matched_indexes],
        ),
        descriptors[keep],
    )


def rank_and_retrieve(
    distances: np.ndarray,
    candidate_grid_value: CandidateGrid,
    gate_mask: np.ndarray,
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    truth_distance: float,
) -> dict:
    selected = np.flatnonzero(gate_mask)
    if len(selected) == 0:
        return {"candidate_count": 0, "accepted": False}
    gated_distances = distances[selected]
    order = np.argsort(gated_distances, kind="stable")
    best_local = int(order[0])
    best_index = int(selected[best_local])
    predicted_xy = candidate_grid_value.map_xy[best_index]
    best_distance = float(gated_distances[best_local])
    second_distance = float(gated_distances[order[1]]) if len(order) > 1 else math.nan
    spatial_errors = np.linalg.norm(candidate_grid_value.map_xy[selected] - target_xy, axis=1)
    quantization_local = int(np.argmin(spatial_errors))
    quantization_index = int(selected[quantization_local])
    quantization_error = float(spatial_errors[quantization_local])
    quantized_truth_distance = float(distances[quantization_index])
    truth_rank = 1 + int(np.count_nonzero(gated_distances < truth_distance))
    quantized_truth_rank = 1 + int(np.count_nonzero(gated_distances < quantized_truth_distance))
    return {
        "accepted": True,
        "candidate_count": int(len(selected)),
        "predicted_x": float(predicted_xy[0]),
        "predicted_y": float(predicted_xy[1]),
        "endpoint_error_m": float(np.linalg.norm(predicted_xy - target_xy)),
        "predicted_displacement_m": float(np.linalg.norm(predicted_xy - source_xy)),
        "best_descriptor_distance": best_distance,
        "second_descriptor_distance": second_distance,
        "descriptor_margin": second_distance - best_distance if np.isfinite(second_distance) else math.nan,
        "truth_descriptor_rank": truth_rank,
        "truth_descriptor_rank_fraction": truth_rank / float(len(selected) + 1),
        "quantized_truth_rank": quantized_truth_rank,
        "quantized_truth_rank_fraction": quantized_truth_rank / float(len(selected) + 1),
        "candidate_quantization_error_m": quantization_error,
        "quantized_truth_descriptor_distance": quantized_truth_distance,
    }


def exact_descriptor(
    extractor,
    image: np.ndarray,
    pixel_xy: np.ndarray,
    variant: DescriptorVariant,
    angle: float,
) -> np.ndarray | None:
    _, descriptor = compute_descriptors(
        extractor,
        image,
        np.asarray(pixel_xy, dtype=float).reshape(1, 2),
        variant,
        angle,
    )
    return descriptor[0] if len(descriptor) == 1 else None


def run_descriptor_variants(
    pairs: pd.DataFrame,
    grids: dict[str, CandidateGrid],
    max_speed_m_per_day: float,
    analysis_epsg: int,
) -> tuple[list[dict], list[dict]]:
    results = []
    timings = []
    for variant in DESCRIPTOR_VARIANTS:
        variant_started = time.perf_counter()
        extractor = build_extractor(variant.extractor_key)
        for target_path, target_pairs in pairs.groupby("target_image_filepath", sort=False):
            target_native, _ = read_scene(target_path)
            target_image = (
                local_contrast_image(target_native)
                if variant.preprocessing == "local_contrast"
                else target_native
            )
            requested_grid = grids[target_path]
            output_xy, descriptors = compute_descriptors(
                extractor,
                target_image,
                requested_grid.pixel_xy,
                variant,
                image_angle(target_path, analysis_epsg),
            )
            usable_grid, descriptors = map_output_descriptors_to_grid(requested_grid, output_xy, descriptors)
            for pair in target_pairs.to_dict("records"):
                source_native, _ = read_scene(pair["source_image_filepath"])
                source_image = (
                    local_contrast_image(source_native)
                    if variant.preprocessing == "local_contrast"
                    else source_native
                )
                source_descriptor = exact_descriptor(
                    extractor,
                    source_image,
                    np.array([pair["source_col"], pair["source_row"]]),
                    variant,
                    pair["source_image_angle_deg"],
                )
                target_descriptor = exact_descriptor(
                    extractor,
                    target_image,
                    np.array([pair["target_col"], pair["target_row"]]),
                    variant,
                    pair["target_image_angle_deg"],
                )
                if source_descriptor is None or target_descriptor is None or len(descriptors) == 0:
                    results.append(
                        {
                            "pair_id": pair["pair_id"],
                            "buoy_id": pair["buoy_id"],
                            "trajectory_id": pair.get("trajectory_id"),
                            "experiment_split": pair.get(
                                "source_experiment_split"
                            ),
                            "month_exclusive_buoy": pair.get(
                                "source_month_exclusive_buoy"
                            ),
                            "method": variant.name,
                            "gate": "unavailable",
                            "accepted": False,
                        }
                    )
                    continue
                distances = descriptor_distances(source_descriptor, descriptors, variant.norm)
                truth_distance = float(
                    descriptor_distances(source_descriptor, target_descriptor[None], variant.norm)[0]
                )
                source_xy = np.array([pair["source_x"], pair["source_y"]], dtype=float)
                target_xy = np.array([pair["target_x"], pair["target_y"]], dtype=float)
                gate_definitions = (
                    ("scene_wide", None),
                    ("physics_50km_day", max_speed_m_per_day * pair["dt_hours"] / 24.0),
                )
                for gate_name, radius_m in gate_definitions:
                    gate_mask = select_gate(usable_grid.map_xy, source_xy, radius_m)
                    retrieval = rank_and_retrieve(
                        distances,
                        usable_grid,
                        gate_mask,
                        source_xy,
                        target_xy,
                        truth_distance,
                    )
                    results.append(
                        {
                            "pair_id": pair["pair_id"],
                            "buoy_id": pair["buoy_id"],
                            "trajectory_id": pair.get("trajectory_id"),
                            "experiment_split": pair.get(
                                "source_experiment_split"
                            ),
                            "month_exclusive_buoy": pair.get(
                                "source_month_exclusive_buoy"
                            ),
                            "source_image": pair["source_image_filename"],
                            "target_image": pair["target_image_filename"],
                            "dt_hours": pair["dt_hours"],
                            "true_displacement_m": float(np.linalg.norm(target_xy - source_xy)),
                            "true_speed_m_per_day": float(
                                np.linalg.norm(target_xy - source_xy) / pair["dt_hours"] * 24.0
                            ),
                            "method": variant.name,
                            "gate": gate_name,
                            "gate_radius_m": radius_m,
                            "descriptor_norm": variant.norm,
                            "truth_descriptor_distance": truth_distance,
                            "normalized_truth_descriptor_distance": normalized_descriptor_distance(
                                truth_distance, len(source_descriptor), variant.norm
                            ),
                            "source_local_mean": pair["source_local_mean"],
                            "target_local_mean": pair["target_local_mean"],
                            "local_mean_change": pair["target_local_mean"] - pair["source_local_mean"],
                            "source_local_std": pair["source_local_std"],
                            "target_local_std": pair["target_local_std"],
                            "local_std_change": pair["target_local_std"] - pair["source_local_std"],
                            **retrieval,
                        }
                    )
        elapsed = time.perf_counter() - variant_started
        timings.append(
            {
                "stage": "descriptor_retrieval",
                "method": variant.name,
                "seconds": elapsed,
                "pairs": len(pairs),
                "seconds_per_pair": elapsed / max(len(pairs), 1),
            }
        )
    return results, timings


def rotate_template(template: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray | None]:
    if abs(angle_deg) < 1.0e-6:
        return template, None
    height, width = template.shape
    transform = cv2.getRotationMatrix2D((width // 2, height // 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(template, transform, (width, height))
    mask = (rotated > 0).astype(np.uint8)
    return rotated, mask


def ncc_gate_mask(
    response_shape: tuple[int, int],
    target_path: str,
    source_xy: np.ndarray,
    radius_m: float | None,
    half_template: int,
    analysis_epsg: int,
) -> np.ndarray:
    height, width = response_shape
    if radius_m is None:
        return np.ones((height, width), dtype=bool)
    center_col, center_row = map_to_pixel(
        target_path, source_xy[0], source_xy[1], analysis_epsg
    )
    x_col, x_row = map_to_pixel(
        target_path, source_xy[0] + 1000.0, source_xy[1], analysis_epsg
    )
    y_col, y_row = map_to_pixel(
        target_path, source_xy[0], source_xy[1] + 1000.0, analysis_epsg
    )
    jacobian = np.array(
        [[x_col - center_col, y_col - center_col], [x_row - center_row, y_row - center_row]],
        dtype=float,
    ) / 1000.0
    inverse = np.linalg.pinv(jacobian)
    max_pixel_radius = radius_m * max(np.linalg.norm(jacobian[:, 0]), np.linalg.norm(jacobian[:, 1]))
    col0 = max(0, int(math.floor(center_col - half_template - max_pixel_radius - 2)))
    col1 = min(width, int(math.ceil(center_col - half_template + max_pixel_radius + 3)))
    row0 = max(0, int(math.floor(center_row - half_template - max_pixel_radius - 2)))
    row1 = min(height, int(math.ceil(center_row - half_template + max_pixel_radius + 3)))
    gate = np.zeros((height, width), dtype=bool)
    if col1 <= col0 or row1 <= row0:
        return gate
    rows, cols = np.mgrid[row0:row1, col0:col1]
    center_pixels = np.stack((cols + half_template, rows + half_template), axis=-1)
    pixel_delta = center_pixels - np.array([center_col, center_row])
    map_delta = pixel_delta @ inverse.T
    gate[row0:row1, col0:col1] = np.linalg.norm(map_delta, axis=-1) <= radius_m
    return gate


def run_ncc_variants(
    pairs: pd.DataFrame,
    max_speed_m_per_day: float,
    template_size: int,
    analysis_epsg: int,
) -> tuple[list[dict], list[dict]]:
    results = []
    method_seconds = {"ncc_geo_native": 0.0, "ncc_geo_gradient": 0.0}
    half = template_size // 2
    for pair in pairs.to_dict("records"):
        source_native, _ = read_scene(pair["source_image_filepath"])
        target_native, target_mask = read_scene(pair["target_image_filepath"])
        source_xy = np.array([pair["source_x"], pair["source_y"]], dtype=float)
        target_xy = np.array([pair["target_x"], pair["target_y"]], dtype=float)
        for method, transform in (
            ("ncc_geo_native", lambda value: value),
            ("ncc_geo_gradient", gradient_image),
        ):
            method_started = time.perf_counter()
            source_image = transform(source_native)
            target_image = transform(target_native)
            template = cv2.getRectSubPix(
                source_image,
                (template_size, template_size),
                (float(pair["source_col"]), float(pair["source_row"])),
            )
            angle_difference = pair["source_image_angle_deg"] - pair["target_image_angle_deg"]
            template, template_mask = rotate_template(template, angle_difference)
            if float(template.var()) < 1.0e-9:
                continue
            if template_mask is None:
                response = cv2.matchTemplate(target_image, template, cv2.TM_CCOEFF_NORMED)
            else:
                response = cv2.matchTemplate(
                    target_image,
                    template,
                    cv2.TM_CCOEFF_NORMED,
                    mask=template_mask,
                )
            valid = np.isfinite(response)
            if target_mask is not None:
                center_mask = target_mask[half : half + response.shape[0], half : half + response.shape[1]]
                valid &= center_mask < 2
            truth_left = int(round(pair["target_col"] - half))
            truth_top = int(round(pair["target_row"] - half))
            truth_valid = (
                0 <= truth_left < response.shape[1]
                and 0 <= truth_top < response.shape[0]
                and valid[truth_top, truth_left]
            )
            truth_score = float(response[truth_top, truth_left]) if truth_valid else math.nan
            for gate_name, radius_m in (
                ("scene_wide", None),
                ("physics_50km_day", max_speed_m_per_day * pair["dt_hours"] / 24.0),
            ):
                gate = valid & ncc_gate_mask(
                    response.shape,
                    pair["target_image_filepath"],
                    source_xy,
                    radius_m,
                    half,
                    analysis_epsg,
                )
                candidate_count = int(gate.sum())
                if candidate_count == 0:
                    results.append(
                        {
                            "pair_id": pair["pair_id"],
                            "buoy_id": pair["buoy_id"],
                            "method": method,
                            "gate": gate_name,
                            "accepted": False,
                        }
                    )
                    continue
                scored = np.where(gate, response, -np.inf)
                flat_index = int(np.argmax(scored))
                top, left = np.unravel_index(flat_index, response.shape)
                predicted_pixel = np.array([[left + half, top + half]], dtype=float)
                predicted_xy = pixels_to_map(
                    pair["target_image_filepath"], predicted_pixel, analysis_epsg
                )[0]
                rank = (
                    1 + int(np.count_nonzero(response[gate] > truth_score))
                    if np.isfinite(truth_score)
                    else None
                )
                results.append(
                    {
                        "pair_id": pair["pair_id"],
                        "buoy_id": pair["buoy_id"],
                        "source_image": pair["source_image_filename"],
                        "target_image": pair["target_image_filename"],
                        "dt_hours": pair["dt_hours"],
                        "true_displacement_m": float(np.linalg.norm(target_xy - source_xy)),
                        "true_speed_m_per_day": float(
                            np.linalg.norm(target_xy - source_xy) / pair["dt_hours"] * 24.0
                        ),
                        "method": method,
                        "gate": gate_name,
                        "gate_radius_m": radius_m,
                        "accepted": True,
                        "candidate_count": candidate_count,
                        "predicted_x": float(predicted_xy[0]),
                        "predicted_y": float(predicted_xy[1]),
                        "endpoint_error_m": float(np.linalg.norm(predicted_xy - target_xy)),
                        "predicted_displacement_m": float(np.linalg.norm(predicted_xy - source_xy)),
                        "best_descriptor_distance": float(-response[top, left]),
                        "truth_descriptor_distance": float(-truth_score),
                        "normalized_truth_descriptor_distance": float((1.0 - truth_score) / 2.0),
                        "truth_descriptor_rank": rank,
                        "truth_descriptor_rank_fraction": rank / float(candidate_count + 1) if rank else math.nan,
                        "source_local_mean": pair["source_local_mean"],
                        "target_local_mean": pair["target_local_mean"],
                        "local_mean_change": pair["target_local_mean"] - pair["source_local_mean"],
                        "source_local_std": pair["source_local_std"],
                        "target_local_std": pair["target_local_std"],
                        "local_std_change": pair["target_local_std"] - pair["source_local_std"],
                    }
                )
            method_seconds[method] += time.perf_counter() - method_started
    timings = [
        {
            "stage": "ncc_retrieval",
            "method": method,
            "seconds": seconds,
            "pairs": len(pairs),
            "seconds_per_pair": seconds / max(len(pairs), 1),
        }
        for method, seconds in method_seconds.items()
    ]
    return results, timings


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    accepted = results[results["accepted"].fillna(False)].copy()
    records = []
    for (method, gate), group in accepted.groupby(["method", "gate"], sort=True):
        errors = group["endpoint_error_m"].dropna().to_numpy(dtype=float)
        records.append(
            {
                "method": method,
                "gate": gate,
                "pairs": len(group),
                "median_error_m": float(np.median(errors)) if len(errors) else math.nan,
                "p90_error_m": float(np.percentile(errors, 90)) if len(errors) else math.nan,
                "max_error_m": float(np.max(errors)) if len(errors) else math.nan,
                "within_2km_fraction": float(np.mean(errors <= 2000.0)) if len(errors) else math.nan,
                "within_5km_fraction": float(np.mean(errors <= 5000.0)) if len(errors) else math.nan,
                "within_10km_fraction": float(np.mean(errors <= 10000.0)) if len(errors) else math.nan,
                "truth_rank_fraction_median": float(group["truth_descriptor_rank_fraction"].median()),
                "truth_distance_normalized_median": float(group["normalized_truth_descriptor_distance"].median()),
                "candidate_count_median": float(group["candidate_count"].median()),
                "quantization_error_median": float(group.get("candidate_quantization_error_m", pd.Series(dtype=float)).median()),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["gate", "median_error_m"])


def plot_results(results: pd.DataFrame, output_path: Path) -> None:
    accepted = results[results["accepted"].fillna(False)].copy()
    methods = sorted(accepted["method"].unique())
    gates = [gate for gate in ("scene_wide", "physics_50km_day") if gate in set(accepted["gate"])]
    fig, axes = plt.subplots(1, len(gates), figsize=(max(8, 5 * len(gates)), 6), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, gate in zip(axes, gates):
        groups = [
            accepted[(accepted["gate"] == gate) & (accepted["method"] == method)]["endpoint_error_m"].dropna().to_numpy() / 1000.0
            for method in methods
        ]
        axis.boxplot(groups, tick_labels=methods, showfliers=True)
        axis.axhline(2.0, color="tab:green", linestyle="--", linewidth=0.8, label="2 km")
        axis.axhline(5.0, color="tab:orange", linestyle="--", linewidth=0.8, label="5 km")
        axis.set_title(gate.replace("_", " "))
        axis.tick_params(axis="x", rotation=55, labelsize=8)
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Buoy endpoint error (km, log scale)")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
    return "\n".join([header, separator, *rows])


def write_report(
    output_path: Path,
    catalog_path: Path | None,
    buoy_path: Path,
    scene_matches_path: Path | None,
    coincidences: pd.DataFrame,
    pairs: pd.DataFrame,
    summary: pd.DataFrame,
    timings: pd.DataFrame,
    args,
) -> None:
    table = summary.copy()
    table["median_error_km"] = table["median_error_m"] / 1000.0
    table["p90_error_km"] = table["p90_error_m"] / 1000.0
    columns = [
        "method",
        "gate",
        "pairs",
        "median_error_km",
        "p90_error_km",
        "within_2km_fraction",
        "within_5km_fraction",
        "truth_rank_fraction_median",
    ]
    timing_columns = ["method", "seconds", "seconds_per_pair"]
    source_lines = [f"- Buoy tracks/coincidences: `{buoy_path}`"]
    if catalog_path is not None:
        source_lines.insert(0, f"- Catalog: `{catalog_path}`")
    if scene_matches_path is not None:
        source_lines.insert(0, f"- Scene matches: `{scene_matches_path}`")
    text = f"""# {args.domain.title()} SAR/buoy descriptor benchmark

Date: {pd.Timestamp.now(tz="UTC").date()}

## Frozen fixture

{chr(10).join(source_lines)}
- Exact-time positions: linear interpolation in EPSG:{args.analysis_epsg} metres using UTC seconds.
- Coincidences: {len(coincidences)} across {coincidences['buoy_id'].nunique()} buoys and {coincidences['image_filename'].nunique()} images.
- Consecutive same-buoy tracking cases: {len(pairs)}.
- Candidate grid: {args.grid_stride} px stride, {args.grid_border} px image border.
- Arms: scene-wide retrieval and the same retrieval with a {args.max_speed_m_per_day / 1000.0:.0f} km/day speed-scaled candidate radius.
- No MAGSAC, affine model, interpolation, correlation threshold, or post-match repair was applied.

## Results

{markdown_table(table, columns)}

## Runtime

{markdown_table(timings, timing_columns)}

The candidate grid sets a finite localization floor; consult `retrieval_results.csv`
for `candidate_quantization_error_m`, exact truth rank, time gap, radiometry changes,
and per-pair failures. NCC methods search at full pixel resolution and therefore do
not have the grid quantization field.

## Interpretation guardrails

- This is a small attribution fixture, not a production trajectory comparison.
- The longest buoy contributes most pairs; pair rows are not independent.
- The input band is already an existing preprocessed uint8 product; preprocessing
  comparisons must keep the coincidence rows and candidate grid frozen.
- Scene-wide retrieval intentionally measures catastrophic ambiguity. The physics arm
  changes only the candidate set, so its gain is attributable to the motion bound.
- Exact buoy locations are used only for source seeding and scoring; target locations
  do not influence candidate ranking.
- Natural-image descriptors remain zero-shot on SAR. A learned model should not be
  integrated until it beats the current ORB+PM path on held-out buoy trajectories and
  deformation-quality metrics.
"""
    output_path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("arctic", "antarctic"), default="arctic")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--buoys", type=Path)
    parser.add_argument(
        "--coincidences",
        type=Path,
        help="Normalized exact-time coincidence CSV; bypasses catalog/track linkage.",
    )
    parser.add_argument("--scene-matches", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--analysis-epsg", type=int)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-time-difference-minutes", type=float, default=60.0)
    parser.add_argument("--max-speed-m-per-day", type=float, default=50000.0)
    parser.add_argument("--grid-stride", type=int, default=16)
    parser.add_argument("--grid-border", type=int, default=128)
    parser.add_argument("--template-size", type=int, default=65)
    parser.add_argument("--skip-ncc", action="store_true")
    parser.add_argument(
        "--invalid-support-policy",
        choices=("error", "skip"),
        default="error",
        help="How to handle exact buoy points outside a scene or on mask values >=2.",
    )
    args = parser.parse_args()
    if args.template_size % 2 != 1:
        raise ValueError("template-size must be odd")
    if args.analysis_epsg is None:
        args.analysis_epsg = 3413 if args.domain == "arctic" else 3412
    if args.coincidences is None and args.buoys is None:
        parser.error("Provide --coincidences or --buoys.")
    if args.coincidences is not None and (
        args.buoys is not None or args.catalog is not None
    ):
        parser.error("Use either --coincidences or catalog/track inputs, not both.")
    if args.coincidences is None and args.domain == "arctic" and args.catalog is None:
        parser.error("--catalog is required for Arctic catalog/track linkage")
    if args.coincidences is None and args.domain == "antarctic" and (
        args.scene_matches is None or args.image_root is None
    ):
        parser.error("--scene-matches and --image-root are required for the Antarctic domain")

    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stage_started = time.perf_counter()
    if args.coincidences is not None:
        coincidences = pd.read_csv(args.coincidences)
        required = {
            "buoy_id",
            "image_id",
            "image_filename",
            "image_filepath",
            "image_time",
            "x",
            "y",
        }
        missing = required - set(coincidences.columns)
        if missing:
            parser.error(f"Coincidence CSV lacks required columns: {sorted(missing)}")
        coincidences["image_time"] = pd.to_datetime(
            coincidences["image_time"], utc=True
        )
    elif args.domain == "arctic":
        catalog = load_catalog(args.catalog)
        coincidences = build_coincidences(
            catalog,
            args.buoys,
            max_time_difference_minutes=args.max_time_difference_minutes,
        )
    else:
        coincidences = build_antarctic_coincidences(
            args.scene_matches,
            args.buoys,
            args.image_root,
        )
    exact_time_count = len(coincidences)
    coincidences = annotate_coincidences(
        coincidences,
        args.analysis_epsg,
        outside_scene_policy=args.invalid_support_policy,
    )
    setup_seconds = time.perf_counter() - stage_started
    invalid = coincidences[
        (coincidences["mask_value"] >= 2)
        | ~np.isfinite(coincidences[["col", "row"]]).all(axis=1)
    ]
    if len(invalid) and args.invalid_support_policy == "error":
        raise ValueError(f"{len(invalid)} coincident buoy points fall on invalid raster support")
    if len(invalid):
        coincidences = coincidences.drop(index=invalid.index).reset_index(drop=True)
    pairs = build_pairs(coincidences)

    target_paths = pairs["target_image_filepath"].drop_duplicates().tolist()
    grid_started = time.perf_counter()
    grids = {
        path: candidate_grid(
            path,
            stride=args.grid_stride,
            border=args.grid_border,
            analysis_epsg=args.analysis_epsg,
        )
        for path in target_paths
    }
    grid_seconds = time.perf_counter() - grid_started
    result_rows, timing_rows = run_descriptor_variants(
        pairs, grids, args.max_speed_m_per_day, args.analysis_epsg
    )
    if not args.skip_ncc:
        ncc_rows, ncc_timings = run_ncc_variants(
            pairs,
            args.max_speed_m_per_day,
            args.template_size,
            args.analysis_epsg,
        )
        result_rows.extend(ncc_rows)
        timing_rows.extend(ncc_timings)
    timing_rows.extend(
        [
            {"stage": "setup", "method": "coincidence_setup", "seconds": setup_seconds, "pairs": len(pairs), "seconds_per_pair": setup_seconds / max(len(pairs), 1)},
            {"stage": "candidate_grid", "method": "candidate_grid", "seconds": grid_seconds, "pairs": len(pairs), "seconds_per_pair": grid_seconds / max(len(pairs), 1)},
        ]
    )
    results = pd.DataFrame.from_records(result_rows)
    summary = summarize_results(results)
    timings = pd.DataFrame.from_records(timing_rows).sort_values(["stage", "method"])

    coincidences.to_csv(args.out_dir / "coincidences.csv", index=False)
    pairs.to_csv(args.out_dir / "pairs.csv", index=False)
    results.to_csv(args.out_dir / "retrieval_results.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    timings.to_csv(args.out_dir / "timings.csv", index=False)
    plot_results(results, args.out_dir / "endpoint_error_by_method.png")
    write_report(
        args.out_dir / "report.md",
        args.catalog,
        args.coincidences if args.coincidences is not None else args.buoys,
        args.scene_matches,
        coincidences,
        pairs,
        summary,
        timings,
        args,
    )
    run_manifest = {
        **vars(args),
        "catalog": str(args.catalog),
        "buoys": str(args.buoys),
        "out_dir": str(args.out_dir),
        "descriptor_variants": [variant.__dict__ for variant in DESCRIPTOR_VARIANTS],
        "elapsed_seconds": time.perf_counter() - started,
        "coincidences": len(coincidences),
        "exact_time_coincidences_before_spatial_filter": exact_time_count,
        "invalid_mask_records": len(invalid),
        "pairs": len(pairs),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, default=str))
    print(summary.to_string(index=False))
    print(json.dumps({"elapsed_seconds": run_manifest["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
