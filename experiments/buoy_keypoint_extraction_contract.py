#!/usr/bin/env python3
"""Compare exact-buoy and locally detected ORB extraction contracts.

The experiment keeps the standard VAE image fixed. Buoy truth is used to place
or label the first descriptor only. Subsequent graph candidates remain the
existing fixed grid and obey the existing 50 km/day physics gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from buoy_descriptor_benchmark import (  # noqa: E402
    descriptor_distances,
    pixels_to_map,
    read_scene,
)
from orb_multiframe_graph import (  # noqa: E402
    GraphSearchConfig,
    precompute_layers,
    search_layered_graph,
)


DEFAULT_PATCH_ROOT = ROOT / "results/buoy_patch_evolution/q2q98_clahe25"
DEFAULT_GRAPH_ROOT = ROOT / "results/orb_multiframe_graph/final_arctic_matrix"
DEFAULT_OUTPUT = ROOT / "results/buoy_keypoint_extraction_contract/q2q98_clahe25"


@dataclass(frozen=True)
class OrbProfile:
    key: str
    report_name: str
    nfeatures: int
    scale_factor: float
    nlevels: int
    edge_threshold: int
    patch_size: int
    fixed_keypoint_size: float
    forced_octave: int


@dataclass(frozen=True)
class RetrievalSpec:
    report_name: str
    source_descriptor: str
    candidate_descriptor: str
    norm: str
    interpretation: str


@dataclass(frozen=True)
class SeedSpec:
    key: str
    report_name: str
    descriptor_key: str
    position_prefix: str


CURRENT_PROFILE = OrbProfile(
    key="current",
    report_name="current research ORB",
    nfeatures=100,
    scale_factor=1.25,
    nlevels=5,
    edge_threshold=16,
    patch_size=64,
    fixed_keypoint_size=31.0,
    forced_octave=5,
)

LIMOSAT_DEFAULT_PROFILE = OrbProfile(
    key="limosat_default",
    report_name="LiMOSAT config default ORB",
    nfeatures=500,
    scale_factor=1.2,
    nlevels=8,
    edge_threshold=31,
    patch_size=31,
    fixed_keypoint_size=31.0,
    forced_octave=8,
)

PROFILES = (CURRENT_PROFILE, LIMOSAT_DEFAULT_PROFILE)

RETRIEVAL_SPECS = (
    RetrievalSpec(
        "current: buoy pixel to buoy pixels",
        "current_exact",
        "current_exact",
        "hamming",
        "symmetric exact-location control",
    ),
    RetrievalSpec(
        "current: nearest detected seed to buoy-pixel candidates",
        "current_nearest_detector_footprint",
        "current_exact",
        "hamming",
        "operational source-to-grid contract",
    ),
    RetrievalSpec(
        "current: nearest detected position with fixed footprint to buoy-pixel candidates",
        "current_nearest_fixed_footprint",
        "current_exact",
        "hamming",
        "isolates moving the descriptor centre",
    ),
    RetrievalSpec(
        "current: strongest local feature to buoy-pixel candidates",
        "current_strongest_detector_footprint",
        "current_exact",
        "hamming",
        "tests response rather than proximity",
    ),
    RetrievalSpec(
        "current: nearest detected feature at both ends",
        "current_nearest_detector_footprint",
        "current_nearest_detector_footprint",
        "hamming",
        "symmetric local-feature diagnostic",
    ),
    RetrievalSpec(
        "LiMOSAT default: buoy pixel to buoy pixels, Hamming",
        "limosat_default_exact",
        "limosat_default_exact",
        "hamming",
        "fixed-grid descriptor control",
    ),
    RetrievalSpec(
        "LiMOSAT default: buoy pixel to buoy pixels, Hamming2",
        "limosat_default_exact",
        "limosat_default_exact",
        "hamming2",
        "current matcher-distance control",
    ),
    RetrievalSpec(
        "LiMOSAT default: nearest detected seed to buoy-pixel candidates, Hamming",
        "limosat_default_nearest_detector_footprint",
        "limosat_default_exact",
        "hamming",
        "production seed/grid geometry with native WTA_K=2 distance",
    ),
    RetrievalSpec(
        "LiMOSAT default: nearest detected seed to buoy-pixel candidates, Hamming2",
        "limosat_default_nearest_detector_footprint",
        "limosat_default_exact",
        "hamming2",
        "current production seed/grid and matcher contract",
    ),
    RetrievalSpec(
        "LiMOSAT default: nearest position with fixed footprint to buoy-pixel candidates, Hamming2",
        "limosat_default_nearest_fixed_footprint",
        "limosat_default_exact",
        "hamming2",
        "isolates moving the descriptor centre",
    ),
    RetrievalSpec(
        "LiMOSAT default: strongest local feature to buoy-pixel candidates, Hamming2",
        "limosat_default_strongest_detector_footprint",
        "limosat_default_exact",
        "hamming2",
        "tests response rather than proximity",
    ),
)

SEED_SPECS = (
    SeedSpec(
        "exact_buoy",
        "buoy coordinate, fixed footprint",
        "current_exact",
        "exact",
    ),
    SeedSpec(
        "nearest_detected",
        "nearest detected feature, detector footprint",
        "current_nearest_detector_footprint",
        "current_nearest",
    ),
    SeedSpec(
        "nearest_fixed",
        "nearest detected position, fixed footprint",
        "current_nearest_fixed_footprint",
        "current_nearest",
    ),
    SeedSpec(
        "strongest_detected",
        "strongest detected feature within 300 m",
        "current_strongest_detector_footprint",
        "current_strongest",
    ),
)

MEMORY_POLICIES = (
    GraphSearchConfig("keep first view only", "anchor", 32, 8),
    GraphSearchConfig(
        "confidence-gated latest confirmed view",
        "confidence_rolling",
        32,
        8,
        update_min_margin=0.032,
        update_max_cost=0.35,
    ),
)


def build_orb(profile: OrbProfile) -> cv2.ORB:
    return cv2.ORB_create(
        nfeatures=profile.nfeatures,
        scaleFactor=profile.scale_factor,
        nlevels=profile.nlevels,
        edgeThreshold=profile.edge_threshold,
        firstLevel=0,
        WTA_K=2,
        patchSize=profile.patch_size,
        scoreType=cv2.ORB_HARRIS_SCORE,
    )


def descriptor_at(
    model,
    image: np.ndarray,
    col: float,
    row: float,
    size: float,
    octave: int,
    angle_deg: float,
    response: float = 0.0,
) -> np.ndarray | None:
    keypoint = cv2.KeyPoint(
        x=float(col),
        y=float(row),
        size=float(size),
        angle=float(angle_deg),
        response=float(response),
        octave=int(octave),
    )
    output, descriptors = model.compute(image, [keypoint])
    if descriptors is None or not output or descriptors.shape != (1, 32):
        return None
    return np.asarray(descriptors[0], dtype=np.uint8)


def local_detected_candidates(
    model,
    image: np.ndarray,
    col: float,
    row: float,
    response_threshold: float,
) -> tuple[list[cv2.KeyPoint], tuple[float, float], int]:
    """Reproduce the local detection window in KeypointDetector.keypoint_from_point."""
    window_size = max(32, int(model.getPatchSize()) + 16)
    half = window_size // 2
    rounded_col = int(round(float(col)))
    rounded_row = int(round(float(row)))
    r0 = max(0, rounded_row - half)
    r1 = min(image.shape[0], rounded_row + half + (window_size % 2))
    c0 = max(0, rounded_col - half)
    c1 = min(image.shape[1], rounded_col + half + (window_size % 2))
    if (r1 - r0) < window_size * 0.8 or (c1 - c0) < window_size * 0.8:
        return [], (math.nan, math.nan), window_size
    patch = image[r0:r1, c0:c1]
    detected = model.detect(patch, None) or []
    candidates = []
    for keypoint in detected:
        if keypoint.response < response_threshold:
            continue
        global_col = c0 + float(keypoint.pt[0])
        global_row = r0 + float(keypoint.pt[1])
        half_descriptor = int(float(keypoint.size) / 2.0)
        if not (
            half_descriptor <= global_col < image.shape[1] - half_descriptor
            and half_descriptor <= global_row < image.shape[0] - half_descriptor
        ):
            continue
        candidates.append(
            cv2.KeyPoint(
                x=global_col,
                y=global_row,
                size=float(keypoint.size),
                angle=float(keypoint.angle),
                response=float(keypoint.response),
                octave=int(keypoint.octave),
            )
        )
    return candidates, (c0 + patch.shape[1] / 2.0, r0 + patch.shape[0] / 2.0), window_size


def candidate_measurements(
    image_path: str,
    candidates: list[cv2.KeyPoint],
    buoy_xy: np.ndarray,
    analysis_epsg: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not candidates:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    pixel_xy = np.asarray([keypoint.pt for keypoint in candidates], dtype=float)
    map_xy = pixels_to_map(image_path, pixel_xy, analysis_epsg)
    offsets = np.linalg.norm(map_xy - np.asarray(buoy_xy, dtype=float), axis=1)
    return map_xy, offsets


def select_local_keypoints(
    candidates: list[cv2.KeyPoint],
    map_xy: np.ndarray,
    buoy_offsets_m: np.ndarray,
    detection_center_pixel: tuple[float, float],
    image_path: str,
    analysis_epsg: int,
    maximum_center_distance_m: float,
) -> dict[str, tuple[int, float] | None]:
    """Return production-nearest and strongest-within-radius selections."""
    if not candidates:
        return {"nearest": None, "strongest": None}
    pixel_xy = np.asarray([keypoint.pt for keypoint in candidates], dtype=float)
    center_pixel = np.asarray(detection_center_pixel, dtype=float).reshape(1, 2)
    center_map = pixels_to_map(image_path, center_pixel, analysis_epsg)[0]
    center_distances = np.linalg.norm(map_xy - center_map, axis=1)

    nearest_index = int(np.argmin(np.sum((pixel_xy - center_pixel) ** 2, axis=1)))
    nearest = None
    if center_distances[nearest_index] <= maximum_center_distance_m:
        nearest = (nearest_index, float(center_distances[nearest_index]))

    within = np.flatnonzero(buoy_offsets_m <= maximum_center_distance_m)
    strongest = None
    if len(within):
        strongest_index = int(
            within[
                np.argmax(
                    np.asarray([candidates[index].response for index in within], dtype=float)
                )
            ]
        )
        strongest = (strongest_index, float(center_distances[strongest_index]))
    return {"nearest": nearest, "strongest": strongest}


def descriptor_or_zero(value: np.ndarray | None) -> tuple[np.ndarray, bool]:
    available = value is not None and np.asarray(value).shape == (32,)
    return (
        np.asarray(value, dtype=np.uint8) if available else np.zeros(32, dtype=np.uint8),
        bool(available),
    )


def extract_profile_observation(
    observation,
    image: np.ndarray,
    profile: OrbProfile,
    model,
    analysis_epsg: int,
    response_threshold: float,
    maximum_center_distance_m: float,
) -> tuple[dict, dict[str, np.ndarray], dict[str, bool]]:
    exact = descriptor_at(
        model,
        image,
        observation.col,
        observation.row,
        profile.fixed_keypoint_size,
        profile.forced_octave,
        observation.image_angle_deg,
    )
    candidates, center_pixel, window_size = local_detected_candidates(
        model,
        image,
        observation.col,
        observation.row,
        response_threshold,
    )
    map_xy, offsets = candidate_measurements(
        observation.image_filepath,
        candidates,
        np.array([observation.x, observation.y], dtype=float),
        analysis_epsg,
    )
    selections = select_local_keypoints(
        candidates,
        map_xy,
        offsets,
        center_pixel,
        observation.image_filepath,
        analysis_epsg,
        maximum_center_distance_m,
    )

    descriptors: dict[str, np.ndarray] = {}
    availability: dict[str, bool] = {}
    exact_value, exact_available = descriptor_or_zero(exact)
    descriptors[f"{profile.key}_exact"] = exact_value
    availability[f"{profile.key}_exact"] = exact_available

    record = {
        f"{profile.key}_detection_window_px": window_size,
        f"{profile.key}_local_candidate_count": len(candidates),
        f"{profile.key}_exact_descriptor_available": exact_available,
    }
    for selection_name in ("nearest", "strongest"):
        selection = selections[selection_name]
        prefix = f"{profile.key}_{selection_name}"
        if selection is None:
            record.update(
                {
                    f"{prefix}_available": False,
                    f"{prefix}_col": math.nan,
                    f"{prefix}_row": math.nan,
                    f"{prefix}_x": math.nan,
                    f"{prefix}_y": math.nan,
                    f"{prefix}_buoy_offset_m": math.nan,
                    f"{prefix}_production_center_distance_m": math.nan,
                    f"{prefix}_response": math.nan,
                    f"{prefix}_detected_size_px": math.nan,
                    f"{prefix}_detected_octave": math.nan,
                }
            )
            for footprint in ("detector_footprint", "fixed_footprint"):
                key = f"{prefix}_{footprint}"
                descriptors[key] = np.zeros(32, dtype=np.uint8)
                availability[key] = False
                record[f"{key}_descriptor_available"] = False
            continue

        selected_index, center_distance = selection
        keypoint = candidates[selected_index]
        selected_map = map_xy[selected_index]
        record.update(
            {
                f"{prefix}_available": True,
                f"{prefix}_col": float(keypoint.pt[0]),
                f"{prefix}_row": float(keypoint.pt[1]),
                f"{prefix}_x": float(selected_map[0]),
                f"{prefix}_y": float(selected_map[1]),
                f"{prefix}_buoy_offset_m": float(offsets[selected_index]),
                f"{prefix}_production_center_distance_m": center_distance,
                f"{prefix}_response": float(keypoint.response),
                f"{prefix}_detected_size_px": float(keypoint.size),
                f"{prefix}_detected_octave": int(keypoint.octave),
            }
        )
        detector_descriptor = descriptor_at(
            model,
            image,
            keypoint.pt[0],
            keypoint.pt[1],
            keypoint.size,
            profile.forced_octave,
            observation.image_angle_deg,
            keypoint.response,
        )
        fixed_descriptor = descriptor_at(
            model,
            image,
            keypoint.pt[0],
            keypoint.pt[1],
            profile.fixed_keypoint_size,
            profile.forced_octave,
            observation.image_angle_deg,
            keypoint.response,
        )
        for footprint, descriptor in (
            ("detector_footprint", detector_descriptor),
            ("fixed_footprint", fixed_descriptor),
        ):
            key = f"{prefix}_{footprint}"
            value, available = descriptor_or_zero(descriptor)
            descriptors[key] = value
            availability[key] = available
            record[f"{key}_descriptor_available"] = available
    return record, descriptors, availability


def extract_sequence(
    sequence_dir: Path,
    analysis_epsg: int,
    response_threshold: float,
    maximum_center_distance_m: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    observations = pd.read_csv(sequence_dir / "observations.csv", dtype={"buoy_id": str})
    observation_records: list[dict | None] = [None] * len(observations)
    descriptor_lists: dict[str, list[np.ndarray | None]] = {}
    availability_lists: dict[str, list[bool | None]] = {}
    models = {profile.key: build_orb(profile) for profile in PROFILES}
    started = time.perf_counter()
    for image_path, image_rows in observations.groupby("image_filepath", sort=False):
        image, _ = read_scene(image_path)
        for observation in image_rows.itertuples(index=True):
            observation_index = int(observation.Index)
            record = {
                "sequence": observation.sequence,
                "role": observation.role,
                "observation_id": observation.observation_id,
                "buoy_id": str(observation.buoy_id),
                "image_id": int(observation.image_id),
                "image_time": observation.image_time,
                "image_filepath": observation.image_filepath,
                "exact_x": float(observation.x),
                "exact_y": float(observation.y),
                "exact_col": float(observation.col),
                "exact_row": float(observation.row),
            }
            for profile in PROFILES:
                profile_record, profile_descriptors, profile_availability = (
                    extract_profile_observation(
                        observation,
                        image,
                        profile,
                        models[profile.key],
                        analysis_epsg,
                        response_threshold,
                        maximum_center_distance_m,
                    )
                )
                record.update(profile_record)
                for key, descriptor in profile_descriptors.items():
                    descriptor_lists.setdefault(key, [None] * len(observations))[
                        observation_index
                    ] = descriptor
                for key, available in profile_availability.items():
                    availability_lists.setdefault(key, [None] * len(observations))[
                        observation_index
                    ] = available
            observation_records[observation_index] = record

    descriptor_arrays = {
        key: np.stack(values).astype(np.uint8) for key, values in descriptor_lists.items()
    }
    availability_arrays = {
        key: np.asarray(values, dtype=bool) for key, values in availability_lists.items()
    }
    audit = audit_current_exact_archive(
        sequence_dir,
        observations,
        descriptor_arrays["current_exact"],
        availability_arrays["current_exact"],
    )
    audit["elapsed_seconds"] = time.perf_counter() - started
    return (
        pd.DataFrame.from_records(observation_records),
        descriptor_arrays,
        availability_arrays,
        audit,
    )


def audit_current_exact_archive(
    sequence_dir: Path,
    observations: pd.DataFrame,
    extracted: np.ndarray,
    available: np.ndarray,
) -> dict:
    with np.load(sequence_dir / "descriptors.npz") as archive:
        archive_ids = archive["observation_id"].astype(str)
        expected_ids = observations.observation_id.astype(str).to_numpy()
        if not np.array_equal(archive_ids, expected_ids):
            raise ValueError(f"Observation ID order differs in {sequence_dir}")
        archive_available = archive["orb_available"].astype(bool)
        common = archive_available & available
        hamming = (
            np.unpackbits(np.bitwise_xor(archive["orb"][common], extracted[common]), axis=1)
            .sum(axis=1)
            .astype(int)
        )
    return {
        "observations": len(observations),
        "archive_available": int(archive_available.sum()),
        "reextracted_available": int(available.sum()),
        "availability_equal": bool(np.array_equal(archive_available, available)),
        "common_descriptors": int(common.sum()),
        "nonzero_hamming_reextractions": int(np.sum(hamming != 0)),
        "maximum_reextraction_hamming": int(hamming.max()) if len(hamming) else 0,
    }


def load_extracted_sequence(
    sequence_out: Path,
    source_sequence_dir: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    observations = pd.read_csv(
        sequence_out / "observation_extraction.csv", dtype={"buoy_id": str}
    )
    descriptors = {}
    availability = {}
    with np.load(sequence_out / "descriptors.npz") as archive:
        for key in archive.files:
            if key == "observation_id":
                continue
            value = archive[key]
            if value.ndim == 2:
                descriptors[key] = value.astype(np.uint8)
            elif value.ndim == 1 and key.endswith("_available"):
                availability[key[: -len("_available")]] = value.astype(bool)
    audit = audit_current_exact_archive(
        source_sequence_dir,
        observations,
        descriptors["current_exact"],
        availability["current_exact"],
    )
    audit["elapsed_seconds"] = 0.0
    audit["loaded_from_existing_extraction"] = True
    return observations, descriptors, availability, audit


def add_detection_center_offsets(
    observations: pd.DataFrame,
    source_sequence_dir: Path,
    analysis_epsg: int,
) -> pd.DataFrame:
    """Measure the production gate centre against the exact-time buoy position."""
    result = observations.copy()
    source = pd.read_csv(
        source_sequence_dir / "observations.csv",
        usecols=["observation_id", "image_height", "image_width"],
    )
    dimensions = source.set_index("observation_id")
    for profile in PROFILES:
        center_pixels = np.zeros((len(result), 2), dtype=float)
        window_size = max(32, profile.patch_size + 16)
        half = window_size // 2
        for index, row in enumerate(result.itertuples(index=False)):
            height = int(dimensions.loc[row.observation_id, "image_height"])
            width = int(dimensions.loc[row.observation_id, "image_width"])
            rounded_col = int(round(float(row.exact_col)))
            rounded_row = int(round(float(row.exact_row)))
            r0 = max(0, rounded_row - half)
            r1 = min(height, rounded_row + half + (window_size % 2))
            c0 = max(0, rounded_col - half)
            c1 = min(width, rounded_col + half + (window_size % 2))
            center_pixels[index] = [c0 + (c1 - c0) / 2.0, r0 + (r1 - r0) / 2.0]
        center_map = np.zeros_like(center_pixels)
        for image_path, indexes in result.groupby("image_filepath", sort=False).groups.items():
            row_indexes = np.asarray(list(indexes), dtype=int)
            center_map[row_indexes] = pixels_to_map(
                image_path,
                center_pixels[row_indexes],
                analysis_epsg,
            )
        exact_map = result[["exact_x", "exact_y"]].to_numpy(dtype=float)
        result[f"{profile.key}_detection_center_buoy_offset_m"] = np.linalg.norm(
            center_map - exact_map, axis=1
        )
    return result


def footprint_equivalence_summary(
    sequence: str,
    descriptors: dict[str, np.ndarray],
    availability: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for profile in PROFILES:
        for selection in ("nearest", "strongest"):
            detector_key = f"{profile.key}_{selection}_detector_footprint"
            fixed_key = f"{profile.key}_{selection}_fixed_footprint"
            common = availability[detector_key] & availability[fixed_key]
            hamming = (
                np.unpackbits(
                    np.bitwise_xor(
                        descriptors[detector_key][common], descriptors[fixed_key][common]
                    ),
                    axis=1,
                )
                .sum(axis=1)
                .astype(int)
            )
            rows.append(
                {
                    "sequence": sequence,
                    "orb_profile": profile.report_name,
                    "selection": selection,
                    "common_descriptors": int(common.sum()),
                    "descriptors_changed_by_size_field": int(np.sum(hamming != 0)),
                    "maximum_hamming_change": int(hamming.max()) if len(hamming) else 0,
                }
            )
    return pd.DataFrame.from_records(rows)


def direct_production_method_spot_check(observations: pd.DataFrame) -> dict:
    """Call KeypointDetector.keypoint_from_point on one real observation."""
    import geopandas as gpd
    from nansat import NSR
    from shapely.geometry import Point

    from limosat.image import Image
    from limosat.keypoint_detector import KeypointDetector

    row = observations.iloc[0]
    point = gpd.GeoDataFrame(
        [{"geometry": Point(row.exact_x, row.exact_y)}], crs="EPSG:3413"
    )
    image = Image(row.image_filepath, srs=NSR(3413))
    profiles = {}
    for profile in PROFILES:
        output = KeypointDetector(build_orb(profile)).keypoint_from_point(
            point,
            octave=profile.forced_octave,
            img=image,
            response_threshold=0.0,
        )
        expected_available = bool(row[f"{profile.key}_nearest_available"])
        if output:
            keypoint = output[0][0]
            expected = np.array(
                [
                    row[f"{profile.key}_nearest_col"],
                    row[f"{profile.key}_nearest_row"],
                ],
                dtype=float,
            )
            pixel_difference = float(np.linalg.norm(np.asarray(keypoint.pt) - expected))
        else:
            pixel_difference = math.nan
        profiles[profile.key] = {
            "production_method_returned_keypoint": bool(output),
            "experiment_helper_returned_keypoint": expected_available,
            "availability_matches": bool(bool(output) == expected_available),
            "selected_pixel_difference": pixel_difference,
        }
    return {
        "observation_id": row.observation_id,
        "profiles": profiles,
    }


def hamming_normalized(reference: np.ndarray, candidates: np.ndarray, norm: str) -> np.ndarray:
    distances = descriptor_distances(reference, candidates, norm)
    maximum = candidates.shape[1] * (8 if norm == "hamming" else 4)
    return distances / float(maximum)


def descriptor_retrieval(
    observations: pd.DataFrame,
    transitions: pd.DataFrame,
    descriptors: dict[str, np.ndarray],
    availability: dict[str, np.ndarray],
    max_speed_m_per_day: float,
    error_threshold_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexes = {
        observation_id: index
        for index, observation_id in enumerate(observations.observation_id.astype(str))
    }
    lookup = observations.set_index("observation_id")
    first_ids = (
        observations.sort_values("image_time")
        .groupby("buoy_id", sort=False)
        .first()
        .observation_id.to_dict()
    )
    pair_rows = []
    summary_rows = []
    for spec in RETRIEVAL_SPECS:
        for reference_memory in ("first view", "previous true view"):
            positive_distances = []
            negative_distance_groups = []
            eligible_rows = []
            reference_target_available = 0
            for transition in transitions.itertuples(index=False):
                reference_id = (
                    first_ids[str(transition.buoy_id)]
                    if reference_memory == "first view"
                    else transition.source_observation_id
                )
                reference_index = indexes[reference_id]
                target_index = indexes[transition.target_observation_id]
                if not (
                    availability[spec.source_descriptor][reference_index]
                    and availability[spec.candidate_descriptor][target_index]
                ):
                    continue
                reference_target_available += 1
                source = lookup.loc[transition.source_observation_id]
                target = lookup.loc[transition.target_observation_id]
                radius_m = max_speed_m_per_day * float(transition.dt_hours) / 24.0
                distractors = observations[
                    (observations.image_id == transition.target_image_id)
                    & (observations.buoy_id != str(transition.buoy_id))
                ].copy()
                distractors = distractors[
                    (
                        np.hypot(distractors.exact_x - source.exact_x, distractors.exact_y - source.exact_y)
                        <= radius_m
                    )
                    & (
                        np.hypot(distractors.exact_x - target.exact_x, distractors.exact_y - target.exact_y)
                        > error_threshold_m
                    )
                ]
                distractor_indexes = [
                    indexes[observation_id]
                    for observation_id in distractors.observation_id
                    if availability[spec.candidate_descriptor][indexes[observation_id]]
                ]
                if not distractor_indexes:
                    continue
                positive = float(
                    hamming_normalized(
                        descriptors[spec.source_descriptor][reference_index],
                        descriptors[spec.candidate_descriptor][target_index : target_index + 1],
                        spec.norm,
                    )[0]
                )
                negative = hamming_normalized(
                    descriptors[spec.source_descriptor][reference_index],
                    descriptors[spec.candidate_descriptor][distractor_indexes],
                    spec.norm,
                )
                rank = 1 + int(np.sum(negative < positive))
                positive_distances.append(positive)
                negative_distance_groups.append(negative)
                row = {
                    "sequence": transition.sequence,
                    "buoy_id": str(transition.buoy_id),
                    "target_observation_id": transition.target_observation_id,
                    "retrieval_contract": spec.report_name,
                    "interpretation": spec.interpretation,
                    "reference_memory": reference_memory,
                    "norm": spec.norm,
                    "same_buoy_distance": positive,
                    "same_buoy_rank": rank,
                    "distractor_count": len(negative),
                }
                pair_rows.append(row)
                eligible_rows.append(row)

            all_transitions = len(transitions)
            if positive_distances:
                positive_values = np.asarray(positive_distances, dtype=float)
                negative_values = np.concatenate(negative_distance_groups)
                labels = np.r_[np.ones(len(positive_values)), np.zeros(len(negative_values))]
                scores = -np.r_[positive_values, negative_values]
                ranks = np.asarray([row["same_buoy_rank"] for row in eligible_rows], dtype=int)
                summary_rows.append(
                    {
                        "sequence": str(observations.sequence.iloc[0]),
                        "retrieval_contract": spec.report_name,
                        "interpretation": spec.interpretation,
                        "reference_memory": reference_memory,
                        "norm": spec.norm,
                        "all_transitions": all_transitions,
                        "reference_target_available": reference_target_available,
                        "eligible_transitions": len(ranks),
                        "unique_buoys": len({row["buoy_id"] for row in eligible_rows}),
                        "distractors": len(negative_values),
                        "same_buoy_top1_fraction_eligible": float(np.mean(ranks == 1)),
                        "same_buoy_top1_fraction_all_transitions": float(np.sum(ranks == 1) / max(all_transitions, 1)),
                        "same_buoy_top3_fraction_eligible": float(np.mean(ranks <= 3)),
                        "same_vs_distractor_auc": float(roc_auc_score(labels, scores)),
                        "median_same_buoy_distance": float(np.median(positive_values)),
                        "median_distractor_distance": float(np.median(negative_values)),
                    }
                )
    return pd.DataFrame.from_records(pair_rows), pd.DataFrame.from_records(summary_rows)


def extraction_summary(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile in PROFILES:
        for selection in ("nearest", "strongest"):
            prefix = f"{profile.key}_{selection}"
            descriptor_column = f"{prefix}_detector_footprint_descriptor_available"
            selected = observations[f"{prefix}_available"].astype(bool)
            descriptor_available = observations[descriptor_column].fillna(False).astype(bool)
            offsets = observations.loc[selected, f"{prefix}_buoy_offset_m"].astype(float)
            center_offsets = observations[
                f"{profile.key}_detection_center_buoy_offset_m"
            ].astype(float)
            rows.append(
                {
                    "sequence": observations.sequence.iloc[0],
                    "orb_profile": profile.report_name,
                    "selection": (
                        "nearest detected feature"
                        if selection == "nearest"
                        else "strongest feature within 300 m"
                    ),
                    "observations": len(observations),
                    "selection_coverage": float(selected.mean()),
                    "descriptor_coverage": float(descriptor_available.mean()),
                    "median_local_candidates": float(
                        observations[f"{profile.key}_local_candidate_count"].median()
                    ),
                    "median_gate_center_buoy_offset_m": float(center_offsets.median()),
                    "p90_gate_center_buoy_offset_m": float(center_offsets.quantile(0.9)),
                    "gate_center_over_300m_count": int((center_offsets > 300.0).sum()),
                    "median_buoy_offset_m": float(offsets.median()) if len(offsets) else math.nan,
                    "p90_buoy_offset_m": float(offsets.quantile(0.9)) if len(offsets) else math.nan,
                    "maximum_buoy_offset_m": float(offsets.max()) if len(offsets) else math.nan,
                    "within_50m_fraction_selected": float((offsets <= 50.0).mean()) if len(offsets) else math.nan,
                    "within_100m_fraction_selected": float((offsets <= 100.0).mean()) if len(offsets) else math.nan,
                    "within_200m_fraction_selected": float((offsets <= 200.0).mean()) if len(offsets) else math.nan,
                    "over_300m_from_true_buoy_count": int((offsets > 300.0).sum()),
                }
            )
    return pd.DataFrame.from_records(rows)


def graph_arguments(manifest: dict) -> SimpleNamespace:
    keys = (
        "analysis_epsg",
        "max_speed_m_per_day",
        "grid_stride",
        "grid_border",
        "orb_nfeatures",
        "orb_scale_factor",
        "orb_nlevels",
        "orb_edge_threshold",
        "orb_patch_size",
        "keypoint_size",
        "octave",
        "angle_mode",
        "descriptor_norm",
    )
    return SimpleNamespace(**{key: manifest[key] for key in keys})


def graph_records_for_seed(
    coincidences: pd.DataFrame,
    layers,
    extraction_observations: pd.DataFrame,
    descriptors: dict[str, np.ndarray],
    availability: dict[str, np.ndarray],
    seed_spec: SeedSpec,
    policy: GraphSearchConfig,
    args,
) -> list[dict]:
    extraction_indexes = {
        observation_id: index
        for index, observation_id in enumerate(extraction_observations.observation_id.astype(str))
    }
    extraction_lookup = extraction_observations.set_index("observation_id")
    rows = []
    for buoy_id, path in coincidences.groupby("buoy_id", sort=True):
        path = path.sort_values("image_time").reset_index(drop=True)
        if len(path) < 2:
            continue
        first = path.iloc[0]
        observation_id = f"{first.sequence}:{str(buoy_id)}:{int(first.image_id)}"
        if observation_id not in extraction_indexes:
            raise KeyError(f"Missing extraction row {observation_id}")
        extraction_index = extraction_indexes[observation_id]
        extraction = extraction_lookup.loc[observation_id]
        if not availability[seed_spec.descriptor_key][extraction_index]:
            rows.append(
                {
                    "sequence": first.sequence,
                    "seed_method": seed_spec.report_name,
                    "memory_method": policy.name,
                    "buoy_id": str(buoy_id),
                    "status": "seed unavailable",
                    "path_observations": len(path),
                }
            )
            continue
        if seed_spec.position_prefix == "exact":
            seed_xy = np.array([extraction.exact_x, extraction.exact_y], dtype=float)
        else:
            seed_xy = np.array(
                [
                    extraction[f"{seed_spec.position_prefix}_x"],
                    extraction[f"{seed_spec.position_prefix}_y"],
                ],
                dtype=float,
            )
        seed_offset_m = float(
            np.linalg.norm(seed_xy - np.array([extraction.exact_x, extraction.exact_y]))
        )
        state = search_layered_graph(
            [layers[path_value] for path_value in path.image_filepath],
            seed_xy=seed_xy,
            seed_descriptor=descriptors[seed_spec.descriptor_key][extraction_index],
            max_speed_m_per_day=args.max_speed_m_per_day,
            config=policy,
            descriptor_norm=args.descriptor_norm,
        )
        if state is None:
            rows.append(
                {
                    "sequence": first.sequence,
                    "seed_method": seed_spec.report_name,
                    "memory_method": policy.name,
                    "buoy_id": str(buoy_id),
                    "status": "graph failed",
                    "path_observations": len(path),
                    "seed_offset_m": seed_offset_m,
                }
            )
            continue
        for observation_index, (predicted, truth) in enumerate(
            zip(state.path_xy, path.itertuples(index=False))
        ):
            skipped = predicted is None
            error = (
                math.nan
                if skipped
                else float(
                    np.linalg.norm(
                        predicted - np.array([truth.x, truth.y], dtype=float)
                    )
                )
            )
            rows.append(
                {
                    "sequence": first.sequence,
                    "seed_method": seed_spec.report_name,
                    "memory_method": policy.name,
                    "buoy_id": str(buoy_id),
                    "status": "skipped" if skipped else "ok",
                    "path_observations": len(path),
                    "observation_index": observation_index,
                    "image_id": int(truth.image_id),
                    "image_time": truth.image_time,
                    "endpoint_error_m": error,
                    "seed_offset_m": seed_offset_m,
                    "descriptor_updated": (
                        False
                        if observation_index == 0
                        else state.descriptor_updates[observation_index - 1]
                    ),
                }
            )
    return rows


def summarize_graph(records: pd.DataFrame, coincidences: pd.DataFrame) -> pd.DataFrame:
    eligible_paths = int((coincidences.groupby("buoy_id").size() >= 2).sum())
    eligible_transitions = int(
        coincidences.groupby("buoy_id").size().sub(1).clip(lower=0).sum()
    )
    rows = []
    for (sequence, seed_method, memory_method), group in records.groupby(
        ["sequence", "seed_method", "memory_method"], sort=False
    ):
        transitions = group[
            group.status.isin(["ok", "skipped"])
            & (group.observation_index.fillna(-1) > 0)
        ]
        tracked = transitions[
            (transitions.status == "ok") & np.isfinite(transitions.endpoint_error_m)
        ]
        errors = tracked.endpoint_error_m.to_numpy(dtype=float)
        within = int(np.sum(errors <= 2000.0))
        catastrophic = int(np.sum(errors > 50000.0))
        seed_rows = group[
            (group.observation_index.fillna(-1) == 0)
            | group.status.isin(["seed unavailable", "graph failed"])
        ]
        offsets = seed_rows.seed_offset_m.dropna().astype(float)
        rows.append(
            {
                "sequence": sequence,
                "seed_method": seed_method,
                "memory_method": memory_method,
                "eligible_paths": eligible_paths,
                "eligible_transitions": eligible_transitions,
                "seed_unavailable_paths": int(
                    group.loc[group.status == "seed unavailable", "buoy_id"].nunique()
                ),
                "graph_failed_paths": int(
                    group.loc[group.status == "graph failed", "buoy_id"].nunique()
                ),
                "tracked_transitions": len(tracked),
                "tracking_fraction_all": float(len(tracked) / max(eligible_transitions, 1)),
                "within_2km_fraction_all": float(within / max(eligible_transitions, 1)),
                "catastrophic_50km_fraction_all": float(
                    catastrophic / max(eligible_transitions, 1)
                ),
                "median_tracked_error_m": float(np.median(errors)) if len(errors) else math.nan,
                "median_seed_offset_m": float(offsets.median()) if len(offsets) else math.nan,
                "p90_seed_offset_m": float(offsets.quantile(0.9)) if len(offsets) else math.nan,
                "memory_updates": int(transitions.descriptor_updated.fillna(False).sum()),
            }
        )
    return pd.DataFrame.from_records(rows)


def run_graph_replay(
    graph_root: Path,
    sequence: str,
    extraction_observations: pd.DataFrame,
    descriptors: dict[str, np.ndarray],
    availability: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sequence_dir = graph_root / sequence
    manifest = json.loads((sequence_dir / "run_manifest.json").read_text())
    args = graph_arguments(manifest)
    coincidences = pd.read_csv(sequence_dir / "coincidences.csv", dtype={"buoy_id": str})
    coincidences["image_time"] = pd.to_datetime(coincidences.image_time, utc=True)
    layers, precompute_seconds = precompute_layers(coincidences, args)
    rows = []
    started = time.perf_counter()
    for seed_spec in SEED_SPECS:
        for policy in MEMORY_POLICIES:
            rows.extend(
                graph_records_for_seed(
                    coincidences,
                    layers,
                    extraction_observations,
                    descriptors,
                    availability,
                    seed_spec,
                    policy,
                    args,
                )
            )
    records = pd.DataFrame.from_records(rows)
    return (
        records,
        summarize_graph(records, coincidences),
        {
            "precompute_seconds": precompute_seconds,
            "search_seconds": time.perf_counter() - started,
            "grid_border_px": args.grid_border,
            "grid_stride_px": args.grid_stride,
            "max_speed_m_per_day": args.max_speed_m_per_day,
        },
    )


def markdown_table(data: pd.DataFrame, columns: list[str]) -> str:
    view = data[columns].copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()],
        ]
    )


def write_report(
    path: Path,
    extraction: pd.DataFrame,
    footprint: pd.DataFrame,
    retrieval: pd.DataFrame,
    graph: pd.DataFrame,
    audits: dict,
    production_spot_checks: dict,
    maximum_center_distance_m: float,
) -> None:
    extraction_columns = [
        "sequence",
        "orb_profile",
        "selection",
        "selection_coverage",
        "descriptor_coverage",
        "median_local_candidates",
        "median_gate_center_buoy_offset_m",
        "gate_center_over_300m_count",
        "median_buoy_offset_m",
        "p90_buoy_offset_m",
        "over_300m_from_true_buoy_count",
    ]
    retrieval_columns = [
        "sequence",
        "retrieval_contract",
        "eligible_transitions",
        "same_buoy_top1_fraction_eligible",
        "same_buoy_top1_fraction_all_transitions",
        "same_vs_distractor_auc",
        "median_same_buoy_distance",
    ]
    if retrieval.empty:
        retrieval_view = pd.DataFrame(columns=retrieval_columns)
    else:
        retrieval_view = retrieval[
            (retrieval.reference_memory == "previous true view")
            & retrieval.retrieval_contract.isin(
                [
                    "current: buoy pixel to buoy pixels",
                    "current: nearest detected seed to buoy-pixel candidates",
                    "current: nearest detected position with fixed footprint to buoy-pixel candidates",
                    "current: strongest local feature to buoy-pixel candidates",
                    "LiMOSAT default: nearest detected seed to buoy-pixel candidates, Hamming",
                    "LiMOSAT default: nearest detected seed to buoy-pixel candidates, Hamming2",
                ]
            )
        ]
    graph_columns = [
        "sequence",
        "seed_method",
        "memory_method",
        "seed_unavailable_paths",
        "graph_failed_paths",
        "tracking_fraction_all",
        "within_2km_fraction_all",
        "catastrophic_50km_fraction_all",
        "median_tracked_error_m",
        "median_seed_offset_m",
    ]
    audit_text = "\n".join(
        f"- {sequence}: availability equal={audit['availability_equal']}, "
        f"nonzero descriptor differences={audit['nonzero_hamming_reextractions']}, "
        f"maximum Hamming difference={audit['maximum_reextraction_hamming']}"
        for sequence, audit in audits.items()
    )
    spot_check_text = "\n".join(
        f"- {sequence}: current helper/production pixel difference="
        f"{check['profiles']['current']['selected_pixel_difference']:.3f} px; "
        f"LiMOSAT-default production keypoint returned="
        f"{check['profiles']['limosat_default']['production_method_returned_keypoint']}"
        for sequence, check in production_spot_checks.items()
    )
    footprint_columns = [
        "sequence",
        "orb_profile",
        "selection",
        "common_descriptors",
        "descriptors_changed_by_size_field",
        "maximum_hamming_change",
    ]
    path.write_text(
        "# Buoy keypoint extraction contract\n\n"
        f"Date: {pd.Timestamp.now(tz='UTC').date()}\n\n"
        "This is an experimental replay on the unchanged standard VAE image. "
        "No LiMOSAT production defaults were changed. Exact-buoy descriptors are "
        "computed at a supplied OpenCV keypoint; they do not call `detect`. The "
        "local-feature arms reproduce the production window and nearest-feature "
        f"selection with its {maximum_center_distance_m:.0f} m rounded-centre gate.\n\n"
        "## Extraction coverage and localization\n\n"
        + markdown_table(extraction, extraction_columns)
        + "\n\nThe LiMOSAT config-default detector has zero coverage because its "
        "47-pixel local window is smaller than two 31-pixel ORB edge exclusions; "
        "there is no valid detection interior. "
        + "\n\nThe selected-feature offset is measured from the exact-time buoy, not "
        "from the detection-window centre used by production. `gate_center` "
        "quantifies that centre-to-buoy geolocation discrepancy before feature "
        "selection. The strongest-feature arm "
        "is a controlled alternative and is not a LiMOSAT default.\n\n"
        "## Detected-size field isolation\n\n"
        + markdown_table(footprint, footprint_columns)
        + "\n\nThe detector reports several keypoint sizes, but LiMOSAT overwrites "
        "the octave before descriptor computation. Under that contract, changing "
        "only the supplied size field did not change an ORB descriptor.\n\n"
        "## Descriptor retrieval\n\n"
        + markdown_table(retrieval_view, retrieval_columns)
        + "\n\nThe operational comparison uses a locally detected source descriptor "
        "against descriptors computed at supplied buoy coordinates, mirroring "
        "the source-to-fixed-grid mismatch. `all_transitions` retains missing "
        "descriptors and cases without a valid distractor in the denominator.\n\n"
        "## Multi-frame tracking with only the first seed changed\n\n"
        + markdown_table(graph, graph_columns)
        + "\n\nThe candidate grid, 128-pixel safety border, Hamming distance, "
        "50 km/day gate, beam width, branching, and memory rules are unchanged. "
        "A detected seed starts at the detected feature position, as production "
        "does; its initial offset is therefore explicit.\n\n"
        "## Reproducibility audit\n\n"
        + audit_text
        + "\n\nDirect calls to production `keypoint_from_point`:\n\n"
        + spot_check_text
        + "\n"
    )


def parse_sequences(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-root", type=Path, default=DEFAULT_PATCH_ROOT)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sequences",
        default="2020_03,2020_02,2015_full15",
    )
    parser.add_argument("--analysis-epsg", type=int, default=3413)
    parser.add_argument("--response-threshold", type=float, default=0.0)
    parser.add_argument("--maximum-center-distance-m", type=float, default=300.0)
    parser.add_argument("--max-speed-m-per-day", type=float, default=50000.0)
    parser.add_argument("--error-threshold-m", type=float, default=2000.0)
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument(
        "--reuse-extracted",
        action="store_true",
        help="Reuse per-sequence observation and descriptor archives in --out-dir.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sequences = parse_sequences(args.sequences)
    all_observations = []
    all_extraction_summary = []
    all_footprint_summary = []
    all_retrieval_pairs = []
    all_retrieval_summary = []
    all_graph_records = []
    all_graph_summary = []
    descriptor_archives = {}
    availability_archives = {}
    audits = {}
    production_spot_checks = {}
    graph_timings = {}
    started = time.perf_counter()

    for sequence in sequences:
        sequence_dir = args.patch_root / sequence
        sequence_out = args.out_dir / sequence
        sequence_out.mkdir(parents=True, exist_ok=True)
        if (
            args.reuse_extracted
            and (sequence_out / "observation_extraction.csv").exists()
            and (sequence_out / "descriptors.npz").exists()
        ):
            observations, descriptors, availability, audit = load_extracted_sequence(
                sequence_out, sequence_dir
            )
        else:
            observations, descriptors, availability, audit = extract_sequence(
                sequence_dir,
                args.analysis_epsg,
                args.response_threshold,
                args.maximum_center_distance_m,
            )
        observations = add_detection_center_offsets(
            observations,
            sequence_dir,
            args.analysis_epsg,
        )
        transitions = pd.read_csv(sequence_dir / "transitions.csv", dtype={"buoy_id": str})
        transitions["sequence"] = sequence
        retrieval_pairs, retrieval_summary = descriptor_retrieval(
            observations,
            transitions,
            descriptors,
            availability,
            args.max_speed_m_per_day,
            args.error_threshold_m,
        )
        all_observations.append(observations)
        all_extraction_summary.append(extraction_summary(observations))
        all_footprint_summary.append(
            footprint_equivalence_summary(sequence, descriptors, availability)
        )
        all_retrieval_pairs.append(retrieval_pairs)
        all_retrieval_summary.append(retrieval_summary)
        descriptor_archives[sequence] = descriptors
        availability_archives[sequence] = availability
        audits[sequence] = audit
        production_spot_checks[sequence] = direct_production_method_spot_check(
            observations
        )

        observations.to_csv(sequence_out / "observation_extraction.csv", index=False)
        np.savez_compressed(
            sequence_out / "descriptors.npz",
            observation_id=observations.observation_id.astype(str).to_numpy(dtype="U"),
            **descriptors,
            **{f"{key}_available": value for key, value in availability.items()},
        )

        if not args.skip_graph:
            graph_records, graph_summary, graph_timing = run_graph_replay(
                args.graph_root,
                sequence,
                observations,
                descriptors,
                availability,
            )
            all_graph_records.append(graph_records)
            all_graph_summary.append(graph_summary)
            graph_timings[sequence] = graph_timing

    observations_all = pd.concat(all_observations, ignore_index=True)
    extraction_all = pd.concat(all_extraction_summary, ignore_index=True)
    footprint_all = pd.concat(all_footprint_summary, ignore_index=True)
    retrieval_pairs_all = pd.concat(all_retrieval_pairs, ignore_index=True)
    retrieval_summary_all = pd.concat(all_retrieval_summary, ignore_index=True)
    graph_records_all = (
        pd.concat(all_graph_records, ignore_index=True)
        if all_graph_records
        else pd.DataFrame()
    )
    graph_summary_all = (
        pd.concat(all_graph_summary, ignore_index=True)
        if all_graph_summary
        else pd.DataFrame()
    )

    observations_all.to_csv(args.out_dir / "observation_extraction_all.csv", index=False)
    extraction_all.to_csv(args.out_dir / "extraction_summary.csv", index=False)
    footprint_all.to_csv(args.out_dir / "footprint_equivalence_summary.csv", index=False)
    retrieval_pairs_all.to_csv(args.out_dir / "descriptor_pair_ranks.csv", index=False)
    retrieval_summary_all.to_csv(args.out_dir / "descriptor_summary.csv", index=False)
    if not graph_records_all.empty:
        graph_records_all.to_csv(args.out_dir / "graph_seed_replay.csv", index=False)
        graph_summary_all.to_csv(args.out_dir / "graph_seed_summary.csv", index=False)
        write_report(
            args.out_dir / "report.md",
            extraction_all,
            footprint_all,
            retrieval_summary_all,
            graph_summary_all,
            audits,
            production_spot_checks,
            args.maximum_center_distance_m,
        )

    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "patch_root": str(args.patch_root.resolve()),
        "graph_root": str(args.graph_root.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "sequences": list(sequences),
        "analysis_crs": f"EPSG:{args.analysis_epsg}",
        "image_input": "balanced_q2q98_clahe25 standard VAE uint8 band",
        "response_threshold": args.response_threshold,
        "maximum_production_center_distance_m": args.maximum_center_distance_m,
        "maximum_drift_m_per_day": args.max_speed_m_per_day,
        "error_threshold_m": args.error_threshold_m,
        "profiles": [profile.__dict__ for profile in PROFILES],
        "retrieval_specs": [spec.__dict__ for spec in RETRIEVAL_SPECS],
        "seed_specs": [spec.__dict__ for spec in SEED_SPECS],
        "memory_policies": [policy.__dict__ for policy in MEMORY_POLICIES],
        "archive_audits": audits,
        "direct_production_method_spot_checks": production_spot_checks,
        "graph_timings": graph_timings,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(extraction_all.to_string(index=False))
    print(retrieval_summary_all.to_string(index=False))
    if not graph_summary_all.empty:
        print(graph_summary_all.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
