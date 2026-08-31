#!/usr/bin/env python3
"""Run the historical complete-footprint ALIKED pair experiment.

Use ``run_learned_drift_pair.py`` for the selected minimal workflow. This
runner remains for buoy, ORB, matcher-call, and alternative-policy audits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import torch
from kornia.feature import ALIKED, laf_from_center_scale_ori
from pyproj import Transformer
from scipy.spatial import Delaunay, cKDTree
from shapely.affinity import translate
from shapely.ops import transform

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_aliked_orb_northup import (
    ANALYSIS_EPSG,
    image_object,
    north_up_patch,
    projected_coordinates,
    retain_best_match_per_source,
    valid_feature_subset,
)
from experiments.aliked_matchers import (
    MutualNearestDescriptorMatcher,
    build_aliked_matcher,
)
from experiments.replay_aliked_candidate_policies import estimate_policy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projected_footprint(image_path: str):
    geometry = shapely.from_geojson(
        image_object(image_path, ANALYSIS_EPSG).get_border_geojson()
    )
    projector = Transformer.from_crs(
        4326, ANALYSIS_EPSG, always_xy=True
    ).transform
    return transform(projector, geometry).buffer(0)


def axis_centres(lower: float, upper: float, core_extent_m: float) -> np.ndarray:
    count = max(1, int(math.ceil((upper - lower) / core_extent_m)))
    return lower + (np.arange(count, dtype=float) + 0.5) * core_extent_m


def anchored_axis_centres(
    lower: float, upper: float, core_extent_m: float, origin_m: float
) -> np.ndarray:
    first_core_edge = (
        math.floor((lower - origin_m) / core_extent_m) * core_extent_m + origin_m
    )
    count = max(1, int(math.ceil((upper - first_core_edge) / core_extent_m)))
    return first_core_edge + (np.arange(count, dtype=float) + 0.5) * core_extent_m


def tile_layout(
    domain,
    tile_pixels: int,
    margin_pixels: int,
    pixel_size_m: float,
    grid_origin_m: float | None = None,
):
    core_extent_m = (tile_pixels - 2 * margin_pixels) * pixel_size_m
    minx, miny, maxx, maxy = domain.bounds
    if grid_origin_m is None:
        centres_x = axis_centres(minx, maxx, core_extent_m)
        centres_y = axis_centres(miny, maxy, core_extent_m)
    else:
        centres_x = anchored_axis_centres(
            minx, maxx, core_extent_m, grid_origin_m
        )
        centres_y = anchored_axis_centres(
            miny, maxy, core_extent_m, grid_origin_m
        )
    records = []
    tile_id = 0
    for row, center_y in enumerate(centres_y):
        for column, center_x in enumerate(centres_x):
            core = shapely.box(
                center_x - core_extent_m / 2.0,
                center_y - core_extent_m / 2.0,
                center_x + core_extent_m / 2.0,
                center_y + core_extent_m / 2.0,
            )
            if not domain.intersects(core):
                continue
            records.append(
                {
                    "tile_id": tile_id,
                    "row": row,
                    "column": column,
                    "center_x": center_x,
                    "center_y": center_y,
                    "core": core,
                }
            )
            tile_id += 1
    return records


def extract_tiles(
    image_path: str,
    domain,
    layout: list[dict],
    model,
    device: torch.device,
    tile_pixels: int,
    margin_pixels: int,
    pixel_size_m: float,
    features_per_tile: int,
    support_radius_pixels: int,
    cache_dir: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    tiles = []
    audit = []
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for tile in layout:
        started = time.perf_counter()
        image_preparation_seconds = 0.0
        detection_description_seconds = 0.0
        cache_read_seconds = 0.0
        cache_write_seconds = 0.0
        cache_path = None
        if cache_dir is not None:
            identity = "|".join(
                [
                    str(Path(image_path).resolve()),
                    f"{tile['center_x']:.3f}",
                    f"{tile['center_y']:.3f}",
                    str(tile_pixels),
                    str(margin_pixels),
                    f"{pixel_size_m:.6f}",
                    str(features_per_tile),
                    str(support_radius_pixels),
                    "aliked-n16-threshold0.2",
                ]
            )
            cache_key = hashlib.sha256(identity.encode()).hexdigest()[:24]
            cache_path = cache_dir / f"{Path(image_path).stem}_{cache_key}.pt"
        cache_hit = bool(cache_path is not None and cache_path.exists())
        if cache_hit:
            cache_started = time.perf_counter()
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            keypoints = cached["keypoints"]
            descriptors = cached["descriptors"]
            scores = cached["scores"]
            valid_fraction = float(cached["valid_fraction"])
            raw_feature_count = int(cached["raw_feature_count"])
            invalid_excluded = int(cached["invalid_support_excluded"])
            cache_read_seconds = time.perf_counter() - cache_started
            image_preparation_seconds += cache_read_seconds
        else:
            preparation_started = time.perf_counter()
            patch, valid, _, _ = north_up_patch(
                image_path,
                tile["center_x"],
                tile["center_y"],
                tile_pixels,
                pixel_size_m,
            )
            tensor = (
                torch.from_numpy(patch.copy()).to(
                    device=device, dtype=torch.float32
                )[None, None]
                / 255.0
            )
            image_preparation_seconds += time.perf_counter() - preparation_started
            detection_started = time.perf_counter()
            with torch.inference_mode():
                raw = model(tensor)[0]
            keypoints, descriptors, scores, invalid_excluded = valid_feature_subset(
                raw,
                valid,
                features_per_tile,
                support_radius_pixels,
            )
            inside_core = (
                (keypoints[:, 0] >= margin_pixels)
                & (keypoints[:, 0] < tile_pixels - margin_pixels)
                & (keypoints[:, 1] >= margin_pixels)
                & (keypoints[:, 1] < tile_pixels - margin_pixels)
            )
            keypoints = keypoints[inside_core].detach().cpu()
            descriptors = descriptors[inside_core].detach().cpu()
            scores = scores[inside_core].detach().cpu()
            valid_fraction = float(valid.mean())
            raw_feature_count = int(len(raw.keypoints))
            detection_description_seconds += (
                time.perf_counter() - detection_started
            )
            if cache_path is not None:
                cache_write_started = time.perf_counter()
                torch.save(
                    {
                        "keypoints": keypoints,
                        "descriptors": descriptors,
                        "scores": scores,
                        "valid_fraction": valid_fraction,
                        "raw_feature_count": raw_feature_count,
                        "invalid_support_excluded": invalid_excluded,
                    },
                    cache_path,
                )
                cache_write_seconds = time.perf_counter() - cache_write_started
        coordinate_started = time.perf_counter()
        xy = projected_coordinates(
            keypoints,
            tile["center_x"],
            tile["center_y"],
            tile_pixels,
            pixel_size_m,
        )
        if len(xy):
            inside_domain = shapely.intersects_xy(domain, xy[:, 0], xy[:, 1])
            keep = np.flatnonzero(inside_domain)
            keypoints = keypoints[keep].detach().cpu()
            descriptors = descriptors[keep]
            scores = scores[keep]
            xy = xy[keep]
        else:
            keypoints = keypoints.detach().cpu()
        image_preparation_seconds += time.perf_counter() - coordinate_started
        tiles.append(
            {
                **tile,
                "keypoints": keypoints,
                "descriptors": descriptors,
                "scores": scores,
                "xy": xy,
            }
        )
        audit.append(
            {
                "tile_id": tile["tile_id"],
                "row": tile["row"],
                "column": tile["column"],
                "center_x": tile["center_x"],
                "center_y": tile["center_y"],
                "valid_fraction": valid_fraction,
                "raw_features": raw_feature_count,
                "invalid_support_excluded": invalid_excluded,
                "retained_core_features": int(len(keypoints)),
                "cache_hit": cache_hit,
                "image_preparation_seconds": image_preparation_seconds,
                "detection_description_seconds": detection_description_seconds,
                "cache_read_seconds": cache_read_seconds,
                "cache_write_seconds": cache_write_seconds,
                "seconds": time.perf_counter() - started,
            }
        )
    return tiles, audit


def spatially_thin_tiles_for_matching(
    tiles: list[dict],
    feature_cap_per_tile: int | None,
    cells_per_axis: int,
    tile_pixels: int,
    margin_pixels: int,
) -> list[dict]:
    """Round-robin detector responses across fixed cells before LightGlue.

    Extraction and its cache remain at the accuracy-first feature budget. This
    creates a matching-only view, so lower compute can be tested without
    redetecting features or allowing high-response clusters to take the whole
    budget.
    """
    if feature_cap_per_tile is None:
        return tiles
    if feature_cap_per_tile < 4:
        raise ValueError("matching feature cap must be at least four")
    if cells_per_axis < 1:
        raise ValueError("matching cells per axis must be positive")
    core_pixels = tile_pixels - 2 * margin_pixels
    if core_pixels <= 0:
        raise ValueError("tile margin leaves no matching core")

    thinned = []
    for tile in tiles:
        feature_count = len(tile["keypoints"])
        if feature_count <= feature_cap_per_tile:
            thinned.append(tile)
            continue
        keypoints = tile["keypoints"].detach().cpu().numpy()
        scores = tile["scores"].detach().cpu().numpy().reshape(-1)
        cell_xy = np.floor(
            (keypoints - margin_pixels) / core_pixels * cells_per_axis
        ).astype(int)
        cell_xy = np.clip(cell_xy, 0, cells_per_axis - 1)
        cell_ids = cell_xy[:, 1] * cells_per_axis + cell_xy[:, 0]
        buckets = []
        for cell_id in np.unique(cell_ids):
            indices = np.flatnonzero(cell_ids == cell_id)
            order = np.lexsort((indices, -scores[indices]))
            buckets.append(indices[order])

        selected: list[int] = []
        rank = 0
        while len(selected) < feature_cap_per_tile:
            candidates = [bucket[rank] for bucket in buckets if rank < len(bucket)]
            if not candidates:
                break
            candidates = sorted(candidates, key=lambda index: (-scores[index], index))
            remaining = feature_cap_per_tile - len(selected)
            selected.extend(candidates[:remaining])
            rank += 1
        keep = np.asarray(sorted(selected), dtype=int)
        thinned.append(
            {
                **{
                    key: value
                    for key, value in tile.items()
                    if key not in {"keypoints", "descriptors", "scores", "xy"}
                },
                "keypoints": tile["keypoints"][keep],
                "descriptors": tile["descriptors"][keep],
                "scores": tile["scores"][keep],
                "xy": tile["xy"][keep],
                "features_before_matching_thinning": feature_count,
            }
        )
    return thinned


def matching_source_subset(
    source_tile: dict,
    target_tile: dict,
    maximum_displacement_m: float,
    physics_subset_matching: bool,
    prior: np.ndarray | None,
    prior_uncertainty_m: float | None,
) -> np.ndarray:
    """Return source features that can physically reach one target core."""
    source_count = len(source_tile["keypoints"])
    if not physics_subset_matching:
        return np.arange(source_count)
    source_x = source_tile["xy"][:, 0]
    source_y = source_tile["xy"][:, 1]
    subset_radius = maximum_displacement_m
    if prior is not None:
        source_x = source_x + prior[0]
        source_y = source_y + prior[1]
        subset_radius = prior_uncertainty_m
    return np.flatnonzero(
        shapely.intersects_xy(
            target_tile["core"].buffer(subset_radius), source_x, source_y
        )
    )


def match_tiles(
    source_tiles: list[dict],
    target_tiles: list[dict],
    matcher,
    device: torch.device,
    tile_pixels: int,
    maximum_displacement_m: float,
    elapsed_days: float,
    maximum_speed_m_per_day: float,
    physics_subset_matching: bool = False,
    reuse_device_features: bool = False,
    matching_prior_displacement_m: tuple[float, float] | None = None,
    matching_prior_uncertainty_m: float | None = None,
    matcher_call_audit: list[dict] | None = None,
    matcher_call_matches: list[dict] | None = None,
    audit_mnn_candidates: bool = False,
    mnn_candidate_limit: int | None = None,
    lightglue_target_batch_size: int = 1,
) -> tuple[pd.DataFrame, list[dict]]:
    if (matching_prior_displacement_m is None) != (
        matching_prior_uncertainty_m is None
    ):
        raise ValueError("matching prior displacement and uncertainty must be paired")
    if matching_prior_uncertainty_m is not None and matching_prior_uncertainty_m <= 0:
        raise ValueError("matching prior uncertainty must be positive")
    if mnn_candidate_limit is not None and mnn_candidate_limit < 1:
        raise ValueError("MNN candidate limit must be positive")
    if lightglue_target_batch_size < 1:
        raise ValueError("LightGlue target batch size must be positive")
    prior = (
        np.asarray(matching_prior_displacement_m, dtype=float)
        if matching_prior_displacement_m is not None
        else None
    )
    matcher_name = getattr(matcher, "matcher_name", "lightglue")
    if mnn_candidate_limit is not None and not matcher_name.startswith("lightglue"):
        raise ValueError("MNN candidate ranking requires a LightGlue matcher")
    uses_laf = bool(getattr(matcher, "uses_laf", True))
    uses_direct_keypoints = bool(
        getattr(matcher, "uses_direct_keypoints", False)
    )
    if lightglue_target_batch_size > 1 and not callable(
        getattr(matcher, "forward_batch", None)
    ):
        raise ValueError(
            "LightGlue target batching requires the direct LightGlue adapter"
        )
    mnn_auditor = MutualNearestDescriptorMatcher().to(device).eval()
    if reuse_device_features:
        for tile in [*source_tiles, *target_tiles]:
            if len(tile["keypoints"]):
                tile["matcher_descriptors"] = tile["descriptors"].to(device)
                if uses_direct_keypoints:
                    tile["matcher_keypoints"] = tile["keypoints"].to(device)
                if uses_laf:
                    tile["matcher_laf"] = laf_from_center_scale_ori(
                        tile["keypoints"].to(device)[None]
                    )
    records = []
    audit = []
    global_source_offset = 0
    for source_tile in source_tiles:
        source_count = len(source_tile["keypoints"])
        reachable_target_domain = source_tile["core"].buffer(maximum_displacement_m)
        if prior is not None:
            reachable_target_domain = translate(
                source_tile["core"], xoff=prior[0], yoff=prior[1]
            ).buffer(matching_prior_uncertainty_m)
        candidate_targets = [
            target_tile
            for target_tile in target_tiles
            if reachable_target_domain.intersects(target_tile["core"])
            and len(target_tile["keypoints"]) >= 4
        ]
        candidate_target_count = len(candidate_targets)
        precomputed_mnn: dict[int, dict] = {}
        candidate_audit_seconds = 0.0
        if mnn_candidate_limit is not None:
            for target_tile in candidate_targets:
                source_subset = matching_source_subset(
                    source_tile,
                    target_tile,
                    maximum_displacement_m,
                    physics_subset_matching,
                    prior,
                    matching_prior_uncertainty_m,
                )
                if len(source_subset) < 4:
                    continue
                source_descriptors = source_tile["descriptors"][source_subset].to(
                    device
                )
                target_descriptors = target_tile["descriptors"].to(device)
                audit_started = time.perf_counter()
                with torch.inference_mode():
                    mnn_scores, mnn_indexes = mnn_auditor(
                        source_descriptors, target_descriptors
                    )
                mnn_seconds = time.perf_counter() - audit_started
                candidate_audit_seconds += mnn_seconds
                mnn_indexes_cpu = mnn_indexes.detach().cpu().numpy()
                mnn_score_values = mnn_scores.detach().cpu().numpy().reshape(-1)
                mnn_valid = np.zeros(len(mnn_indexes_cpu), dtype=bool)
                if len(mnn_indexes_cpu):
                    mnn_source_indices = source_subset[mnn_indexes_cpu[:, 0]]
                    mnn_displacement = (
                        target_tile["xy"][mnn_indexes_cpu[:, 1]]
                        - source_tile["xy"][mnn_source_indices]
                    )
                    mnn_valid = (
                        np.linalg.norm(mnn_displacement, axis=1) / elapsed_days
                        <= maximum_speed_m_per_day
                    )
                    if prior is not None:
                        mnn_valid &= (
                            np.linalg.norm(mnn_displacement - prior, axis=1)
                            <= matching_prior_uncertainty_m
                        )
                valid_count = int(mnn_valid.sum())
                precomputed_mnn[target_tile["tile_id"]] = {
                    "source_features": int(len(source_subset)),
                    "target_features": int(len(target_tile["keypoints"])),
                    "mnn_audit_seconds": mnn_seconds,
                    "mnn_raw_matches": int(len(mnn_indexes_cpu)),
                    "mnn_physics_valid_matches": valid_count,
                    "mnn_median_similarity": (
                        float(np.median(mnn_score_values[mnn_valid]))
                        if valid_count
                        else np.nan
                    ),
                }
            ranked_target_ids = [
                target_id
                for target_id, _ in sorted(
                    precomputed_mnn.items(),
                    key=lambda item: (
                        -item[1]["mnn_physics_valid_matches"],
                        -(
                            item[1]["mnn_median_similarity"]
                            if np.isfinite(item[1]["mnn_median_similarity"])
                            else -np.inf
                        ),
                        item[0],
                    ),
                )[:mnn_candidate_limit]
            ]
            ranked_target_ids = set(ranked_target_ids)
            if matcher_call_audit is not None:
                for target_tile in candidate_targets:
                    target_id = target_tile["tile_id"]
                    if target_id in ranked_target_ids or target_id not in precomputed_mnn:
                        continue
                    matcher_call_audit.append(
                        {
                            "source_tile_id": source_tile["tile_id"],
                            "target_tile_id": target_id,
                            **precomputed_mnn[target_id],
                            "matcher_executed": False,
                            "raw_matches": 0,
                            "physics_valid_matches": 0,
                            "matcher_seconds": 0.0,
                        }
                    )
            candidate_targets = [
                target_tile
                for target_tile in candidate_targets
                if target_tile["tile_id"] in ranked_target_ids
            ]
        all_source_indices = []
        all_target_xy = []
        all_target_tile_ids = []
        all_scores = []
        degenerate_matcher_calls = 0
        source_descriptor_comparisons = 0
        matcher_invocations = 0
        started = time.perf_counter()
        if source_count >= 4:
            prepared_calls = []
            for target_tile in candidate_targets:
                source_subset = matching_source_subset(
                    source_tile,
                    target_tile,
                    maximum_displacement_m,
                    physics_subset_matching,
                    prior,
                    matching_prior_uncertainty_m,
                )
                if len(source_subset) < 4:
                    continue
                source_descriptor_comparisons += len(source_subset)
                if reuse_device_features:
                    source_subset_device = torch.as_tensor(
                        source_subset, device=device, dtype=torch.long
                    )
                    source_descriptors = source_tile[
                        "matcher_descriptors"
                    ].index_select(0, source_subset_device)
                    target_descriptors = target_tile["matcher_descriptors"]
                    if uses_direct_keypoints:
                        source_keypoints = source_tile[
                            "matcher_keypoints"
                        ].index_select(0, source_subset_device)
                        target_keypoints = target_tile["matcher_keypoints"]
                    if uses_laf:
                        source_laf = source_tile["matcher_laf"].index_select(
                            1, source_subset_device
                        )
                        target_laf = target_tile["matcher_laf"]
                    else:
                        source_laf = target_laf = None
                else:
                    source_descriptors = source_tile["descriptors"][
                        source_subset
                    ].to(device)
                    target_descriptors = target_tile["descriptors"].to(device)
                    if uses_direct_keypoints:
                        source_keypoints = source_tile["keypoints"][
                            source_subset
                        ].to(device)
                        target_keypoints = target_tile["keypoints"].to(device)
                    if uses_laf:
                        source_laf = laf_from_center_scale_ori(
                            source_tile["keypoints"][source_subset].to(device)[None]
                        )
                        target_laf = laf_from_center_scale_ori(
                            target_tile["keypoints"].to(device)[None]
                        )
                    else:
                        source_laf = target_laf = None
                prepared_calls.append(
                    {
                        "target_tile": target_tile,
                        "source_subset": source_subset,
                        "source_descriptors": source_descriptors,
                        "target_descriptors": target_descriptors,
                        "source_keypoints": (
                            source_keypoints if uses_direct_keypoints else None
                        ),
                        "target_keypoints": (
                            target_keypoints if uses_direct_keypoints else None
                        ),
                        "source_laf": source_laf,
                        "target_laf": target_laf,
                    }
                )

            batch_id = 0

            def run_one(prepared):
                matcher_kwargs = {
                    "hw1": (tile_pixels, tile_pixels),
                    "hw2": (tile_pixels, tile_pixels),
                }
                if uses_direct_keypoints:
                    matcher_kwargs.update(
                        {
                            "source_keypoints": prepared["source_keypoints"],
                            "target_keypoints": prepared["target_keypoints"],
                        }
                    )
                scores, indexes = matcher(
                    prepared["source_descriptors"],
                    prepared["target_descriptors"],
                    prepared["source_laf"],
                    prepared["target_laf"],
                    **matcher_kwargs,
                )
                return (
                    scores,
                    indexes,
                    dict(getattr(matcher, "last_diagnostics", {})),
                )

            for batch_start in range(
                0, len(prepared_calls), lightglue_target_batch_size
            ):
                batch = prepared_calls[
                    batch_start : batch_start + lightglue_target_batch_size
                ]
                call_started = time.perf_counter()
                fallback_used = False
                try:
                    with torch.inference_mode():
                        if len(batch) > 1:
                            batch_results = matcher.forward_batch(
                                [row["source_descriptors"] for row in batch],
                                [row["target_descriptors"] for row in batch],
                                source_keypoints=[
                                    row["source_keypoints"] for row in batch
                                ],
                                target_keypoints=[
                                    row["target_keypoints"] for row in batch
                                ],
                                hw1=(tile_pixels, tile_pixels),
                                hw2=(tile_pixels, tile_pixels),
                            )
                        else:
                            batch_results = [run_one(batch[0])]
                except IndexError as error:
                    if "non-zero size" not in str(error):
                        raise
                    fallback_used = True
                    matcher_invocations += 1
                    if len(batch) == 1:
                        degenerate_matcher_calls += 1
                        batch_id += 1
                        continue
                    fallback_batch = []
                    fallback_results = []
                    for prepared in batch:
                        matcher_invocations += 1
                        try:
                            with torch.inference_mode():
                                result = run_one(prepared)
                        except IndexError as fallback_error:
                            if "non-zero size" not in str(fallback_error):
                                raise
                            degenerate_matcher_calls += 1
                            continue
                        fallback_batch.append(prepared)
                        fallback_results.append(result)
                    batch = fallback_batch
                    batch_results = fallback_results
                    if not batch:
                        batch_id += 1
                        continue
                batch_seconds = time.perf_counter() - call_started
                if not fallback_used:
                    matcher_invocations += 1
                allocated_matcher_seconds = batch_seconds / len(batch)
                for prepared, result in zip(batch, batch_results, strict=True):
                    target_tile = prepared["target_tile"]
                    source_subset = prepared["source_subset"]
                    source_descriptors = prepared["source_descriptors"]
                    target_descriptors = prepared["target_descriptors"]
                    scores, indexes, diagnostics = result
                    matcher_seconds = allocated_matcher_seconds
                    score_values = scores.detach().cpu().numpy().reshape(-1)
                    indexes = indexes.detach().cpu().numpy()
                    call_source_indices = source_subset[indexes[:, 0]]
                    call_source_xy = source_tile["xy"][call_source_indices]
                    call_target_xy = target_tile["xy"][indexes[:, 1]]
                    call_displacement = call_target_xy - call_source_xy
                    call_speed = (
                        np.linalg.norm(call_displacement, axis=1) / elapsed_days
                    )
                    call_valid = call_speed <= maximum_speed_m_per_day
                    if prior is not None:
                        call_valid &= (
                            np.linalg.norm(call_displacement - prior, axis=1)
                            <= matching_prior_uncertainty_m
                        )
                    mnn_seconds = 0.0
                    mnn_raw_matches = 0
                    mnn_physics_valid_matches = 0
                    mnn_median_similarity = np.nan
                    if target_tile["tile_id"] in precomputed_mnn:
                        mnn_values = precomputed_mnn[target_tile["tile_id"]]
                        mnn_seconds = mnn_values["mnn_audit_seconds"]
                        mnn_raw_matches = mnn_values["mnn_raw_matches"]
                        mnn_physics_valid_matches = mnn_values[
                            "mnn_physics_valid_matches"
                        ]
                        mnn_median_similarity = mnn_values[
                            "mnn_median_similarity"
                        ]
                    elif audit_mnn_candidates:
                        audit_started = time.perf_counter()
                        with torch.inference_mode():
                            mnn_scores, mnn_indexes = mnn_auditor(
                                source_descriptors, target_descriptors
                            )
                        mnn_seconds = time.perf_counter() - audit_started
                        candidate_audit_seconds += mnn_seconds
                        mnn_indexes_cpu = mnn_indexes.detach().cpu().numpy()
                        mnn_score_values = (
                            mnn_scores.detach().cpu().numpy().reshape(-1)
                        )
                        mnn_raw_matches = int(len(mnn_indexes_cpu))
                        if mnn_raw_matches:
                            mnn_source_indices = source_subset[
                                mnn_indexes_cpu[:, 0]
                            ]
                            mnn_displacement = (
                                target_tile["xy"][mnn_indexes_cpu[:, 1]]
                                - source_tile["xy"][mnn_source_indices]
                            )
                            mnn_valid = (
                                np.linalg.norm(mnn_displacement, axis=1)
                                / elapsed_days
                                <= maximum_speed_m_per_day
                            )
                            if prior is not None:
                                mnn_valid &= (
                                    np.linalg.norm(
                                        mnn_displacement - prior, axis=1
                                    )
                                    <= matching_prior_uncertainty_m
                                )
                            mnn_physics_valid_matches = int(mnn_valid.sum())
                            if mnn_physics_valid_matches:
                                mnn_median_similarity = float(
                                    np.median(mnn_score_values[mnn_valid])
                                )
                    if matcher_call_audit is not None:
                        matcher_call_audit.append(
                            {
                                "source_tile_id": source_tile["tile_id"],
                                "target_tile_id": target_tile["tile_id"],
                                "source_features": int(len(source_subset)),
                                "target_features": int(
                                    len(target_tile["keypoints"])
                                ),
                                "raw_matches": int(len(indexes)),
                                "physics_valid_matches": int(call_valid.sum()),
                                "matcher_seconds": matcher_seconds,
                                "matcher_batch_id": batch_id,
                                "matcher_batch_size": len(batch),
                                "matcher_batch_seconds": batch_seconds,
                                "mnn_audit_seconds": mnn_seconds,
                                "mnn_raw_matches": mnn_raw_matches,
                                "mnn_physics_valid_matches": (
                                    mnn_physics_valid_matches
                                ),
                                "mnn_median_similarity": mnn_median_similarity,
                                "matcher_executed": True,
                                **diagnostics,
                            }
                        )
                    if matcher_call_matches is not None:
                        for call_index in range(len(indexes)):
                            matcher_call_matches.append(
                                {
                                    "source_feature_id": int(
                                        global_source_offset
                                        + call_source_indices[call_index]
                                    ),
                                    "source_tile_id": source_tile["tile_id"],
                                    "target_tile_id": target_tile["tile_id"],
                                    "source_x": float(
                                        call_source_xy[call_index, 0]
                                    ),
                                    "source_y": float(
                                        call_source_xy[call_index, 1]
                                    ),
                                    "target_x": float(
                                        call_target_xy[call_index, 0]
                                    ),
                                    "target_y": float(
                                        call_target_xy[call_index, 1]
                                    ),
                                    "dx_m": float(
                                        call_displacement[call_index, 0]
                                    ),
                                    "dy_m": float(
                                        call_displacement[call_index, 1]
                                    ),
                                    "speed_m_per_day": float(
                                        call_speed[call_index]
                                    ),
                                    "matcher_score": float(
                                        score_values[call_index]
                                    ),
                                    "physics_valid": bool(
                                        call_valid[call_index]
                                    ),
                                }
                            )
                    if not len(indexes):
                        continue
                    all_source_indices.append(call_source_indices)
                    all_target_xy.append(call_target_xy)
                    all_target_tile_ids.append(
                        np.full(len(indexes), target_tile["tile_id"], dtype=int)
                    )
                    all_scores.append(score_values)
                batch_id += 1
        raw_matches = sum(len(values) for values in all_source_indices)
        if raw_matches:
            source_indices = np.concatenate(all_source_indices)
            target_xy = np.concatenate(all_target_xy)
            target_tile_ids = np.concatenate(all_target_tile_ids)
            scores = np.concatenate(all_scores)
            keep = retain_best_match_per_source(source_indices, scores)
            source_indices = source_indices[keep]
            target_xy = target_xy[keep]
            target_tile_ids = target_tile_ids[keep]
            scores = scores[keep]
            source_xy = source_tile["xy"][source_indices]
            displacement = target_xy - source_xy
            speed = np.linalg.norm(displacement, axis=1) / elapsed_days
            valid = speed <= maximum_speed_m_per_day
            if prior is not None:
                valid &= (
                    np.linalg.norm(displacement - prior, axis=1)
                    <= matching_prior_uncertainty_m
                )
            for index in np.flatnonzero(valid):
                record = {
                        "source_feature_id": int(
                            global_source_offset + source_indices[index]
                        ),
                        "source_tile_id": source_tile["tile_id"],
                        "target_tile_id": int(target_tile_ids[index]),
                        "source_x": float(source_xy[index, 0]),
                        "source_y": float(source_xy[index, 1]),
                        "target_x": float(target_xy[index, 0]),
                        "target_y": float(target_xy[index, 1]),
                        "dx_m": float(displacement[index, 0]),
                        "dy_m": float(displacement[index, 1]),
                        "speed_m_per_day": float(speed[index]),
                        "matcher_name": matcher_name,
                        "matcher_score": float(scores[index]),
                        "physics_valid": True,
                    }
                if matcher_name.startswith("lightglue"):
                    record["lightglue_score"] = float(scores[index])
                else:
                    record["descriptor_cosine_similarity"] = float(scores[index])
                records.append(record)
        audit.append(
            {
                "source_tile_id": source_tile["tile_id"],
                "source_features": source_count,
                "candidate_target_tiles": candidate_target_count,
                "executed_target_tiles": len(candidate_targets),
                "source_descriptor_comparisons": source_descriptor_comparisons,
                "degenerate_matcher_calls": degenerate_matcher_calls,
                "matcher_invocations": matcher_invocations,
                "raw_matches": raw_matches,
                "unique_source_matches": int(len(source_indices))
                if raw_matches
                else 0,
                "physics_valid_matches": int(np.sum(valid)) if raw_matches else 0,
                "matching_prior_used": prior is not None,
                "matcher_name": matcher_name,
                "candidate_audit_seconds": candidate_audit_seconds,
                "seconds": time.perf_counter() - started - candidate_audit_seconds,
            }
        )
        global_source_offset += source_count
    return pd.DataFrame.from_records(records), audit


def consensus_at_queries(
    matches: pd.DataFrame,
    queries: pd.DataFrame,
    tight_radius_m: float,
    consensus_radius_m: float,
) -> pd.DataFrame:
    tree = (
        cKDTree(matches[["source_x", "source_y"]].to_numpy(dtype=float))
        if len(matches)
        else None
    )
    records = []
    for query in queries.itertuples(index=False):
        local = matches.iloc[:0].copy()
        if tree is not None:
            indices = tree.query_ball_point(
                [query.source_x, query.source_y], tight_radius_m
            )
            local = matches.iloc[indices].copy()
        local["source_distance_m"] = np.hypot(
            local["source_x"] - query.source_x,
            local["source_y"] - query.source_y,
        )
        proposal = estimate_policy(
            local,
            "consensus_within_2km",
            tight_radius_m=tight_radius_m,
            consensus_radius_m=consensus_radius_m,
        )
        records.append({**query._asdict(), **proposal})
    return pd.DataFrame.from_records(records)


def adaptive_consensus_at_queries(
    matches: pd.DataFrame,
    queries: pd.DataFrame,
    support_radii_m: list[float],
    minimum_selected_vectors: int,
    consensus_radius_m: float,
) -> pd.DataFrame:
    """Use the smallest spatial support that supplies enough coherent vectors."""
    if not support_radii_m or any(radius <= 0 for radius in support_radii_m):
        raise ValueError("support radii must be positive")
    if support_radii_m != sorted(support_radii_m):
        raise ValueError("support radii must be sorted")
    if minimum_selected_vectors < 1:
        raise ValueError("minimum_selected_vectors must be positive")
    tree = (
        cKDTree(matches[["source_x", "source_y"]].to_numpy(dtype=float))
        if len(matches)
        else None
    )
    records = []
    for query in queries.itertuples(index=False):
        chosen = {"available": False, "selected_vectors": 0}
        chosen_radius = np.nan
        requirement_met = False
        for radius_m in support_radii_m:
            local = matches.iloc[:0].copy()
            if tree is not None:
                indices = tree.query_ball_point(
                    [query.source_x, query.source_y], radius_m
                )
                local = matches.iloc[indices].copy()
            local["source_distance_m"] = np.hypot(
                local["source_x"] - query.source_x,
                local["source_y"] - query.source_y,
            )
            proposal = estimate_policy(
                local,
                "consensus_within_2km",
                tight_radius_m=radius_m,
                consensus_radius_m=consensus_radius_m,
            )
            if proposal["available"]:
                chosen = proposal
                chosen_radius = radius_m
            if proposal.get("selected_vectors", 0) >= minimum_selected_vectors:
                requirement_met = True
                break
        if not requirement_met:
            chosen = {
                "available": False,
                "selected_vectors": int(chosen.get("selected_vectors", 0)),
            }
            chosen_radius = np.nan
        records.append(
            {
                **query._asdict(),
                **chosen,
                "support_radius_m": chosen_radius,
            }
        )
    return pd.DataFrame.from_records(records)


def nearest_consensus_at_queries(
    matches: pd.DataFrame,
    queries: pd.DataFrame,
    maximum_radius_m: float,
    candidate_count: int,
    minimum_selected_vectors: int,
    consensus_radius_m: float,
) -> pd.DataFrame:
    """Estimate from a bounded number of nearest source-feature matches."""
    if candidate_count < minimum_selected_vectors:
        raise ValueError("candidate_count must be at least minimum_selected_vectors")
    if maximum_radius_m <= 0:
        raise ValueError("maximum_radius_m must be positive")
    tree = (
        cKDTree(matches[["source_x", "source_y"]].to_numpy(dtype=float))
        if len(matches)
        else None
    )
    records = []
    for query in queries.itertuples(index=False):
        local = matches.iloc[:0].copy()
        if tree is not None:
            count = min(candidate_count, len(matches))
            distances, indices = tree.query(
                [query.source_x, query.source_y],
                k=count,
                distance_upper_bound=maximum_radius_m,
            )
            distances = np.atleast_1d(distances)
            indices = np.atleast_1d(indices)
            finite = np.isfinite(distances) & (indices < len(matches))
            local = matches.iloc[indices[finite]].copy()
            local["source_distance_m"] = distances[finite]
        if "source_distance_m" not in local:
            local["source_distance_m"] = pd.Series(dtype=float)
        proposal = estimate_policy(
            local,
            "consensus_within_2km",
            tight_radius_m=maximum_radius_m,
            consensus_radius_m=consensus_radius_m,
        )
        if proposal.get("selected_vectors", 0) < minimum_selected_vectors:
            proposal = {
                "available": False,
                "selected_vectors": int(proposal.get("selected_vectors", 0)),
            }
        records.append(
            {
                **query._asdict(),
                **proposal,
                "candidate_count": int(len(local)),
                "support_radius_m": float(local["source_distance_m"].max())
                if len(local)
                else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def regular_queries(domain, spacing_m: float) -> pd.DataFrame:
    minx, miny, maxx, maxy = domain.bounds
    xs = np.arange(math.ceil(minx / spacing_m) * spacing_m, maxx, spacing_m)
    ys = np.arange(math.ceil(miny / spacing_m) * spacing_m, maxy, spacing_m)
    x_grid, y_grid = np.meshgrid(xs, ys)
    inside = shapely.intersects_xy(domain, x_grid.ravel(), y_grid.ravel())
    return pd.DataFrame(
        {
            "grid_row": np.repeat(np.arange(len(ys)), len(xs))[inside],
            "grid_column": np.tile(np.arange(len(xs)), len(ys))[inside],
            "source_x": x_grid.ravel()[inside],
            "source_y": y_grid.ravel()[inside],
        }
    )


def attach_orb_field(
    field: pd.DataFrame,
    database_path: Path,
    table: str,
    source_run_image_id: int,
    target_run_image_id: int,
    maximum_distance_m: float = 10000.0,
) -> tuple[pd.DataFrame, int]:
    query = f'''SELECT image_id, trajectory_id, geometry FROM "{table}"
                WHERE image_id IN (?, ?)'''
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        points = pd.read_sql_query(
            query, connection, params=(source_run_image_id, target_run_image_id)
        )
    geometry = shapely.from_wkt(points["geometry"].to_numpy())
    points["x"] = shapely.get_x(geometry)
    points["y"] = shapely.get_y(geometry)
    source = points.loc[
        points["image_id"].eq(source_run_image_id), ["trajectory_id", "x", "y"]
    ]
    target = points.loc[
        points["image_id"].eq(target_run_image_id), ["trajectory_id", "x", "y"]
    ]
    paired = source.merge(
        target, on="trajectory_id", suffixes=("_source", "_target"),
        validate="one_to_one",
    )
    source_xy = paired[["x_source", "y_source"]].to_numpy(dtype=float)
    vectors = paired[["x_target", "y_target"]].to_numpy(dtype=float) - source_xy
    tree = cKDTree(source_xy) if len(source_xy) else None
    orb_records = []
    for row in field.itertuples(index=False):
        available = False
        record = {"orb_available_10km": False}
        if tree is not None:
            count = min(4, len(source_xy))
            distances, indices = tree.query([row.source_x, row.source_y], k=count)
            distances = np.atleast_1d(distances)
            indices = np.atleast_1d(indices)
            keep = distances <= maximum_distance_m
            if keep.any():
                weights = 1.0 / np.maximum(distances[keep], 1.0)
                estimate = np.average(vectors[indices[keep]], axis=0, weights=weights)
                available = True
                record.update(
                    {
                        "orb_available_10km": True,
                        "orb_neighbours": int(keep.sum()),
                        "orb_maximum_distance_m": float(distances[keep].max()),
                        "orb_dx_m": float(estimate[0]),
                        "orb_dy_m": float(estimate[1]),
                    }
                )
        if available and bool(row.available):
            record["aliked_orb_vector_difference_m"] = float(
                np.hypot(
                    row.proposal_dx_m - record["orb_dx_m"],
                    row.proposal_dy_m - record["orb_dy_m"],
                )
            )
        orb_records.append(record)
    return pd.concat([field.reset_index(drop=True), pd.DataFrame(orb_records)], axis=1), int(
        len(paired)
    )


def topology_summary(field: pd.DataFrame, spacing_m: float) -> dict:
    available = field.loc[field["available"].fillna(False)].copy()
    if len(available) < 3:
        return {"triangles": 0}
    source = available[["source_x", "source_y"]].to_numpy(dtype=float)
    target = source + available[["proposal_dx_m", "proposal_dy_m"]].to_numpy(
        dtype=float
    )
    triangles = Delaunay(source).simplices
    source_tri = source[triangles]
    target_tri = target[triangles]
    edge_lengths = np.max(
        np.stack(
            [
                np.linalg.norm(source_tri[:, 0] - source_tri[:, 1], axis=1),
                np.linalg.norm(source_tri[:, 1] - source_tri[:, 2], axis=1),
                np.linalg.norm(source_tri[:, 2] - source_tri[:, 0], axis=1),
            ]
        ),
        axis=0,
    )
    keep = edge_lengths <= spacing_m * 1.6
    source_tri = source_tri[keep]
    target_tri = target_tri[keep]

    def signed_twice_area(values):
        first = values[:, 1] - values[:, 0]
        second = values[:, 2] - values[:, 0]
        return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]

    source_area = signed_twice_area(source_tri)
    target_area = signed_twice_area(target_tri)
    area_ratio = np.abs(target_area / source_area)
    return {
        "triangles": int(len(source_tri)),
        "flipped_triangles": int(np.sum(source_area * target_area < 0)),
        "flipped_fraction": float(np.mean(source_area * target_area < 0)),
        "area_ratio_p01": float(np.quantile(area_ratio, 0.01)),
        "area_ratio_median": float(np.median(area_ratio)),
        "area_ratio_p99": float(np.quantile(area_ratio, 0.99)),
    }


def adjacent_vector_differences(
    field: pd.DataFrame, available_column: str, dx_column: str, dy_column: str
) -> list[float]:
    lookup = field.set_index(["grid_row", "grid_column"])
    differences = []
    for (row, column), point in lookup.iterrows():
        if not point[available_column]:
            continue
        for neighbour_index in ((row + 1, column), (row, column + 1)):
            if neighbour_index not in lookup.index:
                continue
            neighbour = lookup.loc[neighbour_index]
            if neighbour[available_column]:
                differences.append(
                    float(
                        np.hypot(
                            point[dx_column] - neighbour[dx_column],
                            point[dy_column] - neighbour[dy_column],
                        )
                    )
                )
    return differences


def summarise(field: pd.DataFrame, matches: pd.DataFrame, paired_orb: int, spacing_m: float):
    available = field["available"].fillna(False)
    orb_available = field["orb_available_10km"].fillna(False)
    both = available & orb_available
    aliked_differences = adjacent_vector_differences(
        field, "available", "proposal_dx_m", "proposal_dy_m"
    )
    orb_differences = adjacent_vector_differences(
        field, "orb_available_10km", "orb_dx_m", "orb_dy_m"
    )
    orb_field = field.assign(
        available=orb_available,
        proposal_dx_m=field["orb_dx_m"],
        proposal_dy_m=field["orb_dy_m"],
    )
    return {
        "physics_valid_feature_matches": int(len(matches)),
        "grid_queries": int(len(field)),
        "aliked_available": int(available.sum()),
        "aliked_coverage_fraction": float(available.mean()),
        "orb_paired_trajectories": paired_orb,
        "orb_available_10km": int(orb_available.sum()),
        "both_available": int(both.sum()),
        "aliked_only_available": int((available & ~orb_available).sum()),
        "orb_only_available": int((~available & orb_available).sum()),
        "neither_available": int((~available & ~orb_available).sum()),
        "median_aliked_orb_vector_difference_m": float(
            field.loc[both, "aliked_orb_vector_difference_m"].median()
        )
        if both.any()
        else None,
        "p90_aliked_orb_vector_difference_m": float(
            field.loc[both, "aliked_orb_vector_difference_m"].quantile(0.90)
        )
        if both.any()
        else None,
        "aliked_adjacent_pairs": len(aliked_differences),
        "aliked_median_adjacent_vector_difference_m": float(
            np.median(aliked_differences)
        )
        if aliked_differences
        else None,
        "aliked_p90_adjacent_vector_difference_m": float(
            np.quantile(aliked_differences, 0.90)
        )
        if aliked_differences
        else None,
        "orb_adjacent_pairs": len(orb_differences),
        "orb_median_adjacent_vector_difference_m": float(
            np.median(orb_differences)
        )
        if orb_differences
        else None,
        "orb_p90_adjacent_vector_difference_m": float(
            np.quantile(orb_differences, 0.90)
        )
        if orb_differences
        else None,
        "aliked_topology": topology_summary(field, spacing_m),
        "orb_topology": topology_summary(orb_field, spacing_m),
    }


def attach_buoy_source_positions(
    pair: pd.DataFrame, observations_path: Path | None
) -> pd.DataFrame:
    """Add projected source coordinates to a transition-only buoy table."""
    if {"source_x", "source_y"}.issubset(pair.columns):
        return pair
    if observations_path is None:
        raise ValueError(
            "case results lack source_x/source_y; provide --observations"
        )
    observations = pd.read_csv(observations_path, dtype={"buoy_id": str})
    required = {"buoy_id", "image_id", "x", "y", "analysis_crs"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"observations lack required columns: {sorted(missing)}")
    positions = observations[list(required)].rename(
        columns={
            "image_id": "source_image_id",
            "x": "source_x",
            "y": "source_y",
            "analysis_crs": "source_analysis_crs",
        }
    )
    positioned = pair.assign(buoy_id=pair["buoy_id"].astype(str)).merge(
        positions,
        on=["buoy_id", "source_image_id"],
        how="left",
        validate="many_to_one",
    )
    missing_position = positioned[["source_x", "source_y"]].isna().any(axis=1)
    if missing_position.any():
        raise ValueError(
            f"missing source positions for {int(missing_position.sum())} transitions"
        )
    if not positioned["source_analysis_crs"].eq(f"EPSG:{ANALYSIS_EPSG}").all():
        raise ValueError(f"buoy source positions must use EPSG:{ANALYSIS_EPSG}")
    return positioned


def select_pair_cases(
    cases: pd.DataFrame,
    source_image_id: int,
    target_image_id: int,
    within_dataset_split: str | None,
) -> pd.DataFrame:
    pair = cases.loc[
        cases["source_image_id"].eq(source_image_id)
        & cases["target_image_id"].eq(target_image_id)
    ].copy()
    if within_dataset_split is not None:
        if "within_dataset_split" not in pair:
            raise ValueError("case results lack within_dataset_split")
        pair = pair.loc[
            pair["within_dataset_split"].eq(within_dataset_split)
        ].copy()
    elif (
        "within_dataset_split" in pair
        and pair["within_dataset_split"].nunique() > 1
    ):
        raise ValueError(
            "requested pair spans multiple data splits; provide "
            "--within-dataset-split"
        )
    if pair.empty:
        raise ValueError("Requested image pair is absent from selected cases")
    return pair


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-results", type=Path, required=True)
    parser.add_argument(
        "--observations",
        type=Path,
        help="Observation table used when case results omit source_x/source_y.",
    )
    parser.add_argument(
        "--within-dataset-split",
        help="Restrict buoy evaluation to one existing split assignment.",
    )
    parser.add_argument("--source-image-id", type=int, required=True)
    parser.add_argument("--target-image-id", type=int, required=True)
    parser.add_argument("--orb-run-dir", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixel-size-m", type=float, default=80.0)
    parser.add_argument("--tile-pixels", type=int, default=512)
    parser.add_argument("--tile-margin-pixels", type=int, default=32)
    parser.add_argument(
        "--tile-grid-origin-m",
        type=float,
        help="Anchor both projected tile axes for cross-pair feature reuse.",
    )
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument(
        "--physics-subset-matching",
        action="store_true",
        help=(
            "For each target tile, include only source features capable of "
            "reaching its core under the configured motion gate."
        ),
    )
    parser.add_argument("--features-per-tile", type=int, default=1024)
    parser.add_argument(
        "--matching-feature-cap-per-tile",
        type=int,
        help=(
            "Optional matching-only cap applied to cached detector features "
            "with spatial round-robin selection."
        ),
    )
    parser.add_argument(
        "--matching-cells-per-axis",
        type=int,
        default=4,
        help="Spatial cells per tile axis used by matching-only thinning.",
    )
    parser.add_argument("--support-radius-pixels", type=int, default=16)
    parser.add_argument("--grid-spacing-m", type=float, default=4000.0)
    parser.add_argument("--tight-radius-m", type=float, default=2000.0)
    parser.add_argument("--consensus-radius-m", type=float, default=1000.0)
    parser.add_argument("--maximum-speed-m-per-day", type=float, default=30000.0)
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
    parser.add_argument("--matcher-call-audit", action="store_true")
    parser.add_argument("--audit-mnn-candidates", action="store_true")
    parser.add_argument("--mnn-candidate-limit", type=int)
    parser.add_argument(
        "--lightglue-target-batch-size",
        type=int,
        default=1,
        help="Reachable target tiles matched per direct LightGlue invocation.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    cases = pd.read_csv(args.case_results, dtype={"buoy_id": str}, low_memory=False)
    pair = select_pair_cases(
        cases,
        args.source_image_id,
        args.target_image_id,
        args.within_dataset_split,
    )
    pair = attach_buoy_source_positions(pair, args.observations)
    first = pair.iloc[0]
    if pair["elapsed_hours"].nunique() != 1:
        raise ValueError("Image-pair elapsed time must be unique")
    elapsed_days = float(first.elapsed_hours) / 24.0
    maximum_displacement_m = elapsed_days * args.maximum_speed_m_per_day
    source_footprint = projected_footprint(first.source_image_filepath)
    target_footprint = projected_footprint(first.target_image_filepath)
    source_domain = source_footprint.intersection(
        target_footprint.buffer(maximum_displacement_m)
    )
    target_domain = target_footprint.intersection(
        source_domain.buffer(maximum_displacement_m)
    )
    if source_domain.is_empty or target_domain.is_empty:
        raise ValueError("Source and target have no physics-reachable overlap")

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

    source_layout = tile_layout(
        source_domain,
        args.tile_pixels,
        args.tile_margin_pixels,
        args.pixel_size_m,
        args.tile_grid_origin_m,
    )
    target_layout = tile_layout(
        target_domain,
        args.tile_pixels,
        args.tile_margin_pixels,
        args.pixel_size_m,
        args.tile_grid_origin_m,
    )
    source_tiles, source_audit = extract_tiles(
        first.source_image_filepath,
        source_domain,
        source_layout,
        model,
        device,
        args.tile_pixels,
        args.tile_margin_pixels,
        args.pixel_size_m,
        args.features_per_tile,
        args.support_radius_pixels,
        args.feature_cache_dir,
    )
    print(
        f"source extraction: {len(source_tiles)} tiles, "
        f"{sum(len(tile['keypoints']) for tile in source_tiles)} features",
        flush=True,
    )
    target_tiles, target_audit = extract_tiles(
        first.target_image_filepath,
        target_domain,
        target_layout,
        model,
        device,
        args.tile_pixels,
        args.tile_margin_pixels,
        args.pixel_size_m,
        args.features_per_tile,
        args.support_radius_pixels,
        args.feature_cache_dir,
    )
    print(
        f"target extraction: {len(target_tiles)} tiles, "
        f"{sum(len(tile['keypoints']) for tile in target_tiles)} features",
        flush=True,
    )
    matching_source_tiles = spatially_thin_tiles_for_matching(
        source_tiles,
        args.matching_feature_cap_per_tile,
        args.matching_cells_per_axis,
        args.tile_pixels,
        args.tile_margin_pixels,
    )
    matching_target_tiles = spatially_thin_tiles_for_matching(
        target_tiles,
        args.matching_feature_cap_per_tile,
        args.matching_cells_per_axis,
        args.tile_pixels,
        args.tile_margin_pixels,
    )
    matcher_calls = [] if args.matcher_call_audit else None
    matcher_call_matches = [] if args.matcher_call_audit else None
    matches, match_audit = match_tiles(
        matching_source_tiles,
        matching_target_tiles,
        matcher,
        device,
        args.tile_pixels,
        maximum_displacement_m,
        elapsed_days,
        args.maximum_speed_m_per_day,
        args.physics_subset_matching,
        matcher_call_audit=matcher_calls,
        matcher_call_matches=matcher_call_matches,
        audit_mnn_candidates=args.audit_mnn_candidates,
        mnn_candidate_limit=args.mnn_candidate_limit,
        lightglue_target_batch_size=args.lightglue_target_batch_size,
    )
    print(f"matching: {len(matches)} physics-valid vectors", flush=True)
    matches.to_csv(args.output_dir / "matches.csv", index=False)
    pd.DataFrame(source_audit).assign(image="source").to_csv(
        args.output_dir / "source_tiles.csv", index=False
    )
    pd.DataFrame(target_audit).assign(image="target").to_csv(
        args.output_dir / "target_tiles.csv", index=False
    )
    pd.DataFrame(match_audit).to_csv(
        args.output_dir / "matching_tiles.csv", index=False
    )
    if matcher_calls is not None:
        pd.DataFrame(matcher_calls).to_csv(
            args.output_dir / "matcher_calls.csv", index=False
        )
        pd.DataFrame(matcher_call_matches).to_csv(
            args.output_dir / "matcher_call_matches.csv", index=False
        )
    queries = regular_queries(source_domain, args.grid_spacing_m)
    field = consensus_at_queries(
        matches, queries, args.tight_radius_m, args.consensus_radius_m
    )

    run_manifest = json.loads((args.orb_run_dir / "run_manifest.json").read_text())
    database_url = run_manifest["engine_url"]
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("Only SQLite ORB baselines are supported")
    image_map = pd.read_csv(args.orb_run_dir / "image_timings.csv")
    catalog_to_run = dict(
        zip(image_map["catalog_image_id"], image_map["run_image_id"], strict=True)
    )
    field, paired_orb = attach_orb_field(
        field,
        Path(database_url[len(prefix) :]),
        run_manifest["effective_run_name"],
        int(catalog_to_run[args.source_image_id]),
        int(catalog_to_run[args.target_image_id]),
    )
    buoy_queries = pair.rename(columns={"source_x": "source_x", "source_y": "source_y"})
    buoy_results = consensus_at_queries(
        matches,
        buoy_queries,
        args.tight_radius_m,
        args.consensus_radius_m,
    )
    if "truth_dx_m" in buoy_results:
        buoy_results["error_m"] = np.hypot(
            buoy_results["proposal_dx_m"] - buoy_results["truth_dx_m"],
            buoy_results["proposal_dy_m"] - buoy_results["truth_dy_m"],
        )

    field.to_csv(args.output_dir / "field_4km.csv", index=False)
    buoy_results.to_csv(args.output_dir / "buoy_results.csv", index=False)
    summary = summarise(field, matches, paired_orb, args.grid_spacing_m)
    summary["matching_source_features"] = int(
        sum(len(tile["keypoints"]) for tile in matching_source_tiles)
    )
    summary["matching_target_features"] = int(
        sum(len(tile["keypoints"]) for tile in matching_target_tiles)
    )
    summary["buoy_cases"] = int(len(buoy_results))
    summary["buoy_available"] = int(buoy_results["available"].sum())
    summary["matcher_calls"] = (
        int(
            sum(bool(row.get("matcher_executed", True)) for row in matcher_calls)
        )
        if matcher_calls is not None
        else None
    )
    summary["matcher_invocations"] = int(
        sum(row["matcher_invocations"] for row in match_audit)
    )
    summary["matcher_seconds"] = float(
        sum(row["seconds"] for row in match_audit)
    )
    summary["candidate_audit_seconds"] = float(
        sum(row["candidate_audit_seconds"] for row in match_audit)
    )
    if "error_m" in buoy_results:
        errors = buoy_results.loc[buoy_results["available"], "error_m"]
        summary["buoy_median_error_m"] = float(errors.median()) if len(errors) else None
        summary["buoy_maximum_error_m"] = float(errors.max()) if len(errors) else None
    summary["elapsed_seconds"] = time.perf_counter() - started
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "status": "complete",
        "source_image_id": args.source_image_id,
        "target_image_id": args.target_image_id,
        "source_image_filepath": first.source_image_filepath,
        "target_image_filepath": first.target_image_filepath,
        "elapsed_hours": float(first.elapsed_hours),
        "analysis_crs": f"EPSG:{ANALYSIS_EPSG}",
        "source_domain_area_km2": float(source_domain.area / 1.0e6),
        "target_domain_area_km2": float(target_domain.area / 1.0e6),
        "source_tiles": len(source_tiles),
        "target_tiles": len(target_tiles),
        "source_feature_cache_hits": int(
            sum(bool(row["cache_hit"]) for row in source_audit)
        ),
        "target_feature_cache_hits": int(
            sum(bool(row["cache_hit"]) for row in target_audit)
        ),
        "parameters": vars(args) | {"device": str(device)},
        "case_results_sha256": sha256(args.case_results),
        "observations_sha256": sha256(args.observations)
        if args.observations is not None
        else None,
        "orb_run_manifest_sha256": sha256(args.orb_run_dir / "run_manifest.json"),
        "interpretation": (
            "Complete physics-reachable image-pair footprint; off-buoy grid nodes "
            "measure coverage, agreement with ORB, and topology without external truth."
        ),
        "summary": summary,
    }
    manifest["parameters"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in manifest["parameters"].items()
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
