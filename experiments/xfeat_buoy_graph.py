#!/usr/bin/env python3
"""Evaluate sparse XFeat descriptors and graph updates on Arctic buoy paths.

XFeat detects its own sparse keypoints; it is not treated as an ORB-compatible
arbitrary-keypoint descriptor. The closest detected feature supplies the seed
descriptor, and feature coverage error is reported separately from tracking
error. Float32 descriptors remain float32 throughout the graph.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from kornia.feature import XFeat
from scipy.spatial import cKDTree

from buoy_descriptor_benchmark import (
    CandidateGrid,
    annotate_coincidences,
    pixels_to_map,
    read_scene,
)
from orb_multiframe_graph import (
    DescriptorLayer,
    GraphSearchConfig,
    UPDATE_POLICY_DESCRIPTIONS,
    search_layered_graph,
    trajectory_column,
)


@dataclass
class XFeatLayer:
    graph: DescriptorLayer
    scores: np.ndarray
    original_shape: tuple[int, int]
    model_shape: tuple[int, int]
    scale_xy: tuple[float, float]


XFEAT_GRAPH_CONFIGS = (
    GraphSearchConfig("xfeat_greedy_rolling", "rolling", 1, 1),
    GraphSearchConfig("xfeat_beam_rolling", "rolling", 32, 8),
    GraphSearchConfig("xfeat_beam_anchor", "anchor", 32, 8),
    GraphSearchConfig("xfeat_beam_anchor_rolling", "anchor_rolling", 32, 8),
    GraphSearchConfig(
        "xfeat_beam_confidence_m005",
        "confidence_rolling",
        32,
        8,
        update_min_margin=0.005,
        update_max_cost=0.35,
    ),
    GraphSearchConfig(
        "xfeat_beam_confidence_m010",
        "confidence_rolling",
        32,
        8,
        update_min_margin=0.010,
        update_max_cost=0.35,
    ),
)


def resize_for_xfeat(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    height, width = image.shape
    scale = min(1.0, max_side / float(max(height, width)))
    target_width = max(32, int(round(width * scale / 32.0)) * 32)
    target_height = max(32, int(round(height * scale / 32.0)) * 32)
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, width / float(target_width), height / float(target_height)


def extract_layer(
    extractor: XFeat,
    row,
    args,
) -> XFeatLayer:
    image, mask = read_scene(row.image_filepath)
    model_image, sx, sy = resize_for_xfeat(image, args.max_side)
    tensor = torch.from_numpy(model_image).float()[None, None].div_(255.0).to(args.device)
    with torch.inference_mode():
        features = extractor.detectAndCompute(
            tensor,
            top_k=args.top_k,
            detection_threshold=args.detection_threshold,
        )[0]
    pixel_xy = features["keypoints"].detach().cpu().numpy().astype(np.float64)
    pixel_xy *= np.array([sx, sy], dtype=np.float64)
    descriptors = features["descriptors"].detach().cpu().numpy().astype(np.float32)
    scores = features["scores"].detach().cpu().numpy().astype(np.float32)

    valid = np.isfinite(pixel_xy).all(axis=1)
    valid &= (pixel_xy[:, 0] >= 0) & (pixel_xy[:, 0] < image.shape[1])
    valid &= (pixel_xy[:, 1] >= 0) & (pixel_xy[:, 1] < image.shape[0])
    if mask is not None and np.any(valid):
        cols = np.clip(np.rint(pixel_xy[:, 0]).astype(int), 0, image.shape[1] - 1)
        rows = np.clip(np.rint(pixel_xy[:, 1]).astype(int), 0, image.shape[0] - 1)
        valid &= mask[rows, cols] < 2
    pixel_xy = pixel_xy[valid]
    descriptors = descriptors[valid]
    scores = scores[valid]
    map_xy = pixels_to_map(row.image_filepath, pixel_xy, args.analysis_epsg)
    finite = np.isfinite(map_xy).all(axis=1)
    pixel_xy = pixel_xy[finite]
    descriptors = descriptors[finite]
    scores = scores[finite]
    map_xy = map_xy[finite]
    grid = CandidateGrid(pixel_xy=pixel_xy, map_xy=map_xy)
    graph = DescriptorLayer(
        image_id=int(row.image_id),
        image_filename=row.image_filename,
        image_time=pd.Timestamp(row.image_time),
        grid=grid,
        descriptors=descriptors,
        spatial_index=cKDTree(map_xy),
    )
    return XFeatLayer(
        graph=graph,
        scores=scores,
        original_shape=image.shape,
        model_shape=model_image.shape,
        scale_xy=(sx, sy),
    )


def precompute_layers(
    coincidences: pd.DataFrame,
    args,
) -> tuple[dict[str, XFeatLayer], pd.DataFrame, float]:
    started = time.perf_counter()
    extractor = XFeat.from_pretrained(
        top_k=args.top_k,
        detection_threshold=args.detection_threshold,
    ).eval().to(args.device)
    layers: dict[str, XFeatLayer] = {}
    rows = []
    unique = coincidences.sort_values("image_time").drop_duplicates("image_filepath")
    for row in unique.itertuples(index=False):
        layer_started = time.perf_counter()
        layer = extract_layer(extractor, row, args)
        layers[row.image_filepath] = layer
        rows.append(
            {
                "image_id": int(row.image_id),
                "image_filename": row.image_filename,
                "image_time": row.image_time,
                "features": len(layer.graph.descriptors),
                "descriptor_dtype": str(layer.graph.descriptors.dtype),
                "descriptor_dimensions": (
                    int(layer.graph.descriptors.shape[1])
                    if layer.graph.descriptors.ndim == 2 and len(layer.graph.descriptors)
                    else 0
                ),
                "original_height": layer.original_shape[0],
                "original_width": layer.original_shape[1],
                "model_height": layer.model_shape[0],
                "model_width": layer.model_shape[1],
                "scale_x": layer.scale_xy[0],
                "scale_y": layer.scale_xy[1],
                "seconds": time.perf_counter() - layer_started,
            }
        )
    return layers, pd.DataFrame.from_records(rows), time.perf_counter() - started


def nearest_feature(layer: XFeatLayer, xy: np.ndarray) -> tuple[int, float]:
    distance, index = layer.graph.spatial_index.query(np.asarray(xy, dtype=float), k=1)
    return int(index), float(distance)


def trajectory_rows(
    coincidences: pd.DataFrame,
    layers: dict[str, XFeatLayer],
    config: GraphSearchConfig,
    args,
) -> list[dict]:
    records = []
    path_column = trajectory_column(coincidences)
    for trajectory_id, group in coincidences.groupby(path_column, sort=True):
        group = group.sort_values("image_time").reset_index(drop=True)
        buoy_id = str(group.iloc[0]["buoy_id"])
        if len(group) < 2:
            continue
        first_layer = layers[group.iloc[0].image_filepath]
        seed_xy = group.loc[0, ["x", "y"]].to_numpy(dtype=float)
        seed_index, seed_offset = nearest_feature(first_layer, seed_xy)
        if seed_offset > args.max_seed_feature_distance_m:
            records.append(
                {
                    "config": config.name,
                    "buoy_id": buoy_id,
                    "trajectory_id": trajectory_id,
                    "status": "seed_unavailable",
                    "path_observations": len(group),
                    "seed_feature_offset_m": seed_offset,
                }
            )
            continue
        graph_layers = [layers[path].graph for path in group.image_filepath]
        state = search_layered_graph(
            graph_layers,
            seed_xy=seed_xy,
            seed_descriptor=first_layer.graph.descriptors[seed_index],
            max_speed_m_per_day=args.max_speed_m_per_day,
            config=config,
            descriptor_norm="cosine",
        )
        if state is None:
            records.append(
                {
                    "config": config.name,
                    "buoy_id": buoy_id,
                    "trajectory_id": trajectory_id,
                    "status": "graph_failed",
                    "path_observations": len(group),
                    "seed_feature_offset_m": seed_offset,
                }
            )
            continue
        for index, (predicted, truth) in enumerate(zip(state.path_xy, group.itertuples(index=False))):
            layer = layers[truth.image_filepath]
            _, coverage_error = nearest_feature(
                layer, np.array([truth.x, truth.y], dtype=float)
            )
            node_index = state.node_indices[index]
            feature_score = (
                float(first_layer.scores[seed_index])
                if index == 0
                else float(layer.scores[node_index])
            )
            records.append(
                {
                    "config": config.name,
                    "update_policy": config.update_policy,
                    "update_policy_description": UPDATE_POLICY_DESCRIPTIONS[
                        config.update_policy
                    ],
                    "buoy_id": buoy_id,
                    "trajectory_id": trajectory_id,
                    "status": "ok",
                    "path_observations": len(group),
                    "observation_index": index,
                    "image_id": truth.image_id,
                    "image_filename": truth.image_filename,
                    "image_time": truth.image_time,
                    "truth_x": truth.x,
                    "truth_y": truth.y,
                    "predicted_x": float(predicted[0]),
                    "predicted_y": float(predicted[1]),
                    "endpoint_error_m": float(
                        np.linalg.norm(predicted - np.array([truth.x, truth.y], dtype=float))
                    ),
                    "feature_coverage_floor_m": coverage_error,
                    "seed_feature_offset_m": seed_offset,
                    "feature_score": feature_score,
                    "edge_cost": 0.0 if index == 0 else state.edge_costs[index - 1],
                    "candidate_count": 0 if index == 0 else state.candidate_counts[index - 1],
                    "node_index": node_index,
                    "descriptor_updated": False if index == 0 else state.descriptor_updates[index - 1],
                }
            )
    return records


def summarize(records: pd.DataFrame, coincidences: pd.DataFrame) -> pd.DataFrame:
    coincidence_path_column = trajectory_column(coincidences)
    eligible_paths = int(
        (coincidences.groupby(coincidence_path_column).size() >= 2).sum()
    )
    rows = []
    for config, group in records.groupby("config", sort=False):
        result_path_column = (
            "trajectory_id" if "trajectory_id" in group else "buoy_id"
        )
        valid = group[(group.status == "ok") & (group.observation_index > 0)].copy()
        completed = int(
            group.loc[group.status == "ok", result_path_column].nunique()
        )
        failures = group[group.status != "ok"]
        if len(valid):
            finals = (
                valid.sort_values("observation_index")
                .groupby(result_path_column)
                .tail(1)
            )
            errors = valid.endpoint_error_m.to_numpy(dtype=float)
            floors = valid.feature_coverage_floor_m.to_numpy(dtype=float)
        else:
            finals = pd.DataFrame()
            errors = np.array([], dtype=float)
            floors = np.array([], dtype=float)
        rows.append(
            {
                "config": config,
                "descriptor_memory": group["update_policy_description"].dropna().iloc[0]
                if "update_policy_description" in group
                and group["update_policy_description"].notna().any()
                else "",
                "eligible_paths": eligible_paths,
                "completed_paths": completed,
                "failed_paths": int(failures[result_path_column].nunique()),
                "transitions": len(errors),
                "median_error_m": float(np.median(errors)) if len(errors) else math.nan,
                "p90_error_m": float(np.percentile(errors, 90)) if len(errors) else math.nan,
                "within_2km_fraction": float(np.mean(errors <= 2000.0)) if len(errors) else math.nan,
                "catastrophic_50km_fraction": float(np.mean(errors > 50000.0)) if len(errors) else math.nan,
                "median_feature_coverage_floor_m": float(np.median(floors)) if len(floors) else math.nan,
                "median_final_error_m": float(finals.endpoint_error_m.median()) if len(finals) else math.nan,
                "descriptor_update_fraction": float(valid.descriptor_updated.mean()) if len(valid) else math.nan,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["failed_paths", "catastrophic_50km_fraction", "median_error_m"]
    )


def write_report(path: Path, summary: pd.DataFrame, feature_summary: pd.DataFrame, args, elapsed: float) -> None:
    view = summary.copy()
    for column in [
        "median_error_m",
        "p90_error_m",
        "median_feature_coverage_floor_m",
        "median_final_error_m",
    ]:
        view[column.replace("_m", "_km")] = view[column] / 1000.0
    columns = [
        "config",
        "descriptor_memory",
        "completed_paths",
        "failed_paths",
        "median_error_km",
        "p90_error_km",
        "within_2km_fraction",
        "catastrophic_50km_fraction",
        "median_final_error_km",
        "descriptor_update_fraction",
    ]
    table = view[columns].copy()
    for column in table.select_dtypes(include=["float"]).columns:
        table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in table.to_numpy()],
    ]
    path.write_text(
        f"""# Arctic XFeat sparse-descriptor buoy graph

- Standard VAE-preprocessed band; no additional preprocessing.
- XFeat sparse detection at max side {args.max_side}px, top_k={args.top_k}, detection threshold={args.detection_threshold}.
- 64-dimensional float32 descriptors with cosine distance; no ORB dtype or Hamming assumptions.
- Exact buoy positions are used for the initial position and evaluation only.
- The seed descriptor comes from the nearest detected feature within {args.max_seed_feature_distance_m / 1000:.1f} km.
- Median sparse feature count: {feature_summary.features.median():.0f} features/image; see `image_features.csv` for spatial sampling details.
- Runtime: {elapsed:.2f} seconds.

## Results

{chr(10).join(lines)}

This is a descriptor/candidate graph test, not full LiMOSAT. It excludes PM,
MAGSAC, correlation, neighbourhood deformation, and joint track assignment.
Feature-coverage error is retained in the CSV and must not be attributed to
descriptor matching.
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coincidences", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--analysis-epsg", type=int, default=3413)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=8192)
    parser.add_argument("--detection-threshold", type=float, default=0.05)
    parser.add_argument("--max-seed-feature-distance-m", type=float, default=5000.0)
    parser.add_argument("--max-speed-m-per-day", type=float, default=50000.0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--experiment-split")
    parser.add_argument("--month-exclusive-buoys-only", action="store_true")
    parser.add_argument("--graph-configs", default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    coincidences = pd.read_csv(args.coincidences)
    coincidences["image_time"] = pd.to_datetime(coincidences["image_time"], utc=True)
    if args.experiment_split is not None:
        if "experiment_split" not in coincidences:
            parser.error("--experiment-split requires an experiment_split fixture column.")
        coincidences = coincidences[
            coincidences["experiment_split"].eq(args.experiment_split)
        ].copy()
    if args.month_exclusive_buoys_only:
        if "month_exclusive_buoy" not in coincidences:
            parser.error(
                "--month-exclusive-buoys-only requires a month_exclusive_buoy fixture column."
            )
        values = coincidences["month_exclusive_buoy"]
        if values.dtype != bool:
            values = values.astype(str).str.lower().isin({"true", "1"})
        coincidences = coincidences[values].copy()
    if coincidences.empty:
        raise ValueError("No coincidence observations remain after fixture filtering.")
    before_spatial = len(coincidences)
    coincidences = annotate_coincidences(
        coincidences,
        args.analysis_epsg,
        outside_scene_policy="skip",
    )
    invalid = (coincidences.mask_value >= 2) | ~np.isfinite(
        coincidences[["col", "row"]]
    ).all(axis=1)
    coincidences = coincidences.loc[~invalid].reset_index(drop=True)
    if coincidences.empty:
        raise ValueError("No spatially valid exact-time buoy coincidences.")

    layers, feature_summary, extraction_seconds = precompute_layers(coincidences, args)
    requested = None if args.graph_configs is None else set(args.graph_configs.split(","))
    configs = tuple(
        config for config in XFEAT_GRAPH_CONFIGS if requested is None or config.name in requested
    )
    if requested is not None:
        missing = requested - {config.name for config in configs}
        if missing:
            parser.error(f"Unknown graph configurations: {sorted(missing)}")

    records = []
    timings = []
    for config in configs:
        config_started = time.perf_counter()
        records.extend(trajectory_rows(coincidences, layers, config, args))
        timings.append({"config": config.name, "seconds": time.perf_counter() - config_started})
    records = pd.DataFrame.from_records(records)
    summary = summarize(records, coincidences)
    elapsed = time.perf_counter() - started

    coincidences.to_csv(args.out_dir / "coincidences.csv", index=False)
    feature_summary.to_csv(args.out_dir / "image_features.csv", index=False)
    records.to_csv(args.out_dir / "trajectory_results.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    pd.DataFrame.from_records(timings).to_csv(args.out_dir / "timings.csv", index=False)
    write_report(args.out_dir / "report.md", summary, feature_summary, args, elapsed)
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "descriptor_dtype": "float32",
                "descriptor_dimensions": 64,
                "descriptor_norm": "cosine",
                "arbitrary_supplied_keypoints": False,
                "preprocessing": "standard VAE band only",
                "exact_time_coincidences_before_spatial_filter": before_spatial,
                "spatially_valid_coincidences": len(coincidences),
                "feature_extraction_seconds": extraction_seconds,
                "elapsed_seconds": elapsed,
                "configs": [config.__dict__ for config in configs],
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False))
    print(json.dumps({"feature_extraction_seconds": extraction_seconds, "elapsed_seconds": elapsed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
