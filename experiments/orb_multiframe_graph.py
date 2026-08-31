#!/usr/bin/env python3
"""Evaluate multi-frame ORB candidate graphs on exact-time Arctic buoy paths.

The experiment keeps the existing VAE-preprocessed image band fixed. It does
not change production LiMOSAT. Candidate positions are a fixed grid, graph
edges obey only a speed-scaled displacement limit, and buoy target positions
are used solely for scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from experiments.buoy_descriptor_benchmark import (
        CandidateGrid,
        DescriptorVariant,
        annotate_coincidences,
        build_coincidences,
        candidate_grid,
        compute_descriptors,
        descriptor_distances,
        exact_descriptor,
        image_angle,
        load_catalog,
        map_output_descriptors_to_grid,
        read_scene,
    )
except ModuleNotFoundError:  # Direct execution from the experiments directory.
    from buoy_descriptor_benchmark import (  # type: ignore[no-redef]
        CandidateGrid,
        DescriptorVariant,
        annotate_coincidences,
        build_coincidences,
        candidate_grid,
        compute_descriptors,
        descriptor_distances,
        exact_descriptor,
        image_angle,
        load_catalog,
        map_output_descriptors_to_grid,
        read_scene,
    )


@dataclass(frozen=True)
class GraphSearchConfig:
    name: str
    update_policy: str
    beam_width: int
    branching: int
    motion_weight: float = 0.0
    displacement_weight: float = 0.0
    bank_size: int = 4
    update_min_margin: float = 0.0
    update_max_cost: float = 1.0
    max_consecutive_skips: int = 0
    skip_penalty: float = 0.45
    preferred_speed_m_per_day: float | None = None
    excess_speed_weight: float = 0.0


@dataclass
class DescriptorLayer:
    image_id: int
    image_filename: str
    image_time: pd.Timestamp
    grid: CandidateGrid
    descriptors: np.ndarray
    spatial_index: cKDTree | None = None


@dataclass
class PathState:
    score: float
    position_xy: np.ndarray
    velocity_xy_per_second: np.ndarray | None
    descriptors: tuple[np.ndarray, ...]
    path_xy: tuple[np.ndarray | None, ...]
    node_indices: tuple[int, ...]
    edge_costs: tuple[float, ...]
    candidate_counts: tuple[int, ...]
    descriptor_updates: tuple[bool, ...] = ()
    elapsed_since_position_seconds: float = 0.0
    consecutive_skips: int = 0


GRAPH_CONFIGS = (
    GraphSearchConfig("greedy_rolling", "rolling", beam_width=1, branching=1),
    GraphSearchConfig("greedy_anchor", "anchor", beam_width=1, branching=1),
    GraphSearchConfig("beam_rolling", "rolling", beam_width=32, branching=8),
    GraphSearchConfig("beam_anchor_b4", "anchor", beam_width=4, branching=4),
    GraphSearchConfig("beam_anchor_b8", "anchor", beam_width=8, branching=8),
    GraphSearchConfig("beam_anchor", "anchor", beam_width=32, branching=8),
    GraphSearchConfig("beam_anchor_b128", "anchor", beam_width=128, branching=8),
    GraphSearchConfig("beam_majority_bit", "majority_bit", beam_width=32, branching=8),
    GraphSearchConfig("beam_bank_min", "bank_min", beam_width=32, branching=8),
    GraphSearchConfig("beam_anchor_rolling", "anchor_rolling", beam_width=32, branching=8),
    GraphSearchConfig("beam_anchor_guarded", "anchor_guarded", beam_width=32, branching=8),
    GraphSearchConfig(
        "beam_bank_min_motion",
        "bank_min",
        beam_width=32,
        branching=8,
        motion_weight=0.25,
    ),
    GraphSearchConfig("beam_anchor_speed025", "anchor", 32, 8, displacement_weight=0.025),
    GraphSearchConfig("beam_anchor_speed05", "anchor", 32, 8, displacement_weight=0.05),
    GraphSearchConfig("beam_anchor_speed10", "anchor", 32, 8, displacement_weight=0.10),
    GraphSearchConfig("beam_anchor_motion05", "anchor", 32, 8, motion_weight=0.05),
    GraphSearchConfig("beam_anchor_motion10", "anchor", 32, 8, motion_weight=0.10),
    GraphSearchConfig(
        "beam_anchor_speed05_motion05",
        "anchor",
        32,
        8,
        motion_weight=0.05,
        displacement_weight=0.05,
    ),
    GraphSearchConfig(
        "beam_anchor_rolling_speed05_motion05",
        "anchor_rolling",
        32,
        8,
        motion_weight=0.05,
        displacement_weight=0.05,
    ),
    GraphSearchConfig(
        "beam_confidence_update_m004",
        "confidence_rolling",
        32,
        8,
        update_min_margin=0.004,
        update_max_cost=0.35,
    ),
    GraphSearchConfig(
        "beam_confidence_update_m008",
        "confidence_rolling",
        32,
        8,
        update_min_margin=0.008,
        update_max_cost=0.35,
    ),
    GraphSearchConfig(
        "beam_confidence_update_m016",
        "confidence_rolling",
        32,
        8,
        update_min_margin=0.016,
        update_max_cost=0.35,
    ),
    GraphSearchConfig(
        "beam_confidence_update_m032",
        "confidence_rolling",
        32,
        8,
        update_min_margin=0.032,
        update_max_cost=0.35,
    ),
    GraphSearchConfig(
        "beam_anchor_skip1",
        "anchor",
        32,
        8,
        max_consecutive_skips=1,
        skip_penalty=0.45,
    ),
    GraphSearchConfig(
        "beam_confidence_m032_skip1",
        "confidence_rolling",
        32,
        8,
        update_min_margin=0.032,
        update_max_cost=0.35,
        max_consecutive_skips=1,
        skip_penalty=0.45,
    ),
    GraphSearchConfig(
        "beam_fixed_first_prefer40_w0025",
        "anchor",
        32,
        8,
        preferred_speed_m_per_day=40_000.0,
        excess_speed_weight=0.025,
    ),
    GraphSearchConfig(
        "beam_fixed_first_prefer40_w005",
        "anchor",
        32,
        8,
        preferred_speed_m_per_day=40_000.0,
        excess_speed_weight=0.05,
    ),
    GraphSearchConfig(
        "beam_fixed_first_prefer40_w010",
        "anchor",
        32,
        8,
        preferred_speed_m_per_day=40_000.0,
        excess_speed_weight=0.10,
    ),
    GraphSearchConfig(
        "beam_confidence_prefer40_w005",
        "confidence_rolling",
        32,
        8,
        update_min_margin=0.032,
        update_max_cost=0.35,
        preferred_speed_m_per_day=40_000.0,
        excess_speed_weight=0.05,
    ),
)

UPDATE_POLICY_DESCRIPTIONS = {
    "rolling": "compare with the descriptor from the previous selected position",
    "anchor": "compare with the fixed descriptor at the first known buoy position",
    "majority_bit": "compare with the bitwise-majority descriptor of selected positions",
    "bank_min": "use the closest descriptor in a recent selected-position memory bank",
    "anchor_rolling": "average fixed-first and previous-selected descriptor costs",
    "anchor_guarded": "require agreement with fixed-first and previous-selected descriptors",
    "confidence_rolling": "update the previous-selected descriptor only after a confident match",
}


def trajectory_column(frame: pd.DataFrame) -> str:
    """Prefer explicit split-safe trajectories, retaining legacy buoy fixtures."""
    return (
        "experiment_trajectory_id"
        if "experiment_trajectory_id" in frame.columns
        else "buoy_id"
    )


def majority_binary_descriptor(descriptors: tuple[np.ndarray, ...]) -> np.ndarray:
    """Return bitwise majority, resolving exact ties with the latest state."""
    values = np.asarray(descriptors, dtype=np.uint8)
    bits = np.unpackbits(values, axis=1)
    bit_sum = bits.sum(axis=0)
    result = bit_sum * 2 > len(values)
    ties = bit_sum * 2 == len(values)
    if np.any(ties):
        latest = np.unpackbits(values[-1])
        result[ties] = latest[ties].astype(bool)
    return np.packbits(result.astype(np.uint8))


def appearance_costs(
    state_descriptors: tuple[np.ndarray, ...],
    candidates: np.ndarray,
    policy: str,
    descriptor_norm: str = "hamming",
) -> np.ndarray:
    """Normalized ORB Hamming cost for one explicit update policy."""
    if policy == "rolling":
        distances = descriptor_distances(state_descriptors[-1], candidates, descriptor_norm)
    elif policy == "anchor":
        distances = descriptor_distances(state_descriptors[0], candidates, descriptor_norm)
    elif policy == "majority_bit":
        prototype = majority_binary_descriptor(state_descriptors)
        distances = descriptor_distances(prototype, candidates, descriptor_norm)
    elif policy == "bank_min":
        distances = np.min(
            np.vstack(
                [descriptor_distances(descriptor, candidates, descriptor_norm) for descriptor in state_descriptors]
            ),
            axis=0,
        )
    elif policy in {"anchor_rolling", "confidence_rolling"}:
        anchor = descriptor_distances(state_descriptors[0], candidates, descriptor_norm)
        rolling = descriptor_distances(state_descriptors[-1], candidates, descriptor_norm)
        distances = 0.5 * (anchor + rolling)
    elif policy == "anchor_guarded":
        anchor = descriptor_distances(state_descriptors[0], candidates, descriptor_norm)
        rolling = descriptor_distances(state_descriptors[-1], candidates, descriptor_norm)
        distances = np.maximum(anchor, rolling)
    else:
        raise ValueError(f"Unknown update policy: {policy}")
    if descriptor_norm == "hamming":
        scale = float(candidates.shape[1] * 8)
    elif descriptor_norm == "hamming2":
        scale = float(candidates.shape[1] * 4)
    elif descriptor_norm in {"cosine", "l2"}:
        scale = 2.0
    else:
        raise ValueError(f"Unknown descriptor norm: {descriptor_norm}")
    return distances.astype(float) / scale


def updated_descriptor_bank(
    state_descriptors: tuple[np.ndarray, ...],
    candidate: np.ndarray,
    policy: str,
    bank_size: int,
    should_update: bool = True,
) -> tuple[np.ndarray, ...]:
    candidate = np.asarray(candidate).copy()
    if policy == "anchor":
        return state_descriptors
    if policy == "rolling":
        return (candidate,)
    if policy == "majority_bit":
        return (*state_descriptors, candidate)
    if policy == "bank_min":
        return (*state_descriptors, candidate)[-bank_size:]
    if policy in {"anchor_rolling", "anchor_guarded"}:
        return (state_descriptors[0], candidate)
    if policy == "confidence_rolling":
        if not should_update:
            return state_descriptors
        return (state_descriptors[0], candidate)
    raise ValueError(f"Unknown update policy: {policy}")


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) <= count:
        return np.argsort(values, kind="stable")
    subset = np.argpartition(values, count - 1)[:count]
    return subset[np.argsort(values[subset], kind="stable")]


def skipped_path_state(
    state: PathState,
    effective_dt_seconds: float,
    candidate_count: int,
    penalty: float,
) -> PathState:
    return PathState(
        score=state.score + penalty,
        position_xy=state.position_xy.copy(),
        velocity_xy_per_second=state.velocity_xy_per_second,
        descriptors=state.descriptors,
        path_xy=(*state.path_xy, None),
        node_indices=(*state.node_indices, -2),
        edge_costs=(*state.edge_costs, penalty),
        candidate_counts=(*state.candidate_counts, candidate_count),
        descriptor_updates=(*state.descriptor_updates, False),
        elapsed_since_position_seconds=effective_dt_seconds,
        consecutive_skips=state.consecutive_skips + 1,
    )


def expand_graph_layer(
    states: list[PathState],
    layer: DescriptorLayer,
    dt_seconds: float,
    max_speed_m_per_day: float,
    config: GraphSearchConfig,
    descriptor_norm: str = "hamming",
) -> list[PathState]:
    expanded = []
    if layer.spatial_index is None:
        layer.spatial_index = cKDTree(layer.grid.map_xy)
    for state in states:
        effective_dt_seconds = dt_seconds + state.elapsed_since_position_seconds
        radius_m = max_speed_m_per_day * effective_dt_seconds / 86400.0
        gate_indices = np.asarray(
            sorted(layer.spatial_index.query_ball_point(state.position_xy, radius_m)),
            dtype=int,
        )
        if len(gate_indices) == 0:
            if state.consecutive_skips < config.max_consecutive_skips:
                expanded.append(
                    skipped_path_state(
                        state,
                        effective_dt_seconds,
                        candidate_count=0,
                        penalty=config.skip_penalty,
                    )
                )
            continue
        candidate_descriptors = layer.descriptors[gate_indices]
        gated_spatial_distance = np.linalg.norm(
            layer.grid.map_xy[gate_indices] - state.position_xy,
            axis=1,
        )
        appearance = appearance_costs(
            state.descriptors,
            candidate_descriptors,
            config.update_policy,
            descriptor_norm,
        )
        if state.velocity_xy_per_second is None or config.motion_weight == 0:
            motion_cost = np.zeros(len(gate_indices), dtype=float)
        else:
            predicted = state.position_xy + state.velocity_xy_per_second * effective_dt_seconds
            prediction_error = np.linalg.norm(layer.grid.map_xy[gate_indices] - predicted, axis=1)
            motion_cost = prediction_error / max(radius_m, 1.0)
        displacement_cost = gated_spatial_distance / max(radius_m, 1.0)
        if (
            config.preferred_speed_m_per_day is not None
            and config.excess_speed_weight > 0
        ):
            preferred_radius_m = (
                config.preferred_speed_m_per_day
                * effective_dt_seconds
                / 86400.0
            )
            excess_speed_cost = np.clip(
                (gated_spatial_distance - preferred_radius_m)
                / max(radius_m - preferred_radius_m, 1.0),
                0.0,
                None,
            )
        else:
            excess_speed_cost = np.zeros(len(gate_indices), dtype=float)
        edge_cost = (
            appearance
            + config.motion_weight * motion_cost
            + config.displacement_weight * displacement_cost
            + config.excess_speed_weight * excess_speed_cost
        )
        appearance_order = np.argsort(appearance, kind="stable")
        appearance_best = int(appearance_order[0])
        appearance_margin = (
            float(appearance[appearance_order[1]] - appearance[appearance_best])
            if len(appearance_order) > 1
            else math.inf
        )
        for local_index in _top_indices(edge_cost, config.branching):
            node_index = int(gate_indices[local_index])
            position = layer.grid.map_xy[node_index].copy()
            velocity = (position - state.position_xy) / max(effective_dt_seconds, 1.0)
            descriptor = layer.descriptors[node_index]
            if config.update_policy == "anchor":
                descriptor_updated = False
            elif config.update_policy == "confidence_rolling":
                descriptor_updated = not (
                    int(local_index) != appearance_best
                    or appearance_margin < config.update_min_margin
                    or float(appearance[local_index]) > config.update_max_cost
                )
            else:
                descriptor_updated = True
            expanded.append(
                PathState(
                    score=state.score + float(edge_cost[local_index]),
                    position_xy=position,
                    velocity_xy_per_second=velocity,
                    descriptors=updated_descriptor_bank(
                        state.descriptors,
                        descriptor,
                        config.update_policy,
                        config.bank_size,
                        descriptor_updated,
                    ),
                    path_xy=(*state.path_xy, position),
                    node_indices=(*state.node_indices, node_index),
                    edge_costs=(*state.edge_costs, float(edge_cost[local_index])),
                    candidate_counts=(*state.candidate_counts, int(len(gate_indices))),
                    descriptor_updates=(*state.descriptor_updates, descriptor_updated),
                    elapsed_since_position_seconds=0.0,
                    consecutive_skips=0,
                )
            )
        if state.consecutive_skips < config.max_consecutive_skips:
            expanded.append(
                skipped_path_state(
                    state,
                    effective_dt_seconds,
                    candidate_count=int(len(gate_indices)),
                    penalty=config.skip_penalty,
                )
            )
    expanded.sort(key=lambda state: state.score)
    return expanded[: config.beam_width]


def search_layered_graph(
    layers: list[DescriptorLayer],
    seed_xy: np.ndarray,
    seed_descriptor: np.ndarray,
    max_speed_m_per_day: float,
    config: GraphSearchConfig,
    descriptor_norm: str = "hamming",
) -> PathState | None:
    if not layers:
        return None
    states = [
        PathState(
            score=0.0,
            position_xy=np.asarray(seed_xy, dtype=float),
            velocity_xy_per_second=None,
            descriptors=(np.asarray(seed_descriptor).copy(),),
            path_xy=(np.asarray(seed_xy, dtype=float).copy(),),
            node_indices=(-1,),
            edge_costs=(),
            candidate_counts=(),
            descriptor_updates=(),
            elapsed_since_position_seconds=0.0,
            consecutive_skips=0,
        )
    ]
    previous_time = layers[0].image_time
    for layer in layers[1:]:
        dt_seconds = (layer.image_time - previous_time).total_seconds()
        if dt_seconds <= 0:
            raise ValueError("Image times must increase within a graph.")
        states = expand_graph_layer(
            states,
            layer,
            dt_seconds,
            max_speed_m_per_day,
            config,
            descriptor_norm,
        )
        if not states:
            return None
        previous_time = layer.image_time
    return states[0]


def build_orb(args) -> cv2.ORB:
    return cv2.ORB_create(
        nfeatures=args.orb_nfeatures,
        scaleFactor=args.orb_scale_factor,
        nlevels=args.orb_nlevels,
        edgeThreshold=args.orb_edge_threshold,
        firstLevel=0,
        WTA_K=2,
        patchSize=args.orb_patch_size,
        scoreType=cv2.ORB_HARRIS_SCORE,
    )


def precompute_layers(
    coincidences: pd.DataFrame,
    args,
) -> tuple[dict[str, DescriptorLayer], float]:
    started = time.perf_counter()
    orb = build_orb(args)
    variant = DescriptorVariant(
        name="orb_graph_hamming",
        extractor_key="orb",
        norm="hamming",
        angle_mode=args.angle_mode,
        keypoint_size=args.keypoint_size,
        octave=args.octave,
    )
    layers = {}
    for row in coincidences.sort_values("image_time").drop_duplicates("image_filepath").itertuples(index=False):
        requested = candidate_grid(
            row.image_filepath,
            stride=args.grid_stride,
            border=args.grid_border,
            analysis_epsg=args.analysis_epsg,
        )
        image, _ = read_scene(row.image_filepath)
        angle = image_angle(row.image_filepath, args.analysis_epsg) if args.angle_mode == "geographic" else 0.0
        output_xy, descriptors = compute_descriptors(
            orb,
            image,
            requested.pixel_xy,
            variant,
            angle,
        )
        usable, descriptors = map_output_descriptors_to_grid(requested, output_xy, descriptors)
        layers[row.image_filepath] = DescriptorLayer(
            image_id=int(row.image_id),
            image_filename=row.image_filename,
            image_time=pd.Timestamp(row.image_time),
            grid=usable,
            descriptors=np.asarray(descriptors, dtype=np.uint8),
            spatial_index=cKDTree(usable.map_xy),
        )
    return layers, time.perf_counter() - started


def seed_descriptor(row: pd.Series, orb: cv2.ORB, variant: DescriptorVariant, args) -> np.ndarray | None:
    image, _ = read_scene(row.image_filepath)
    angle = float(row.image_angle_deg) if args.angle_mode == "geographic" else 0.0
    return exact_descriptor(
        orb,
        image,
        np.array([row.col, row.row]),
        variant,
        angle,
    )


def trajectory_rows(
    coincidences: pd.DataFrame,
    layer_lookup: dict[str, DescriptorLayer],
    config: GraphSearchConfig,
    args,
) -> list[dict]:
    orb = build_orb(args)
    variant = DescriptorVariant(
        name="orb_graph_hamming",
        extractor_key="orb",
        norm="hamming",
        angle_mode=args.angle_mode,
        keypoint_size=args.keypoint_size,
        octave=args.octave,
    )
    records = []
    path_column = trajectory_column(coincidences)
    for trajectory_id, group in coincidences.groupby(path_column, sort=True):
        group = group.sort_values("image_time").reset_index(drop=True)
        buoy_id = str(group.iloc[0]["buoy_id"])
        if len(group) < 2:
            continue
        descriptor = seed_descriptor(group.iloc[0], orb, variant, args)
        if descriptor is None:
            records.append(
                {
                    "config": config.name,
                    "buoy_id": buoy_id,
                    "trajectory_id": trajectory_id,
                    "status": "seed_unavailable",
                    "path_observations": len(group),
                }
            )
            continue
        layers = [layer_lookup[path] for path in group.image_filepath]
        state = search_layered_graph(
            layers,
            seed_xy=group.loc[0, ["x", "y"]].to_numpy(dtype=float),
            seed_descriptor=descriptor,
            max_speed_m_per_day=args.max_speed_m_per_day,
            config=config,
            descriptor_norm=args.descriptor_norm,
        )
        if state is None:
            records.append(
                {
                    "config": config.name,
                    "buoy_id": buoy_id,
                    "trajectory_id": trajectory_id,
                    "status": "graph_failed",
                    "path_observations": len(group),
                }
            )
            continue
        for index, (predicted, truth) in enumerate(zip(state.path_xy, group.itertuples(index=False))):
            skipped = predicted is None
            error = (
                math.nan
                if skipped
                else float(np.linalg.norm(predicted - np.array([truth.x, truth.y], dtype=float)))
            )
            records.append(
                {
                    "config": config.name,
                    "update_policy": config.update_policy,
                    "update_policy_description": UPDATE_POLICY_DESCRIPTIONS[
                        config.update_policy
                    ],
                    "beam_width": config.beam_width,
                    "branching": config.branching,
                    "motion_weight": config.motion_weight,
                    "displacement_weight": config.displacement_weight,
                    "preferred_speed_m_per_day": config.preferred_speed_m_per_day,
                    "excess_speed_weight": config.excess_speed_weight,
                    "buoy_id": buoy_id,
                    "trajectory_id": trajectory_id,
                    "status": "skipped" if skipped else "ok",
                    "path_observations": len(group),
                    "observation_index": index,
                    "image_id": truth.image_id,
                    "image_filename": truth.image_filename,
                    "image_time": truth.image_time,
                    "truth_x": truth.x,
                    "truth_y": truth.y,
                    "predicted_x": math.nan if skipped else float(predicted[0]),
                    "predicted_y": math.nan if skipped else float(predicted[1]),
                    "endpoint_error_m": error,
                    "cumulative_graph_cost": state.score,
                    "edge_cost": 0.0 if index == 0 else state.edge_costs[index - 1],
                    "candidate_count": 0 if index == 0 else state.candidate_counts[index - 1],
                    "descriptor_updated": False if index == 0 else state.descriptor_updates[index - 1],
                    "node_index": state.node_indices[index],
                }
            )
    return records


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config, all_config in records.groupby("config", sort=False):
        path_column = "trajectory_id" if "trajectory_id" in records else "buoy_id"
        valid = all_config[
            (all_config.status == "ok") & (all_config.observation_index > 0)
        ].copy()
        observed_path = all_config[all_config.status.isin(["ok", "skipped"])].copy()
        skipped = observed_path[observed_path.status == "skipped"]
        eligible_transitions = int(
            (
                all_config.groupby(path_column)["path_observations"].max() - 1
            ).clip(lower=0).sum()
        )
        final_rows = (
            observed_path[observed_path.path_observations >= 3]
            .sort_values("observation_index")
            .groupby(path_column)
            .tail(1)
        )
        final_errors = final_rows.loc[final_rows.status == "ok", "endpoint_error_m"]
        errors = valid.endpoint_error_m.to_numpy(dtype=float)
        long_paths = valid[valid.path_observations >= 3]
        long_errors = long_paths.endpoint_error_m.to_numpy(dtype=float)
        policy = next(item.update_policy for item in GRAPH_CONFIGS if item.name == config)
        rows.append(
            {
                "config": config,
                "descriptor_memory": UPDATE_POLICY_DESCRIPTIONS[policy],
                "eligible_transitions": eligible_transitions,
                "tracked_transitions": len(errors),
                "paths": int(all_config[path_column].nunique()),
                "graph_failed_paths": int(
                    all_config.loc[
                        all_config.status == "graph_failed", path_column
                    ].nunique()
                ),
                "seed_unavailable_paths": int(
                    all_config.loc[
                        all_config.status == "seed_unavailable", path_column
                    ].nunique()
                ),
                "skipped_observations": len(skipped),
                "observation_coverage_fraction": float(
                    len(errors) / max(eligible_transitions, 1)
                ),
                "median_error_m": float(np.median(errors)) if len(errors) else math.nan,
                "p90_error_m": float(np.percentile(errors, 90)) if len(errors) else math.nan,
                "max_error_m": float(np.max(errors)) if len(errors) else math.nan,
                "within_2km_fraction_tracked": float(np.mean(errors <= 2000.0))
                if len(errors)
                else math.nan,
                "within_2km_fraction_all": float(
                    np.count_nonzero(errors <= 2000.0) / max(eligible_transitions, 1)
                ),
                "within_5km_fraction_all": float(
                    np.count_nonzero(errors <= 5000.0) / max(eligible_transitions, 1)
                ),
                "catastrophic_50km_fraction_all": float(
                    np.count_nonzero(errors > 50000.0) / max(eligible_transitions, 1)
                ),
                "long_path_observations": len(long_errors),
                "long_path_median_error_m": float(np.median(long_errors)) if len(long_errors) else math.nan,
                "long_path_final_error_m": float(final_errors.median()) if len(final_errors) else math.nan,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("long_path_final_error_m")


def plot_errors(records: pd.DataFrame, output_path: Path) -> None:
    valid = records[
        (records.status == "ok")
        & (records.path_observations >= 3)
        & (records.observation_index > 0)
    ].copy()
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for config, group in valid.groupby("config", sort=False):
        group = group.sort_values("observation_index")
        axis.plot(
            group.observation_index,
            group.endpoint_error_m / 1000.0,
            marker="o",
            linewidth=1.2,
            label=config,
        )
    axis.axhline(2.0, color="tab:green", linestyle="--", linewidth=0.8)
    axis.axhline(5.0, color="tab:orange", linestyle="--", linewidth=0.8)
    axis.set_yscale("log")
    axis.set_xlabel("Observation index after exact-time seed")
    axis.set_ylabel("Buoy endpoint error (km, log scale)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(output_path: Path, summary: pd.DataFrame, args, elapsed: float) -> None:
    table = summary.copy()
    for column in ("median_error_m", "p90_error_m", "long_path_median_error_m", "long_path_final_error_m"):
        table[column.replace("_m", "_km")] = table[column] / 1000.0
    columns = [
        "config",
        "descriptor_memory",
        "median_error_km",
        "p90_error_km",
        "tracked_transitions",
        "eligible_transitions",
        "within_2km_fraction_all",
        "catastrophic_50km_fraction_all",
        "graph_failed_paths",
        "seed_unavailable_paths",
        "skipped_observations",
        "observation_coverage_fraction",
        "long_path_final_error_km",
    ]
    view = table[columns].copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()],
    ]
    output_path.write_text(
        f"""# Arctic ORB multi-frame candidate graph

Date: {pd.Timestamp.now(tz='UTC').date()}

- Standard VAE-preprocessed band; no additional preprocessing arm.
- Exact-time buoy location is used only for the first seed and evaluation.
- Candidate grid: {args.grid_stride} px; physics edge gate: {args.max_speed_m_per_day / 1000:.0f} km/day.
- ORB: `WTA_K=2`, {args.descriptor_norm}, nlevels={args.orb_nlevels}, patchSize={args.orb_patch_size}, supplied keypoint size={args.keypoint_size}, octave={args.octave}.
- Runtime: {elapsed:.2f} seconds.

## Results

{chr(10).join(lines)}

`greedy_rolling` means one candidate is selected using the descriptor from the
previous selected position; it is the deployable one-step baseline. The
`descriptor_memory` column spells out every update rule. Beam configurations
retain multiple hypotheses and select the lowest-cost complete path. The graph
has no access to target buoy positions. Short two-image buoy paths contribute
to the overall columns but not the long-path final-error column.

This first pass deliberately excludes correlation, MAGSAC, neighbourhood
deformation, cycle checks, descriptor thresholds, missing observations, and
confidence-gated updates. Those terms should be introduced individually after
the appearance-update policies are attributed.
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--buoys", type=Path)
    parser.add_argument(
        "--coincidences",
        type=Path,
        help="Normalized exact-time coincidence CSV from build_arctic_fixture_ledger.py.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--analysis-epsg", type=int, default=3413)
    parser.add_argument("--max-time-difference-minutes", type=float, default=60.0)
    parser.add_argument(
        "--outside-track-policy",
        choices=("error", "skip"),
        default="error",
        help="How to handle catalog matches outside the recorded buoy-track interval.",
    )
    parser.add_argument(
        "--invalid-support-policy",
        choices=("error", "skip"),
        default="error",
        help="How to handle exact buoy positions outside a scene or on invalid mask support.",
    )
    parser.add_argument("--max-speed-m-per-day", type=float, default=50000.0)
    parser.add_argument("--grid-stride", type=int, default=16)
    parser.add_argument("--grid-border", type=int, default=128)
    parser.add_argument("--orb-nfeatures", type=int, default=100)
    parser.add_argument("--orb-scale-factor", type=float, default=1.25)
    parser.add_argument("--orb-nlevels", type=int, default=5)
    parser.add_argument("--orb-edge-threshold", type=int, default=16)
    parser.add_argument("--orb-patch-size", type=int, default=64)
    parser.add_argument("--keypoint-size", type=float, default=31.0)
    parser.add_argument("--octave", type=int, default=5)
    parser.add_argument("--angle-mode", choices=("geographic", "zero"), default="geographic")
    parser.add_argument("--descriptor-norm", choices=("hamming", "hamming2"), default="hamming")
    parser.add_argument(
        "--experiment-split",
        help="Optional fixture split to run (for example development or validation).",
    )
    parser.add_argument(
        "--month-exclusive-buoys-only",
        action="store_true",
        help="Use only buoys observed in one month, eliminating buoy identity overlap between splits.",
    )
    parser.add_argument(
        "--graph-configs",
        default=None,
        help="Optional comma-separated subset of graph configuration names.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.coincidences is not None:
        if args.catalog is not None or args.buoys is not None:
            parser.error("Use either --coincidences or --catalog plus --buoys, not both.")
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
        coincidences["image_time"] = pd.to_datetime(coincidences["image_time"], utc=True)
    else:
        if args.catalog is None or args.buoys is None:
            parser.error("Provide --coincidences or both --catalog and --buoys.")
        catalog = load_catalog(args.catalog)
        coincidences = build_coincidences(
            catalog,
            args.buoys,
            max_time_difference_minutes=args.max_time_difference_minutes,
            outside_track_policy=args.outside_track_policy,
        )
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
    exact_time_count = len(coincidences)
    coincidences = annotate_coincidences(
        coincidences,
        args.analysis_epsg,
        outside_scene_policy=args.invalid_support_policy,
    )
    outside_scene_count = exact_time_count - len(coincidences)
    if coincidences.empty:
        raise ValueError("No exact-time buoy points fall inside the selected SAR scenes.")
    invalid_mask = (coincidences["mask_value"] >= 2) | ~np.isfinite(
        coincidences[["col", "row"]]
    ).all(axis=1)
    invalid_mask_count = int(invalid_mask.sum())
    if invalid_mask_count:
        if args.invalid_support_policy == "error":
            raise ValueError(
                f"{invalid_mask_count} coincident buoy points fall on invalid raster support"
            )
        coincidences = coincidences.loc[~invalid_mask].reset_index(drop=True)
    layers, precompute_seconds = precompute_layers(coincidences, args)
    rows = []
    config_timings = []
    requested_configs = None if args.graph_configs is None else set(args.graph_configs.split(","))
    active_configs = tuple(
        config for config in GRAPH_CONFIGS if requested_configs is None or config.name in requested_configs
    )
    if requested_configs is not None:
        missing_configs = requested_configs - {config.name for config in active_configs}
        if missing_configs:
            raise ValueError(f"Unknown graph configurations: {sorted(missing_configs)}")
    for config in active_configs:
        config_started = time.perf_counter()
        rows.extend(trajectory_rows(coincidences, layers, config, args))
        config_timings.append(
            {
                "config": config.name,
                "seconds": time.perf_counter() - config_started,
            }
        )
    records = pd.DataFrame.from_records(rows)
    summary = summarize(records)
    elapsed = time.perf_counter() - started

    coincidences.to_csv(args.out_dir / "coincidences.csv", index=False)
    records.to_csv(args.out_dir / "trajectory_results.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    pd.DataFrame(config_timings).to_csv(args.out_dir / "timings.csv", index=False)
    plot_errors(records, args.out_dir / "long_path_errors.png")
    write_report(args.out_dir / "report.md", summary, args, elapsed)
    manifest = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "graph_configs": [config.__dict__ for config in active_configs],
        "precompute_seconds": precompute_seconds,
        "elapsed_seconds": elapsed,
        "coincidences": len(coincidences),
        "exact_time_coincidences_before_spatial_filter": exact_time_count,
        "outside_scene_records": outside_scene_count,
        "invalid_mask_records": invalid_mask_count,
        "buoys": int(coincidences.buoy_id.nunique()),
        "trajectories": int(coincidences[trajectory_column(coincidences)].nunique()),
        "images": int(coincidences.image_filepath.nunique()),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps({"precompute_seconds": precompute_seconds, "elapsed_seconds": elapsed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
