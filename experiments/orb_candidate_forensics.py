#!/usr/bin/env python3
"""Replay the frozen ORB graph and attribute buoy-tracking failures.

Buoy truth is used only after each graph layer has expanded and pruned. The
archive distinguishes raster/grid coverage, physics-gate exclusion, candidate
ranking, beam pruning, final path selection, false descriptor updates, and
missed recovery opportunities. Production LiMOSAT is not modified.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from buoy_descriptor_benchmark import descriptor_distances, read_scene
from buoy_patch_evolution import (
    map_aligned_patch,
    native_patch,
    patch_pair_metrics,
    patch_statistics,
)
from orb_multiframe_graph import (
    GraphSearchConfig,
    PathState,
    appearance_costs,
    build_orb,
    expand_graph_layer,
    precompute_layers,
    seed_descriptor,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_ROOT = ROOT / "results/orb_multiframe_graph/final_arctic_matrix"
DEFAULT_PATCH_ROOT = ROOT / "results/buoy_patch_evolution/q2q98_clahe25"


@dataclass
class LayerTrace:
    parents: list[PathState]
    expanded: list[PathState]
    retained: list[PathState]


@dataclass
class SearchTrace:
    final_state: PathState | None
    layers: list[LayerTrace]


def initial_state(seed_xy: np.ndarray, descriptor: np.ndarray) -> PathState:
    xy = np.asarray(seed_xy, dtype=float)
    return PathState(
        score=0.0,
        position_xy=xy,
        velocity_xy_per_second=None,
        descriptors=(np.asarray(descriptor).copy(),),
        path_xy=(xy.copy(),),
        node_indices=(-1,),
        edge_costs=(),
        candidate_counts=(),
        descriptor_updates=(),
        elapsed_since_position_seconds=0.0,
        consecutive_skips=0,
    )


def search_with_trace(
    layers,
    seed_xy: np.ndarray,
    descriptor: np.ndarray,
    max_speed_m_per_day: float,
    config: GraphSearchConfig,
    descriptor_norm: str,
) -> SearchTrace:
    """Run the graph unchanged while retaining states before and after pruning."""
    states = [initial_state(seed_xy, descriptor)]
    traces: list[LayerTrace] = []
    previous_time = layers[0].image_time
    unpruned_config = replace(config, beam_width=1_000_000)
    for layer in layers[1:]:
        dt_seconds = (layer.image_time - previous_time).total_seconds()
        if dt_seconds <= 0:
            raise ValueError("Image times must increase within a graph.")
        expanded = expand_graph_layer(
            states,
            layer,
            dt_seconds,
            max_speed_m_per_day,
            unpruned_config,
            descriptor_norm,
        )
        retained = expanded[: config.beam_width]
        traces.append(LayerTrace(states, expanded, retained))
        states = retained
        if not states:
            return SearchTrace(None, traces)
        previous_time = layer.image_time
    return SearchTrace(states[0], traces)


def state_with_prefix(states: list[PathState], prefix: tuple[int, ...]) -> PathState | None:
    for state in states:
        if state.node_indices == prefix:
            return state
    return None


def selected_states(trace: SearchTrace, layer_number: int) -> tuple[PathState | None, PathState | None, str]:
    """Return parent and selected state for a 1-based target layer."""
    layer_trace = trace.layers[layer_number - 1]
    if trace.final_state is not None:
        parent_prefix = trace.final_state.node_indices[:layer_number]
        selected_prefix = trace.final_state.node_indices[: layer_number + 1]
        return (
            state_with_prefix(layer_trace.parents, parent_prefix),
            state_with_prefix(layer_trace.retained, selected_prefix),
            "lowest_cost_complete_path",
        )
    if layer_trace.retained:
        selected = layer_trace.retained[0]
        parent = state_with_prefix(layer_trace.parents, selected.node_indices[:-1])
        return parent, selected, "online_best_before_graph_failure"
    return (layer_trace.parents[0] if layer_trace.parents else None), None, "graph_exhaustion"


def stable_ranks(values: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(values), dtype=int)
    ranks[np.argsort(values, kind="stable")] = np.arange(1, len(values) + 1)
    return ranks


def candidate_cost_table(
    parent: PathState,
    layer,
    dt_seconds: float,
    max_speed_m_per_day: float,
    config: GraphSearchConfig,
    descriptor_norm: str,
) -> dict[str, np.ndarray | float]:
    effective_dt = dt_seconds + parent.elapsed_since_position_seconds
    radius_m = max_speed_m_per_day * effective_dt / 86400.0
    gate_indices = np.asarray(
        sorted(layer.spatial_index.query_ball_point(parent.position_xy, radius_m)),
        dtype=int,
    )
    if parent.velocity_xy_per_second is None:
        motion_prediction = parent.position_xy.copy()
    else:
        motion_prediction = (
            parent.position_xy + parent.velocity_xy_per_second * effective_dt
        )
    if len(gate_indices) == 0:
        empty = np.empty(0, dtype=float)
        return {
            "effective_dt_seconds": effective_dt,
            "radius_m": radius_m,
            "prediction_xy": motion_prediction,
            "gate_indices": gate_indices,
            "appearance": empty,
            "anchor": empty,
            "rolling": empty,
            "edge": empty,
            "edge_ranks": np.empty(0, dtype=int),
            "appearance_ranks": np.empty(0, dtype=int),
            "anchor_ranks": np.empty(0, dtype=int),
            "rolling_ranks": np.empty(0, dtype=int),
            "spatial_distance_m": empty,
            "prediction_error_m": empty,
        }
    descriptors = layer.descriptors[gate_indices]
    appearance = appearance_costs(
        parent.descriptors, descriptors, config.update_policy, descriptor_norm
    )
    scale = descriptors.shape[1] * (8 if descriptor_norm == "hamming" else 4)
    anchor = descriptor_distances(
        parent.descriptors[0], descriptors, descriptor_norm
    ).astype(float) / scale
    rolling = descriptor_distances(
        parent.descriptors[-1], descriptors, descriptor_norm
    ).astype(float) / scale
    spatial = np.linalg.norm(layer.grid.map_xy[gate_indices] - parent.position_xy, axis=1)
    prediction_error = np.linalg.norm(
        layer.grid.map_xy[gate_indices] - motion_prediction, axis=1
    )
    motion = (
        np.zeros(len(gate_indices), dtype=float)
        if parent.velocity_xy_per_second is None or config.motion_weight == 0
        else prediction_error / max(radius_m, 1.0)
    )
    edge = (
        appearance
        + config.motion_weight * motion
        + config.displacement_weight * spatial / max(radius_m, 1.0)
    )
    return {
        "effective_dt_seconds": effective_dt,
        "radius_m": radius_m,
        "prediction_xy": motion_prediction,
        "gate_indices": gate_indices,
        "appearance": appearance,
        "anchor": anchor,
        "rolling": rolling,
        "edge": edge,
        "edge_ranks": stable_ranks(edge),
        "appearance_ranks": stable_ranks(appearance),
        "anchor_ranks": stable_ranks(anchor),
        "rolling_ranks": stable_ranks(rolling),
        "spatial_distance_m": spatial,
        "prediction_error_m": prediction_error,
    }


def local_index(table: dict, node_index: int) -> int | None:
    matches = np.flatnonzero(table["gate_indices"] == node_index)
    return int(matches[0]) if len(matches) else None


def add_provisional_costs(
    table: dict,
    layer,
    provisional_descriptor: np.ndarray,
    descriptor_norm: str,
) -> None:
    gate_indices = table["gate_indices"]
    if len(gate_indices) == 0:
        table["provisional"] = np.empty(0, dtype=float)
        table["provisional_ranks"] = np.empty(0, dtype=int)
        return
    candidates = layer.descriptors[gate_indices]
    scale = candidates.shape[1] * (8 if descriptor_norm == "hamming" else 4)
    provisional = descriptor_distances(
        provisional_descriptor,
        candidates,
        descriptor_norm,
    ).astype(float) / scale
    table["provisional"] = provisional
    table["provisional_ranks"] = stable_ranks(provisional)


def minimum_state_error(states: list[PathState], truth_xy: np.ndarray) -> float:
    observed = [
        state
        for state in states
        if state.node_indices[-1] != -2 and state.path_xy[-1] is not None
    ]
    if not observed:
        return math.nan
    return float(min(np.linalg.norm(state.position_xy - truth_xy) for state in observed))


def aggregate_parent_access(
    parents: list[PathState],
    layer,
    truth_node_index: int,
    dt_seconds: float,
    max_speed_m_per_day: float,
    config: GraphSearchConfig,
    descriptor_norm: str,
) -> tuple[bool, float, float]:
    edge_ranks = []
    anchor_ranks = []
    for parent in parents:
        table = candidate_cost_table(
            parent,
            layer,
            dt_seconds,
            max_speed_m_per_day,
            config,
            descriptor_norm,
        )
        index = local_index(table, truth_node_index)
        if index is not None:
            edge_ranks.append(float(table["edge_ranks"][index]))
            anchor_ranks.append(float(table["anchor_ranks"][index]))
    return (
        bool(edge_ranks),
        min(edge_ranks) if edge_ranks else math.nan,
        min(anchor_ranks) if anchor_ranks else math.nan,
    )


def classify_failure(row: dict, branching: int, error_threshold_m: float) -> str:
    if row["path_status"] == "seed_unavailable":
        return "seed_descriptor_unavailable"
    error = row.get("selected_error_m", math.nan)
    if np.isfinite(error) and error <= error_threshold_m:
        return "success"
    if row["truth_grid_distance_m"] > error_threshold_m:
        return (
            "candidate_border_exclusion"
            if row.get("raster_border_distance_px", math.inf)
            < row.get("candidate_grid_border_px", -math.inf)
            else "candidate_descriptor_coverage_failure"
        )
    if not row["truth_accessible_from_any_parent"]:
        return "state_gate_exclusion"
    if not row["truth_in_selected_parent_gate"]:
        return "selected_path_gate_exclusion"
    truth_rank = row.get("truth_edge_rank", math.nan)
    if np.isfinite(truth_rank) and truth_rank > branching:
        return (
            "observation_appearance_ranking"
            if row["truth_appearance_hard"]
            else "candidate_ranking_failure"
        )
    expanded_error = row.get("expanded_min_truth_error_m", math.nan)
    retained_error = row.get("retained_min_truth_error_m", math.nan)
    if np.isfinite(expanded_error) and expanded_error <= error_threshold_m:
        if not np.isfinite(retained_error) or retained_error > error_threshold_m:
            return "beam_pruning_failure"
    if np.isfinite(retained_error) and retained_error <= error_threshold_m:
        return "final_path_selection_failure"
    if row["path_status"] in {"graph_exhaustion", "downstream_graph_failure"}:
        return "graph_exhaustion_other"
    return "path_selection_or_scoring_failure"


def truth_metric_lookup(patch_root: Path, sequence: str) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    sequence_dir = patch_root / sequence
    transitions = pd.read_csv(
        sequence_dir / "transitions.csv",
        dtype={"buoy_id": str},
    ).set_index("target_observation_id")
    with np.load(sequence_dir / "patches.npz") as archive:
        ids = archive["observation_id"].astype(str)
        patches = archive["map_5000m"][:, ::2, ::2]
        masks = archive["map_5000m_valid"][:, ::2, ::2]
    patch_lookup = {
        observation_id: (patches[index], masks[index])
        for index, observation_id in enumerate(ids)
    }
    return transitions, patch_lookup


def graph_args(manifest: dict) -> SimpleNamespace:
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


def config_from_manifest(manifest: dict, name: str) -> GraphSearchConfig:
    matches = [config for config in manifest["graph_configs"] if config["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Configuration {name!r} is not unique in frozen manifest.")
    return GraphSearchConfig(**matches[0])


def add_candidate_records(
    records: list[dict],
    descriptors: list[np.ndarray],
    transition_id: str,
    sequence: str,
    config: GraphSearchConfig,
    buoy_id: str,
    image_row,
    parent: PathState,
    selected: PathState | None,
    layer_trace: LayerTrace,
    layer,
    table: dict,
    truth_xy: np.ndarray,
    truth_node_index: int,
    top_candidates: int,
) -> None:
    if not len(table["gate_indices"]):
        return
    selected_node = -2 if selected is None else selected.node_indices[-1]
    required = set(np.flatnonzero(table["edge_ranks"] <= top_candidates).tolist())
    truth_local = local_index(table, truth_node_index)
    selected_local = local_index(table, selected_node)
    if truth_local is not None:
        required.add(truth_local)
    if selected_local is not None:
        required.add(selected_local)
    retained_prefixes = {state.node_indices for state in layer_trace.retained}
    for index in sorted(required, key=lambda item: table["edge_ranks"][item]):
        node_index = int(table["gate_indices"][index])
        map_xy = layer.grid.map_xy[node_index]
        pixel_xy = layer.grid.pixel_xy[node_index]
        candidate_id = f"{transition_id}:node{node_index}"
        records.append(
            {
                "candidate_id": candidate_id,
                "transition_id": transition_id,
                "sequence": sequence,
                "config": config.name,
                "buoy_id": buoy_id,
                "image_id": int(image_row.image_id),
                "image_filepath": image_row.image_filepath,
                "node_index": node_index,
                "map_x": float(map_xy[0]),
                "map_y": float(map_xy[1]),
                "pixel_col": float(pixel_xy[0]),
                "pixel_row": float(pixel_xy[1]),
                "truth_error_m": float(np.linalg.norm(map_xy - truth_xy)),
                "parent_displacement_m": float(table["spatial_distance_m"][index]),
                "motion_prediction_error_m": float(table["prediction_error_m"][index]),
                "appearance_cost": float(table["appearance"][index]),
                "anchor_cost": float(table["anchor"][index]),
                "rolling_cost": float(table["rolling"][index]),
                "provisional_previous_cost": (
                    float(table["provisional"][index])
                    if "provisional" in table
                    else math.nan
                ),
                "edge_cost": float(table["edge"][index]),
                "appearance_rank": int(table["appearance_ranks"][index]),
                "anchor_rank": int(table["anchor_ranks"][index]),
                "rolling_rank": int(table["rolling_ranks"][index]),
                "provisional_previous_rank": (
                    int(table["provisional_ranks"][index])
                    if "provisional_ranks" in table
                    else math.nan
                ),
                "edge_rank": int(table["edge_ranks"][index]),
                "is_truth_nearest_grid_node": node_index == truth_node_index,
                "is_selected_final_path_node": node_index == selected_node,
                "retained_from_selected_parent": (
                    (*parent.node_indices, node_index) in retained_prefixes
                ),
            }
        )
        descriptors.append(layer.descriptors[node_index].astype(np.uint8))


def replay_path(
    sequence: str,
    group: pd.DataFrame,
    layer_lookup,
    config: GraphSearchConfig,
    args,
    truth_metrics: pd.DataFrame,
    top_candidates: int,
    error_threshold_m: float,
) -> tuple[list[dict], list[dict], list[np.ndarray], SearchTrace | None]:
    group = group.sort_values("image_time").reset_index(drop=True)
    orb = build_orb(args)
    variant = SimpleNamespace(
        name="orb_graph_hamming",
        extractor_key="orb",
        norm="hamming",
        angle_mode=args.angle_mode,
        keypoint_size=args.keypoint_size,
        octave=args.octave,
    )
    descriptor = seed_descriptor(group.iloc[0], orb, variant, args)
    transitions: list[dict] = []
    candidates: list[dict] = []
    candidate_descriptors: list[np.ndarray] = []
    buoy_id = str(group.buoy_id.iloc[0])
    if descriptor is None:
        for layer_number, target in enumerate(group.itertuples(index=False), start=0):
            if layer_number == 0:
                continue
            transition_id = f"{sequence}:{config.name}:{buoy_id}:{int(target.image_id)}"
            record = {
                "transition_id": transition_id,
                "sequence": sequence,
                "config": config.name,
                "buoy_id": buoy_id,
                "observation_index": layer_number,
                "image_id": int(target.image_id),
                "image_time": target.image_time,
                "image_filepath": target.image_filepath,
                "truth_x": float(target.x),
                "truth_y": float(target.y),
                "raster_border_distance_px": float(
                    min(
                        target.col,
                        target.row,
                        target.image_width - 1 - target.col,
                        target.image_height - 1 - target.row,
                    )
                ),
                "candidate_grid_border_px": args.grid_border,
                "path_status": "seed_unavailable",
                "selected_error_m": math.nan,
                "truth_grid_distance_m": math.nan,
                "truth_accessible_from_any_parent": False,
                "truth_in_selected_parent_gate": False,
                "truth_appearance_hard": True,
            }
            record["primary_failure_mechanism"] = classify_failure(
                record, config.branching, error_threshold_m
            )
            transitions.append(record)
        return transitions, candidates, candidate_descriptors, None

    layers = [layer_lookup[path] for path in group.image_filepath]
    trace = search_with_trace(
        layers,
        group.loc[0, ["x", "y"]].to_numpy(dtype=float),
        descriptor,
        args.max_speed_m_per_day,
        config,
        args.descriptor_norm,
    )
    for layer_number, target in enumerate(group.itertuples(index=False), start=0):
        if layer_number == 0:
            continue
        transition_id = f"{sequence}:{config.name}:{buoy_id}:{int(target.image_id)}"
        observation_id = f"{sequence}:{buoy_id}:{int(target.image_id)}"
        truth_row = truth_metrics.loc[observation_id]
        truth_xy = np.array([target.x, target.y], dtype=float)
        layer = layers[layer_number]
        _, truth_node_index = layer.spatial_index.query(truth_xy, k=1)
        truth_node_index = int(truth_node_index)
        truth_grid_distance = float(
            np.linalg.norm(layer.grid.map_xy[truth_node_index] - truth_xy)
        )
        truth_candidate_count_2km = len(
            layer.spatial_index.query_ball_point(truth_xy, error_threshold_m)
        )
        truth_candidate_count_5km = len(
            layer.spatial_index.query_ball_point(truth_xy, 5000.0)
        )
        if layer_number > len(trace.layers):
            record = {
                "transition_id": transition_id,
                "sequence": sequence,
                "config": config.name,
                "buoy_id": buoy_id,
                "observation_index": layer_number,
                "image_id": int(target.image_id),
                "image_time": target.image_time,
                "image_filepath": target.image_filepath,
                "truth_x": float(target.x),
                "truth_y": float(target.y),
                "raster_border_distance_px": float(
                    min(
                        target.col,
                        target.row,
                        target.image_width - 1 - target.col,
                        target.image_height - 1 - target.row,
                    )
                ),
                "candidate_grid_border_px": args.grid_border,
                "path_status": "downstream_graph_failure",
                "selected_error_m": math.nan,
                "truth_grid_distance_m": truth_grid_distance,
                "truth_candidate_count_2km": truth_candidate_count_2km,
                "truth_candidate_count_5km": truth_candidate_count_5km,
                "truth_accessible_from_any_parent": False,
                "truth_in_selected_parent_gate": False,
                "truth_appearance_hard": bool(
                    not np.isfinite(truth_row.orb_anchor_hamming_norm)
                    or truth_row.orb_anchor_hamming_norm > config.update_max_cost
                    or truth_row.map_5000m_prev_ncc < 0.25
                ),
            }
            record["primary_failure_mechanism"] = classify_failure(
                record, config.branching, error_threshold_m
            )
            transitions.append(record)
            continue

        layer_trace = trace.layers[layer_number - 1]
        parent, selected, selection_basis = selected_states(trace, layer_number)
        dt_seconds = (layer.image_time - layers[layer_number - 1].image_time).total_seconds()
        any_access, best_truth_rank, best_anchor_rank = aggregate_parent_access(
            layer_trace.parents,
            layer,
            truth_node_index,
            dt_seconds,
            args.max_speed_m_per_day,
            config,
            args.descriptor_norm,
        )
        if parent is None:
            table = None
            truth_local = None
            selected_error = math.nan
            selected_node = -2
            prediction_xy = np.array([math.nan, math.nan])
        else:
            table = candidate_cost_table(
                parent,
                layer,
                dt_seconds,
                args.max_speed_m_per_day,
                config,
                args.descriptor_norm,
            )
            previous_node = parent.node_indices[-1]
            provisional_descriptor = (
                layers[layer_number - 1].descriptors[previous_node]
                if previous_node >= 0
                else parent.descriptors[-1]
            )
            add_provisional_costs(
                table,
                layer,
                provisional_descriptor,
                args.descriptor_norm,
            )
            truth_local = local_index(table, truth_node_index)
            selected_node = -2 if selected is None else selected.node_indices[-1]
            selected_error = (
                math.nan
                if selected is None or selected_node == -2
                else float(np.linalg.norm(selected.position_xy - truth_xy))
            )
            prediction_xy = np.asarray(table["prediction_xy"], dtype=float)
        source_truth_xy = group.loc[layer_number - 1, ["x", "y"]].to_numpy(
            dtype=float
        )
        parent_source_error = (
            math.nan
            if parent is None
            else float(np.linalg.norm(parent.position_xy - source_truth_xy))
        )
        record = {
            "transition_id": transition_id,
            "sequence": sequence,
            "config": config.name,
            "buoy_id": buoy_id,
            "observation_index": layer_number,
            "image_id": int(target.image_id),
            "image_time": target.image_time,
            "image_filepath": target.image_filepath,
            "truth_x": float(target.x),
            "truth_y": float(target.y),
            "raster_border_distance_px": float(
                min(
                    target.col,
                    target.row,
                    target.image_width - 1 - target.col,
                    target.image_height - 1 - target.row,
                )
            ),
            "candidate_grid_border_px": args.grid_border,
            "selection_basis": selection_basis,
            "path_status": (
                "graph_exhaustion"
                if selected is None
                else "skipped" if selected_node == -2 else "ok"
            ),
            "selected_node_index": selected_node,
            "selected_x": (
                math.nan if selected is None or selected_node == -2 else float(selected.position_xy[0])
            ),
            "selected_y": (
                math.nan if selected is None or selected_node == -2 else float(selected.position_xy[1])
            ),
            "selected_error_m": selected_error,
            "motion_predicted_x": float(prediction_xy[0]),
            "motion_predicted_y": float(prediction_xy[1]),
            "motion_prediction_error_m": (
                float(np.linalg.norm(prediction_xy - truth_xy))
                if np.all(np.isfinite(prediction_xy))
                else math.nan
            ),
            "parent_source_error_m": parent_source_error,
            "parent_descriptor_was_updated": bool(
                parent is not None
                and len(parent.descriptor_updates)
                and parent.descriptor_updates[-1]
            ),
            "physics_radius_m": math.nan if table is None else float(table["radius_m"]),
            "selected_parent_candidate_count": (
                0 if table is None else len(table["gate_indices"])
            ),
            "beam_parent_states": len(layer_trace.parents),
            "expanded_states": len(layer_trace.expanded),
            "retained_states": len(layer_trace.retained),
            "expanded_min_truth_error_m": minimum_state_error(
                layer_trace.expanded, truth_xy
            ),
            "retained_min_truth_error_m": minimum_state_error(
                layer_trace.retained, truth_xy
            ),
            "truth_node_index": truth_node_index,
            "truth_grid_distance_m": truth_grid_distance,
            "truth_candidate_count_2km": truth_candidate_count_2km,
            "truth_candidate_count_5km": truth_candidate_count_5km,
            "truth_accessible_from_any_parent": any_access,
            "best_truth_edge_rank_any_parent": best_truth_rank,
            "best_truth_anchor_rank_any_parent": best_anchor_rank,
            "truth_in_selected_parent_gate": truth_local is not None,
            "truth_edge_rank": (
                math.nan if truth_local is None else int(table["edge_ranks"][truth_local])
            ),
            "truth_anchor_rank": (
                math.nan if truth_local is None else int(table["anchor_ranks"][truth_local])
            ),
            "truth_rolling_rank": (
                math.nan if truth_local is None else int(table["rolling_ranks"][truth_local])
            ),
            "truth_provisional_previous_rank": (
                math.nan
                if truth_local is None
                else int(table["provisional_ranks"][truth_local])
            ),
            "truth_grid_edge_cost": (
                math.nan if truth_local is None else float(table["edge"][truth_local])
            ),
            "truth_grid_anchor_cost": (
                math.nan if truth_local is None else float(table["anchor"][truth_local])
            ),
            "truth_grid_rolling_cost": (
                math.nan if truth_local is None else float(table["rolling"][truth_local])
            ),
            "truth_grid_provisional_previous_cost": (
                math.nan
                if truth_local is None
                else float(table["provisional"][truth_local])
            ),
            "descriptor_updated": (
                False if selected is None else bool(selected.descriptor_updates[-1])
            ),
            "truth_orb_prev_hamming_norm": float(truth_row.orb_prev_hamming_norm),
            "truth_orb_anchor_hamming_norm": float(truth_row.orb_anchor_hamming_norm),
            "truth_patch_prev_ncc": float(truth_row.map_5000m_prev_ncc),
            "truth_patch_anchor_ncc": float(truth_row.map_5000m_anchor_ncc),
            "truth_patch_prev_histogram_js_distance": float(
                truth_row.map_5000m_prev_histogram_js_distance
            ),
        }
        record["truth_appearance_hard"] = bool(
            not np.isfinite(record["truth_orb_anchor_hamming_norm"])
            or record["truth_orb_anchor_hamming_norm"] > config.update_max_cost
            or record["truth_patch_prev_ncc"] < 0.25
        )
        record["false_update"] = bool(
            record["descriptor_updated"]
            and (not np.isfinite(selected_error) or selected_error > error_threshold_m)
        )
        record["provisional_bridge_signature"] = bool(
            np.isfinite(parent_source_error)
            and parent_source_error <= error_threshold_m
            and not record["parent_descriptor_was_updated"]
            and np.isfinite(record["truth_provisional_previous_rank"])
            and record["truth_provisional_previous_rank"] <= config.branching
            and (
                not np.isfinite(record["truth_edge_rank"])
                or record["truth_edge_rank"] > config.branching
            )
        )
        record["primary_failure_mechanism"] = classify_failure(
            record, config.branching, error_threshold_m
        )
        transitions.append(record)
        if parent is not None and table is not None:
            add_candidate_records(
                candidates,
                candidate_descriptors,
                transition_id,
                sequence,
                config,
                buoy_id,
                target,
                parent,
                selected,
                layer_trace,
                layer,
                table,
                truth_xy,
                truth_node_index,
                top_candidates,
            )
    return transitions, candidates, candidate_descriptors, trace


def add_temporal_labels(records: pd.DataFrame, branching: int, error_threshold_m: float) -> pd.DataFrame:
    records = records.sort_values(
        ["sequence", "config", "buoy_id", "observation_index"]
    ).reset_index(drop=True)
    records["false_update"] = records.false_update.fillna(False).astype(bool)
    records["prior_false_update"] = False
    records["probable_update_poisoning"] = False
    records["next_observation_clearer"] = False
    records["recovery_opportunity_next"] = False
    records["recovered_next"] = False
    records["missed_recovery_next"] = False
    for _, group in records.groupby(["sequence", "config", "buoy_id"], sort=False):
        indexes = group.index.to_numpy()
        for current_index, next_index in zip(indexes[:-1], indexes[1:]):
            current = records.loc[current_index]
            following = records.loc[next_index]
            records.loc[next_index, "prior_false_update"] = bool(current.false_update)
            poisoning = bool(
                current.false_update
                and following.truth_in_selected_parent_gate
                and np.isfinite(following.truth_anchor_rank)
                and following.truth_anchor_rank <= branching
                and (
                    not np.isfinite(following.truth_edge_rank)
                    or following.truth_edge_rank > branching
                )
            )
            records.loc[next_index, "probable_update_poisoning"] = poisoning
            current_failure = (
                not np.isfinite(current.selected_error_m)
                or current.selected_error_m > error_threshold_m
            )
            next_anchor = following.truth_orb_anchor_hamming_norm
            current_anchor = current.truth_orb_anchor_hamming_norm
            clearer = bool(
                current_failure
                and np.isfinite(next_anchor)
                and next_anchor <= 0.35
                and (
                    not np.isfinite(current_anchor)
                    or next_anchor <= current_anchor - 0.05
                )
            )
            opportunity = bool(
                clearer
                and following.truth_accessible_from_any_parent
                and following.best_truth_anchor_rank_any_parent <= branching
            )
            next_success = bool(
                np.isfinite(following.selected_error_m)
                and following.selected_error_m <= error_threshold_m
            )
            records.loc[current_index, "next_observation_clearer"] = clearer
            records.loc[current_index, "recovery_opportunity_next"] = opportunity
            records.loc[current_index, "recovered_next"] = opportunity and next_success
            records.loc[current_index, "missed_recovery_next"] = opportunity and not next_success
    return records


def validate_against_frozen_results(
    records: pd.DataFrame,
    frozen_results: pd.DataFrame,
    sequence: str,
    config: str,
) -> dict:
    expected = frozen_results[
        (frozen_results.config == config)
        & frozen_results.status.isin(["ok", "skipped"])
        & (frozen_results.observation_index > 0)
    ].copy()
    completed_buoys = set(expected.buoy_id.astype(str))
    actual = records[
        (records.sequence == sequence)
        & (records.config == config)
        & records.path_status.isin(["ok", "skipped"])
        & records.buoy_id.astype(str).isin(completed_buoys)
    ].copy()
    merged = actual.merge(
        expected[
            [
                "buoy_id",
                "image_id",
                "status",
                "node_index",
                "predicted_x",
                "predicted_y",
            ]
        ],
        on=["buoy_id", "image_id"],
        how="outer",
        indicator=True,
    )
    if np.any(merged._merge != "both"):
        raise AssertionError("Forensic replay does not reproduce frozen trajectory rows.")
    comparable = merged[(merged.path_status == "ok") & (merged.status == "ok")]
    position_difference = np.hypot(
        comparable.selected_x - comparable.predicted_x,
        comparable.selected_y - comparable.predicted_y,
    )
    node_match = comparable.selected_node_index == comparable.node_index
    if np.any(position_difference > 1.0e-6) or not np.all(node_match):
        raise AssertionError("Forensic replay differs from frozen selected graph path.")
    return {
        "sequence": sequence,
        "config": config,
        "frozen_rows": len(expected),
        "replayed_rows": len(actual),
        "maximum_position_difference_m": (
            float(position_difference.max()) if len(position_difference) else 0.0
        ),
        "node_indices_match": bool(np.all(node_match)),
    }


def extract_patch_archives(
    transitions: pd.DataFrame,
    candidates: pd.DataFrame,
    truth_patch_lookup: dict[str, tuple[np.ndarray, np.ndarray]],
    analysis_epsg: int,
    width_m: float,
    map_pixels: int,
    native_pixels: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    count = len(transitions)
    shape = (count, map_pixels, map_pixels)
    transition_arrays = {
        "truth": np.zeros(shape, dtype=np.uint8),
        "truth_valid": np.zeros(shape, dtype=bool),
        "selected": np.zeros(shape, dtype=np.uint8),
        "selected_valid": np.zeros(shape, dtype=bool),
        "motion_predicted": np.zeros(shape, dtype=np.uint8),
        "motion_predicted_valid": np.zeros(shape, dtype=bool),
    }
    metric_records = []
    for image_path, group in transitions.groupby("image_filepath", sort=False):
        image, mask = read_scene(image_path)
        for index, row in group.iterrows():
            observation_id = f"{row.sequence}:{row.buoy_id}:{int(row.image_id)}"
            truth_patch, truth_valid = truth_patch_lookup[observation_id]
            transition_arrays["truth"][index] = truth_patch
            transition_arrays["truth_valid"][index] = truth_valid
            selected_patch = np.zeros((map_pixels, map_pixels), dtype=np.uint8)
            selected_valid = np.zeros((map_pixels, map_pixels), dtype=bool)
            if np.isfinite(row.selected_x) and np.isfinite(row.selected_y):
                selected_patch, selected_valid = map_aligned_patch(
                    image_path,
                    image,
                    mask,
                    row.selected_x,
                    row.selected_y,
                    width_m,
                    map_pixels,
                    analysis_epsg,
                )
            predicted_patch = np.zeros((map_pixels, map_pixels), dtype=np.uint8)
            predicted_valid = np.zeros((map_pixels, map_pixels), dtype=bool)
            if np.isfinite(row.motion_predicted_x) and np.isfinite(row.motion_predicted_y):
                predicted_patch, predicted_valid = map_aligned_patch(
                    image_path,
                    image,
                    mask,
                    row.motion_predicted_x,
                    row.motion_predicted_y,
                    width_m,
                    map_pixels,
                    analysis_epsg,
                )
            transition_arrays["selected"][index] = selected_patch
            transition_arrays["selected_valid"][index] = selected_valid
            transition_arrays["motion_predicted"][index] = predicted_patch
            transition_arrays["motion_predicted_valid"][index] = predicted_valid
            selected_stats = patch_statistics(selected_patch, selected_valid)
            predicted_stats = patch_statistics(predicted_patch, predicted_valid)
            truth_stats = patch_statistics(truth_patch, truth_valid)
            selected_truth = patch_pair_metrics(
                truth_patch,
                selected_patch,
                truth_valid,
                selected_valid,
            )
            metric_records.append(
                {
                    "index": index,
                    **{f"truth_patch_{key}": value for key, value in truth_stats.items()},
                    **{f"selected_patch_{key}": value for key, value in selected_stats.items()},
                    **{f"motion_predicted_patch_{key}": value for key, value in predicted_stats.items()},
                    **{f"selected_vs_truth_{key}": value for key, value in selected_truth.items()},
                }
            )
    metrics = pd.DataFrame.from_records(metric_records).set_index("index")
    for column in metrics:
        transitions.loc[metrics.index, column] = metrics[column]

    candidate_shape = (len(candidates), native_pixels, native_pixels)
    candidate_arrays = {
        "patch": np.zeros(candidate_shape, dtype=np.uint8),
        "valid": np.zeros(candidate_shape, dtype=bool),
    }
    candidate_metrics = []
    for image_path, group in candidates.groupby("image_filepath", sort=False):
        image, mask = read_scene(image_path)
        for index, row in group.iterrows():
            patch, valid = native_patch(
                image,
                mask,
                row.pixel_col,
                row.pixel_row,
                native_pixels,
            )
            candidate_arrays["patch"][index] = patch
            candidate_arrays["valid"][index] = valid
            candidate_metrics.append(
                {
                    "index": index,
                    **{f"patch_{key}": value for key, value in patch_statistics(patch, valid).items()},
                }
            )
    if candidate_metrics:
        metrics = pd.DataFrame.from_records(candidate_metrics).set_index("index")
        for column in metrics:
            candidates.loc[metrics.index, column] = metrics[column]
    return transitions, candidates, transition_arrays, candidate_arrays


def mechanism_summary(records: pd.DataFrame, error_threshold_m: float) -> pd.DataFrame:
    rows = []
    for (sequence, config), group in records.groupby(["sequence", "config"], sort=False):
        failures = group[
            ~np.isfinite(group.selected_error_m)
            | (group.selected_error_m > error_threshold_m)
        ]
        for mechanism, mechanism_rows in failures.groupby(
            "primary_failure_mechanism", sort=False
        ):
            rows.append(
                {
                    "sequence": sequence,
                    "config": config,
                    "primary_failure_mechanism": mechanism,
                    "count": len(mechanism_rows),
                    "failure_fraction": len(mechanism_rows) / max(len(failures), 1),
                    "unique_buoys": mechanism_rows.buoy_id.nunique(),
                    "false_updates": int(mechanism_rows.false_update.sum()),
                    "probable_update_poisoning": int(
                        mechanism_rows.probable_update_poisoning.sum()
                    ),
                    "recovery_opportunities": int(
                        mechanism_rows.recovery_opportunity_next.sum()
                    ),
                    "missed_recoveries": int(
                        mechanism_rows.missed_recovery_next.sum()
                    ),
                    "provisional_bridge_signatures": int(
                        mechanism_rows.provisional_bridge_signature.sum()
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(map(str, row)) + " |" for row in frame.to_numpy()
    )
    return "\n".join(lines)


def storyboard_pages(
    transitions: pd.DataFrame,
    patches: dict[str, np.ndarray],
    output_dir: Path,
    error_threshold_m: float,
    maximum_paths: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = transitions[
        ~np.isfinite(transitions.selected_error_m)
        | (transitions.selected_error_m > error_threshold_m)
    ]
    ranked_paths = (
        failures.groupby("buoy_id")
        .selected_error_m.max()
        .sort_values(ascending=False, na_position="first")
        .head(maximum_paths)
        .index
    )
    created = 0
    for buoy_id in ranked_paths:
        group = transitions[transitions.buoy_id == buoy_id].sort_values(
            "observation_index"
        )
        for page, start in enumerate(range(0, len(group), 4), start=1):
            page_rows = group.iloc[start : start + 4]
            fig, axes = plt.subplots(
                3,
                len(page_rows),
                figsize=(3.2 * len(page_rows), 8.4),
                squeeze=False,
            )
            for column, (index, row) in enumerate(page_rows.iterrows()):
                for axis, key, label in zip(
                    axes[:, column],
                    ("truth", "motion_predicted", "selected"),
                    ("truth", "motion prediction", "selected"),
                ):
                    axis.imshow(patches[key][index], cmap="gray", vmin=0, vmax=255)
                    axis.axis("off")
                    axis.set_title(label, fontsize=8)
                error_text = (
                    "untracked"
                    if not np.isfinite(row.selected_error_m)
                    else f"{row.selected_error_m / 1000:.2f} km"
                )
                mechanism = str(row.primary_failure_mechanism).replace("_", " ")
                axes[0, column].set_title(
                    f"{pd.Timestamp(row.image_time):%Y-%m-%d}\ntruth | {error_text}\n{mechanism}",
                    fontsize=7,
                )
            fig.suptitle(
                f"{group.sequence.iloc[0]} {group.config.iloc[0]} buoy {buoy_id}",
                fontsize=10,
            )
            fig.tight_layout()
            fig.savefig(output_dir / f"{buoy_id}_p{page:02d}.png", dpi=150)
            plt.close(fig)
            created += 1
    return created


def write_report(
    path: Path,
    summary: pd.DataFrame,
    transitions: pd.DataFrame,
    validations: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    summary_view = summary.copy()
    summary_view["failure_fraction"] = summary_view.failure_fraction.map(
        lambda value: f"{value:.3f}"
    )
    columns = [
        "sequence",
        "config",
        "primary_failure_mechanism",
        "count",
        "failure_fraction",
        "unique_buoys",
        "false_updates",
        "probable_update_poisoning",
        "recovery_opportunities",
        "missed_recoveries",
        "provisional_bridge_signatures",
    ]
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *[
            "| " + " | ".join(map(str, row)) + " |"
            for row in summary_view[columns].to_numpy()
        ],
    ]
    totals = (
        transitions.groupby(["sequence", "config"])
        .agg(
            transitions=("transition_id", "size"),
            false_updates=("false_update", "sum"),
            probable_update_poisoning=("probable_update_poisoning", "sum"),
            recovery_opportunities=("recovery_opportunity_next", "sum"),
            recovered_next=("recovered_next", "sum"),
            missed_recovery_next=("missed_recovery_next", "sum"),
            provisional_bridge_signatures=("provisional_bridge_signature", "sum"),
        )
        .reset_index()
    )
    path.write_text(
        "# ORB candidate failure forensics\n\n"
        "The frozen graph was replayed without using buoy truth during matching. "
        "Truth was applied only after expansion and beam pruning. Selected paths "
        "reproduce the frozen node indices and map positions exactly.\n\n"
        "## Primary mechanisms among untracked or >2 km transitions\n\n"
        + "\n".join(table)
        + "\n\n## Temporal update and recovery signatures\n\n"
        + markdown_table(totals)
        + "\n\n`false_update` means the evaluation buoy is more than 2 km from a "
        "selected node that changed descriptor memory. `probable_update_poisoning` "
        "requires the next truth-near node to be branch-eligible under the immutable "
        "anchor but not under the actual updated cost. These are causal diagnostics, "
        "not deployable inputs.\n\n"
        "`provisional_bridge_signature` means the previous selected node was within "
        "2 km, was not committed to descriptor memory, and its actual descriptor "
        "would place the next truth-near candidate inside the branching set when the "
        "current confirmed-memory cost would not.\n\n"
        "`candidate_border_exclusion` means that the SAR raster covers the buoy but "
        "the hard candidate-grid border removes its neighbourhood. A descriptor "
        "coverage failure is the corresponding hole away from that border. Gate "
        "failures occur later in state space; ranking and pruning failures require "
        "a truth-near descriptor to exist and remain physically reachable.\n\n"
        "## Reproducibility\n\n"
        f"Replay runtime: {elapsed_seconds:.2f} seconds. Frozen comparisons:\n\n"
        + markdown_table(validations)
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument(
        "--frozen-results-root",
        type=Path,
        help="Optional border-sweep root containing per-sequence border_NNNpx trajectories.",
    )
    parser.add_argument("--patch-root", type=Path, default=DEFAULT_PATCH_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/orb_candidate_forensics/q2q98_clahe25",
    )
    parser.add_argument("--sequences", default="2020_03,2020_02,2015_full15")
    parser.add_argument("--configs", default="beam_confidence_update_m032")
    parser.add_argument("--grid-border-override", type=int)
    parser.add_argument("--top-candidates", type=int, default=8)
    parser.add_argument("--error-threshold-m", type=float, default=2000.0)
    parser.add_argument("--map-patch-width-m", type=float, default=5000.0)
    parser.add_argument("--map-patch-pixels", type=int, default=65)
    parser.add_argument("--candidate-native-pixels", type=int, default=65)
    parser.add_argument("--maximum-storyboard-paths", type=int, default=12)
    args = parser.parse_args()
    sequences = tuple(item.strip() for item in args.sequences.split(",") if item.strip())
    configs = tuple(item.strip() for item in args.configs.split(",") if item.strip())
    if args.map_patch_pixels != 65:
        parser.error("The current truth archive downsampling contract requires 65 pixels.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    all_transitions = []
    all_candidates = []
    validations = []
    sequence_manifests = {}

    for sequence in sequences:
        graph_dir = args.graph_root / sequence
        manifest = json.loads((graph_dir / "run_manifest.json").read_text())
        sequence_manifests[sequence] = manifest
        graph_parameters = graph_args(manifest)
        if args.grid_border_override is not None:
            graph_parameters.grid_border = args.grid_border_override
        coincidences = pd.read_csv(
            graph_dir / "coincidences.csv",
            dtype={"buoy_id": str},
        )
        coincidences["image_time"] = pd.to_datetime(coincidences.image_time, utc=True)
        frozen_path = (
            graph_dir / "trajectory_results.csv"
            if args.frozen_results_root is None
            else args.frozen_results_root
            / sequence
            / f"border_{graph_parameters.grid_border:03d}px"
            / "trajectory_results.csv"
        )
        frozen = pd.read_csv(
            frozen_path,
            dtype={"buoy_id": str},
        )
        truth_metrics, truth_patches = truth_metric_lookup(args.patch_root, sequence)
        layers, layer_seconds = precompute_layers(coincidences, graph_parameters)
        for config_name in configs:
            config = config_from_manifest(manifest, config_name)
            transition_records = []
            candidate_records = []
            descriptor_records = []
            for _, group in coincidences.groupby("buoy_id", sort=True):
                transitions, candidates, descriptors, _ = replay_path(
                    sequence,
                    group,
                    layers,
                    config,
                    graph_parameters,
                    truth_metrics,
                    args.top_candidates,
                    args.error_threshold_m,
                )
                transition_records.extend(transitions)
                candidate_records.extend(candidates)
                descriptor_records.extend(descriptors)
            transitions = add_temporal_labels(
                pd.DataFrame.from_records(transition_records),
                config.branching,
                args.error_threshold_m,
            )
            candidates = pd.DataFrame.from_records(candidate_records).reset_index(drop=True)
            validation = validate_against_frozen_results(
                transitions,
                frozen,
                sequence,
                config_name,
            )
            validation["layer_precompute_seconds"] = layer_seconds
            validations.append(validation)
            config_dir = args.out_dir / sequence / config_name
            config_dir.mkdir(parents=True, exist_ok=True)
            transitions, candidates, patch_arrays, candidate_patch_arrays = extract_patch_archives(
                transitions,
                candidates,
                truth_patches,
                graph_parameters.analysis_epsg,
                args.map_patch_width_m,
                args.map_patch_pixels,
                args.candidate_native_pixels,
            )
            transitions.to_csv(config_dir / "transition_forensics.csv", index=False)
            candidates.to_csv(config_dir / "candidate_forensics.csv", index=False)
            transition_ids = transitions.transition_id.astype(str).to_numpy(dtype="U")
            np.savez_compressed(
                config_dir / "forensic_patches.npz",
                transition_id=transition_ids,
                **patch_arrays,
            )
            candidate_ids = candidates.candidate_id.astype(str).to_numpy(dtype="U")
            np.savez_compressed(
                config_dir / "candidate_descriptors.npz",
                candidate_id=candidate_ids,
                descriptor=np.stack(descriptor_records).astype(np.uint8),
            )
            np.savez_compressed(
                config_dir / "candidate_patches.npz",
                candidate_id=candidate_ids,
                **candidate_patch_arrays,
            )
            storyboard_pages(
                transitions,
                patch_arrays,
                config_dir / "storyboards",
                args.error_threshold_m,
                args.maximum_storyboard_paths,
            )
            all_transitions.append(transitions)
            all_candidates.append(candidates)

    transitions = pd.concat(all_transitions, ignore_index=True, sort=False)
    candidates = pd.concat(all_candidates, ignore_index=True, sort=False)
    summary = mechanism_summary(transitions, args.error_threshold_m)
    validation_frame = pd.DataFrame.from_records(validations)
    transitions.to_csv(args.out_dir / "transition_forensics_all.csv", index=False)
    candidates.to_csv(args.out_dir / "candidate_forensics_all.csv", index=False)
    summary.to_csv(args.out_dir / "mechanism_summary.csv", index=False)
    validation_frame.to_csv(args.out_dir / "frozen_replay_validation.csv", index=False)
    elapsed = time.perf_counter() - started
    write_report(
        args.out_dir / "report.md",
        summary,
        transitions,
        validation_frame,
        elapsed,
    )
    output_manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "graph_root": str(args.graph_root),
        "patch_root": str(args.patch_root),
        "sequences": sequences,
        "configs": configs,
        "grid_border_px": (
            args.grid_border_override
            if args.grid_border_override is not None
            else "from frozen graph manifest"
        ),
        "frozen_results_root": (
            str(args.frozen_results_root) if args.frozen_results_root is not None else None
        ),
        "top_candidates": args.top_candidates,
        "error_threshold_m": args.error_threshold_m,
        "map_patch_width_m": args.map_patch_width_m,
        "map_patch_pixels": args.map_patch_pixels,
        "candidate_native_pixels": args.candidate_native_pixels,
        "truth_use": "evaluation after graph expansion and pruning only",
        "truth_appearance_hard_contract": (
            "exact ORB anchor distance > config update_max_cost or previous 5 km NCC < 0.25"
        ),
        "elapsed_seconds": elapsed,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(output_manifest, indent=2))
    print(summary.to_string(index=False))
    print(validation_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
