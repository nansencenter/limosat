#!/usr/bin/env python3
"""Match only source tiles near trajectories lost before a later SAR image."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_efficientloftr_sequence import (
    PairSpec,
    file_sha256,
    field_from_csv,
    load_completed_pair,
    pair_domains,
    pair_identity,
    track_pair,
)
from limosat.learned_drift import (
    EfficientLoFTRConfig,
    FieldEdge,
    advect_trajectory_graph,
    coarse_phase_translation,
)
from limosat.learned_drift.efficientloftr import load_optimized_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjacent-sequence-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-pair-run-dir", type=Path, action="append", required=True
    )
    parser.add_argument("--efficientloftr-repo", type=Path, required=True)
    parser.add_argument("--efficientloftr-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--selection-buffer-m", type=float)
    return parser.parse_args()


def load_adjacent_sequence(
    run_dir: Path,
) -> tuple[list[str], list[FieldEdge]]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    pairs = manifest["pairs_summary"]
    image_ids = [str(pairs[0]["source_image_id"])] + [
        str(pair["target_image_id"]) for pair in pairs
    ]
    edges = []
    for pair in pairs:
        source = str(pair["source_image_id"])
        target = str(pair["target_image_id"])
        field = field_from_csv(run_dir / f"pair_{source}_{target}" / "field_4km.csv")
        edges.append(
            FieldEdge(source, target, float(pair["elapsed_hours"]), field)
        )
    return image_ids, edges


def load_reference(run_dir: Path) -> PairSpec:
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    buoy_path = run_dir / "buoy_results.csv"
    return PairSpec(
        int(manifest["source_image_id"]),
        int(manifest["target_image_id"]),
        str(manifest["source_image_filepath"]),
        str(manifest["target_image_filepath"]),
        float(manifest["elapsed_hours"]),
        buoy_path if buoy_path.exists() else None,
    )


def recovery_positions(
    trajectories: pd.DataFrame,
    image_ids: list[str],
    source_id: str,
    target_id: str,
) -> pd.DataFrame:
    source_index = image_ids.index(source_id)
    target_index = image_ids.index(target_id)
    if target_index <= source_index + 1:
        raise ValueError("targeted recovery requires a non-consecutive image pair")
    source = trajectories.loc[
        trajectories.image_index.eq(source_index),
        ["trajectory_id", "x_m", "y_m", "active"],
    ].rename(columns={"active": "source_active"})
    target = trajectories.loc[
        trajectories.image_index.eq(target_index),
        ["trajectory_id", "active"],
    ].rename(columns={"active": "target_active"})
    rows = source.merge(target, on="trajectory_id", validate="one_to_one")
    rows = rows.loc[rows.source_active & ~rows.target_active].copy()
    rows["source_image_id"] = source_id
    rows["target_image_id"] = target_id
    return rows


def main() -> int:
    args = parse_args()
    config = EfficientLoFTRConfig()
    selection_buffer_m = (
        float(args.selection_buffer_m)
        if args.selection_buffer_m is not None
        else max(
            config.maximum_neighbour_distance_m,
            config.maximum_triangle_edge_m,
        )
    )
    if not np.isfinite(selection_buffer_m) or selection_buffer_m <= 0:
        raise ValueError("selection buffer must be finite and positive")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    image_ids, adjacent_edges = load_adjacent_sequence(args.adjacent_sequence_dir)
    trajectories = advect_trajectory_graph(
        adjacent_edges,
        image_ids,
        config.grid_spacing_m,
        maximum_triangle_edge_m=config.maximum_triangle_edge_m,
    )
    references = [load_reference(path) for path in args.reference_pair_run_dir]
    model = None
    model_setup_seconds = 0.0
    pair_summaries = []
    selection_rows = []
    checkpoint_sha256 = file_sha256(args.efficientloftr_checkpoint)
    started = time.perf_counter()

    for spec in references:
        source_id = str(spec.source_image_id)
        target_id = str(spec.target_image_id)
        selected = recovery_positions(
            trajectories, image_ids, source_id, target_id
        )
        if selected.empty:
            raise ValueError(f"no dormant trajectories for {source_id}->{target_id}")
        selection_rows.append(selected)
        source_index = image_ids.index(source_id)
        previous_field = None
        previous_elapsed_days = None
        initial_displacement_m = None
        initial_routing = "preceding_adjacent_field"
        if source_index == 0:
            initial_domain, _ = pair_domains(spec, config)
            translation = coarse_phase_translation(
                spec.source_path,
                spec.target_path,
                initial_domain,
                config.maximum_displacement_m(spec.elapsed_hours),
                config.analysis_epsg,
                config.transform_grid_spacing_px,
            )
            initial_displacement_m = translation.displacement_m
            initial_routing = "phase_correlation"
        else:
            previous_edge = adjacent_edges[source_index - 1]
            previous_field = previous_edge.field
            previous_elapsed_days = previous_edge.elapsed_hours / 24.0

        positions = selected[["x_m", "y_m"]].to_numpy(np.float64)
        identity = pair_identity(
            spec,
            config,
            "sequential_local",
            initial_routing,
            initial_displacement_m,
            checkpoint_sha256,
            previous_field,
            previous_elapsed_days,
            positions,
            selection_buffer_m,
        )
        pair_dir = args.output_dir / f"pair_{source_id}_{target_id}"
        completed = load_completed_pair(pair_dir, identity)
        if completed is not None:
            _field, summary = completed
        else:
            if model is None:
                model_started = time.perf_counter()
                model = load_optimized_model(
                    args.efficientloftr_repo,
                    args.efficientloftr_checkpoint,
                    device,
                )
                model_setup_seconds = time.perf_counter() - model_started
            _field, summary = track_pair(
                spec,
                model,
                device,
                config,
                "sequential_local",
                previous_field,
                previous_elapsed_days,
                initial_displacement_m,
                identity,
                pair_dir,
                positions,
                selection_buffer_m,
            )
        pair_summaries.append(summary)
        print(json.dumps(summary), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(selection_rows, ignore_index=True).to_csv(
        args.output_dir / "targeted_source_positions.csv", index=False
    )
    report = {
        "status": "complete",
        "matcher": "official EfficientLoFTR optimized",
        "purpose": "non-consecutive matching near trajectories lost by adjacent fields",
        "device": device.type,
        "adjacent_sequence_dir": str(args.adjacent_sequence_dir),
        "image_ids": image_ids,
        "selection_buffer_m": selection_buffer_m,
        "selection_basis": "active at skip source and inactive at skip target in the adjacent-only graph",
        "checkpoint_sha256": checkpoint_sha256,
        "model_setup_seconds": model_setup_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "pairs_summary": pair_summaries,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
