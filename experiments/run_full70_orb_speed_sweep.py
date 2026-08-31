#!/usr/bin/env python3
"""Run a physics-speed sensitivity sweep while reusing extracted ORB layers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from experiments.orb_multiframe_graph import (
        GRAPH_CONFIGS,
        annotate_coincidences,
        precompute_layers,
        summarize,
        trajectory_rows,
    )
except ModuleNotFoundError:  # Direct execution from the experiments directory.
    from orb_multiframe_graph import (  # type: ignore[no-redef]
        GRAPH_CONFIGS,
        annotate_coincidences,
        precompute_layers,
        summarize,
        trajectory_rows,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COINCIDENCES = (
    ROOT
    / "results/iabp_s1_stratified_coverage/full70_level1_tracking_observations.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/orb_multiframe_graph/full70_level1/speed_sweep"


def parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or min(values) <= 0:
        raise ValueError("Speed values must be positive")
    return values


def summarize_subset(
    records: pd.DataFrame,
    coincidences: pd.DataFrame,
    experiment_split: str,
    speed_km_day: float,
) -> list[pd.DataFrame]:
    tables = []
    for subset, path_ids in (
        ("all_temporal", None),
        (
            "strict_month_exclusive_buoy",
            set(
                coincidences.loc[
                    coincidences["month_exclusive_buoy"],
                    "experiment_trajectory_id",
                ]
            ),
        ),
    ):
        selected = (
            records
            if path_ids is None
            else records[records["trajectory_id"].isin(path_ids)]
        )
        if selected.empty:
            continue
        table = summarize(selected)
        table.insert(0, "speed_limit_km_day", speed_km_day)
        table.insert(0, "evaluation_subset", subset)
        table.insert(0, "experiment_split", experiment_split)
        tables.append(table)
    return tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coincidences", type=Path, default=DEFAULT_COINCIDENCES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--speeds-km-day", default="40,50,75,100")
    parser.add_argument(
        "--graph-configs",
        default="greedy_rolling,beam_anchor,beam_confidence_update_m032",
    )
    parser.add_argument("--analysis-epsg", type=int, default=3413)
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
    args = parser.parse_args()
    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    coincidences = pd.read_csv(args.coincidences, low_memory=False)
    coincidences["image_time"] = pd.to_datetime(coincidences["image_time"], utc=True)
    if coincidences["month_exclusive_buoy"].dtype != bool:
        coincidences["month_exclusive_buoy"] = (
            coincidences["month_exclusive_buoy"]
            .astype(str)
            .str.lower()
            .isin({"true", "1"})
        )
    exact_count = len(coincidences)
    coincidences = annotate_coincidences(
        coincidences, args.analysis_epsg, outside_scene_policy="skip"
    )
    invalid = (coincidences["mask_value"] >= 2) | ~np.isfinite(
        coincidences[["col", "row"]]
    ).all(axis=1)
    invalid_count = int(invalid.sum())
    coincidences = coincidences.loc[~invalid].reset_index(drop=True)
    layers, precompute_seconds = precompute_layers(coincidences, args)

    requested = set(args.graph_configs.split(","))
    configs = tuple(config for config in GRAPH_CONFIGS if config.name in requested)
    missing = requested - {config.name for config in configs}
    if missing:
        raise ValueError(f"Unknown graph configurations: {sorted(missing)}")

    record_tables = []
    summary_tables = []
    timing_rows = []
    for experiment_split, split_rows in coincidences.groupby(
        "experiment_split", sort=False
    ):
        for speed_km_day in parse_floats(args.speeds_km_day):
            args.max_speed_m_per_day = speed_km_day * 1000.0
            speed_records = []
            for config in configs:
                config_started = time.perf_counter()
                config_records = trajectory_rows(split_rows, layers, config, args)
                timing_rows.append(
                    {
                        "experiment_split": experiment_split,
                        "speed_limit_km_day": speed_km_day,
                        "config": config.name,
                        "seconds": time.perf_counter() - config_started,
                    }
                )
                speed_records.extend(config_records)
            records = pd.DataFrame.from_records(speed_records)
            records.insert(0, "speed_limit_km_day", speed_km_day)
            records.insert(0, "experiment_split", experiment_split)
            record_tables.append(records)
            summary_tables.extend(
                summarize_subset(
                    records,
                    split_rows,
                    str(experiment_split),
                    speed_km_day,
                )
            )

    all_records = pd.concat(record_tables, ignore_index=True)
    summaries = pd.concat(summary_tables, ignore_index=True)
    timings = pd.DataFrame.from_records(timing_rows)
    all_records.to_csv(args.out_dir / "trajectory_results.csv", index=False)
    summaries.to_csv(args.out_dir / "summary.csv", index=False)
    timings.to_csv(args.out_dir / "timings.csv", index=False)
    payload = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "coincidences_before_spatial_filter": exact_count,
        "invalid_support_observations": invalid_count,
        "spatially_valid_observations": len(coincidences),
        "images": int(coincidences["image_filepath"].nunique()),
        "speeds_km_day": list(parse_floats(args.speeds_km_day)),
        "configs": [config.name for config in configs],
        "orb_contract": {
            "nfeatures": args.orb_nfeatures,
            "scale_factor": args.orb_scale_factor,
            "nlevels": args.orb_nlevels,
            "edge_threshold": args.orb_edge_threshold,
            "patch_size": args.orb_patch_size,
            "supplied_keypoint_size": args.keypoint_size,
            "supplied_octave": args.octave,
            "angle_mode": args.angle_mode,
            "descriptor_norm": args.descriptor_norm,
            "grid_stride_pixels": args.grid_stride,
            "grid_border_pixels": args.grid_border,
        },
        "precompute_seconds": precompute_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "reuse_contract": "ORB candidate layers extracted once and reused for every speed arm",
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
