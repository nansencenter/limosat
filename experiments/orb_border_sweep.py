#!/usr/bin/env python3
"""Test the hard ORB candidate-grid border on frozen Arctic buoy sequences."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from orb_multiframe_graph import GraphSearchConfig, precompute_layers, trajectory_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_ROOT = ROOT / "results/orb_multiframe_graph/final_arctic_matrix"


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def graph_args(manifest: dict, border: int) -> SimpleNamespace:
    keys = (
        "analysis_epsg",
        "max_speed_m_per_day",
        "grid_stride",
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
    values = {key: manifest[key] for key in keys}
    values["grid_border"] = border
    return SimpleNamespace(**values)


def frozen_config(manifest: dict, name: str) -> GraphSearchConfig:
    values = [item for item in manifest["graph_configs"] if item["name"] == name]
    if len(values) != 1:
        raise ValueError(f"Missing or ambiguous frozen config {name!r}.")
    return GraphSearchConfig(**values[0])


def target_rows(coincidences: pd.DataFrame) -> pd.DataFrame:
    return (
        coincidences.sort_values(["buoy_id", "image_time"])
        .groupby("buoy_id", sort=False)
        .apply(lambda group: group.iloc[1:], include_groups=False)
        .reset_index(level=0)
        .reset_index(drop=True)
    )


def layer_coverage(coincidences: pd.DataFrame, layers: dict) -> dict[str, float]:
    distances = []
    for row in target_rows(coincidences).itertuples(index=False):
        layer = layers[row.image_filepath]
        distance, _ = layer.spatial_index.query(np.array([row.x, row.y]), k=1)
        distances.append(float(distance))
    values = np.asarray(distances, dtype=float)
    return {
        "target_candidate_coverage_2km": float(np.mean(values <= 2000.0)),
        "target_candidate_coverage_5km": float(np.mean(values <= 5000.0)),
        "median_nearest_candidate_m": float(np.median(values)),
        "p90_nearest_candidate_m": float(np.percentile(values, 90)),
    }


def summarize_run(
    sequence: str,
    border: int,
    coincidences: pd.DataFrame,
    records: pd.DataFrame,
    layers: dict,
    precompute_seconds: float,
    tracking_seconds: float,
) -> dict:
    eligible_paths = coincidences.groupby("buoy_id").size()
    eligible_paths = eligible_paths[eligible_paths >= 2]
    eligible_transitions = int((eligible_paths - 1).sum())
    valid = records[
        (records.status == "ok") & (records.observation_index > 0)
    ].copy()
    success = valid.endpoint_error_m <= 2000.0
    catastrophic = valid.endpoint_error_m > 50000.0
    return {
        "sequence": sequence,
        "grid_border_px": border,
        "eligible_paths": len(eligible_paths),
        "eligible_transitions": eligible_transitions,
        "tracked_paths": valid.buoy_id.nunique(),
        "tracked_transitions": len(valid),
        "tracking_coverage_all": len(valid) / max(eligible_transitions, 1),
        "within_2km_count": int(success.sum()),
        "within_2km_fraction_all": float(success.sum() / max(eligible_transitions, 1)),
        "catastrophic_50km_count": int(catastrophic.sum()),
        "catastrophic_50km_fraction_all": float(
            catastrophic.sum() / max(eligible_transitions, 1)
        ),
        "median_error_tracked_m": float(valid.endpoint_error_m.median()),
        "p90_error_tracked_m": float(valid.endpoint_error_m.quantile(0.9)),
        "seed_unavailable_paths": int(
            records.loc[records.status == "seed_unavailable", "buoy_id"].nunique()
        ),
        "graph_failed_paths": int(
            records.loc[records.status == "graph_failed", "buoy_id"].nunique()
        ),
        "total_candidate_descriptors": int(
            sum(len(layer.descriptors) for layer in layers.values())
        ),
        "precompute_seconds": precompute_seconds,
        "tracking_seconds": tracking_seconds,
        **layer_coverage(coincidences, layers),
    }


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


def write_report(path: Path, summary: pd.DataFrame, config: str) -> None:
    columns = [
        "sequence",
        "grid_border_px",
        "target_candidate_coverage_2km",
        "tracking_coverage_all",
        "within_2km_fraction_all",
        "catastrophic_50km_fraction_all",
        "median_error_tracked_m",
        "precompute_seconds",
        "tracking_seconds",
    ]
    view = summary[columns].copy()
    for column in view.select_dtypes(include="float").columns:
        view[column] = view[column].map(lambda value: f"{value:.3f}")
    validation = summary[summary.sequence == "2020_02"].sort_values(
        [
            "within_2km_fraction_all",
            "catastrophic_50km_fraction_all",
            "grid_border_px",
        ],
        ascending=[False, True, False],
    )
    selected_border = int(validation.iloc[0].grid_border_px)
    path.write_text(
        "# ORB candidate-border sweep\n\n"
        f"Frozen graph configuration: `{config}`. Border selection uses February "
        "validation only; N-ICE2015 remains the reported holdout. All fractions use "
        "every eligible transition as the denominator.\n\n"
        + markdown_table(view)
        + f"\n\nValidation selects `{selected_border}` pixels by <=2 km coverage, "
        "then catastrophic-error rate. Candidate extraction is allowed to reject "
        "individual keypoints internally; the sweep changes only the requested grid "
        "extent.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/orb_border_sweep/q2q98_clahe25",
    )
    parser.add_argument("--sequences", default="2020_03,2020_02,2015_full15")
    parser.add_argument("--borders", default="16,32,64,96,128")
    parser.add_argument("--config", default="beam_confidence_update_m032")
    args = parser.parse_args()
    sequences = tuple(item.strip() for item in args.sequences.split(",") if item.strip())
    borders = parse_ints(args.borders)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summaries = []
    for sequence in sequences:
        graph_dir = args.graph_root / sequence
        manifest = json.loads((graph_dir / "run_manifest.json").read_text())
        config = frozen_config(manifest, args.config)
        coincidences = pd.read_csv(
            graph_dir / "coincidences.csv", dtype={"buoy_id": str}
        )
        coincidences["image_time"] = pd.to_datetime(coincidences.image_time, utc=True)
        for border in borders:
            run_args = graph_args(manifest, border)
            layers, precompute_seconds = precompute_layers(coincidences, run_args)
            tracking_started = time.perf_counter()
            records = pd.DataFrame.from_records(
                trajectory_rows(coincidences, layers, config, run_args)
            )
            tracking_seconds = time.perf_counter() - tracking_started
            run_dir = args.out_dir / sequence / f"border_{border:03d}px"
            run_dir.mkdir(parents=True, exist_ok=True)
            records.to_csv(run_dir / "trajectory_results.csv", index=False)
            summaries.append(
                summarize_run(
                    sequence,
                    border,
                    coincidences,
                    records,
                    layers,
                    precompute_seconds,
                    tracking_seconds,
                )
            )
    summary = pd.DataFrame.from_records(summaries)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    write_report(args.out_dir / "report.md", summary, args.config)
    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "graph_root": str(args.graph_root),
        "sequences": sequences,
        "borders_px": borders,
        "config": args.config,
        "selection_split": "2020_02 validation",
        "holdout_split": "2015_full15",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
