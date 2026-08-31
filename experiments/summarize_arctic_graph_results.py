#!/usr/bin/env python3
"""Consolidate the frozen Arctic ORB and XFeat graph experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ORB_RUNS = {
    "2020_03": ("development", ROOT / "results/orb_multiframe_graph/final_arctic_matrix/2020_03"),
    "2020_02": ("validation", ROOT / "results/orb_multiframe_graph/final_arctic_matrix/2020_02"),
    "2015_full15": ("holdout", ROOT / "results/orb_multiframe_graph/final_arctic_matrix/2015_full15"),
}
XFEAT_RUNS = {
    "2020_03": (
        "development",
        ROOT / "results/xfeat_buoy_graph/arctic_2020_03/q2q98_clahe25_max1536_top16000",
    ),
    "2020_02": (
        "validation",
        ROOT / "results/xfeat_buoy_graph/arctic_2020_02/q2q98_clahe25_max1536_top16000",
    ),
}


def load_orb(sequence: str, role: str, run: Path) -> pd.DataFrame:
    data = pd.read_csv(run / "summary.csv")
    manifest = json.loads((run / "run_manifest.json").read_text())
    data.insert(0, "sequence", sequence)
    data.insert(1, "role", role)
    data.insert(2, "backend", "ORB supplied-grid")
    data["eligible_paths"] = data.paths + data.graph_failed_paths + data.seed_unavailable_paths
    data["completed_paths"] = data.paths
    data["transitions"] = data.observations
    data["feature_extraction_seconds"] = manifest["precompute_seconds"]
    data["elapsed_seconds"] = manifest["elapsed_seconds"]
    data["descriptor_dtype"] = "uint8"
    data["descriptor_dimensions"] = 32
    data["descriptor_norm"] = "hamming"
    return data


def load_xfeat(sequence: str, role: str, run: Path) -> pd.DataFrame:
    data = pd.read_csv(run / "summary.csv")
    manifest = json.loads((run / "run_manifest.json").read_text())
    data.insert(0, "sequence", sequence)
    data.insert(1, "role", role)
    data.insert(2, "backend", "XFeat sparse")
    failures = pd.read_csv(run / "trajectory_results.csv")
    graph_failed = failures[failures.status == "graph_failed"].groupby("config").buoy_id.nunique()
    unavailable = failures[failures.status == "seed_unavailable"].groupby("config").buoy_id.nunique()
    data["graph_failed_paths"] = data.config.map(graph_failed).fillna(0).astype(int)
    data["seed_unavailable_paths"] = data.config.map(unavailable).fillna(0).astype(int)
    data["observations"] = data.transitions
    data["skipped_observations"] = 0
    data["observation_coverage_fraction"] = 1.0
    data["long_path_final_error_m"] = data.median_final_error_m
    data["feature_extraction_seconds"] = manifest["feature_extraction_seconds"]
    data["elapsed_seconds"] = manifest["elapsed_seconds"]
    data["descriptor_dtype"] = manifest["descriptor_dtype"]
    data["descriptor_dimensions"] = manifest["descriptor_dimensions"]
    data["descriptor_norm"] = manifest["descriptor_norm"]
    return data


def main() -> int:
    out_dir = ROOT / "results/arctic_descriptor_graph_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [load_orb(sequence, role, run) for sequence, (role, run) in ORB_RUNS.items()]
    frames.extend(load_xfeat(sequence, role, run) for sequence, (role, run) in XFEAT_RUNS.items())
    data = pd.concat(frames, ignore_index=True, sort=False)
    columns = [
        "sequence",
        "role",
        "backend",
        "config",
        "eligible_paths",
        "completed_paths",
        "graph_failed_paths",
        "seed_unavailable_paths",
        "observations",
        "transitions",
        "skipped_observations",
        "observation_coverage_fraction",
        "median_error_m",
        "p90_error_m",
        "within_2km_fraction",
        "catastrophic_50km_fraction",
        "long_path_final_error_m",
        "median_feature_coverage_floor_m",
        "feature_extraction_seconds",
        "elapsed_seconds",
        "descriptor_dtype",
        "descriptor_dimensions",
        "descriptor_norm",
    ]
    for column in columns:
        if column not in data:
            data[column] = pd.NA
    data[columns].to_csv(out_dir / "comparison.csv", index=False)

    selected = data[
        data.config.isin(
            [
                "greedy_rolling",
                "beam_anchor",
                "beam_confidence_update_m032",
                "beam_confidence_m032_skip1",
                "xfeat_greedy_rolling",
                "xfeat_beam_anchor_rolling",
            ]
        )
    ].copy()
    selected["median_error_km"] = selected.median_error_m / 1000.0
    selected["p90_error_km"] = selected.p90_error_m / 1000.0
    selected["final_error_km"] = selected.long_path_final_error_m / 1000.0
    view_columns = [
        "sequence",
        "backend",
        "config",
        "completed_paths",
        "median_error_km",
        "p90_error_km",
        "within_2km_fraction",
        "catastrophic_50km_fraction",
        "final_error_km",
    ]
    view = selected[view_columns].copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    lines = [
        "| " + " | ".join(view_columns) + " |",
        "| " + " | ".join(["---"] * len(view_columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()],
    ]
    (out_dir / "report.md").write_text(
        "# Arctic descriptor and graph comparison\n\n"
        "Frozen input: standard `balanced_q2q98_clahe25` VAE band, EPSG:3413, "
        "exact-time buoy interpolation, 50 km/day hard physics gate.\n\n"
        + "\n".join(lines)
        + "\n\nMetrics are conditional on completed paths; failure, seed-unavailable, skip, "
        "and sparse-feature coverage columns remain in `comparison.csv`. XFeat was "
        "not promoted to the 2015 holdout because it underperformed ORB on both "
        "development and validation.\n"
    )
    print(data[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
