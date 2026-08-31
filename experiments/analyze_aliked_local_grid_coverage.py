#!/usr/bin/env python3
"""Measure spatial coverage and roughness of cached ALIKED match vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.replay_aliked_candidate_policies import (
    estimate_policy,
    recenter_matches,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_grid(
    transitions: pd.DataFrame,
    vectors: pd.DataFrame,
    grid_half_extent_m: float,
    grid_spacing_m: float,
    tight_radius_m: float,
    consensus_radius_m: float,
    maximum_speed_m_per_day: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    offsets = np.arange(
        -grid_half_extent_m,
        grid_half_extent_m + grid_spacing_m / 2.0,
        grid_spacing_m,
    )
    vector_groups = {
        transition_id: group
        for transition_id, group in vectors.groupby("transition_id", sort=False)
    }
    empty = vectors.iloc[:0]
    records = []
    for row in transitions.loc[
        transitions["representative_panel"].fillna(False)
    ].itertuples(index=False):
        matches = vector_groups.get(row.transition_id, empty)
        for row_index, offset_y in enumerate(offsets):
            for column_index, offset_x in enumerate(offsets):
                query = np.array(
                    [row.source_x + offset_x, row.source_y + offset_y], dtype=float
                )
                local = recenter_matches(
                    matches,
                    query,
                    source_radius_m=10000.0,
                    maximum_speed_m_per_day=maximum_speed_m_per_day,
                )
                proposal = estimate_policy(
                    local,
                    "consensus_within_2km",
                    tight_radius_m=tight_radius_m,
                    consensus_radius_m=consensus_radius_m,
                )
                records.append(
                    {
                        "transition_id": row.transition_id,
                        "row": row_index,
                        "column": column_index,
                        "offset_x_m": offset_x,
                        "offset_y_m": offset_y,
                        "radius_m": float(np.hypot(offset_x, offset_y)),
                        **proposal,
                    }
                )
    grid = pd.DataFrame.from_records(records)
    neighbours = []
    for transition_id, group in grid.groupby("transition_id", sort=False):
        lookup = group.set_index(["row", "column"])
        for (row_index, column_index), point in lookup.iterrows():
            if not point["available"]:
                continue
            for neighbour_index in (
                (row_index + 1, column_index),
                (row_index, column_index + 1),
            ):
                if neighbour_index not in lookup.index:
                    continue
                neighbour = lookup.loc[neighbour_index]
                if not neighbour["available"]:
                    continue
                vector_difference = np.hypot(
                    point["proposal_dx_m"] - neighbour["proposal_dx_m"],
                    point["proposal_dy_m"] - neighbour["proposal_dy_m"],
                )
                neighbours.append(
                    {
                        "transition_id": transition_id,
                        "row": row_index,
                        "column": column_index,
                        "neighbour_row": neighbour_index[0],
                        "neighbour_column": neighbour_index[1],
                        "vector_difference_m": float(vector_difference),
                        "difference_per_km": float(
                            vector_difference / (grid_spacing_m / 1000.0)
                        ),
                    }
                )
    return grid, pd.DataFrame.from_records(neighbours)


def summarize(grid: pd.DataFrame, neighbours: pd.DataFrame) -> dict:
    by_case = grid.groupby("transition_id")["available"].agg(["count", "sum"])
    by_case["coverage_fraction"] = by_case["sum"] / by_case["count"]
    center = grid.loc[grid["radius_m"].eq(0.0)]
    return {
        "cases": int(grid["transition_id"].nunique()),
        "queries": int(len(grid)),
        "covered_queries": int(grid["available"].sum()),
        "overall_coverage_fraction": float(grid["available"].mean()),
        "median_case_coverage_fraction": float(by_case["coverage_fraction"].median()),
        "p10_case_coverage_fraction": float(
            by_case["coverage_fraction"].quantile(0.10)
        ),
        "minimum_case_coverage_fraction": float(by_case["coverage_fraction"].min()),
        "center_available_cases": int(center["available"].sum()),
        "neighbour_pairs": int(len(neighbours)),
        "median_neighbour_vector_difference_m": (
            float(neighbours["vector_difference_m"].median())
            if len(neighbours)
            else None
        ),
        "p90_neighbour_vector_difference_m": (
            float(neighbours["vector_difference_m"].quantile(0.90))
            if len(neighbours)
            else None
        ),
        "interpretation": (
            "coverage and local vector roughness only; buoy truth is not assumed "
            "at off-centre query locations"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-half-extent-m", type=float, default=8000.0)
    parser.add_argument("--grid-spacing-m", type=float, default=4000.0)
    parser.add_argument("--tight-radius-m", type=float, default=2000.0)
    parser.add_argument("--consensus-radius-m", type=float, default=1000.0)
    parser.add_argument("--maximum-speed-m-per-day", type=float, default=30000.0)
    args = parser.parse_args()

    vectors = pd.read_csv(args.vectors, low_memory=False)
    transitions = pd.read_csv(args.transitions, low_memory=False)
    grid, neighbours = evaluate_grid(
        transitions,
        vectors,
        args.grid_half_extent_m,
        args.grid_spacing_m,
        args.tight_radius_m,
        args.consensus_radius_m,
        args.maximum_speed_m_per_day,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.output_dir / "grid_results.csv", index=False)
    neighbours.to_csv(args.output_dir / "neighbour_results.csv", index=False)
    summary = summarize(grid, neighbours)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "vectors_sha256": sha256(args.vectors),
        "transitions_sha256": sha256(args.transitions),
        "grid_half_extent_m": args.grid_half_extent_m,
        "grid_spacing_m": args.grid_spacing_m,
        "tight_radius_m": args.tight_radius_m,
        "consensus_radius_m": args.consensus_radius_m,
        "maximum_speed_m_per_day": args.maximum_speed_m_per_day,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
