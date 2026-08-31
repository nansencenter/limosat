#!/usr/bin/env python3
"""Apply tight pattern refinement only to nodes in flipped ALIKED triangles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_aliked_orb_northup import pattern_refine
from experiments.run_aliked_dense_pair import summarise


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flipped_node_indices(field: pd.DataFrame, spacing_m: float) -> np.ndarray:
    available_indices = np.flatnonzero(field["available"].fillna(False).to_numpy())
    if len(available_indices) < 3:
        return np.empty(0, dtype=int)
    available = field.iloc[available_indices]
    source = available[["source_x", "source_y"]].to_numpy(dtype=float)
    target = source + available[["proposal_dx_m", "proposal_dy_m"]].to_numpy(
        dtype=float
    )
    triangles = Delaunay(source).simplices
    source_triangles = source[triangles]
    target_triangles = target[triangles]
    edges = np.max(
        np.stack(
            [
                np.linalg.norm(
                    source_triangles[:, 0] - source_triangles[:, 1], axis=1
                ),
                np.linalg.norm(
                    source_triangles[:, 1] - source_triangles[:, 2], axis=1
                ),
                np.linalg.norm(
                    source_triangles[:, 2] - source_triangles[:, 0], axis=1
                ),
            ]
        ),
        axis=0,
    )
    keep = edges <= spacing_m * 1.6
    triangles = triangles[keep]
    source_triangles = source_triangles[keep]
    target_triangles = target_triangles[keep]

    def signed_twice_area(values):
        first = values[:, 1] - values[:, 0]
        second = values[:, 2] - values[:, 0]
        return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]

    flipped = (
        signed_twice_area(source_triangles)
        * signed_twice_area(target_triangles)
        < 0
    )
    local_indices = np.unique(triangles[flipped].ravel())
    return available_indices[local_indices]


def reject_flipped_nodes_until_stable(
    field: pd.DataFrame, spacing_m: float
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    """Reject fold vertices until retriangulation introduces no new folds."""
    rejected = field.copy()
    rejected_by_iteration: list[np.ndarray] = []
    available_column = rejected.columns.get_loc("available")
    while True:
        selected = flipped_node_indices(rejected, spacing_m)
        if len(selected) == 0:
            break
        rejected.iloc[selected, available_column] = False
        rejected_by_iteration.append(selected)
    if rejected_by_iteration:
        rejected_indices = np.unique(np.concatenate(rejected_by_iteration))
    else:
        rejected_indices = np.empty(0, dtype=int)
    return rejected, rejected_indices, [
        int(len(selected)) for selected in rejected_by_iteration
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-run-dir", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-spacing-m", type=float, default=4000.0)
    parser.add_argument("--search-border-pixels", type=int, default=4)
    args = parser.parse_args()

    field = pd.read_csv(args.field)
    matches = pd.read_csv(args.dense_run_dir / "matches.csv")
    dense_manifest = json.loads(
        (args.dense_run_dir / "run_manifest.json").read_text()
    )
    dense_summary = json.loads((args.dense_run_dir / "summary.json").read_text())
    selected_indices = flipped_node_indices(field, args.grid_spacing_m)
    refined = field.copy()
    records = []
    for node_index in selected_indices:
        row = field.iloc[node_index]
        source_xy = row[["source_x", "source_y"]].to_numpy(dtype=float)
        proposal_xy = source_xy + row[
            ["proposal_dx_m", "proposal_dy_m"]
        ].to_numpy(dtype=float)
        started = time.perf_counter()
        result = pattern_refine(
            dense_manifest["source_image_filepath"],
            dense_manifest["target_image_filepath"],
            source_xy,
            proposal_xy,
            template_half_size=16,
            search_border=args.search_border_pixels,
            subpixel_method="quadratic",
            template_sampling="bilinear",
        )
        accepted = bool(result.get("accepted", False))
        if accepted:
            refined.loc[node_index, "proposal_dx_m"] = (
                result["corrected_x"] - row.source_x
            )
            refined.loc[node_index, "proposal_dy_m"] = (
                result["corrected_y"] - row.source_y
            )
        records.append(
            {
                "node_index": int(node_index),
                "grid_row": int(row.grid_row),
                "grid_column": int(row.grid_column),
                "source_x": row.source_x,
                "source_y": row.source_y,
                "direct_dx_m": row.proposal_dx_m,
                "direct_dy_m": row.proposal_dy_m,
                "pattern_available": bool(result.get("available", False)),
                "pattern_accepted": accepted,
                "correlation": result.get("correlation", np.nan),
                "correction_pixels": result.get("correction_pixels", np.nan),
                "subpixel_status": result.get("subpixel_status"),
                "seconds": time.perf_counter() - started,
            }
        )
    both = refined["available"].fillna(False) & refined[
        "orb_available_10km"
    ].fillna(False)
    refined.loc[:, "aliked_orb_vector_difference_m"] = np.nan
    refined.loc[both, "aliked_orb_vector_difference_m"] = np.hypot(
        refined.loc[both, "proposal_dx_m"] - refined.loc[both, "orb_dx_m"],
        refined.loc[both, "proposal_dy_m"] - refined.loc[both, "orb_dy_m"],
    )
    summary = summarise(
        refined,
        matches,
        int(dense_summary["orb_paired_trajectories"]),
        args.grid_spacing_m,
    )
    summary.update(
        {
            "selected_flip_nodes": int(len(selected_indices)),
            "pattern_accepted_nodes": int(
                sum(record["pattern_accepted"] for record in records)
            ),
            "pattern_seconds": float(
                sum(record["seconds"] for record in records)
            ),
        }
    )
    rejected, rejected_indices, rejection_iteration_counts = (
        reject_flipped_nodes_until_stable(field, args.grid_spacing_m)
    )
    rejection_summary = summarise(
        rejected,
        matches,
        int(dense_summary["orb_paired_trajectories"]),
        args.grid_spacing_m,
    )
    rejection_summary.update(
        {
            "rejected_flip_nodes": int(len(rejected_indices)),
            "rejection_iterations": int(len(rejection_iteration_counts)),
            "rejected_nodes_per_iteration": rejection_iteration_counts,
            "rule": (
                "iteratively mark every node in a flipped local triangle "
                "unavailable until retriangulation contains no folds"
            ),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    refined.to_csv(args.output_dir / "refined_field.csv", index=False)
    rejected.to_csv(args.output_dir / "flip_rejected_field.csv", index=False)
    pd.DataFrame(records).to_csv(
        args.output_dir / "pattern_refinement.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.output_dir / "rejection_summary.json").write_text(
        json.dumps(rejection_summary, indent=2) + "\n"
    )
    manifest = {
        "status": "complete",
        "dense_run_dir": str(args.dense_run_dir),
        "field": str(args.field),
        "field_sha256": sha256(args.field),
        "selection": "all nodes participating in a flipped local Delaunay triangle",
        "grid_spacing_m": args.grid_spacing_m,
        "maximum_triangle_edge_m": args.grid_spacing_m * 1.6,
        "pattern_matching": {
            "template_half_size_pixels": 16,
            "search_border_pixels": args.search_border_pixels,
            "subpixel_method": "quadratic",
            "template_sampling": "bilinear",
            "minimum_correlation": 0.30,
        },
        "application": "apply every accepted refinement; retain direct otherwise",
        "summary": summary,
        "flip_node_rejection_summary": rejection_summary,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
