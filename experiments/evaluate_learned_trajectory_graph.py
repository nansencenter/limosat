#!/usr/bin/env python3
"""Evaluate observed adjacent and skip-edge fields as Lagrangian trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_efficientloftr_sequence import field_from_csv
from limosat.learned_drift import FieldEdge, advect_trajectory_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjacent-sequence-dir", type=Path, required=True)
    parser.add_argument("--skip-sequence-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--reference-pair-run-dir", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--new-point-exclusion-radius-m", type=float, default=2000.0
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_edges(run_dir: Path) -> list[FieldEdge]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    return [
        FieldEdge(
            pair["source_image_id"],
            pair["target_image_id"],
            float(pair["elapsed_hours"]),
            field_from_csv(
                run_dir
                / f"pair_{pair['source_image_id']}_{pair['target_image_id']}"
                / "field_4km.csv"
            ),
        )
        for pair in manifest["pairs_summary"]
    ]


def trajectory_summary(rows: pd.DataFrame) -> dict:
    by_image = rows.groupby("image_index", sort=True)["active"].agg(["sum", "count"])
    final = rows.loc[rows.image_index == rows.image_index.max()]
    initial_trajectories = (
        int(rows.loc[rows.seed_image_index == 0, "trajectory_id"].nunique())
        if "seed_image_index" in rows
        else int(by_image.iloc[0]["count"])
    )
    return {
        "seeded": initial_trajectories,
        "new_trajectories": int(
            rows.loc[rows.trajectory_state == "new_trajectory", "trajectory_id"].nunique()
        ),
        "trajectory_count": int(rows.trajectory_id.nunique()),
        "complete": int(final.active.sum()),
        "complete_fraction": float(final.active.mean()),
        "active_by_image": by_image["sum"].astype(int).tolist(),
        "trajectory_count_by_image": by_image["count"].astype(int).tolist(),
        "observed_skip_edge_rows": int(
            (rows.trajectory_state == "observed_skip_edge").sum()
        ),
        "reconnected_rows": int(
            rows.get("reconnected_after_gap", pd.Series(False, index=rows.index)).sum()
        ),
        "dormant_rows": int((rows.trajectory_state == "dormant").sum()),
    }


def main() -> int:
    args = parse_args()
    adjacent_manifest = json.loads(
        (args.adjacent_sequence_dir / "run_manifest.json").read_text()
    )
    image_ids = [adjacent_manifest["pairs_summary"][0]["source_image_id"]] + [
        pair["target_image_id"] for pair in adjacent_manifest["pairs_summary"]
    ]
    edges = load_edges(args.adjacent_sequence_dir)
    for run_dir in args.skip_sequence_dir:
        edges.extend(load_edges(run_dir))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid = advect_trajectory_graph(edges, image_ids, 4000.0)
    grid.to_csv(args.output_dir / "grid_trajectory_graph.csv", index=False)
    summary = trajectory_summary(grid)
    points_graph = advect_trajectory_graph(
        edges,
        image_ids,
        4000.0,
        maximum_triangle_edge_m=6400.0,
        add_new_trajectories=True,
        new_point_exclusion_radius_m=args.new_point_exclusion_radius_m,
    )
    points_graph.to_csv(
        args.output_dir / "grid_trajectory_graph_with_new_points.csv", index=False
    )
    summary["with_new_points"] = {
        **trajectory_summary(points_graph),
        "new_point_exclusion_radius_m": args.new_point_exclusion_radius_m,
    }

    references = [
        pd.read_csv(path / "buoy_results.csv", dtype={"buoy_id": str})
        for path in args.reference_pair_run_dir
    ]
    first = references[0].drop_duplicates("buoy_id").reset_index(drop=True)
    buoy = advect_trajectory_graph(
        edges,
        image_ids,
        4000.0,
        first[["source_x", "source_y"]].to_numpy(np.float64),
    )
    buoy["buoy_id"] = first.loc[buoy["trajectory_id"], "buoy_id"].to_numpy()
    truth_rows = []
    for image_index, reference in enumerate(references, start=1):
        truth = reference.drop_duplicates("buoy_id").copy()
        truth["truth_x_m"] = truth["source_x"] + truth["truth_dx_m"]
        truth["truth_y_m"] = truth["source_y"] + truth["truth_dy_m"]
        truth["image_index"] = image_index
        truth_rows.append(
            truth[["buoy_id", "image_index", "truth_x_m", "truth_y_m"]]
        )
    buoy = buoy.merge(
        pd.concat(truth_rows, ignore_index=True),
        on=["buoy_id", "image_index"],
        how="left",
        validate="many_to_one",
    )
    buoy["error_m"] = np.where(
        buoy["active"] & buoy["truth_x_m"].notna(),
        np.hypot(buoy["x_m"] - buoy["truth_x_m"], buoy["y_m"] - buoy["truth_y_m"]),
        np.nan,
    )
    buoy.to_csv(args.output_dir / "buoy_trajectory_graph.csv", index=False)
    errors = buoy["error_m"].dropna().to_numpy(float)
    final = buoy.loc[
        buoy.image_index == len(image_ids) - 1, "error_m"
    ].dropna().to_numpy(float)
    summary["buoys"] = {
        **trajectory_summary(buoy),
        "comparisons": len(errors),
        "median_error_m": float(np.median(errors)) if len(errors) else None,
        "p90_error_m": float(np.quantile(errors, 0.90)) if len(errors) else None,
        "final_median_error_m": float(np.median(final)) if len(final) else None,
        "final_p90_error_m": float(np.quantile(final, 0.90)) if len(final) else None,
        "final_maximum_error_m": float(np.max(final)) if len(final) else None,
    }
    summary["edges"] = [
        {
            "source_image_id": edge.source_image_id,
            "target_image_id": edge.target_image_id,
            "elapsed_hours": edge.elapsed_hours,
        }
        for edge in edges
    ]
    (args.output_dir / "report.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
