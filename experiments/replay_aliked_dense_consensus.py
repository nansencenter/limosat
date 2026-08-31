#!/usr/bin/env python3
"""Replay spatial support radii on a cached ALIKED dense-pair field."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_aliked_dense_pair import (
    adaptive_consensus_at_queries,
    consensus_at_queries,
    nearest_consensus_at_queries,
    summarise,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_radii(value: str) -> list[float]:
    radii = [float(item) for item in value.split(",")]
    if not radii or any(radius <= 0 for radius in radii):
        raise argparse.ArgumentTypeError("radii must be positive comma-separated values")
    return radii


def parse_counts(value: str) -> list[int]:
    counts = [int(item) for item in value.split(",")]
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("counts must be positive comma-separated values")
    return counts


def evaluate_policy(
    field: pd.DataFrame,
    buoy: pd.DataFrame,
    orb_context: pd.DataFrame,
    query_columns: list[str],
    matches: pd.DataFrame,
    paired_orb: int,
    grid_spacing_m: float,
    metadata: dict,
    started: float,
) -> tuple[pd.DataFrame, dict]:
    field = field.merge(orb_context, on=query_columns, validate="one_to_one")
    both = field["available"].fillna(False) & field[
        "orb_available_10km"
    ].fillna(False)
    field["aliked_orb_vector_difference_m"] = np.nan
    field.loc[both, "aliked_orb_vector_difference_m"] = np.hypot(
        field.loc[both, "proposal_dx_m"] - field.loc[both, "orb_dx_m"],
        field.loc[both, "proposal_dy_m"] - field.loc[both, "orb_dy_m"],
    )
    summary = summarise(field, matches, paired_orb, grid_spacing_m)
    available_buoy = buoy["available"].fillna(False)
    errors = np.hypot(
        buoy.loc[available_buoy, "proposal_dx_m"]
        - buoy.loc[available_buoy, "truth_dx_m"],
        buoy.loc[available_buoy, "proposal_dy_m"]
        - buoy.loc[available_buoy, "truth_dy_m"],
    )
    summary.update(
        metadata
        | {
            "buoy_cases": int(len(buoy)),
            "buoy_available": int(available_buoy.sum()),
            "buoy_correct_within_2km": int((errors <= 2000.0).sum()),
            "buoy_median_error_m": float(errors.median()) if len(errors) else None,
            "buoy_p90_error_m": float(errors.quantile(0.90)) if len(errors) else None,
            "replay_seconds": time.perf_counter() - started,
        }
    )
    return field, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tight-radii-m",
        type=parse_radii,
        default=[2000, 3000, 4000, 6000],
    )
    parser.add_argument("--consensus-radius-m", type=float, default=1000.0)
    parser.add_argument("--grid-spacing-m", type=float, default=4000.0)
    parser.add_argument(
        "--adaptive-min-vectors", type=parse_counts, default=[3, 5, 8]
    )
    parser.add_argument(
        "--nearest-candidate-counts", type=parse_counts, default=[8, 12]
    )
    parser.add_argument("--nearest-min-vectors", type=int, default=8)
    parser.add_argument("--nearest-maximum-radius-m", type=float, default=6000.0)
    parser.add_argument(
        "--nearest-only",
        action="store_true",
        help="Skip development sweeps and replay only requested nearest counts.",
    )
    args = parser.parse_args()
    if args.nearest_only:
        args.tight_radii_m = []
        args.adaptive_min_vectors = []

    matches_path = args.dense_run_dir / "matches.csv"
    field_path = args.dense_run_dir / "field_4km.csv"
    buoy_path = args.dense_run_dir / "buoy_results.csv"
    matches = pd.read_csv(matches_path)
    base_field = pd.read_csv(field_path)
    buoy_queries = pd.read_csv(buoy_path, low_memory=False)
    query_columns = ["grid_row", "grid_column", "source_x", "source_y"]
    orb_columns = [
        column
        for column in base_field.columns
        if column.startswith("orb_")
    ]
    orb_context = base_field[query_columns + orb_columns].copy()
    run_summary = json.loads((args.dense_run_dir / "summary.json").read_text())
    paired_orb = int(run_summary["orb_paired_trajectories"])

    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tight_radius_m in args.tight_radii_m:
        started = time.perf_counter()
        field = consensus_at_queries(
            matches,
            base_field[query_columns],
            tight_radius_m,
            args.consensus_radius_m,
        ).merge(orb_context, on=query_columns, validate="one_to_one")
        both = field["available"].fillna(False) & field[
            "orb_available_10km"
        ].fillna(False)
        field["aliked_orb_vector_difference_m"] = np.nan
        field.loc[both, "aliked_orb_vector_difference_m"] = np.hypot(
            field.loc[both, "proposal_dx_m"] - field.loc[both, "orb_dx_m"],
            field.loc[both, "proposal_dy_m"] - field.loc[both, "orb_dy_m"],
        )
        summary = summarise(field, matches, paired_orb, args.grid_spacing_m)
        buoy = consensus_at_queries(
            matches,
            buoy_queries,
            tight_radius_m,
            args.consensus_radius_m,
        )
        available_buoy = buoy["available"].fillna(False)
        errors = np.hypot(
            buoy.loc[available_buoy, "proposal_dx_m"]
            - buoy.loc[available_buoy, "truth_dx_m"],
            buoy.loc[available_buoy, "proposal_dy_m"]
            - buoy.loc[available_buoy, "truth_dy_m"],
        )
        summary.update(
            {
                "policy": "fixed_radius",
                "tight_radius_m": tight_radius_m,
                "minimum_selected_vectors": None,
                "consensus_radius_m": args.consensus_radius_m,
                "buoy_cases": int(len(buoy)),
                "buoy_available": int(available_buoy.sum()),
                "buoy_correct_within_2km": int((errors <= 2000.0).sum()),
                "buoy_median_error_m": float(errors.median()) if len(errors) else None,
                "buoy_p90_error_m": float(errors.quantile(0.90)) if len(errors) else None,
                "replay_seconds": time.perf_counter() - started,
            }
        )
        rows.append(
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, dict)
                else value
                for key, value in summary.items()
            }
        )
        field.to_csv(
            args.output_dir / f"field_radius_{int(tight_radius_m)}m.csv",
            index=False,
        )
        buoy.to_csv(
            args.output_dir / f"buoy_radius_{int(tight_radius_m)}m.csv",
            index=False,
        )
        print(
            f"radius={tight_radius_m:.0f} m coverage="
            f"{summary['aliked_coverage_fraction']:.3f} adjacent_p90="
            f"{summary['aliked_p90_adjacent_vector_difference_m']:.1f} m",
            flush=True,
        )
    for minimum_selected_vectors in args.adaptive_min_vectors:
        started = time.perf_counter()
        field = adaptive_consensus_at_queries(
            matches,
            base_field[query_columns],
            args.tight_radii_m,
            minimum_selected_vectors,
            args.consensus_radius_m,
        ).merge(orb_context, on=query_columns, validate="one_to_one")
        both = field["available"].fillna(False) & field[
            "orb_available_10km"
        ].fillna(False)
        field["aliked_orb_vector_difference_m"] = np.nan
        field.loc[both, "aliked_orb_vector_difference_m"] = np.hypot(
            field.loc[both, "proposal_dx_m"] - field.loc[both, "orb_dx_m"],
            field.loc[both, "proposal_dy_m"] - field.loc[both, "orb_dy_m"],
        )
        summary = summarise(field, matches, paired_orb, args.grid_spacing_m)
        buoy = adaptive_consensus_at_queries(
            matches,
            buoy_queries,
            args.tight_radii_m,
            minimum_selected_vectors,
            args.consensus_radius_m,
        )
        available_buoy = buoy["available"].fillna(False)
        errors = np.hypot(
            buoy.loc[available_buoy, "proposal_dx_m"]
            - buoy.loc[available_buoy, "truth_dx_m"],
            buoy.loc[available_buoy, "proposal_dy_m"]
            - buoy.loc[available_buoy, "truth_dy_m"],
        )
        summary.update(
            {
                "policy": "adaptive_support",
                "tight_radius_m": None,
                "minimum_selected_vectors": minimum_selected_vectors,
                "consensus_radius_m": args.consensus_radius_m,
                "buoy_cases": int(len(buoy)),
                "buoy_available": int(available_buoy.sum()),
                "buoy_correct_within_2km": int((errors <= 2000.0).sum()),
                "buoy_median_error_m": float(errors.median()) if len(errors) else None,
                "buoy_p90_error_m": float(errors.quantile(0.90)) if len(errors) else None,
                "replay_seconds": time.perf_counter() - started,
            }
        )
        rows.append(
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, dict)
                else value
                for key, value in summary.items()
            }
        )
        field.to_csv(
            args.output_dir
            / f"field_adaptive_min_{minimum_selected_vectors}.csv",
            index=False,
        )
        buoy.to_csv(
            args.output_dir
            / f"buoy_adaptive_min_{minimum_selected_vectors}.csv",
            index=False,
        )
        print(
            f"adaptive_min={minimum_selected_vectors} coverage="
            f"{summary['aliked_coverage_fraction']:.3f} adjacent_p90="
            f"{summary['aliked_p90_adjacent_vector_difference_m']:.1f} m",
            flush=True,
        )
    for candidate_count in args.nearest_candidate_counts:
        started = time.perf_counter()
        field = nearest_consensus_at_queries(
            matches,
            base_field[query_columns],
            args.nearest_maximum_radius_m,
            candidate_count,
            args.nearest_min_vectors,
            args.consensus_radius_m,
        )
        buoy = nearest_consensus_at_queries(
            matches,
            buoy_queries,
            args.nearest_maximum_radius_m,
            candidate_count,
            args.nearest_min_vectors,
            args.consensus_radius_m,
        )
        field, summary = evaluate_policy(
            field,
            buoy,
            orb_context,
            query_columns,
            matches,
            paired_orb,
            args.grid_spacing_m,
            {
                "policy": "nearest_consensus",
                "tight_radius_m": None,
                "minimum_selected_vectors": args.nearest_min_vectors,
                "candidate_count": candidate_count,
                "maximum_radius_m": args.nearest_maximum_radius_m,
                "consensus_radius_m": args.consensus_radius_m,
            },
            started,
        )
        rows.append(
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, dict)
                else value
                for key, value in summary.items()
            }
        )
        field.to_csv(
            args.output_dir / f"field_nearest_{candidate_count}.csv", index=False
        )
        buoy.to_csv(
            args.output_dir / f"buoy_nearest_{candidate_count}.csv", index=False
        )
        print(
            f"nearest={candidate_count} coverage="
            f"{summary['aliked_coverage_fraction']:.3f} adjacent_p90="
            f"{summary['aliked_p90_adjacent_vector_difference_m']:.1f} m",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "summary.csv", index=False)
    manifest = {
        "status": "complete",
        "dense_run_dir": str(args.dense_run_dir),
        "matches_sha256": sha256(matches_path),
        "base_field_sha256": sha256(field_path),
        "buoy_queries_sha256": sha256(buoy_path),
        "tight_radii_m": args.tight_radii_m,
        "consensus_radius_m": args.consensus_radius_m,
        "grid_spacing_m": args.grid_spacing_m,
        "adaptive_minimum_selected_vectors": args.adaptive_min_vectors,
        "nearest_candidate_counts": args.nearest_candidate_counts,
        "nearest_minimum_selected_vectors": args.nearest_min_vectors,
        "nearest_maximum_radius_m": args.nearest_maximum_radius_m,
        "nearest_only": args.nearest_only,
        "selection_rule": (
            "Choose the smallest radius that materially reduces control-pair "
            "roughness without reducing buoy correctness or spatial coverage."
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
