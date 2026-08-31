#!/usr/bin/env python3
"""Run EfficientLoFTR over the physically connected full-70 image paths."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = (
    ROOT / "experiments/configs/efficientloftr_full70_sequences_20260824.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--efficientloftr-repo", type=Path, required=True)
    parser.add_argument("--efficientloftr-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument(
        "--routing-mode",
        choices=("same_center", "sequential", "sequential_global", "sequential_local"),
        default="sequential_local",
    )
    parser.add_argument(
        "--initial-routing",
        choices=("same_center", "phase_correlation"),
        default="phase_correlation",
    )
    return parser.parse_args()


def read_inputs(
    plan_path: Path, transitions_path: Path, observations_path: Path
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    plan = json.loads(plan_path.read_text())
    transitions = pd.read_csv(
        transitions_path,
        dtype={"buoy_id": str},
        parse_dates=["source_image_time", "target_image_time"],
    )
    observations = pd.read_csv(
        observations_path,
        dtype={"buoy_id": str},
        parse_dates=["image_time"],
    )
    return plan, transitions, observations


def validate_plan(plan: dict, transitions: pd.DataFrame) -> dict:
    available_edges = set(
        map(
            tuple,
            transitions[["source_image_id", "target_image_id"]]
            .drop_duplicates()
            .astype(int)
            .to_numpy(),
        )
    )
    available_images = set(transitions["source_image_id"].astype(int)) | set(
        transitions["target_image_id"].astype(int)
    )
    planned_images: set[int] = set()
    planned_edges: list[tuple[int, int]] = []
    names: set[str] = set()
    for sequence in plan["sequences"]:
        name = str(sequence["name"])
        if name in names:
            raise ValueError(f"duplicate sequence name: {name}")
        names.add(name)
        image_ids = [int(value) for value in sequence["image_ids"]]
        if len(image_ids) < 2 or len(set(image_ids)) != len(image_ids):
            raise ValueError(f"invalid image path: {name}")
        planned_images.update(image_ids)
        for edge in zip(image_ids, image_ids[1:]):
            if edge not in available_edges:
                raise ValueError(f"planned edge is absent from transitions: {edge}")
            planned_edges.append(edge)
    if planned_images != available_images:
        missing = sorted(available_images - planned_images)
        extra = sorted(planned_images - available_images)
        raise ValueError(f"plan/image mismatch; missing={missing}, extra={extra}")
    return {
        "unique_images": len(planned_images),
        "sequence_paths": len(plan["sequences"]),
        "pair_runs": len(planned_edges),
        "unique_pair_runs": len(set(planned_edges)),
    }


def pair_reference_rows(
    source_id: int,
    target_id: int,
    transitions: pd.DataFrame,
    observations: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    pair = transitions.loc[
        (transitions["source_image_id"] == source_id)
        & (transitions["target_image_id"] == target_id)
    ].copy()
    if pair.empty:
        raise ValueError(f"no transition rows for {source_id}->{target_id}")
    fixed = pair[
        [
            "source_image_time",
            "source_image_filepath",
            "target_image_time",
            "target_image_filepath",
            "elapsed_hours",
        ]
    ].drop_duplicates()
    if len(fixed) != 1:
        raise ValueError(f"inconsistent metadata for {source_id}->{target_id}")
    fixed_row = fixed.iloc[0]

    source_positions = observations.loc[
        observations["image_id"].eq(source_id),
        ["buoy_id", "x", "y"],
    ].drop_duplicates("buoy_id")
    if source_positions["buoy_id"].duplicated().any():
        raise ValueError(f"duplicate source buoy positions for image {source_id}")
    buoy = pair.merge(
        source_positions,
        on="buoy_id",
        how="left",
        validate="many_to_one",
    )
    if buoy[["x", "y"]].isna().any().any():
        raise ValueError(f"missing source buoy positions for {source_id}->{target_id}")
    buoy = buoy.rename(columns={"x": "source_x", "y": "source_y"})
    buoy = buoy[
        [
            "buoy_id",
            "source_x",
            "source_y",
            "truth_dx_m",
            "truth_dy_m",
            "elapsed_hours",
            "cadence_band",
            "experiment_split",
            "source_sic_regime",
            "target_sic_regime",
            "source_spatial_block",
            "target_spatial_block",
        ]
    ].drop_duplicates("buoy_id")
    manifest = {
        "source_image_id": source_id,
        "target_image_id": target_id,
        "source_image_time": fixed_row["source_image_time"].isoformat(),
        "target_image_time": fixed_row["target_image_time"].isoformat(),
        "source_image_filepath": str(fixed_row["source_image_filepath"]),
        "target_image_filepath": str(fixed_row["target_image_filepath"]),
        "elapsed_hours": float(fixed_row["elapsed_hours"]),
        "truth_source": "official_iabp_level1_linear_interpolation",
        "analysis_crs": "EPSG:3413",
        "buoys": len(buoy),
    }
    return manifest, buoy


def prepare_references(
    plan: dict,
    transitions: pd.DataFrame,
    observations: pd.DataFrame,
    output_dir: Path,
) -> dict[tuple[int, int], Path]:
    reference_root = output_dir / "reference_pairs"
    references: dict[tuple[int, int], Path] = {}
    for sequence in plan["sequences"]:
        image_ids = [int(value) for value in sequence["image_ids"]]
        for source_id, target_id in zip(image_ids, image_ids[1:]):
            edge = (source_id, target_id)
            if edge in references:
                continue
            pair_dir = reference_root / f"pair_{source_id}_{target_id}"
            manifest, buoy = pair_reference_rows(
                source_id, target_id, transitions, observations
            )
            pair_dir.mkdir(parents=True, exist_ok=True)
            (pair_dir / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            buoy.to_csv(pair_dir / "buoy_results.csv", index=False)
            references[edge] = pair_dir
    return references


def write_batch_manifest(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n")


def buoy_metrics(rows: pd.DataFrame) -> dict:
    available = rows["available"].fillna(False).astype(bool)
    errors = rows.loc[available, "error_m"].dropna().to_numpy(float)
    within_2km = available & rows["error_m"].le(2_000.0)
    return {
        "expected": len(rows),
        "available": int(available.sum()),
        "within_2km": int(within_2km.sum()),
        "availability_fraction": float(available.mean()) if len(rows) else None,
        "within_2km_of_expected_fraction": (
            float(within_2km.mean()) if len(rows) else None
        ),
        "within_2km_of_available_fraction": (
            float(within_2km.sum() / available.sum())
            if available.any()
            else None
        ),
        "median_error_m": float(np.median(errors)) if len(errors) else None,
        "p90_error_m": float(np.quantile(errors, 0.90)) if len(errors) else None,
        "p95_error_m": float(np.quantile(errors, 0.95)) if len(errors) else None,
        "maximum_error_m": float(np.max(errors)) if len(errors) else None,
    }


def summarize_completed_batch(plan: dict, output_dir: Path) -> dict:
    sequence_rows = []
    pair_rows = []
    buoy_frames = []
    for index, sequence in enumerate(plan["sequences"], start=1):
        sequence_dir = output_dir / f"sequence_{index:02d}_{sequence['name']}"
        manifest = json.loads((sequence_dir / "run_manifest.json").read_text())
        trajectories = manifest["trajectories"]
        sequence_rows.append(
            {
                "sequence": sequence["name"],
                "images": manifest["images"],
                "pairs": manifest["pairs"],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "pair_compute_seconds": manifest["pair_compute_seconds"],
                "trajectories_seeded": trajectories["seeded"],
                "trajectories_complete": trajectories["complete"],
                "complete_fraction": trajectories["complete_fraction"],
            }
        )
        for pair in manifest["pairs_summary"]:
            source_id = int(pair["source_image_id"])
            target_id = int(pair["target_image_id"])
            pair_rows.append(
                {
                    "sequence": sequence["name"],
                    "source_image_id": source_id,
                    "target_image_id": target_id,
                    "elapsed_hours": pair["elapsed_hours"],
                    "grid_nodes": pair["grid_nodes"],
                    "available_nodes": pair["available_after_fold_rejection"],
                    "coverage": pair["coverage_after_fold_rejection"],
                    "fold_rejected_nodes": pair["fold_rejected_nodes"],
                    "final_flipped_fraction": pair["topology_after_rejection"].get(
                        "flipped_fraction", 0.0
                    ),
                    "pair_seconds": pair["timing_seconds"]["pair_total"],
                    "sampling_seconds": pair["timing_seconds"]["sampling"],
                    "matching_seconds": pair["timing_seconds"]["matching"],
                }
            )
            buoy_path = sequence_dir / f"pair_{source_id}_{target_id}/buoy_results.csv"
            if buoy_path.exists():
                buoy = pd.read_csv(buoy_path, dtype={"buoy_id": str})
                buoy["sequence"] = sequence["name"]
                buoy["source_image_id"] = source_id
                buoy["target_image_id"] = target_id
                buoy_frames.append(buoy)

    sequences = pd.DataFrame(sequence_rows)
    pairs = pd.DataFrame(pair_rows)
    buoys = pd.concat(buoy_frames, ignore_index=True)
    unique_buoys = buoys.drop_duplicates(
        ["source_image_id", "target_image_id", "buoy_id"]
    ).copy()
    sequences.to_csv(output_dir / "sequence_summary.csv", index=False)
    pairs.to_csv(output_dir / "pair_summary.csv", index=False)
    buoys.to_csv(output_dir / "buoy_pair_run_results.csv", index=False)
    for column in (
        "cadence_band",
        "experiment_split",
        "source_sic_regime",
        "target_sic_regime",
    ):
        records = []
        for value, group in unique_buoys.groupby(column, dropna=False):
            records.append({column: value, **buoy_metrics(group)})
        pd.DataFrame(records).to_csv(
            output_dir / f"buoy_metrics_by_{column}.csv", index=False
        )

    weighted_coverage = float(pairs["available_nodes"].sum() / pairs["grid_nodes"].sum())
    strict_weighted = float(
        sequences["trajectories_complete"].sum()
        / sequences["trajectories_seeded"].sum()
    )
    return {
        "completed_sequence_science_elapsed_seconds": float(
            sequences["elapsed_seconds"].sum()
        ),
        "pair_compute_seconds": float(pairs["pair_seconds"].sum()),
        "matching_seconds": float(pairs["matching_seconds"].sum()),
        "sampling_seconds": float(pairs["sampling_seconds"].sum()),
        "pair_run_buoys": buoy_metrics(buoys),
        "unique_edge_buoys": buoy_metrics(unique_buoys),
        "coverage": {
            "pair_median": float(pairs["coverage"].median()),
            "pair_p10": float(pairs["coverage"].quantile(0.10)),
            "node_weighted": weighted_coverage,
            "pairs_at_least_50pct": int(pairs["coverage"].ge(0.50).sum()),
            "pairs_under_10pct": int(pairs["coverage"].lt(0.10).sum()),
        },
        "topology": {
            "fold_rejected_nodes": int(pairs["fold_rejected_nodes"].sum()),
            "final_fields_with_flips": int(
                pairs["final_flipped_fraction"].gt(0.0).sum()
            ),
        },
        "strict_trajectories": {
            "sequence_median_complete_fraction": float(
                sequences["complete_fraction"].median()
            ),
            "point_weighted_complete_fraction": strict_weighted,
            "zero_complete_sequences": int(
                sequences["complete_fraction"].eq(0.0).sum()
            ),
            "minimum_complete_fraction": float(
                sequences["complete_fraction"].min()
            ),
            "maximum_complete_fraction": float(
                sequences["complete_fraction"].max()
            ),
        },
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    plan, transitions, observations = read_inputs(
        args.plan, args.transitions, args.observations
    )
    coverage = validate_plan(plan, transitions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    references = prepare_references(
        plan, transitions, observations, args.output_dir
    )
    report_path = args.output_dir / "batch_manifest.json"
    report = {
        "status": "running",
        "plan": str(args.plan),
        "transitions": str(args.transitions),
        "observations": str(args.observations),
        "device": args.device,
        "routing_mode": args.routing_mode,
        "initial_routing": args.initial_routing,
        **coverage,
        "completed_sequences": 0,
        "sequences": [],
    }
    write_batch_manifest(report_path, report)

    runner = ROOT / "experiments/run_efficientloftr_sequence.py"
    try:
        for index, sequence in enumerate(plan["sequences"], start=1):
            image_ids = [int(value) for value in sequence["image_ids"]]
            sequence_dir = args.output_dir / (
                f"sequence_{index:02d}_{sequence['name']}"
            )
            command = [
                sys.executable,
                str(runner),
                "--efficientloftr-repo",
                str(args.efficientloftr_repo),
                "--efficientloftr-checkpoint",
                str(args.efficientloftr_checkpoint),
                "--output-dir",
                str(sequence_dir),
                "--routing-mode",
                args.routing_mode,
                "--initial-routing",
                args.initial_routing,
                "--device",
                args.device,
            ]
            for edge in zip(image_ids, image_ids[1:]):
                command.extend(["--reference-pair-run-dir", str(references[edge])])
            print(
                json.dumps(
                    {
                        "sequence": index,
                        "name": sequence["name"],
                        "image_ids": image_ids,
                    }
                ),
                flush=True,
            )
            subprocess.run(command, check=True)
            sequence_manifest = json.loads(
                (sequence_dir / "run_manifest.json").read_text()
            )
            report["sequences"].append(
                {
                    "name": sequence["name"],
                    "image_ids": image_ids,
                    "output_dir": str(sequence_dir),
                    "status": sequence_manifest["status"],
                    "elapsed_seconds": sequence_manifest["elapsed_seconds"],
                    "current_execution_elapsed_seconds": sequence_manifest[
                        "current_execution_elapsed_seconds"
                    ],
                    "trajectory_summary": sequence_manifest["trajectories"],
                }
            )
            report["completed_sequences"] = index
            report["current_execution_elapsed_seconds"] = (
                time.perf_counter() - started
            )
            write_batch_manifest(report_path, report)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["current_execution_elapsed_seconds"] = time.perf_counter() - started
        write_batch_manifest(report_path, report)
        raise

    report["aggregate_summary"] = summarize_completed_batch(
        plan, args.output_dir
    )
    report["status"] = "complete"
    report["current_execution_elapsed_seconds"] = time.perf_counter() - started
    write_batch_manifest(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
