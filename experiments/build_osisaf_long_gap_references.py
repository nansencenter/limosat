#!/usr/bin/env python3
"""Build predeclared first-to-last 2020 gap references from buoy observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "experiments/configs/efficientloftr_full70_sequences_20260824.json"
DEFAULT_OBSERVATIONS = Path(
    "/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/"
    "splits/full70_2020/observations.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/osisaf_routing_prior_audit_20260831"
    / "expanded_gap_references"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-hours", type=float, default=30.0)
    return parser.parse_args()


def one_value(rows: pd.DataFrame, column: str):
    values = rows[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"expected one {column}, found {values.tolist()}")
    return values.iloc[0]


def build_reference_rows(
    observations: pd.DataFrame,
    source_id: int,
    target_id: int,
) -> tuple[dict, pd.DataFrame]:
    source = observations.loc[observations["image_id"] == source_id].copy()
    target = observations.loc[observations["image_id"] == target_id].copy()
    if source.empty or target.empty:
        raise ValueError(f"missing observations for {source_id}->{target_id}")
    if source["buoy_id"].duplicated().any() or target["buoy_id"].duplicated().any():
        raise ValueError("buoy ids must be unique within each image")
    source_time = pd.Timestamp(one_value(source, "image_time"))
    target_time = pd.Timestamp(one_value(target, "image_time"))
    elapsed_hours = (target_time - source_time).total_seconds() / 3600.0
    shared = source.merge(
        target,
        on="buoy_id",
        how="inner",
        suffixes=("_source", "_target"),
        validate="one_to_one",
    )
    rows = pd.DataFrame(
        {
            "transition_id": [
                f"gap_{source_id}_{target_id}_{buoy_id}"
                for buoy_id in shared["buoy_id"]
            ],
            "buoy_id": shared["buoy_id"].astype(str),
            "source_image_id": source_id,
            "target_image_id": target_id,
            "source_image_filepath": one_value(source, "image_filepath"),
            "target_image_filepath": one_value(target, "image_filepath"),
            "elapsed_hours": elapsed_hours,
            "source_x": shared["x_source"].to_numpy(float),
            "source_y": shared["y_source"].to_numpy(float),
            "truth_dx_m": (
                shared["x_target"].to_numpy(float)
                - shared["x_source"].to_numpy(float)
            ),
            "truth_dy_m": (
                shared["y_target"].to_numpy(float)
                - shared["y_source"].to_numpy(float)
            ),
            "analysis_crs": "EPSG:3413",
            "experiment_split": shared["experiment_split_source"],
            "source_sic_regime": shared["sic_regime_source"],
            "target_sic_regime": shared["sic_regime_target"],
            "source_spatial_block": shared["spatial_block_source"],
            "target_spatial_block": shared["spatial_block_target"],
        }
    )
    if len(rows):
        truth_distance = np.hypot(rows["truth_dx_m"], rows["truth_dy_m"])
        truth_speed = truth_distance / elapsed_hours * 24.0 / 1000.0
    else:
        truth_distance = pd.Series(dtype=float)
        truth_speed = pd.Series(dtype=float)
    manifest = {
        "status": "complete",
        "source_image_id": source_id,
        "target_image_id": target_id,
        "source_image_time": source_time.isoformat(),
        "target_image_time": target_time.isoformat(),
        "source_image_filepath": one_value(source, "image_filepath"),
        "target_image_filepath": one_value(target, "image_filepath"),
        "elapsed_hours": elapsed_hours,
        "truth_source": "official_iabp_level1_linear_interpolation",
        "analysis_crs": "EPSG:3413",
        "buoys": len(rows),
        "median_truth_distance_km": (
            float(truth_distance.median() / 1000.0) if len(rows) else None
        ),
        "median_truth_speed_km_per_day": (
            float(truth_speed.median()) if len(rows) else None
        ),
    }
    return manifest, rows


def build_all(
    plan: dict,
    observations: pd.DataFrame,
    output_dir: Path,
    minimum_hours: float,
) -> dict:
    if minimum_hours <= 0:
        raise ValueError("minimum hours must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[int, int]] = set()
    selected = []
    excluded = []
    for sequence in plan["sequences"]:
        source_id = int(sequence["image_ids"][0])
        target_id = int(sequence["image_ids"][-1])
        edge = (source_id, target_id)
        if edge in seen:
            excluded.append(
                {"sequence": sequence["name"], "edge": list(edge), "reason": "duplicate_edge"}
            )
            continue
        seen.add(edge)
        manifest, rows = build_reference_rows(
            observations, source_id, target_id
        )
        if manifest["elapsed_hours"] < minimum_hours:
            excluded.append(
                {"sequence": sequence["name"], "edge": list(edge), "reason": "below_minimum_hours"}
            )
            continue
        if rows.empty:
            excluded.append(
                {"sequence": sequence["name"], "edge": list(edge), "reason": "no_shared_buoy_truth"}
            )
            continue
        pair_dir = output_dir / f"pair_{source_id}_{target_id}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        rows.to_csv(pair_dir / "buoy_results.csv", index=False)
        (pair_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        selected.append(
            {
                "sequence": sequence["name"],
                "case_id": f"full70_simulated_gap_{source_id}_{target_id}",
                "reference_pair_dir": str(pair_dir),
                **manifest,
            }
        )
    report = {
        "status": "complete",
        "selection_rule": (
            "unique first-to-last edge from every frozen full70 component; "
            f"elapsed >= {minimum_hours:g} h; at least one shared buoy"
        ),
        "minimum_hours": minimum_hours,
        "selected_pairs": len(selected),
        "selected": selected,
        "excluded": excluded,
    }
    (output_dir / "expansion_plan.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    observations = pd.read_csv(
        args.observations,
        dtype={"buoy_id": str},
        parse_dates=["image_time"],
    )
    report = build_all(plan, observations, args.output_dir, args.minimum_hours)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
