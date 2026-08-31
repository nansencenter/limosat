#!/usr/bin/env python3
"""Build strict and gap-aware trajectories from persisted pair drift fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_efficientloftr_sequence import (
    PairSpec,
    field_from_csv,
    save_trajectory_products,
)
from limosat.learned_drift import ALIKEDConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-run-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-pair-run-dir", type=Path, action="append", required=True
    )
    parser.add_argument("--comparison-input", type=Path)
    parser.add_argument("--method-name", default="efficientloftr_nearest12")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = []
    fields = []
    for reference_dir in args.reference_pair_run_dir:
        manifest = json.loads((reference_dir / "run_manifest.json").read_text())
        source_id = int(manifest["source_image_id"])
        target_id = int(manifest["target_image_id"])
        buoy_path = reference_dir / "buoy_results.csv"
        specs.append(
            PairSpec(
                source_id,
                target_id,
                manifest["source_image_filepath"],
                manifest["target_image_filepath"],
                float(manifest["elapsed_hours"]),
                buoy_path if buoy_path.exists() else None,
            )
        )
        fields.append(
            field_from_csv(
                args.sequence_run_dir
                / f"pair_{source_id}_{target_id}"
                / "field_4km.csv"
            )
        )
    for previous, current in zip(specs, specs[1:]):
        if previous.target_image_id != current.source_image_id:
            raise ValueError("reference pairs do not form a contiguous chain")
    summary = save_trajectory_products(
        specs, fields, args.sequence_run_dir, ALIKEDConfig()
    )
    if args.comparison_input is not None:
        learned_rows = []
        for spec in specs:
            rows = pd.read_csv(
                args.sequence_run_dir
                / f"pair_{spec.source_image_id}_{spec.target_image_id}"
                / "buoy_results.csv",
                dtype={"buoy_id": str},
            )
            rows["method"] = args.method_name
            rows["estimated_dx_m"] = rows["proposal_dx_m"]
            rows["estimated_dy_m"] = rows["proposal_dy_m"]
            rows["analysis_crs"] = "EPSG:3413"
            learned_rows.append(rows)
        base = pd.read_csv(args.comparison_input, dtype={"buoy_id": str})
        base = base.loc[base["method"] != args.method_name]
        columns = [
            "transition_id",
            "source_image_id",
            "target_image_id",
            "method",
            "buoy_id",
            "source_x",
            "source_y",
            "truth_dx_m",
            "truth_dy_m",
            "estimated_dx_m",
            "estimated_dy_m",
            "available",
            "elapsed_hours",
            "analysis_crs",
        ]
        pd.concat(
            [base[columns], pd.concat(learned_rows, ignore_index=True)[columns]],
            ignore_index=True,
        ).to_csv(
            args.sequence_run_dir / "buoy_deformation_comparison_input.csv",
            index=False,
        )
    output = args.sequence_run_dir / "trajectory_summary_v2.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
