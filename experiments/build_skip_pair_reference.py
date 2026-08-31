#!/usr/bin/env python3
"""Compose a non-adjacent buoy/SAR reference from two adjacent pair fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-pair-dir", type=Path, required=True)
    parser.add_argument("--second-pair-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    first_manifest = json.loads(
        (args.first_pair_dir / "run_manifest.json").read_text()
    )
    second_manifest = json.loads(
        (args.second_pair_dir / "run_manifest.json").read_text()
    )
    if first_manifest["target_image_id"] != second_manifest["source_image_id"]:
        raise ValueError("input pairs are not adjacent")
    if (
        first_manifest["target_image_filepath"]
        != second_manifest["source_image_filepath"]
    ):
        raise ValueError("shared image paths disagree")

    first = pd.read_csv(
        args.first_pair_dir / "buoy_results.csv", dtype={"buoy_id": str}
    )
    second = pd.read_csv(
        args.second_pair_dir / "buoy_results.csv", dtype={"buoy_id": str}
    )
    shared = first.merge(
        second,
        on="buoy_id",
        suffixes=("_first", "_second"),
        how="inner",
        validate="one_to_one",
    )
    first_target = shared[["source_x_first", "source_y_first"]].to_numpy() + shared[
        ["truth_dx_m_first", "truth_dy_m_first"]
    ].to_numpy()
    second_source = shared[["source_x_second", "source_y_second"]].to_numpy()
    shared_image_position_difference_m = np.linalg.norm(
        first_target - second_source, axis=1
    )
    final_target = second_source + shared[
        ["truth_dx_m_second", "truth_dy_m_second"]
    ].to_numpy()
    source = shared[["source_x_first", "source_y_first"]].to_numpy()
    truth = final_target - source
    rows = pd.DataFrame(
        {
            "transition_id": [
                f"skip_{first_manifest['source_image_id']}_{second_manifest['target_image_id']}_{buoy_id}"
                for buoy_id in shared["buoy_id"]
            ],
            "buoy_id": shared["buoy_id"],
            "source_image_id": first_manifest["source_image_id"],
            "target_image_id": second_manifest["target_image_id"],
            "source_image_filepath": first_manifest["source_image_filepath"],
            "target_image_filepath": second_manifest["target_image_filepath"],
            "elapsed_hours": float(first_manifest["elapsed_hours"])
            + float(second_manifest["elapsed_hours"]),
            "source_x": source[:, 0],
            "source_y": source[:, 1],
            "truth_dx_m": truth[:, 0],
            "truth_dy_m": truth[:, 1],
            "analysis_crs": "EPSG:3413",
        }
    )
    manifest = {
        "status": "complete",
        "source_image_id": first_manifest["source_image_id"],
        "target_image_id": second_manifest["target_image_id"],
        "source_image_filepath": first_manifest["source_image_filepath"],
        "target_image_filepath": second_manifest["target_image_filepath"],
        "elapsed_hours": float(rows["elapsed_hours"].iloc[0]),
        "buoys": len(rows),
        "shared_image_position_difference_median": float(
            np.median(shared_image_position_difference_m)
        ),
        "shared_image_position_difference_maximum": float(
            np.max(shared_image_position_difference_m)
        ),
        "source_pair_directories": [
            str(args.first_pair_dir),
            str(args.second_pair_dir),
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "buoy_results.csv", index=False)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
