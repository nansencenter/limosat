#!/usr/bin/env python3
"""Run a reproducible ORB contract matrix through the Arctic graph benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


MATRIX = (
    ("current_hamming", {}),
    ("current_hamming2", {"descriptor_norm": "hamming2"}),
    ("valid_octave4", {"octave": 4}),
    ("octave0", {"octave": 0}),
    ("keypoint_size64_octave0", {"octave": 0, "keypoint_size": 64}),
    ("keypoint_size64_octave5", {"octave": 5, "keypoint_size": 64}),
    (
        "default_n8_invalid_octave8",
        {
            "orb_nlevels": 8,
            "orb_patch_size": 31,
            "orb_edge_threshold": 31,
            "keypoint_size": 31,
            "octave": 8,
        },
    ),
    (
        "default_n8_invalid_octave8_hamming2",
        {
            "orb_nlevels": 8,
            "orb_patch_size": 31,
            "orb_edge_threshold": 31,
            "keypoint_size": 31,
            "octave": 8,
            "descriptor_norm": "hamming2",
        },
    ),
    (
        "default_n8_valid_octave7",
        {
            "orb_nlevels": 8,
            "orb_patch_size": 31,
            "orb_edge_threshold": 31,
            "keypoint_size": 31,
            "octave": 7,
        },
    ),
    ("zero_angle", {"angle_mode": "zero"}),
)


def flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--buoys", type=Path)
    parser.add_argument(
        "--coincidences",
        type=Path,
        help="Normalized exact-time coincidence CSV used by orb_multiframe_graph.py.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--graph-configs",
        default="greedy_rolling,beam_anchor",
    )
    args = parser.parse_args()
    if args.coincidences is None and (args.catalog is None or args.buoys is None):
        parser.error("Provide --coincidences or both --catalog and --buoys.")
    if args.coincidences is not None and (args.catalog is not None or args.buoys is not None):
        parser.error("Use --coincidences or --catalog plus --buoys, not both.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    graph_script = Path(__file__).with_name("orb_multiframe_graph.py")
    rows = []
    timings = []
    started = time.perf_counter()
    for case_name, overrides in MATRIX:
        case_dir = args.out_dir / case_name
        command = [
            sys.executable,
            str(graph_script),
            "--out-dir",
            str(case_dir),
            "--graph-configs",
            args.graph_configs,
        ]
        if args.coincidences is not None:
            command.extend(
                [
                    "--coincidences",
                    str(args.coincidences),
                    "--invalid-support-policy",
                    "skip",
                ]
            )
        else:
            command.extend(
                ["--catalog", str(args.catalog), "--buoys", str(args.buoys)]
            )
        for key, value in overrides.items():
            command.extend([flag_name(key), str(value)])
        case_started = time.perf_counter()
        subprocess.run(command, check=True)
        timings.append(
            {
                "contract_case": case_name,
                "seconds": time.perf_counter() - case_started,
            }
        )
        summary = pd.read_csv(case_dir / "summary.csv")
        for row in summary.to_dict("records"):
            rows.append({"contract_case": case_name, **overrides, **row})

    results = pd.DataFrame.from_records(rows)
    results.to_csv(args.out_dir / "contract_matrix_summary.csv", index=False)
    pd.DataFrame(timings).to_csv(args.out_dir / "contract_matrix_timings.csv", index=False)
    anchor = results[results.config == "beam_anchor"].copy()
    anchor["median_error_km"] = anchor.median_error_m / 1000.0
    anchor["p90_error_km"] = anchor.p90_error_m / 1000.0
    anchor["long_path_final_error_km"] = anchor.long_path_final_error_m / 1000.0
    columns = [
        "contract_case",
        "median_error_km",
        "p90_error_km",
        "within_2km_fraction",
        "catastrophic_50km_fraction",
        "long_path_final_error_km",
    ]
    view = anchor.sort_values(
        ["within_2km_fraction", "long_path_final_error_km"],
        ascending=[False, True],
    )[columns]
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(lambda value: f"{value:.3f}")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()],
    ]
    (args.out_dir / "report.md").write_text(
        f"""# ORB graph contract matrix

Date: {pd.Timestamp.now(tz='UTC').date()}

The standard Arctic VAE input, buoy coincidences, 16-pixel grid, 50 km/day
physics gate, and graph configurations are fixed. Each arm changes only the
listed ORB norm, supplied-keypoint, scale, or orientation contract.

## Beam anchor results

{chr(10).join(lines)}

The current arm is `nlevels=5`, `patchSize=64`, `edgeThreshold=16`, supplied
keypoint `size=31`, `octave=5`, geographic orientation, `WTA_K=2`, and Hamming.
`current_hamming2` changes only the distance definition to reproduce LiMOSAT's
current matcher default.
"""
    )
    manifest = {
        "catalog": None if args.catalog is None else str(args.catalog),
        "buoys": None if args.buoys is None else str(args.buoys),
        "coincidences": None if args.coincidences is None else str(args.coincidences),
        "out_dir": str(args.out_dir),
        "graph_configs": args.graph_configs,
        "matrix": [{"name": name, **values} for name, values in MATRIX],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(view.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
