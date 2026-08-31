#!/usr/bin/env python3
"""Re-summarize an ORB contract matrix with all transitions retained."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_case(case_dir: Path, contract_case: str, overrides: dict) -> list[dict]:
    coincidences = pd.read_csv(case_dir / "coincidences.csv", dtype={"buoy_id": str})
    records = pd.read_csv(case_dir / "trajectory_results.csv", dtype={"buoy_id": str})
    eligible_paths = int((coincidences.groupby("buoy_id").size() >= 2).sum())
    eligible_transitions = int(
        coincidences.groupby("buoy_id").size().sub(1).clip(lower=0).sum()
    )
    rows = []
    for config, group in records.groupby("config", sort=False):
        tracked = group[
            (group.status == "ok")
            & (group.observation_index.fillna(-1) > 0)
            & np.isfinite(group.endpoint_error_m)
        ]
        errors = tracked.endpoint_error_m.to_numpy(dtype=float)
        rows.append(
            {
                "contract_case": contract_case,
                **overrides,
                "config": config,
                "eligible_paths": eligible_paths,
                "eligible_transitions": eligible_transitions,
                "seed_unavailable_paths": int(
                    group.loc[group.status == "seed_unavailable", "buoy_id"].nunique()
                ),
                "graph_failed_paths": int(
                    group.loc[group.status == "graph_failed", "buoy_id"].nunique()
                ),
                "tracked_transitions": len(tracked),
                "tracking_fraction_all": float(
                    len(tracked) / max(eligible_transitions, 1)
                ),
                "within_2km_fraction_all": float(
                    np.sum(errors <= 2000.0) / max(eligible_transitions, 1)
                ),
                "within_5km_fraction_all": float(
                    np.sum(errors <= 5000.0) / max(eligible_transitions, 1)
                ),
                "catastrophic_50km_fraction_all": float(
                    np.sum(errors > 50000.0) / max(eligible_transitions, 1)
                ),
                "median_tracked_error_m": (
                    float(np.median(errors)) if len(errors) else math.nan
                ),
            }
        )
    return rows


def markdown_table(data: pd.DataFrame, columns: list[str]) -> str:
    view = data[columns].copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()],
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.matrix_dir / "run_manifest.json").read_text())
    rows = []
    for case in manifest["matrix"]:
        name = case["name"]
        overrides = {key: value for key, value in case.items() if key != "name"}
        rows.extend(summarize_case(args.matrix_dir / name, name, overrides))
    summary = pd.DataFrame.from_records(rows)
    summary.to_csv(
        args.matrix_dir / "contract_matrix_all_transition_summary.csv", index=False
    )

    view = summary[summary.config == "beam_confidence_update_m032"].sort_values(
        [
            "within_2km_fraction_all",
            "catastrophic_50km_fraction_all",
            "tracking_fraction_all",
        ],
        ascending=[False, True, False],
    )
    columns = [
        "contract_case",
        "seed_unavailable_paths",
        "graph_failed_paths",
        "tracking_fraction_all",
        "within_2km_fraction_all",
        "catastrophic_50km_fraction_all",
        "median_tracked_error_m",
    ]
    (args.matrix_dir / "report_all_transitions.md").write_text(
        "# ORB contract matrix: all-transition summary\n\n"
        "Every exact-time transition remains in the denominator. A missing seed "
        "or failed graph therefore cannot improve the accuracy fractions by "
        "removing a difficult path. The table shows the confidence-gated memory "
        "policy; the CSV also contains the fixed-first-view policy.\n\n"
        + markdown_table(view, columns)
        + "\n"
    )
    print(view[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
