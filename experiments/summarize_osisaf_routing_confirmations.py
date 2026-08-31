#!/usr/bin/env python3
"""Summarize bounded MPS confirmations of OSI-455 tile routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    ROOT
    / "results/osisaf_routing_prior_audit_20260831"
    / "mps_confirmations"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def optional_metric(values: dict | None, key: str) -> float:
    if values is None or values.get(key) is None:
        return np.nan
    return float(values[key])


def confirmation_row(manifest: dict) -> dict:
    baseline = manifest["baseline"]
    assisted = manifest["osisaf_assisted"]
    baseline_buoy = baseline.get("buoys")
    assisted_buoy = assisted.get("buoys")
    baseline_coverage = float(baseline["coverage_after_fold_rejection"])
    assisted_coverage = float(assisted["coverage_after_fold_rejection"])
    return {
        "case_id": manifest["case_id"],
        "role": manifest["role"],
        "elapsed_hours": manifest["elapsed_hours"],
        "device": manifest["device"],
        "fallback": manifest["fallback"],
        "osi455_available_tile_fraction": manifest[
            "osi455_available_tile_fraction"
        ],
        "osi455_available_tiles": manifest["osi455_available_tiles"],
        "source_tiles": manifest["source_tiles"],
        "physics_clipped_tiles": manifest["physics_clipped_tiles"],
        "baseline_routing_sources": json.dumps(
            baseline["routing"]["source_counts"], sort_keys=True
        ),
        "baseline_matches": baseline["physics_valid_matches"],
        "assisted_matches": assisted["physics_valid_matches"],
        "match_ratio": (
            assisted["physics_valid_matches"] / baseline["physics_valid_matches"]
            if baseline["physics_valid_matches"]
            else np.nan
        ),
        "baseline_coverage": baseline_coverage,
        "assisted_coverage": assisted_coverage,
        "coverage_change_percentage_points": 100.0
        * (assisted_coverage - baseline_coverage),
        "baseline_buoys_available": optional_metric(baseline_buoy, "available"),
        "assisted_buoys_available": optional_metric(assisted_buoy, "available"),
        "baseline_buoys_within_2km": optional_metric(
            baseline_buoy, "correct_within_2km"
        ),
        "assisted_buoys_within_2km": optional_metric(
            assisted_buoy, "correct_within_2km"
        ),
        "baseline_buoy_median_error_m": optional_metric(
            baseline_buoy, "median_error_m"
        ),
        "assisted_buoy_median_error_m": optional_metric(
            assisted_buoy, "median_error_m"
        ),
        "baseline_pair_seconds": baseline["timing_seconds"]["pair_total"],
        "assisted_pair_seconds": assisted["timing_seconds"]["pair_total"],
    }


def routing_provenance_metrics(path: Path) -> dict:
    rows = pd.read_csv(path)
    available = rows.loc[rows["osi455_available"].fillna(False)]
    return {
        "osi455_wind_involved_tiles": int(
            (available["osi455_wind_fraction"] > 0).sum()
        ),
        "osi455_wind_involved_fraction_of_available": (
            float((available["osi455_wind_fraction"] > 0).mean())
            if len(available)
            else np.nan
        ),
        "osi455_median_accumulated_uncertainty_km": (
            float(available["osi455_uncertainty_m"].median() / 1000.0)
            if len(available)
            else np.nan
        ),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    printable = frame.copy()
    numeric = printable.select_dtypes(include=[np.number]).columns
    printable[numeric] = printable[numeric].round(3)
    headings = list(map(str, printable.columns))
    lines = ["| " + " | ".join(headings) + " |"]
    lines.append("| " + " | ".join("---" for _ in headings) + " |")
    for row in printable.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    paths = sorted(args.root.glob("*/run_manifest.json"))
    if not paths:
        raise FileNotFoundError(f"no confirmation manifests below {args.root}")
    manifests = [json.loads(path.read_text()) for path in paths]
    if any(manifest.get("device") != "mps" for manifest in manifests):
        raise ValueError("all confirmation runs must use MPS")
    records = []
    for path, manifest in zip(paths, manifests, strict=True):
        record = confirmation_row(manifest)
        record.update(routing_provenance_metrics(path.parent / "routing_prior_tiles.csv"))
        records.append(record)
    rows = pd.DataFrame(records)
    rows = rows.sort_values(["elapsed_hours", "case_id"]).reset_index(drop=True)
    rows.to_csv(args.root / "confirmation_summary.csv", index=False)
    display = rows[
        [
            "case_id",
            "role",
            "elapsed_hours",
            "osi455_available_tile_fraction",
            "osi455_wind_involved_fraction_of_available",
            "osi455_median_accumulated_uncertainty_km",
            "baseline_coverage",
            "assisted_coverage",
            "coverage_change_percentage_points",
            "baseline_buoys_available",
            "assisted_buoys_available",
            "baseline_buoy_median_error_m",
            "assisted_buoy_median_error_m",
        ]
    ]
    report = "\n".join(
        [
            "# OSI-455 MPS routing confirmations",
            "",
            "Four preselected mechanistic cases were run with the same EfficientLoFTR "
            "matcher and field estimator. Only target-tile centres changed: OSI-455 "
            "where strictly available, same-centre otherwise.",
            "",
            markdown_table(display),
            "",
            "## Guardrails",
            "",
            "- These cases confirm mechanism and compute cost; they are not an independent accuracy test.",
            "- OSI-455 availability is strict at every integrated daily step; no nearest-valid spatial filling was used.",
            "- The baseline for the 71.6-hour case is inherited SAR-field routing; the other baselines use global phase correlation.",
            "- Accepted vectors remain SAR-to-SAR EfficientLoFTR matches and the unchanged local consensus field.",
        ]
    )
    (args.root / "REPORT.md").write_text(report + "\n")
    print(rows.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
