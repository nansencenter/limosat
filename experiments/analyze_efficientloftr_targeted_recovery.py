#!/usr/bin/env python3
"""Evaluate targeted non-consecutive matching against routing-matched controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_efficientloftr_leave_one_out import compare_same_pair
from experiments.evaluate_closure_trajectory_graph import compare_graphs
from experiments.run_efficientloftr_sequence import field_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted-run-dir", type=Path, required=True)
    parser.add_argument(
        "--full-pair-run-dir", type=Path, action="append", required=True
    )
    parser.add_argument("--targeted-graph-report", type=Path, required=True)
    parser.add_argument("--full-graph-report", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def pair_key(summary: dict) -> tuple[str, str]:
    return str(summary["source_image_id"]), str(summary["target_image_id"])


def comparison_metrics(
    targeted_run_dir: Path, full_pair_run_dirs: list[Path]
) -> tuple[list[dict], dict]:
    targeted_manifest = read_json(targeted_run_dir / "run_manifest.json")
    targeted = {pair_key(row): row for row in targeted_manifest["pairs_summary"]}
    full = {}
    full_paths = {}
    for pair_dir in full_pair_run_dirs:
        summary = read_json(pair_dir / "summary.json")
        key = pair_key(summary)
        if key in full:
            raise ValueError(f"duplicate full control for {key}")
        full[key] = summary
        full_paths[key] = pair_dir
    if targeted.keys() != full.keys():
        raise ValueError("targeted and full controls must contain the same pairs")

    pair_reports = []
    for key, targeted_summary in targeted.items():
        source, target = key
        targeted_pair_dir = targeted_run_dir / f"pair_{source}_{target}"
        field_report, _rows = compare_same_pair(
            field_from_csv(full_paths[key] / "field_4km.csv"),
            field_from_csv(targeted_pair_dir / "field_4km.csv"),
            float(targeted_summary["elapsed_hours"]),
        )
        pair_reports.append(
            {
                "source_image_id": source,
                "target_image_id": target,
                "targeted_source_tiles": int(targeted_summary["source_tiles"]),
                "full_source_tiles": int(full[key]["source_tiles"]),
                "targeted_topology": targeted_summary["topology_after_rejection"],
                "field_comparison": field_report,
            }
        )

    totals = {
        "targeted_source_tiles": sum(row["targeted_source_tiles"] for row in pair_reports),
        "full_source_tiles": sum(row["full_source_tiles"] for row in pair_reports),
        "targeted_matching_seconds": sum(
            float(row["timing_seconds"]["matching"]) for row in targeted.values()
        ),
        "full_matching_seconds": sum(
            float(row["timing_seconds"]["matching"]) for row in full.values()
        ),
        "targeted_pair_seconds": sum(
            float(row["timing_seconds"]["pair_total"]) for row in targeted.values()
        ),
        "full_pair_seconds": sum(
            float(row["timing_seconds"]["pair_total"]) for row in full.values()
        ),
    }
    totals["matcher_call_reduction_fraction"] = 1.0 - (
        totals["targeted_source_tiles"] / totals["full_source_tiles"]
    )
    totals["matching_time_reduction_fraction"] = 1.0 - (
        totals["targeted_matching_seconds"] / totals["full_matching_seconds"]
    )
    totals["pair_time_reduction_fraction"] = 1.0 - (
        totals["targeted_pair_seconds"] / totals["full_pair_seconds"]
    )
    return pair_reports, totals


def evaluate_gate(
    pair_reports: list[dict],
    totals: dict,
    targeted_graph: dict,
    full_graph: dict,
    graph_comparison: dict,
    gate: dict,
) -> tuple[dict, dict]:
    adjacent_complete = int(full_graph["adjacent_only_graph"]["complete"])
    full_complete = int(full_graph["shortest_graph"]["complete"])
    targeted_complete = int(targeted_graph["shortest_graph"]["complete"])
    full_gain = full_complete - adjacent_complete
    targeted_gain = targeted_complete - adjacent_complete
    recovery_fraction = targeted_gain / full_gain if full_gain > 0 else 1.0
    trajectory = {
        "adjacent_only_complete": adjacent_complete,
        "full_complete": full_complete,
        "targeted_complete": targeted_complete,
        "full_gain": full_gain,
        "targeted_gain": targeted_gain,
        "gain_recovery_fraction": recovery_fraction,
    }

    full_buoy = full_graph["buoys_unsealed"]["shortest"]
    targeted_buoy = targeted_graph["buoys_unsealed"]["shortest"]
    buoy = {
        "full": full_buoy,
        "targeted": targeted_buoy,
        "availability_loss": int(full_buoy["available"] - targeted_buoy["available"]),
        "correct_within_2km_loss": int(
            full_buoy["correct_within_2km"] - targeted_buoy["correct_within_2km"]
        ),
        "median_error_increase_m": float(
            targeted_buoy["median_error_m"] - full_buoy["median_error_m"]
        ),
        "p90_error_increase_m": float(
            targeted_buoy["p90_error_m"] - full_buoy["p90_error_m"]
        ),
    }

    checks = {
        "trajectory_gain": recovery_fraction
        >= gate["trajectory_gain_recovery_fraction_min"],
        "matcher_calls": totals["matcher_call_reduction_fraction"]
        >= gate["matcher_call_reduction_fraction_min"],
        "buoy_availability": buoy["availability_loss"]
        <= gate["buoy_availability_loss_max"],
        "buoy_correct_within_2km": buoy["correct_within_2km_loss"]
        <= gate["buoy_correct_within_2km_loss_max"],
        "buoy_median_error": buoy["median_error_increase_m"]
        <= gate["buoy_median_error_increase_m_max"],
        "buoy_p90_error": buoy["p90_error_increase_m"]
        <= gate["buoy_p90_error_increase_m_max"],
        "topology": all(
            row["targeted_topology"]["flipped_triangles"]
            <= gate["flipped_triangles_max"]
            for row in pair_reports
        ),
        "trajectory_position_median": all(
            row["position_difference_m"]["median"]
            <= gate["trajectory_position_median_difference_m_max"]
            for row in graph_comparison["by_image"].values()
        ),
        "trajectory_position_p90": all(
            row["position_difference_m"]["p90"]
            <= gate["trajectory_position_p90_difference_m_max"]
            for row in graph_comparison["by_image"].values()
        ),
        "trajectory_total_deformation": all(
            row["spearman_total_per_day"]
            >= gate["trajectory_total_deformation_spearman_min"]
            for row in graph_comparison["deformation_by_image"].values()
        ),
    }
    return {
        "trajectory": trajectory,
        "buoys": buoy,
        "trajectory_graph_comparison": graph_comparison,
    }, checks


def markdown(report: dict) -> str:
    trajectory = report["trajectory"]
    totals = report["totals"]
    buoy = report["buoys"]
    lines = [
        "# EfficientLoFTR targeted recovery gate",
        "",
        f"Overall gate: **{'PASS' if report['gate_passed'] else 'FAIL'}**",
        "",
        f"- Trajectory gain recovered: {trajectory['gain_recovery_fraction']:.1%} "
        f"({trajectory['targeted_gain']} of {trajectory['full_gain']})",
        f"- Matcher-call reduction: {totals['matcher_call_reduction_fraction']:.1%} "
        f"({totals['targeted_source_tiles']} vs {totals['full_source_tiles']})",
        f"- Matching-time reduction: {totals['matching_time_reduction_fraction']:.1%}",
        f"- Correct buoy comparisons: {buoy['targeted']['correct_within_2km']} targeted "
        f"vs {buoy['full']['correct_within_2km']} full",
        f"- Targeted post-rejection flipped triangles: "
        f"{sum(row['targeted_topology']['flipped_triangles'] for row in report['pairs'])}",
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: {name.replace('_', ' ')}"
        for name, passed in report["checks"].items()
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config = read_json(args.gate_config)
    pair_reports, totals = comparison_metrics(
        args.targeted_run_dir, args.full_pair_run_dir
    )
    targeted_graph = read_json(args.targeted_graph_report)
    full_graph = read_json(args.full_graph_report)
    elapsed_hours = [
        float(row["elapsed_hours"])
        for row in full_graph["edges"]
        if row["role"] == "adjacent"
    ]
    graph_comparison = compare_graphs(
        pd.read_csv(args.full_graph_report.parent / "shortest_observed_graph.csv"),
        pd.read_csv(args.targeted_graph_report.parent / "shortest_observed_graph.csv"),
        elapsed_hours,
    )
    metrics, checks = evaluate_gate(
        pair_reports,
        totals,
        targeted_graph,
        full_graph,
        graph_comparison,
        config["gate"],
    )
    report = {
        "status": "complete",
        "policy_name": config["policy_name"],
        "gate_passed": all(checks.values()),
        "checks": checks,
        **metrics,
        "totals": totals,
        "pairs": pair_reports,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
