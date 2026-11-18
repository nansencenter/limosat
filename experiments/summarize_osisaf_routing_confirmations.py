#!/usr/bin/env python3
"""Summarize bounded MPS confirmations of OSI-455 tile routing."""

from __future__ import annotations

import argparse
import json
import math
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


def infer_cluster(manifest: dict) -> str:
    if manifest.get("cluster"):
        return str(manifest["cluster"])
    case_id = manifest["case_id"]
    if case_id.startswith("nice_"):
        return "nice_apr27_may02"
    if case_id == "full70_2020_10107_10217":
        return "mar27_29_central"
    if case_id == "full70_2020_9535_9872":
        return "mar21_26"
    return case_id


def confirmation_row(manifest: dict) -> dict:
    baseline = manifest["baseline"]
    assisted = manifest["osisaf_assisted"]
    baseline_buoy = baseline.get("buoys")
    assisted_buoy = assisted.get("buoys")
    baseline_coverage = float(baseline["coverage_after_fold_rejection"])
    assisted_coverage = float(assisted["coverage_after_fold_rejection"])
    baseline_sources = baseline["routing"]["source_counts"]
    return {
        "case_id": manifest["case_id"],
        "role": manifest["role"],
        "cluster": infer_cluster(manifest),
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
            baseline_sources, sort_keys=True
        ),
        "baseline_is_direct_phase": set(baseline_sources)
        == {"coarse_phase_translation"},
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


def bootstrap_mean_interval(
    values: np.ndarray, repetitions: int = 20_000, seed: int = 455
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a non-empty finite vector")
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(repetitions, len(values)), replace=True)
    return tuple(np.quantile(samples.mean(axis=1), [0.025, 0.975]))


def exact_two_sided_sign_test(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return np.nan
    extreme = max(wins, losses)
    upper_tail = sum(math.comb(trials, value) for value in range(extreme, trials + 1))
    return min(1.0, 2.0 * upper_tail / (2**trials))


def long_gap_statistics(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cases = rows.loc[
        (rows["elapsed_hours"] >= 30) & rows["baseline_is_direct_phase"]
    ].copy()
    if cases["case_id"].duplicated().any():
        duplicates = cases.loc[cases["case_id"].duplicated(), "case_id"].tolist()
        raise ValueError(f"duplicate direct-phase cases: {duplicates}")
    cases = cases.sort_values(["elapsed_hours", "case_id"]).reset_index(drop=True)
    clusters = (
        cases.groupby("cluster", as_index=False)
        .agg(
            cases=("case_id", "size"),
            minimum_elapsed_hours=("elapsed_hours", "min"),
            maximum_elapsed_hours=("elapsed_hours", "max"),
            mean_coverage_change_percentage_points=(
                "coverage_change_percentage_points",
                "mean",
            ),
            mean_match_ratio=("match_ratio", "mean"),
            mean_osi455_available_tile_fraction=(
                "osi455_available_tile_fraction",
                "mean",
            ),
        )
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    changes = cases["coverage_change_percentage_points"]
    cluster_changes = clusters["mean_coverage_change_percentage_points"].to_numpy()
    ci_low, ci_high = bootstrap_mean_interval(cluster_changes)
    case_wins = int((changes > 1e-9).sum())
    case_losses = int((changes < -1e-9).sum())
    cluster_wins = int((cluster_changes > 1e-9).sum())
    cluster_losses = int((cluster_changes < -1e-9).sum())
    both_buoy = cases.loc[
        cases["baseline_buoy_median_error_m"].notna()
        & cases["assisted_buoy_median_error_m"].notna()
    ]
    statistics = {
        "cases": len(cases),
        "clusters": len(clusters),
        "minimum_elapsed_hours": float(cases["elapsed_hours"].min()),
        "maximum_elapsed_hours": float(cases["elapsed_hours"].max()),
        "coverage_increased_cases": case_wins,
        "coverage_unchanged_cases": int((changes.abs() <= 1e-9).sum()),
        "coverage_decreased_cases": case_losses,
        "case_level_two_sided_sign_test_p": exact_two_sided_sign_test(
            case_wins, case_losses
        ),
        "cluster_coverage_increased": cluster_wins,
        "cluster_coverage_unchanged": int((np.abs(cluster_changes) <= 1e-9).sum()),
        "cluster_coverage_decreased": cluster_losses,
        "cluster_level_two_sided_sign_test_p": exact_two_sided_sign_test(
            cluster_wins, cluster_losses
        ),
        "median_coverage_change_percentage_points": float(changes.median()),
        "cluster_balanced_mean_coverage_change_percentage_points": float(
            cluster_changes.mean()
        ),
        "cluster_bootstrap_95ci_low_percentage_points": float(ci_low),
        "cluster_bootstrap_95ci_high_percentage_points": float(ci_high),
        "case_buoy_available_baseline": int(cases["baseline_buoys_available"].sum()),
        "case_buoy_available_assisted": int(cases["assisted_buoys_available"].sum()),
        "case_buoy_within_2km_baseline": int(cases["baseline_buoys_within_2km"].sum()),
        "case_buoy_within_2km_assisted": int(cases["assisted_buoys_within_2km"].sum()),
        "cases_with_buoy_estimates_from_both": len(both_buoy),
        "assisted_lower_buoy_median_error_cases": int(
            (
                both_buoy["assisted_buoy_median_error_m"]
                < both_buoy["baseline_buoy_median_error_m"]
            ).sum()
        ),
    }
    return cases, clusters, statistics


def agreement_statistics(cases: pd.DataFrame, pair_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = pair_summary[
        ["case_id", "phase_response", "phase_osi455_median_distance_km"]
    ]
    joined = cases.merge(metrics, on="case_id", how="left", validate="one_to_one")
    distance = joined["phase_osi455_median_distance_km"]
    joined["routing_agreement"] = pd.cut(
        distance,
        [-np.inf, 5.0, 20.0, np.inf],
        labels=("agree_le_5km", "differ_5_to_20km", "differ_gt_20km"),
    ).astype(object)
    joined.loc[distance.isna(), "routing_agreement"] = "osi_unavailable_at_buoy"
    return (
        joined.groupby("routing_agreement", as_index=False)
        .agg(
            cases=("case_id", "size"),
            coverage_increased_cases=(
                "coverage_change_percentage_points",
                lambda values: int((values > 1e-9).sum()),
            ),
            median_coverage_change_percentage_points=(
                "coverage_change_percentage_points",
                "median",
            ),
            mean_coverage_change_percentage_points=(
                "coverage_change_percentage_points",
                "mean",
            ),
            baseline_case_buoys_available=("baseline_buoys_available", "sum"),
            assisted_case_buoys_available=("assisted_buoys_available", "sum"),
        )
        .sort_values("routing_agreement")
        .reset_index(drop=True)
    )


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
    long_cases, clusters, statistics = long_gap_statistics(rows)
    long_cases.to_csv(args.root / "long_gap_case_summary.csv", index=False)
    clusters.to_csv(args.root / "long_gap_cluster_summary.csv", index=False)
    pair_summary_path = args.root.parent / "pair_summary.csv"
    agreement = agreement_statistics(
        long_cases, pd.read_csv(pair_summary_path)
    )
    agreement.to_csv(args.root / "long_gap_agreement_summary.csv", index=False)
    (args.root / "long_gap_statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n"
    )
    display = long_cases[
        [
            "case_id",
            "cluster",
            "elapsed_hours",
            "osi455_available_tile_fraction",
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
            f"{statistics['cases']} matched >=30-hour pairs from "
            f"{statistics['clusters']} independent sequence/components were run with "
            "the same EfficientLoFTR matcher and field estimator. Only target-tile "
            "centres changed: direct phase correlation in the baseline; OSI-455 where "
            "strictly available and same-centre otherwise in the assisted arm.",
            "",
            "## Main result",
            "",
            f"- Field coverage increased in {statistics['coverage_increased_cases']}/"
            f"{statistics['cases']} cases, was unchanged in "
            f"{statistics['coverage_unchanged_cases']}, and decreased in "
            f"{statistics['coverage_decreased_cases']}.",
            f"- Median case-level coverage change was "
            f"{statistics['median_coverage_change_percentage_points']:.2f} percentage "
            "points.",
            f"- The component-balanced mean was "
            f"{statistics['cluster_balanced_mean_coverage_change_percentage_points']:.2f} "
            "percentage points (component bootstrap 95% interval "
            f"{statistics['cluster_bootstrap_95ci_low_percentage_points']:.2f} to "
            f"{statistics['cluster_bootstrap_95ci_high_percentage_points']:.2f}).",
            f"- Directional consistency was suggestive rather than conventionally "
            f"significant: {statistics['cluster_coverage_increased']}/"
            f"{statistics['clusters']} component means increased "
            f"(two-sided sign-test p="
            f"{statistics['cluster_level_two_sided_sign_test_p']:.3f}).",
            f"- Available case-buoy evaluations changed from "
            f"{statistics['case_buoy_available_baseline']} to "
            f"{statistics['case_buoy_available_assisted']}; evaluations within 2 km "
            f"changed from {statistics['case_buoy_within_2km_baseline']} to "
            f"{statistics['case_buoy_within_2km_assisted']}.",
            f"- Where both arms produced buoy estimates, OSI routing had lower median "
            f"error in {statistics['assisted_lower_buoy_median_error_cases']}/"
            f"{statistics['cases_with_buoy_estimates_from_both']} cases.",
            "",
            "## Observable phase/OSI agreement",
            "",
            "This is an exploratory post-hoc stratification using information available "
            "at inference time; it was not used to select the frozen expansion.",
            "",
            markdown_table(agreement),
            "",
            markdown_table(display),
            "",
            "## Guardrails",
            "",
            "- The 11-pair expansion was frozen using only duration and shared-buoy availability before its matching results were inspected.",
            "- The extra edges overlap two 2020 components and one N-ICE component; uncertainty is therefore resampled over components, not individual pairs or buoys.",
            "- Case-buoy totals are descriptive and not independent; one pair contributes 88 observations.",
            "- OSI-455 availability is strict at every integrated daily step; no nearest-valid spatial filling was used.",
            "- Every pair in the main result uses a direct global phase-correlation baseline. The older inherited-field diagnostic remains in confirmation_summary.csv but is excluded.",
            "- Accepted vectors remain SAR-to-SAR EfficientLoFTR matches and the unchanged local consensus field.",
            "- This is an Arctic 2015/2020 experiment, not Antarctic validation; MIZ and fast-ice sample sizes remain inadequate for conclusions.",
        ]
    )
    (args.root / "REPORT.md").write_text(report + "\n")
    print(rows.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
