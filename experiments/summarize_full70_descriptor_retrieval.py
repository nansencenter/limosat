#!/usr/bin/env python3
"""Stratify full-70 exact-buoy descriptor retrieval without dropping failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results/buoy_descriptor_benchmark/full70_level1_border128"


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1"})


def summarize_retrieval(
    pairs: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    splits = pairs["source_experiment_split"].drop_duplicates()
    methods = results["method"].drop_duplicates()
    gates = [gate for gate in ("physics_50km_day", "scene_wide") if gate in set(results["gate"])]
    strict = as_bool(pairs["source_month_exclusive_buoy"])
    for split in splits:
        split_pairs = pairs[pairs["source_experiment_split"].eq(split)]
        for subset, expected_pairs in (
            ("all_temporal", split_pairs),
            (
                "strict_month_exclusive_buoy",
                split_pairs[strict.loc[split_pairs.index]],
            ),
        ):
            expected_ids = set(expected_pairs["pair_id"])
            for method in methods:
                for gate in gates:
                    group = results[
                        results["pair_id"].isin(expected_ids)
                        & results["method"].eq(method)
                        & results["gate"].eq(gate)
                        & results["accepted"].fillna(False)
                    ]
                    errors = group["endpoint_error_m"].dropna().to_numpy(dtype=float)
                    denominator = len(expected_ids)
                    rows.append(
                        {
                            "experiment_split": split,
                            "evaluation_subset": subset,
                            "method": method,
                            "gate": gate,
                            "expected_pairs": denominator,
                            "retrieved_pairs": len(errors),
                            "retrieval_coverage_fraction": len(errors)
                            / max(denominator, 1),
                            "median_error_m": float(np.median(errors))
                            if len(errors)
                            else np.nan,
                            "p90_error_m": float(np.percentile(errors, 90))
                            if len(errors)
                            else np.nan,
                            "within_2km_fraction_all": float(
                                np.count_nonzero(errors <= 2000.0)
                                / max(denominator, 1)
                            ),
                            "catastrophic_50km_fraction_all": float(
                                np.count_nonzero(errors > 50000.0)
                                / max(denominator, 1)
                            ),
                            "truth_descriptor_top1_fraction_all": float(
                                group["truth_descriptor_rank"].eq(1).sum()
                                / max(denominator, 1)
                            ),
                            "median_normalized_truth_descriptor_distance": float(
                                group["normalized_truth_descriptor_distance"].median()
                            ),
                            "median_candidate_quantization_error_m": float(
                                group["candidate_quantization_error_m"].median()
                            ),
                        }
                    )
    return pd.DataFrame.from_records(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    pairs = pd.read_csv(args.results_dir / "pairs.csv", low_memory=False)
    results = pd.read_csv(args.results_dir / "retrieval_results.csv", low_memory=False)
    summary = summarize_retrieval(pairs, results)
    summary.to_csv(args.results_dir / "stratified_summary.csv", index=False)
    payload = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "expected_pairs": int(pairs["pair_id"].nunique()),
        "splits": summary["experiment_split"].drop_duplicates().tolist(),
        "evaluation_subsets": summary["evaluation_subset"].drop_duplicates().tolist(),
        "methods": summary["method"].drop_duplicates().tolist(),
        "gates": summary["gate"].drop_duplicates().tolist(),
        "denominator_policy": "all expected pairs, including descriptor-unavailable rows",
    }
    (args.results_dir / "stratified_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
