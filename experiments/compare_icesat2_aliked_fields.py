#!/usr/bin/env python3
"""Compare ORB and several ALIKED fields on identical ATL07 observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.validate_icesat2_deformation import (
    TriangleDisplacementField,
    aggregate_atl07_bins,
    circular_shift_test,
    colocate_method,
    json_safe,
    load_aliked_vectors,
    load_atl07,
    load_orb_vectors,
    safe_spearman,
    summarize_atl07_bins,
)


LABEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not LABEL_PATTERN.fullmatch(label) or not path:
        raise argparse.ArgumentTypeError(
            "ALIKED fields must use a unique LABEL=/path/to/field.csv value"
        )
    if label == "orb":
        raise argparse.ArgumentTypeError("The label 'orb' is reserved")
    return label, Path(path)


def compare_method_bins(bins_by_method: dict[str, pd.DataFrame]) -> dict:
    labels = list(bins_by_method)
    reference_label = labels[0]
    reference = bins_by_method[reference_label]
    result: dict[str, dict] = {}
    for label in labels[1:]:
        candidate = bins_by_method[label]
        paired = reference.merge(
            candidate,
            on=["beam", "track_bin"],
            suffixes=("_reference", "_candidate"),
            validate="one_to_one",
        )
        result[label] = {
            "reference": reference_label,
            "bins": int(len(paired)),
            "shear_spearman_between_fields": safe_spearman(
                paired[f"{reference_label}_shear_per_day"].to_numpy(dtype=float),
                paired[f"{label}_shear_per_day"].to_numpy(dtype=float),
            ),
            "median_absolute_shear_difference_per_day": float(
                np.nanmedian(
                    np.abs(
                        paired[f"{reference_label}_shear_per_day"]
                        - paired[f"{label}_shear_per_day"]
                    )
                )
            ),
            "divergence_spearman_between_fields": safe_spearman(
                paired[f"{reference_label}_divergence_per_day"].to_numpy(
                    dtype=float
                ),
                paired[f"{label}_divergence_per_day"].to_numpy(dtype=float),
            ),
            "median_absolute_divergence_difference_per_day": float(
                np.nanmedian(
                    np.abs(
                        paired[f"{reference_label}_divergence_per_day"]
                        - paired[f"{label}_divergence_per_day"]
                    )
                )
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atl07", type=Path, required=True)
    parser.add_argument("--orb-database", type=Path, required=True)
    parser.add_argument("--orb-table", required=True)
    parser.add_argument("--orb-source-image-id", type=int, required=True)
    parser.add_argument("--orb-target-image-id", type=int, required=True)
    parser.add_argument(
        "--aliked-field",
        action="append",
        type=parse_labeled_path,
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--pair-start", required=True)
    parser.add_argument("--pair-end", required=True)
    parser.add_argument("--bin-size-m", type=float, default=4000.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = [label for label, _ in args.aliked_field]
    if len(labels) != len(set(labels)):
        parser.error("ALIKED field labels must be unique")

    pair_start = pd.Timestamp(args.pair_start)
    pair_end = pd.Timestamp(args.pair_end)
    if pair_start.tzinfo is None or pair_end.tzinfo is None:
        parser.error("Pair times must be timezone-aware")

    observations = load_atl07(args.atl07)
    observations = observations.loc[
        observations["time_utc"].between(pair_start, pair_end)
    ].reset_index(drop=True)
    observations_in_interval = int(len(observations))

    vector_tables = {
        "orb": load_orb_vectors(
            args.orb_database,
            args.orb_table,
            args.orb_source_image_id,
            args.orb_target_image_id,
        )
    }
    vector_tables.update(
        {label: load_aliked_vectors(path) for label, path in args.aliked_field}
    )
    fields = {
        "orb": TriangleDisplacementField.build(
            vector_tables["orb"], maximum_edge_m=20_000.0, minimum_quality=0.05
        )
    }
    fields.update(
        {
            label: TriangleDisplacementField.build(
                vector_tables[label], maximum_edge_m=6_400.0
            )
            for label in labels
        }
    )
    for label, field in fields.items():
        observations = pd.concat(
            [
                observations,
                colocate_method(observations, field, pair_start, pair_end, label),
            ],
            axis=1,
        )

    common = np.logical_and.reduce(
        [observations[f"{label}_available"].to_numpy() for label in fields]
    )
    observations = observations.loc[common].reset_index(drop=True)
    bins_by_method = {
        label: aggregate_atl07_bins(observations, label, args.bin_size_m)
        for label in fields
    }
    bins = pd.concat(bins_by_method.values(), ignore_index=True)

    summary = {
        "status": "complete" if len(observations) else "insufficient_common_support",
        "pair_start_utc": pair_start.isoformat(),
        "pair_end_utc": pair_end.isoformat(),
        "atl07_path": str(args.atl07),
        "bin_size_m": args.bin_size_m,
        "observations_in_sar_interval": observations_in_interval,
        "observations_on_all_method_support": int(len(observations)),
        "strong_beam_observations_on_all_method_support": int(
            observations["beam_type"].eq("strong").sum()
        ),
        "topography_valid_on_all_method_support": int(
            observations["topography_valid"].sum()
        ),
        "ridge_events_on_all_method_support": int(
            observations["ridge_event"].sum()
        ),
        "fields": {
            "orb": {
                "vectors": int(len(vector_tables["orb"])),
                "path": str(args.orb_database),
            },
            **{
                label: {
                    "vectors": int(len(vector_tables[label])),
                    "path": str(dict(args.aliked_field)[label]),
                }
                for label in labels
            },
        },
        "methods": {},
        "aliked_field_agreement": compare_method_bins(
            {label: bins_by_method[label] for label in labels}
        ),
        "interpretation": (
            "Every method is evaluated on exactly the same ATL07 observations. "
            "Associations are structural validation, not displacement truth."
        ),
    }
    for label, method_bins in bins_by_method.items():
        method_summary = summarize_atl07_bins(method_bins, label)
        method_summary["spatial_nulls"] = {
            "convergence_vs_ridge_density": circular_shift_test(
                method_bins,
                f"{label}_cumulative_convergence",
                "ridge_density_per_km",
                seed=20260819,
            ),
            "shear_vs_relative_roughness": circular_shift_test(
                method_bins,
                f"{label}_shear_per_day",
                "relative_roughness_m",
                seed=20260820,
            ),
        }
        summary["methods"][label] = method_summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(args.output_dir / "atl07_exact_common_points.csv", index=False)
    bins.to_csv(args.output_dir / "atl07_exact_common_bins.csv", index=False)
    encoded = json.dumps(json_safe(summary), indent=2, allow_nan=False)
    (args.output_dir / "summary.json").write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
