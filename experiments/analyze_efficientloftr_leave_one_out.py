#!/usr/bin/env python3
"""Compare an EfficientLoFTR image chain with leave-one-image-out paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_pair_deformation_comparison import triangle_field
from experiments.run_efficientloftr_sequence import field_from_csv
from limosat.learned_drift.trajectory import sample_field
from limosat.learned_drift.types import DriftField


MAXIMUM_TRIANGLE_EDGE_M = 6_400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-sequence-dir", type=Path, required=True)
    parser.add_argument("--omission-dir", type=Path, action="append", required=True)
    parser.add_argument("--direct-first-to-last-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_run(
    run_dir: Path,
) -> tuple[list[str], dict[tuple[str, str], tuple[float, DriftField]], dict]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    pairs = manifest["pairs_summary"]
    image_ids = [str(pairs[0]["source_image_id"])] + [
        str(pair["target_image_id"]) for pair in pairs
    ]
    edges = {}
    for pair in pairs:
        source = str(pair["source_image_id"])
        target = str(pair["target_image_id"])
        field = field_from_csv(run_dir / f"pair_{source}_{target}" / "field_4km.csv")
        edges[(source, target)] = (float(pair["elapsed_hours"]), field)
    return image_ids, edges, manifest


def valid_source_points(field: DriftField) -> np.ndarray:
    valid = field.available & np.isfinite(field.displacement_m).all(axis=1)
    return np.asarray(field.source_xy_m[valid], dtype=np.float64)


def unique_points(*values: np.ndarray) -> np.ndarray:
    points = np.vstack([value for value in values if len(value)])
    return np.unique(np.rint(points).astype(np.int64), axis=0).astype(np.float64)


def compose(fields: list[DriftField], query_xy_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(query_xy_m, dtype=np.float64).copy()
    active = np.ones(len(positions), dtype=bool)
    for field in fields:
        indices = np.flatnonzero(active)
        sampled = sample_field(field, positions[indices], MAXIMUM_TRIANGLE_EDGE_M)
        active[indices] = sampled.available
        supported = indices[sampled.available]
        positions[supported] += sampled.displacement_m[sampled.available]
    return positions - query_xy_m, active


def quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "median": None, "p90": None, "p95": None, "maximum": None}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def field_vectors(field: DriftField) -> pd.DataFrame:
    rows = field.to_frame().rename(
        columns={"proposal_dx_m": "dx_m", "proposal_dy_m": "dy_m"}
    )
    return rows.loc[
        rows.available,
        ["source_x", "source_y", "dx_m", "dy_m"],
    ].copy()


def deformation_comparison(
    first: pd.DataFrame,
    second: pd.DataFrame,
    queries: np.ndarray,
    elapsed_hours: float,
) -> dict:
    query_rows = pd.DataFrame(queries, columns=["source_x", "source_y"])
    elapsed_days = elapsed_hours / 24.0
    first_field, _ = triangle_field(
        first, query_rows, elapsed_days, MAXIMUM_TRIANGLE_EDGE_M
    )
    second_field, _ = triangle_field(
        second, query_rows, elapsed_days, MAXIMUM_TRIANGLE_EDGE_M
    )
    common = first_field.available & second_field.available
    report = {"common_cells": int(common.sum())}
    for component in ("divergence_per_day", "shear_per_day", "total_per_day"):
        left = first_field.loc[common, component].to_numpy(float)
        right = second_field.loc[common, component].to_numpy(float)
        report[f"median_absolute_{component}_difference"] = (
            float(np.median(np.abs(left - right))) if len(left) else None
        )
        report[f"spearman_{component}"] = (
            float(pd.Series(left).corr(pd.Series(right), method="spearman"))
            if len(left) > 1
            else None
        )
    return report


def direct_vs_composed(
    direct: DriftField,
    component_fields: list[DriftField],
    elapsed_hours: float,
) -> tuple[dict, pd.DataFrame]:
    valid = direct.available & np.isfinite(direct.displacement_m).all(axis=1)
    query = np.asarray(direct.source_xy_m[valid], dtype=np.float64)
    direct_displacement = np.asarray(direct.displacement_m[valid], dtype=np.float64)
    composed_displacement, composed_available = compose(component_fields, query)
    difference = np.linalg.norm(
        direct_displacement[composed_available]
        - composed_displacement[composed_available],
        axis=1,
    )
    samples = pd.DataFrame(
        {
            "source_x": query[:, 0],
            "source_y": query[:, 1],
            "direct_dx_m": direct_displacement[:, 0],
            "direct_dy_m": direct_displacement[:, 1],
            "composed_available": composed_available,
            "composed_dx_m": composed_displacement[:, 0],
            "composed_dy_m": composed_displacement[:, 1],
        }
    )
    samples["vector_difference_m"] = np.where(
        composed_available,
        np.linalg.norm(direct_displacement - composed_displacement, axis=1),
        np.nan,
    )
    composed_vectors = samples.loc[
        samples.composed_available,
        ["source_x", "source_y", "composed_dx_m", "composed_dy_m"],
    ].rename(columns={"composed_dx_m": "dx_m", "composed_dy_m": "dy_m"})
    direct_vectors = samples[
        ["source_x", "source_y", "direct_dx_m", "direct_dy_m"]
    ].rename(columns={"direct_dx_m": "dx_m", "direct_dy_m": "dy_m"})
    report = {
        "direct_nodes": int(valid.sum()),
        "composed_available": int(composed_available.sum()),
        "composed_available_fraction": float(composed_available.mean()),
        "vector_difference_m": quantiles(difference),
        "deformation": deformation_comparison(
            direct_vectors,
            composed_vectors,
            query[composed_available],
            elapsed_hours,
        ),
    }
    return report, samples


def compare_same_pair(
    first: DriftField,
    second: DriftField,
    elapsed_hours: float,
) -> tuple[dict, pd.DataFrame]:
    first_rows = field_vectors(first).rename(columns={"dx_m": "first_dx_m", "dy_m": "first_dy_m"})
    second_rows = field_vectors(second).rename(columns={"dx_m": "second_dx_m", "dy_m": "second_dy_m"})
    common = first_rows.merge(second_rows, on=["source_x", "source_y"], how="outer", indicator=True)
    both = common["_merge"].eq("both")
    common["vector_difference_m"] = np.where(
        both,
        np.hypot(
            common.first_dx_m - common.second_dx_m,
            common.first_dy_m - common.second_dy_m,
        ),
        np.nan,
    )
    queries = common.loc[both, ["source_x", "source_y"]].to_numpy(float)
    report = {
        "first_available_nodes": int(len(first_rows)),
        "second_available_nodes": int(len(second_rows)),
        "common_nodes": int(both.sum()),
        "first_only_nodes": int(common["_merge"].eq("left_only").sum()),
        "second_only_nodes": int(common["_merge"].eq("right_only").sum()),
        "vector_difference_m": quantiles(common.vector_difference_m.to_numpy(float)),
        "deformation": deformation_comparison(
            first_rows.rename(columns={"first_dx_m": "dx_m", "first_dy_m": "dy_m"}),
            second_rows.rename(columns={"second_dx_m": "dx_m", "second_dy_m": "dy_m"}),
            queries,
            elapsed_hours,
        ),
    }
    return report, common


def path_comparison(
    first_fields: list[DriftField],
    second_fields: list[DriftField],
    seeds: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    first_displacement, first_available = compose(first_fields, seeds)
    second_displacement, second_available = compose(second_fields, seeds)
    common = first_available & second_available
    difference = np.linalg.norm(
        first_displacement[common] - second_displacement[common], axis=1
    )
    rows = pd.DataFrame(
        {
            "source_x": seeds[:, 0],
            "source_y": seeds[:, 1],
            "full_available": first_available,
            "leave_one_out_available": second_available,
            "full_dx_m": first_displacement[:, 0],
            "full_dy_m": first_displacement[:, 1],
            "leave_one_out_dx_m": second_displacement[:, 0],
            "leave_one_out_dy_m": second_displacement[:, 1],
        }
    )
    rows["endpoint_difference_m"] = np.where(
        common,
        np.linalg.norm(first_displacement - second_displacement, axis=1),
        np.nan,
    )
    return {
        "seed_nodes": int(len(seeds)),
        "full_available": int(first_available.sum()),
        "leave_one_out_available": int(second_available.sum()),
        "common_available": int(common.sum()),
        "full_only": int((first_available & ~second_available).sum()),
        "leave_one_out_only": int((~first_available & second_available).sum()),
        "endpoint_difference_m": quantiles(difference),
    }, rows


def main() -> int:
    args = parse_args()
    full_ids, full, full_manifest = load_run(args.full_sequence_dir)
    if len(full_ids) < 3:
        raise ValueError("the full experiment must contain at least three images")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_fields = [full[edge][1] for edge in zip(full_ids, full_ids[1:])]
    report = {
        "status": "complete",
        "full_image_ids": full_ids,
        "maximum_triangle_edge_m": MAXIMUM_TRIANGLE_EDGE_M,
        "full_trajectories": full_manifest.get("trajectories"),
        "omissions": {},
    }
    for omission_dir in args.omission_dir:
        omitted_ids, omitted, omitted_manifest = load_run(omission_dir)
        missing = [image_id for image_id in full_ids if image_id not in omitted_ids]
        if len(missing) != 1:
            raise ValueError(f"{omission_dir} does not omit exactly one image")
        missing_id = missing[0]
        missing_index = full_ids.index(missing_id)
        if missing_index in (0, len(full_ids) - 1):
            raise ValueError("only interior-image omissions are supported")
        expected = full_ids[:missing_index] + full_ids[missing_index + 1 :]
        if omitted_ids != expected:
            raise ValueError(f"{omission_dir} is not the expected ordered omission")

        previous_id = full_ids[missing_index - 1]
        next_id = full_ids[missing_index + 1]
        direct_hours, direct = omitted[(previous_id, next_id)]
        first_hours, first_component = full[(previous_id, missing_id)]
        second_hours, second_component = full[(missing_id, next_id)]
        if not np.isclose(direct_hours, first_hours + second_hours):
            raise ValueError("direct and composed elapsed times disagree")
        direct_report, direct_samples = direct_vs_composed(
            direct,
            [first_component, second_component],
            direct_hours,
        )

        same_pairs = {}
        for edge in zip(omitted_ids, omitted_ids[1:]):
            if edge not in full:
                continue
            elapsed_hours, full_field = full[edge]
            _, omitted_field = omitted[edge]
            pair_report, pair_samples = compare_same_pair(
                full_field, omitted_field, elapsed_hours
            )
            pair_name = f"{edge[0]}_to_{edge[1]}"
            same_pairs[pair_name] = pair_report
            pair_samples.to_csv(
                args.output_dir
                / f"remove_{missing_id}_same_pair_{pair_name}.csv",
                index=False,
            )

        omitted_fields = [
            omitted[edge][1] for edge in zip(omitted_ids, omitted_ids[1:])
        ]
        seeds = unique_points(
            valid_source_points(full_fields[0]),
            valid_source_points(omitted_fields[0]),
        )
        path_report, path_samples = path_comparison(
            full_fields, omitted_fields, seeds
        )
        key = f"remove_{missing_id}"
        report["omissions"][key] = {
            "omitted_image_id": missing_id,
            "image_ids": omitted_ids,
            "direct_edge": f"{previous_id}_to_{next_id}",
            "direct_vs_composed": direct_report,
            "same_pair_comparisons": same_pairs,
            "end_to_end_path": path_report,
            "trajectories": omitted_manifest.get("trajectories"),
        }
        direct_samples.to_csv(
            args.output_dir / f"{key}_direct_vs_composed.csv", index=False
        )
        path_samples.to_csv(args.output_dir / f"{key}_path.csv", index=False)

    if args.direct_first_to_last_dir is not None:
        _, direct_edges, _ = load_run(args.direct_first_to_last_dir)
        edge = (full_ids[0], full_ids[-1])
        direct_hours, direct = direct_edges[edge]
        composed_hours = sum(
            full[component][0] for component in zip(full_ids, full_ids[1:])
        )
        if not np.isclose(direct_hours, composed_hours):
            raise ValueError("first-to-last direct and composed elapsed times disagree")
        direct_report, direct_samples = direct_vs_composed(
            direct, full_fields, direct_hours
        )
        report["direct_first_to_last_vs_composed"] = direct_report
        direct_samples.to_csv(
            args.output_dir / "direct_first_to_last_vs_composed.csv", index=False
        )

    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
