#!/usr/bin/env python3
"""Compare one EfficientLoFTR chain with frozen ALIKED and production ORB."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import shapely

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_buoy_array_deformation import analyze as analyze_buoys
from experiments.analyze_pair_deformation_comparison import (
    compare_fields,
    field_distribution,
    triangle_field,
)
from experiments.run_aliked_dense_pair import (
    nearest_consensus_at_queries,
    topology_summary,
)


METHODS = ("Production ORB", "ALIKED", "EfficientLoFTR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--efficient-sequence-dir", type=Path, required=True)
    parser.add_argument("--efficient-control-dir", type=Path, required=True)
    parser.add_argument("--aliked-sequence-dir", type=Path, required=True)
    parser.add_argument("--orb-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def orb_context(run_dir: Path):
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    prefix = "sqlite:///"
    if not manifest["engine_url"].startswith(prefix):
        raise ValueError("only SQLite ORB baselines are supported")
    image_map = pd.read_csv(run_dir / "image_timings.csv")
    catalog_to_run = dict(
        zip(image_map.catalog_image_id, image_map.run_image_id, strict=True)
    )
    return (
        Path(manifest["engine_url"][len(prefix) :]),
        manifest["effective_run_name"],
        catalog_to_run,
        manifest,
        image_map,
    )


def load_orb_pair(
    database: Path,
    table: str,
    source_run_id: int,
    target_run_id: int,
) -> pd.DataFrame:
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        points = pd.read_sql_query(
            f'''SELECT image_id, trajectory_id, geometry FROM "{table}"
                WHERE image_id IN (?, ?)''',
            connection,
            params=(source_run_id, target_run_id),
        )
    geometry = shapely.from_wkt(points.geometry.to_numpy())
    points["x"] = shapely.get_x(geometry)
    points["y"] = shapely.get_y(geometry)
    source = points.loc[
        points.image_id.eq(source_run_id), ["trajectory_id", "x", "y"]
    ].rename(columns={"x": "source_x", "y": "source_y"})
    target = points.loc[
        points.image_id.eq(target_run_id), ["trajectory_id", "x", "y"]
    ].rename(columns={"x": "target_x", "y": "target_y"})
    paired = source.merge(target, on="trajectory_id", validate="one_to_one")
    paired["dx_m"] = paired.target_x - paired.source_x
    paired["dy_m"] = paired.target_y - paired.source_y
    return paired


def idw_field(vectors: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    result = queries.copy()
    result["available"] = False
    result["proposal_dx_m"] = np.nan
    result["proposal_dy_m"] = np.nan
    result["selected_vectors"] = 0
    source = vectors[["source_x", "source_y"]].to_numpy(float)
    displacement = vectors[["dx_m", "dy_m"]].to_numpy(float)
    if not len(source):
        return result
    count = min(4, len(source))
    distances, indices = cKDTree(source).query(
        queries[["source_x", "source_y"]].to_numpy(float), k=count
    )
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)
    if len(queries) == 1:
        distances = distances.reshape(1, -1)
        indices = indices.reshape(1, -1)
    for row in range(len(result)):
        keep = distances[row] <= 10_000.0
        if not keep.any():
            continue
        weights = 1.0 / np.maximum(distances[row, keep], 1.0)
        estimate = np.average(displacement[indices[row, keep]], axis=0, weights=weights)
        result.loc[row, ["available", "proposal_dx_m", "proposal_dy_m", "selected_vectors"]] = [
            True,
            estimate[0],
            estimate[1],
            int(keep.sum()),
        ]
    result["available"] = result.available.astype(bool)
    return result


def load_efficient_matches(path: Path) -> pd.DataFrame:
    with np.load(path) as values:
        source = values["source_xy_m"]
        target = values["target_xy_m"]
        score = values["score"]
    return pd.DataFrame(
        {
            "source_x": source[:, 0],
            "source_y": source[:, 1],
            "dx_m": target[:, 0] - source[:, 0],
            "dy_m": target[:, 1] - source[:, 1],
            "matcher_score": score,
            "physics_valid": True,
        }
    )


def normalized_field(path: Path) -> pd.DataFrame:
    field = pd.read_csv(path)
    field["available"] = field.available.fillna(False).astype(bool)
    return field


def align_field(field: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_x",
        "source_y",
        "available",
        "proposal_dx_m",
        "proposal_dy_m",
        "selected_vectors",
    ]
    result = queries.merge(field[columns], on=["source_x", "source_y"], how="left")
    result["available"] = result.available.fillna(False).astype(bool)
    return result


def endpoint_rows(
    truth: pd.DataFrame,
    estimate: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    result = truth.copy()
    result["method"] = method
    result["available"] = estimate.available.fillna(False).to_numpy(bool)
    result["estimated_dx_m"] = estimate.proposal_dx_m.to_numpy(float)
    result["estimated_dy_m"] = estimate.proposal_dy_m.to_numpy(float)
    result["analysis_crs"] = "EPSG:3413"
    columns = [
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
    for identity in ("experiment_trajectory_id", "continuous_trajectory_id"):
        if identity in result:
            columns.append(identity)
    return result[columns]


def pooled_endpoint_summary(rows: pd.DataFrame) -> dict:
    summaries = {}
    for method, group in rows.groupby("method", sort=False):
        available = group.available.to_numpy(bool)
        error = np.hypot(
            group.estimated_dx_m.to_numpy(float) - group.truth_dx_m.to_numpy(float),
            group.estimated_dy_m.to_numpy(float) - group.truth_dy_m.to_numpy(float),
        )
        selected = error[available]
        summaries[method] = {
            "expected": int(len(group)),
            "available": int(available.sum()),
            "correct_within_2km": int(np.sum(selected <= 2_000.0)),
            "median_error_m": float(np.median(selected)) if len(selected) else None,
            "p90_error_m": float(np.quantile(selected, 0.90)) if len(selected) else None,
            "maximum_error_m": float(np.max(selected)) if len(selected) else None,
        }
    return summaries


def pooled_deformation_summary(outputs: dict[str, pd.DataFrame]) -> dict:
    pairs = outputs["pairwise_relative_displacement"]
    pair_summary = outputs["pair_summary"]
    summaries = {}
    for method in METHODS:
        method_pairs = pairs.loc[pairs.method.eq(method)]
        available = method_pairs.available.fillna(False)
        errors = method_pairs.loc[
            available, "relative_displacement_error_m"
        ].dropna().to_numpy(float)
        gradients = pair_summary.loc[
            pair_summary.method.eq(method), "gradient_frobenius_error"
        ].dropna().to_numpy(float)
        summaries[method] = {
            "available_buoy_pairs": int(available.sum()),
            "expected_buoy_pairs": int(len(method_pairs)),
            "median_relative_displacement_error_m": float(np.median(errors)),
            "p90_relative_displacement_error_m": float(np.quantile(errors, 0.90)),
            "median_affine_gradient_frobenius_error": float(np.median(gradients)),
        }
    return summaries


def vector_agreement(fields: dict[str, pd.DataFrame]) -> dict:
    result = {}
    for left_name, right_name in (
        ("Production ORB", "ALIKED"),
        ("Production ORB", "EfficientLoFTR"),
        ("ALIKED", "EfficientLoFTR"),
    ):
        left = fields[left_name]
        right = fields[right_name]
        common = left.available.to_numpy(bool) & right.available.to_numpy(bool)
        difference = np.hypot(
            left.loc[common, "proposal_dx_m"].to_numpy(float)
            - right.loc[common, "proposal_dx_m"].to_numpy(float),
            left.loc[common, "proposal_dy_m"].to_numpy(float)
            - right.loc[common, "proposal_dy_m"].to_numpy(float),
        )
        result[f"{left_name}_vs_{right_name}"] = {
            "common_nodes": int(common.sum()),
            "median_m": float(np.median(difference)),
            "p90_m": float(np.quantile(difference, 0.90)),
            "p99_m": float(np.quantile(difference, 0.99)),
            "maximum_m": float(np.max(difference)),
        }
    return result


def routing_control(primary: pd.DataFrame, control: pd.DataFrame) -> dict:
    common = primary.available.to_numpy(bool) & control.available.to_numpy(bool)
    difference = np.hypot(
        primary.loc[common, "proposal_dx_m"].to_numpy(float)
        - control.loc[common, "proposal_dx_m"].to_numpy(float),
        primary.loc[common, "proposal_dy_m"].to_numpy(float)
        - control.loc[common, "proposal_dy_m"].to_numpy(float),
    )
    return {
        "primary_available": int(primary.available.sum()),
        "control_available": int(control.available.sum()),
        "common_nodes": int(common.sum()),
        "median_difference_m": float(np.median(difference)),
        "p90_difference_m": float(np.quantile(difference, 0.90)),
        "maximum_difference_m": float(np.max(difference)),
    }


def tile_seam_audit(field: pd.DataFrame, tile_core_size_m: float = 35_840.0) -> dict:
    available = field.loc[field.available].copy()
    lookup = {
        (int(row.grid_row), int(row.grid_column)): row
        for row in available.itertuples(index=False)
    }
    internal = []
    crossing = []
    for (grid_row, grid_column), point in lookup.items():
        for neighbour_key in ((grid_row, grid_column + 1), (grid_row + 1, grid_column)):
            neighbour = lookup.get(neighbour_key)
            if neighbour is None:
                continue
            difference = float(
                np.hypot(
                    point.proposal_dx_m - neighbour.proposal_dx_m,
                    point.proposal_dy_m - neighbour.proposal_dy_m,
                )
            )
            point_tile = (
                np.floor(point.source_x / tile_core_size_m),
                np.floor(point.source_y / tile_core_size_m),
            )
            neighbour_tile = (
                np.floor(neighbour.source_x / tile_core_size_m),
                np.floor(neighbour.source_y / tile_core_size_m),
            )
            (crossing if point_tile != neighbour_tile else internal).append(difference)

    def summary(values: list[float]) -> dict:
        return {
            "edges": len(values),
            "median_vector_difference_m": float(np.median(values)),
            "p90_vector_difference_m": float(np.quantile(values, 0.90)),
        }

    result = {"internal": summary(internal), "tile_boundary_crossing": summary(crossing)}
    result["crossing_to_internal_median_ratio"] = (
        result["tile_boundary_crossing"]["median_vector_difference_m"]
        / result["internal"]["median_vector_difference_m"]
    )
    return result


def propagate_buoy_paths(pair_inputs: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    ordered_pairs = list(pair_inputs)
    truth = pd.concat(
        [pair_inputs[pair]["truth"] for pair in ordered_pairs], ignore_index=True
    )
    path_counts = truth.groupby("experiment_trajectory_id", sort=False).size()
    complete_path_ids = set(path_counts[path_counts.eq(len(ordered_pairs))].index)
    records = []
    for method in METHODS:
        for path_id in sorted(complete_path_ids):
            position = None
            path_available = True
            for step, pair in enumerate(ordered_pairs, start=1):
                inputs = pair_inputs[pair]
                row = inputs["truth"].loc[
                    inputs["truth"].experiment_trajectory_id.eq(path_id)
                ]
                if len(row) != 1:
                    raise ValueError(f"path {path_id} is not unique on pair {pair}")
                row = row.iloc[0]
                if position is None:
                    position = np.array([row.source_x, row.source_y], dtype=float)
                estimate = None
                if path_available:
                    query = pd.DataFrame(
                        {"source_x": [position[0]], "source_y": [position[1]]}
                    )
                    if method == "Production ORB":
                        estimate = idw_field(inputs["matches"][method], query).iloc[0]
                    else:
                        estimate = nearest_consensus_at_queries(
                            inputs["matches"][method],
                            query,
                            maximum_radius_m=6_000.0,
                            candidate_count=12,
                            minimum_selected_vectors=8,
                            consensus_radius_m=1_000.0,
                        ).iloc[0]
                    path_available = bool(estimate.available)
                if path_available:
                    position = position + np.array(
                        [estimate.proposal_dx_m, estimate.proposal_dy_m], dtype=float
                    )
                truth_target = np.array(
                    [row.source_x + row.truth_dx_m, row.source_y + row.truth_dy_m],
                    dtype=float,
                )
                records.append(
                    {
                        "method": method,
                        "experiment_trajectory_id": path_id,
                        "step": step,
                        "source_image_id": int(row.source_image_id),
                        "target_image_id": int(row.target_image_id),
                        "available": path_available,
                        "predicted_x": position[0] if path_available else np.nan,
                        "predicted_y": position[1] if path_available else np.nan,
                        "truth_x": truth_target[0],
                        "truth_y": truth_target[1],
                        "error_m": float(np.linalg.norm(position - truth_target))
                        if path_available
                        else np.nan,
                    }
                )

    paths = pd.DataFrame.from_records(records)
    summary = {}
    for method, group in paths.groupby("method", sort=False):
        per_path = group.groupby("experiment_trajectory_id", sort=False)
        complete = per_path.available.all()
        maximum_error = per_path.error_m.max()
        final = group.loc[group.step.eq(len(ordered_pairs)) & group.available, "error_m"]
        summary[method] = {
            "expected_complete_paths": len(complete_path_ids),
            "complete_available_paths": int(complete.sum()),
            "complete_paths_with_all_step_errors_within_2km": int(
                (complete & maximum_error.le(2_000.0)).sum()
            ),
            "final_error_median_m": float(final.median()) if len(final) else None,
            "final_error_p90_m": float(final.quantile(0.90)) if len(final) else None,
            "final_error_maximum_m": float(final.max()) if len(final) else None,
        }
    return paths, summary


def plot_deformation(
    deformation: dict[str, dict[str, pd.DataFrame]],
    elapsed_hours: dict[str, float],
    output: Path,
) -> None:
    values = np.concatenate(
        [
            field.loc[field.available, "total_per_day"].dropna().to_numpy(float)
            for pair in deformation.values()
            for field in pair.values()
        ]
    )
    positive = values[values > 0]
    lower = max(1.0e-3, float(np.quantile(positive, 0.02)))
    upper = max(lower * 10.0, float(np.quantile(positive, 0.98)))
    norm = LogNorm(lower, upper)
    fig, axes = plt.subplots(
        len(deformation), len(METHODS), figsize=(17, 14), constrained_layout=True
    )
    artist = None
    for row, (pair, fields) in enumerate(deformation.items()):
        for column, method in enumerate(METHODS):
            axis = axes[row, column]
            field = fields[method]
            available = field.available.fillna(False) & field.total_per_day.notna()
            artist = axis.scatter(
                field.loc[available, "source_x"] / 1000.0,
                field.loc[available, "source_y"] / 1000.0,
                c=field.loc[available, "total_per_day"],
                s=5,
                marker="s",
                linewidths=0,
                cmap="magma",
                norm=norm,
                rasterized=True,
            )
            axis.set_aspect("equal")
            axis.set_title(
                f"{method} · {pair.replace('_', '→')}\n"
                f"{elapsed_hours[pair]:.2f} h · deformation support {available.mean():.1%}"
            )
            axis.set_xlabel("EPSG:3413 x (km)")
            axis.set_ylabel("EPSG:3413 y (km)")
    fig.colorbar(artist, ax=axes, shrink=0.80, label="total deformation (day⁻¹)")
    fig.suptitle(
        "Production ORB, ALIKED, and optimized EfficientLoFTR · frozen January 2020 chain",
        fontsize=15,
    )
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_pair_deformation(
    pair: str,
    fields: dict[str, pd.DataFrame],
    elapsed_hours: float,
    output: Path,
) -> None:
    values = np.concatenate(
        [
            field.loc[field.available, "total_per_day"].dropna().to_numpy(float)
            for field in fields.values()
        ]
    )
    positive = values[values > 0]
    lower = max(1.0e-3, float(np.quantile(positive, 0.02)))
    upper = max(lower * 10.0, float(np.quantile(positive, 0.98)))
    norm = LogNorm(lower, upper)
    fig, axes = plt.subplots(1, len(METHODS), figsize=(17, 5.5), constrained_layout=True)
    artist = None
    for axis, method in zip(axes, METHODS, strict=True):
        field = fields[method]
        available = field.available.fillna(False) & field.total_per_day.notna()
        artist = axis.scatter(
            field.loc[available, "source_x"] / 1000.0,
            field.loc[available, "source_y"] / 1000.0,
            c=field.loc[available, "total_per_day"],
            s=7,
            marker="s",
            linewidths=0,
            cmap="magma",
            norm=norm,
            rasterized=True,
        )
        axis.set_aspect("equal")
        axis.set_title(f"{method}\ndeformation support {available.mean():.1%}")
        axis.set_xlabel("EPSG:3413 x (km)")
        axis.set_ylabel("EPSG:3413 y (km)")
    fig.colorbar(artist, ax=axes, shrink=0.82, label="total deformation (day⁻¹)")
    fig.suptitle(
        f"January 2020 pair {pair.replace('_', '→')} · {elapsed_hours:.2f} h",
        fontsize=15,
    )
    fig.savefig(output, dpi=190)
    plt.close(fig)


def write_markdown(report: dict, output: Path) -> None:
    endpoint = report["pooled_buoy_endpoint"]
    propagated = report["propagated_buoy_paths"]
    deformation = report["pooled_buoy_deformation"]
    lines = [
        "# EfficientLoFTR chain comparison",
        "",
        "| Method | Available | ≤2 km | Median error | P90 error | Maximum error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = endpoint[method]
        lines.append(
            f"| {method} | {row['available']}/{row['expected']} | "
            f"{row['correct_within_2km']}/{row['expected']} | "
            f"{row['median_error_m']:.1f} m | {row['p90_error_m']:.1f} m | "
            f"{row['maximum_error_m']:.1f} m |"
        )
    lines.extend(
        [
            "",
            "## Propagated three-step buoy paths",
            "",
            "| Method | Complete paths | All steps ≤2 km | Final median | Final P90 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = propagated[method]
        lines.append(
            f"| {method} | {row['complete_available_paths']}/{row['expected_complete_paths']} | "
            f"{row['complete_paths_with_all_step_errors_within_2km']}/{row['expected_complete_paths']} | "
            f"{row['final_error_median_m']:.1f} m | {row['final_error_p90_m']:.1f} m |"
        )
    lines.extend(
        [
            "",
            "## Buoy-array deformation",
            "",
            "| Method | Available buoy pairs | Relative-error median | Relative-error P90 | Median affine-gradient error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = deformation[method]
        lines.append(
            f"| {method} | {row['available_buoy_pairs']}/{row['expected_buoy_pairs']} | "
            f"{row['median_relative_displacement_error_m']:.1f} m | "
            f"{row['p90_relative_displacement_error_m']:.1f} m | "
            f"{row['median_affine_gradient_frobenius_error']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- EfficientLoFTR optimized, MPS, complete four-image run: "
            f"{report['runtime_seconds']['efficientloftr_end_to_end']:.2f} s.",
            f"- ALIKED five-layer/direct, CPU, warm complete run: "
            f"{report['runtime_seconds']['aliked_end_to_end']:.2f} s.",
            f"- Production ORB, CPU, the four relevant image updates inside the 70-image run: "
            f"{report['runtime_seconds']['orb_four_image_updates']:.2f} s.",
            "",
            "Runtime devices differ; scientific outputs use the same images and reporting grid.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    efficient_manifest = json.loads(
        (args.efficient_sequence_dir / "run_manifest.json").read_text()
    )
    control_manifest = json.loads(
        (args.efficient_control_dir / "run_manifest.json").read_text()
    )
    aliked_manifest = json.loads(
        (args.aliked_sequence_dir / "run_manifest.json").read_text()
    )
    database, table, catalog_to_run, orb_manifest, image_timings = orb_context(
        args.orb_run_dir
    )

    pair_reports = {}
    deformation = {}
    elapsed_lookup = {}
    endpoint_frames = []
    pair_inputs = {}
    for pair_summary in efficient_manifest["pairs_summary"]:
        source_id = int(pair_summary["source_image_id"])
        target_id = int(pair_summary["target_image_id"])
        pair = f"{source_id}_{target_id}"
        elapsed_hours = float(pair_summary["elapsed_hours"])
        elapsed_lookup[pair] = elapsed_hours
        efficient_dir = args.efficient_sequence_dir / f"pair_{pair}"
        control_dir = args.efficient_control_dir / f"pair_{pair}"
        aliked_dir = args.aliked_sequence_dir / f"pair_{pair}"

        efficient = normalized_field(efficient_dir / "field_4km.csv")
        control = normalized_field(control_dir / "field_4km.csv")
        queries = efficient[["grid_row", "grid_column", "source_x", "source_y"]].copy()
        aliked = align_field(
            normalized_field(aliked_dir / "field_nearest12_fold_rejected.csv"), queries
        )
        orb_vectors = load_orb_pair(
            database,
            table,
            int(catalog_to_run[source_id]),
            int(catalog_to_run[target_id]),
        )
        orb = idw_field(orb_vectors, queries)
        fields = {
            "Production ORB": orb,
            "ALIKED": aliked,
            "EfficientLoFTR": efficient,
        }

        truth = pd.read_csv(efficient_dir / "buoy_results.csv", dtype={"buoy_id": str})
        truth_columns = truth[
            [
                "source_image_id",
                "target_image_id",
                "buoy_id",
                "experiment_trajectory_id",
                "continuous_trajectory_id",
                "source_x",
                "source_y",
                "truth_dx_m",
                "truth_dy_m",
                "elapsed_hours",
            ]
        ].copy()
        efficient_buoy = truth[["available", "proposal_dx_m", "proposal_dy_m"]]
        aliked_buoy = pd.read_csv(aliked_dir / "buoy_nearest12.csv")
        orb_buoy = idw_field(
            orb_vectors,
            truth_columns[["source_x", "source_y"]].copy(),
        )
        pair_inputs[pair] = {
            "truth": truth_columns,
            "matches": {
                "Production ORB": orb_vectors,
                "ALIKED": pd.read_csv(aliked_dir / "matches.csv"),
                "EfficientLoFTR": load_efficient_matches(efficient_dir / "matches.npz"),
            },
        }
        endpoint_frames.extend(
            [
                endpoint_rows(truth_columns, orb_buoy, "Production ORB"),
                endpoint_rows(truth_columns, aliked_buoy, "ALIKED"),
                endpoint_rows(truth_columns, efficient_buoy, "EfficientLoFTR"),
            ]
        )

        deformation_fields = {}
        deformation_geometry = {}
        for method, field in fields.items():
            vectors = (
                orb_vectors
                if method == "Production ORB"
                else field.loc[field.available]
                .rename(columns={"proposal_dx_m": "dx_m", "proposal_dy_m": "dy_m"})
            )
            result, geometry = triangle_field(
                vectors,
                queries,
                elapsed_hours / 24.0,
                maximum_edge_m=20_000.0 if method == "Production ORB" else 6_400.0,
                minimum_quality=0.05 if method == "Production ORB" else 0.0,
            )
            deformation_fields[method] = result
            deformation_geometry[method] = geometry
        deformation[pair] = deformation_fields
        pair_reports[pair] = {
            "elapsed_hours": elapsed_hours,
            "displacement_coverage": {
                method: {
                    "available_nodes": int(field.available.sum()),
                    "grid_nodes": int(len(field)),
                    "coverage_fraction": float(field.available.mean()),
                    "topology": topology_summary(field, 4_000.0),
                }
                for method, field in fields.items()
            },
            "field_vector_agreement": vector_agreement(fields),
            "routing_control": routing_control(efficient, control),
            "learned_tile_seam_audit": {
                "ALIKED": tile_seam_audit(aliked),
                "EfficientLoFTR": tile_seam_audit(efficient),
            },
            "deformation": {
                "fields": {
                    method: field_distribution(field)
                    for method, field in deformation_fields.items()
                },
                "geometry": deformation_geometry,
                "agreement": {
                    "ORB_vs_ALIKED": compare_fields(
                        deformation_fields["Production ORB"], deformation_fields["ALIKED"]
                    ),
                    "ORB_vs_EfficientLoFTR": compare_fields(
                        deformation_fields["Production ORB"], deformation_fields["EfficientLoFTR"]
                    ),
                    "ALIKED_vs_EfficientLoFTR": compare_fields(
                        deformation_fields["ALIKED"], deformation_fields["EfficientLoFTR"]
                    ),
                },
            },
        }

    endpoints = pd.concat(endpoint_frames, ignore_index=True)
    endpoints.to_csv(args.output_dir / "buoy_endpoint_estimates.csv", index=False)
    buoy_outputs = analyze_buoys(endpoints, bootstrap_replicates=0)
    buoy_dir = args.output_dir / "buoy_deformation"
    buoy_dir.mkdir(exist_ok=True)
    for name, frame in buoy_outputs.items():
        frame.to_csv(buoy_dir / f"{name}.csv", index=False)
    propagated_paths, propagated_summary = propagate_buoy_paths(pair_inputs)
    propagated_paths.to_csv(args.output_dir / "propagated_buoy_paths.csv", index=False)

    relevant_ids = {721, 731, 740, 849}
    orb_four_image_seconds = float(
        image_timings.loc[image_timings.catalog_image_id.isin(relevant_ids), "seconds"].sum()
    )
    report = {
        "status": "complete",
        "primary_efficient_routing": efficient_manifest["routing_mode"],
        "control_efficient_routing": control_manifest["routing_mode"],
        "pooled_buoy_endpoint": pooled_endpoint_summary(endpoints),
        "propagated_buoy_paths": propagated_summary,
        "pooled_buoy_deformation": pooled_deformation_summary(buoy_outputs),
        "buoy_deformation_pair_summary": buoy_outputs["pair_summary"].to_dict("records"),
        "pairs": pair_reports,
        "runtime_seconds": {
            "efficientloftr_end_to_end": float(efficient_manifest["elapsed_seconds"]),
            "efficientloftr_model_setup": float(efficient_manifest["model_setup_seconds"]),
            "efficientloftr_device": efficient_manifest["device"],
            "aliked_end_to_end": float(aliked_manifest["summary"]["elapsed_seconds"]),
            "aliked_device": aliked_manifest["summary"]["device"],
            "orb_four_image_updates": orb_four_image_seconds,
            "orb_full70_end_to_end": float(orb_manifest["elapsed_seconds"]),
            "orb_device": "cpu",
        },
        "runtime_scope_note": (
            "EfficientLoFTR and ALIKED are isolated complete four-image runs. "
            "ORB is the sum of the four relevant updates inside the production 70-image run, "
            "so its state includes earlier images. EfficientLoFTR used MPS; ALIKED and ORB used CPU."
        ),
        "interpretation": (
            "Buoy endpoints and buoy-array deformation are accuracy evidence. Field agreement, "
            "coverage, topology, and mapped deformation are diagnostics without independent dense truth."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output_dir / "report.md")
    plot_deformation(deformation, elapsed_lookup, args.output_dir / "deformation_comparison.png")
    longest_pair = max(elapsed_lookup, key=elapsed_lookup.get)
    plot_pair_deformation(
        longest_pair,
        deformation[longest_pair],
        elapsed_lookup[longest_pair],
        args.output_dir / "deformation_long_gap_comparison.png",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
