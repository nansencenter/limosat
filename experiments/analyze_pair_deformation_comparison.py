#!/usr/bin/env python3
"""Compare ORB and ALIKED pair deformation on one common reporting grid."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree
import shapely


COMPONENTS = ("divergence_per_day", "shear_per_day", "total_per_day")


def deformation_components(gradient: np.ndarray, elapsed_days: float) -> dict:
    velocity_gradient = gradient / elapsed_days
    divergence = np.trace(velocity_gradient, axis1=-2, axis2=-1)
    shear = np.hypot(
        velocity_gradient[..., 0, 0] - velocity_gradient[..., 1, 1],
        velocity_gradient[..., 0, 1] + velocity_gradient[..., 1, 0],
    )
    return {
        "divergence_per_day": divergence,
        "shear_per_day": shear,
        "total_per_day": np.hypot(divergence, shear),
        "vorticity_per_day": (
            velocity_gradient[..., 1, 0] - velocity_gradient[..., 0, 1]
        ),
    }


def load_orb_pairs(database: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(database) as connection:
        rows = pd.read_sql_query(
            f'SELECT image_id, trajectory_id, geometry, interpolated, corr '
            f'FROM "{table}" WHERE image_id IN (1, 2)',
            connection,
        )
    geometry = shapely.from_wkt(rows["geometry"].to_numpy())
    rows["x"] = shapely.get_x(geometry)
    rows["y"] = shapely.get_y(geometry)
    source = rows.loc[rows.image_id.eq(1), ["trajectory_id", "x", "y"]].rename(
        columns={"x": "source_x", "y": "source_y"}
    )
    target = rows.loc[
        rows.image_id.eq(2), ["trajectory_id", "x", "y", "interpolated", "corr"]
    ].rename(columns={"x": "target_x", "y": "target_y"})
    paired = source.merge(target, on="trajectory_id", how="inner", validate="one_to_one")
    paired["dx_m"] = paired.target_x - paired.source_x
    paired["dy_m"] = paired.target_y - paired.source_y
    return paired


def triangle_field(
    vectors: pd.DataFrame,
    queries: pd.DataFrame,
    elapsed_days: float,
    maximum_edge_m: float,
    minimum_quality: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    vectors = vectors.dropna(subset=["source_x", "source_y", "dx_m", "dy_m"])
    vectors = vectors.drop_duplicates(["source_x", "source_y"], keep="first")
    result = queries.copy()
    for column in COMPONENTS:
        result[column] = np.nan
    result["available"] = False
    if len(vectors) < 3:
        return result, {"vectors": int(len(vectors)), "triangles": 0}

    source = vectors[["source_x", "source_y"]].to_numpy(dtype=float)
    displacement = vectors[["dx_m", "dy_m"]].to_numpy(dtype=float)
    target = source + displacement
    triangulation = Delaunay(source)
    triangles = triangulation.simplices
    source_triangles = source[triangles]
    target_triangles = target[triangles]

    source_edges = np.stack(
        (
            source_triangles[:, 1] - source_triangles[:, 0],
            source_triangles[:, 2] - source_triangles[:, 0],
        ),
        axis=2,
    )
    target_edges = np.stack(
        (
            target_triangles[:, 1] - target_triangles[:, 0],
            target_triangles[:, 2] - target_triangles[:, 0],
        ),
        axis=2,
    )
    edge_lengths = np.stack(
        (
            np.linalg.norm(source_triangles[:, 1] - source_triangles[:, 0], axis=1),
            np.linalg.norm(source_triangles[:, 2] - source_triangles[:, 1], axis=1),
            np.linalg.norm(source_triangles[:, 0] - source_triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    source_cross = np.linalg.det(source_edges)
    target_cross = np.linalg.det(target_edges)
    quality = 2.0 * np.sqrt(3.0) * np.abs(source_cross) / np.maximum(
        np.square(edge_lengths).sum(axis=1), 1.0
    )
    finite = np.isfinite(source_cross) & (np.abs(source_cross) > 1.0)
    valid = (
        finite
        & edge_lengths.max(axis=1).__le__(maximum_edge_m)
        & quality.__ge__(minimum_quality)
        & (source_cross * target_cross > 0.0)
    )

    gradients = np.full((len(triangles), 2, 2), np.nan)
    gradients[finite] = (
        target_edges[finite] @ np.linalg.inv(source_edges[finite]) - np.eye(2)
    )
    components = deformation_components(gradients, elapsed_days)
    query_xy = queries[["source_x", "source_y"]].to_numpy(dtype=float)
    simplex = triangulation.find_simplex(query_xy)
    available = simplex >= 0
    available[available] &= valid[simplex[available]]
    result.loc[available, "available"] = True
    for column in COMPONENTS:
        values = np.asarray(components[column])
        result.loc[available, column] = values[simplex[available]]

    return result, {
        "vectors": int(len(vectors)),
        "triangles": int(len(triangles)),
        "valid_triangles": int(valid.sum()),
        "folded_triangles_excluded": int((source_cross * target_cross <= 0.0).sum()),
        "long_or_low_quality_triangles_excluded": int((finite & ~valid & (source_cross * target_cross > 0.0)).sum()),
        "maximum_edge_m": float(maximum_edge_m),
        "minimum_quality": float(minimum_quality),
    }


def coherent_match_mask(
    matches: pd.DataFrame,
    neighbour_count: int = 12,
    maximum_radius_m: float = 6000.0,
    consensus_radius_m: float = 1000.0,
    minimum_agreement: int = 8,
) -> np.ndarray:
    source = matches[["source_x", "source_y"]].to_numpy(dtype=float)
    displacement = matches[["dx_m", "dy_m"]].to_numpy(dtype=float)
    count = min(neighbour_count, len(matches))
    distances, indices = cKDTree(source).query(
        source, k=count, distance_upper_bound=maximum_radius_m
    )
    keep = np.zeros(len(matches), dtype=bool)
    for row in range(len(matches)):
        local = indices[row][np.isfinite(distances[row]) & (indices[row] < len(matches))]
        if len(local) < minimum_agreement:
            continue
        agreement = np.linalg.norm(displacement[local] - displacement[row], axis=1)
        keep[row] = int((agreement <= consensus_radius_m).sum()) >= minimum_agreement
    return keep


def robust_local_affine(
    source_xy: np.ndarray,
    displacement_xy: np.ndarray,
    query_xy: np.ndarray,
    spatial_scale_m: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    centred_km = (source_xy - query_xy) / 1000.0
    design = np.column_stack((np.ones(len(source_xy)), centred_km))
    if len(source_xy) < 3 or np.linalg.matrix_rank(design) < 3:
        return None
    distance = np.linalg.norm(source_xy - query_xy, axis=1)
    base_weight = np.exp(-0.5 * np.square(distance / spatial_scale_m))
    weight = base_weight.copy()
    coefficients = np.zeros((3, 2))
    residual = np.full(len(source_xy), np.nan)
    for _ in range(5):
        root = np.sqrt(np.maximum(weight, 1.0e-8))
        coefficients = np.linalg.lstsq(
            design * root[:, None], displacement_xy * root[:, None], rcond=None
        )[0]
        residual = np.linalg.norm(design @ coefficients - displacement_xy, axis=1)
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        threshold = max(80.0, 1.5 * scale)
        robust_weight = np.minimum(1.0, threshold / np.maximum(residual, 1.0e-6))
        weight = base_weight * robust_weight
    gradient = coefficients[1:].T / 1000.0
    return gradient, residual


def direct_local_affine_field(
    matches: pd.DataFrame,
    queries: pd.DataFrame,
    elapsed_days: float,
    maximum_radius_m: float = 12000.0,
    candidate_count: int = 64,
    consensus_radius_m: float = 1000.0,
    minimum_agreement: int = 8,
) -> tuple[pd.DataFrame, dict]:
    source = matches[["source_x", "source_y"]].to_numpy(dtype=float)
    displacement = matches[["dx_m", "dy_m"]].to_numpy(dtype=float)
    query_xy = queries[["source_x", "source_y"]].to_numpy(dtype=float)
    count = min(candidate_count, len(matches))
    distances, indices = cKDTree(source).query(
        query_xy, k=count, distance_upper_bound=maximum_radius_m
    )
    rows = []
    for row, point in enumerate(query_xy):
        local = indices[row][np.isfinite(distances[row]) & (indices[row] < len(matches))]
        record = {"available": False, "selected_matches": 0, "median_residual_m": np.nan}
        if len(local) >= minimum_agreement:
            local_displacement = displacement[local]
            pair_distance = np.linalg.norm(
                local_displacement[:, None] - local_displacement[None, :], axis=2
            )
            seed = int(np.argmax((pair_distance <= consensus_radius_m).sum(axis=1)))
            selected = pair_distance[seed] <= consensus_radius_m
            centre = np.median(local_displacement[selected], axis=0)
            selected = np.linalg.norm(local_displacement - centre, axis=1) <= consensus_radius_m
            record["selected_matches"] = int(selected.sum())
            if selected.sum() >= minimum_agreement:
                fit = robust_local_affine(
                    source[local][selected],
                    local_displacement[selected],
                    point,
                    spatial_scale_m=maximum_radius_m / 2.0,
                )
                if fit is not None:
                    gradient, residual = fit
                    components = deformation_components(gradient, elapsed_days)
                    record.update(
                        available=True,
                        median_residual_m=float(np.median(residual)),
                        **{key: float(np.asarray(value)) for key, value in components.items()},
                    )
        rows.append(record)
    return pd.concat([queries.reset_index(drop=True), pd.DataFrame(rows)], axis=1), {
        "raw_matches": int(len(matches)),
        "maximum_radius_m": float(maximum_radius_m),
        "candidate_count": int(candidate_count),
        "consensus_radius_m": float(consensus_radius_m),
        "minimum_agreement": int(minimum_agreement),
    }


def field_distribution(field: pd.DataFrame) -> dict:
    available = field["available"].fillna(False)
    values = field.loc[available, "total_per_day"].dropna().to_numpy(dtype=float)
    return {
        "available_cells": int(len(values)),
        "coverage_fraction": float(len(values) / len(field)) if len(field) else 0.0,
        "median_total_per_day": float(np.median(values)) if len(values) else None,
        "p90_total_per_day": float(np.quantile(values, 0.90)) if len(values) else None,
        "p99_total_per_day": float(np.quantile(values, 0.99)) if len(values) else None,
    }


def compare_fields(first: pd.DataFrame, second: pd.DataFrame) -> dict:
    first_values = first["total_per_day"].to_numpy(dtype=float)
    second_values = second["total_per_day"].to_numpy(dtype=float)
    common = np.isfinite(first_values) & np.isfinite(second_values)
    if not common.any():
        return {"common_cells": 0}
    left = first_values[common]
    right = second_values[common]
    correlation = np.corrcoef(np.log10(left + 1.0e-4), np.log10(right + 1.0e-4))[0, 1]
    return {
        "common_cells": int(common.sum()),
        "median_absolute_difference_per_day": float(np.median(np.abs(left - right))),
        "log10_correlation": float(correlation),
    }


def plot_fields(fields: dict[str, pd.DataFrame], output: Path) -> None:
    values = np.concatenate(
        [field.loc[field.available, "total_per_day"].dropna().to_numpy() for field in fields.values()]
    )
    positive = values[values > 0.0]
    lower = max(1.0e-3, float(np.quantile(positive, 0.02)))
    upper = max(lower * 10.0, float(np.quantile(positive, 0.98)))
    norm = LogNorm(vmin=lower, vmax=upper)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    artist = None
    for axis, (label, field) in zip(axes.ravel(), fields.items(), strict=True):
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
        axis.set_title(f"{label}\ncoverage {available.mean():.1%}")
        axis.set_aspect("equal")
        axis.set_xlabel("EPSG:3413 x (km)")
        axis.set_ylabel("EPSG:3413 y (km)")
    fig.colorbar(artist, ax=axes, shrink=0.82, label="total deformation (day⁻¹)")
    fig.suptitle("Fram Strait · 28–29 March 2015 · 32.84 h", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orb-database", type=Path, required=True)
    parser.add_argument("--orb-table", required=True)
    parser.add_argument("--aliked-field", type=Path, required=True)
    parser.add_argument("--aliked-matches", type=Path, required=True)
    parser.add_argument("--elapsed-hours", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    elapsed_days = args.elapsed_hours / 24.0

    virtual = pd.read_csv(args.aliked_field)
    queries = virtual[["grid_row", "grid_column", "source_x", "source_y"]].copy()
    virtual_vectors = virtual.loc[virtual.available.fillna(False)].copy()
    virtual_vectors = virtual_vectors.rename(
        columns={"proposal_dx_m": "dx_m", "proposal_dy_m": "dy_m"}
    )
    orb_vectors = load_orb_pairs(args.orb_database, args.orb_table)
    matches = pd.read_csv(args.aliked_matches)

    orb_field, orb_geometry = triangle_field(
        orb_vectors, queries, elapsed_days, maximum_edge_m=20000.0, minimum_quality=0.05
    )
    virtual_field, virtual_geometry = triangle_field(
        virtual_vectors, queries, elapsed_days, maximum_edge_m=6400.0
    )
    direct_field, direct_geometry = direct_local_affine_field(
        matches, queries, elapsed_days
    )
    coherent = coherent_match_mask(matches)
    raw_field, raw_geometry = triangle_field(
        matches.loc[coherent],
        queries,
        elapsed_days,
        maximum_edge_m=6400.0,
        minimum_quality=0.10,
    )
    raw_geometry["coherent_matches"] = int(coherent.sum())

    fields = {
        "Production ORB": orb_field,
        "ALIKED virtual 4 km": virtual_field,
        "ALIKED direct local affine": direct_field,
        "ALIKED raw triangles": raw_field,
    }
    combined = queries.copy()
    for label, field in fields.items():
        prefix = label.lower().replace(" ", "_")
        for column in ("available", *COMPONENTS):
            combined[f"{prefix}_{column}"] = field[column].to_numpy()
    combined.to_csv(args.output_dir / "deformation_common_grid.csv", index=False)
    direct_field.to_csv(args.output_dir / "direct_local_affine_field.csv", index=False)

    summary = {
        "elapsed_hours": args.elapsed_hours,
        "reporting_grid_m": 4000.0,
        "orb_vectors": int(len(orb_vectors)),
        "orb_direct_vectors": int((orb_vectors.interpolated == 0).sum()),
        "orb_interpolated_vectors": int((orb_vectors.interpolated == 1).sum()),
        "aliked_raw_matches": int(len(matches)),
        "fields": {label: field_distribution(field) for label, field in fields.items()},
        "geometry": {
            "Production ORB": orb_geometry,
            "ALIKED virtual 4 km": virtual_geometry,
            "ALIKED direct local affine": direct_geometry,
            "ALIKED raw triangles": raw_geometry,
        },
        "comparisons": {
            "orb_vs_virtual": compare_fields(orb_field, virtual_field),
            "virtual_vs_direct_local_affine": compare_fields(virtual_field, direct_field),
            "virtual_vs_raw_triangles": compare_fields(virtual_field, raw_field),
        },
        "interpretation": (
            "Unlabelled pair: coverage, topology, method agreement, and visual structure are "
            "diagnostics rather than deformation accuracy estimates."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_fields(fields, args.output_dir / "deformation_comparison.png")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
