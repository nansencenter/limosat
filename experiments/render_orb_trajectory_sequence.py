#!/usr/bin/env python3
"""Render production ORB trajectories and a matched deformation composite."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
from PIL import Image
import shapely

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.render_efficientloftr_trajectory_sequence import (
    area_weighted_grid,
    image_metadata,
    pair_triangle_gradients,
    plot_deformation,
    sha256,
    valid_time_pairs,
    verify_affine_calculation,
    warp_background,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catalog-image-ids", type=int, nargs="+", required=True)
    parser.add_argument("--minimum-days", type=float, default=0.5)
    parser.add_argument("--maximum-days", type=float, default=2.5)
    parser.add_argument("--grid-spacing-m", type=float, default=4000.0)
    parser.add_argument("--maximum-triangle-edge-m", type=float, default=20_000.0)
    parser.add_argument("--minimum-triangle-quality", type=float, default=0.05)
    return parser.parse_args()


def load_orb_rows(
    run_dir: Path, catalog_image_ids: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, dict, Path, str]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    timings = pd.read_csv(run_dir / "image_timings.csv")
    selected = (
        timings.set_index("catalog_image_id")
        .reindex(catalog_image_ids)
        .reset_index()
    )
    if selected.run_image_id.isna().any():
        missing = selected.loc[selected.run_image_id.isna(), "catalog_image_id"].tolist()
        raise ValueError(f"ORB run is missing catalog image IDs: {missing}")
    run_image_ids = selected.run_image_id.astype(int).tolist()
    engine_prefix = "sqlite:///"
    if not manifest["engine_url"].startswith(engine_prefix):
        raise ValueError("Only SQLite production runs are supported")
    database = Path(manifest["engine_url"][len(engine_prefix) :])
    table = manifest["effective_run_name"]
    placeholders = ",".join("?" for _ in run_image_ids)
    query = f'''SELECT image_id, trajectory_id, geometry, interpolated, corr
                FROM "{table}" WHERE image_id IN ({placeholders})'''
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        rows = pd.read_sql_query(query, connection, params=run_image_ids)
    geometry = shapely.from_wkt(rows.geometry.to_numpy())
    rows["x_m"] = shapely.get_x(geometry)
    rows["y_m"] = shapely.get_y(geometry)
    run_to_catalog = dict(zip(run_image_ids, catalog_image_ids, strict=True))
    rows["catalog_image_id"] = rows.image_id.map(run_to_catalog).astype(int)
    if rows.duplicated(["image_id", "trajectory_id"]).any():
        raise ValueError("Production ORB table has duplicate trajectory rows in an image")
    return rows, selected, manifest, database, table


def sparse_trajectory_arrays(
    rows: pd.DataFrame, catalog_image_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trajectory_ids = np.sort(rows.trajectory_id.unique())
    trajectory_index = {value: index for index, value in enumerate(trajectory_ids)}
    image_index = {value: index for index, value in enumerate(catalog_image_ids)}
    shape = (len(trajectory_ids), len(catalog_image_ids))
    xy = np.full((*shape, 2), np.nan, dtype=float)
    present = np.zeros(shape, dtype=bool)
    interpolated = np.zeros(shape, dtype=bool)
    correlation = np.full(shape, np.nan, dtype=float)
    for row in rows.itertuples(index=False):
        i = trajectory_index[row.trajectory_id]
        j = image_index[row.catalog_image_id]
        xy[i, j] = (row.x_m, row.y_m)
        present[i, j] = True
        interpolated[i, j] = bool(row.interpolated)
        correlation[i, j] = row.corr
    return trajectory_ids, xy, present, interpolated, correlation


def observed_paths(
    xy: np.ndarray, present: np.ndarray, frame: int
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    current_paths: list[np.ndarray] = []
    absent_paths: list[np.ndarray] = []
    absent_endpoints = []
    for trajectory in range(len(xy)):
        indices = np.flatnonzero(present[trajectory, : frame + 1])
        if not len(indices):
            continue
        path = xy[trajectory, indices]
        if present[trajectory, frame]:
            if len(path) > 1:
                current_paths.append(path)
        else:
            if len(path) > 1:
                absent_paths.append(path)
            absent_endpoints.append(path[-1])
    return current_paths, absent_paths, np.asarray(absent_endpoints).reshape(-1, 2)


def latest_links(
    xy: np.ndarray, present: np.ndarray, frame: int
) -> tuple[np.ndarray, np.ndarray]:
    adjacent = []
    skipped = []
    if frame == 0:
        return np.empty((0, 2, 2)), np.empty((0, 2, 2))
    for trajectory in np.flatnonzero(present[:, frame]):
        prior = np.flatnonzero(present[trajectory, :frame])
        if not len(prior):
            continue
        previous = prior[-1]
        segment = np.stack((xy[trajectory, previous], xy[trajectory, frame]))
        (adjacent if previous == frame - 1 else skipped).append(segment)
    return np.asarray(adjacent).reshape(-1, 2, 2), np.asarray(skipped).reshape(-1, 2, 2)


def draw_orb_frame(
    path: Path,
    background: np.ma.MaskedArray,
    bounds: tuple[float, float, float, float],
    xy: np.ndarray,
    present: np.ndarray,
    interpolated: np.ndarray,
    frame: int,
    timestamp: pd.Timestamp,
    start_time: pd.Timestamp,
) -> None:
    fig, axis = plt.subplots(figsize=(11.5, 8.2))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.09, top=0.90)
    axis.set_facecolor("#0d1720")
    axis.imshow(
        background,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        extent=bounds,
        origin="upper",
        interpolation="bilinear",
        alpha=0.78,
    )
    current_paths, absent_paths, absent_endpoints = observed_paths(xy, present, frame)
    adjacent, skipped = latest_links(xy, present, frame)
    if absent_paths:
        axis.add_collection(LineCollection(absent_paths, colors="#9aa4ad", linewidths=0.45, alpha=0.20))
    if current_paths:
        axis.add_collection(LineCollection(current_paths, colors="#36d6d0", linewidths=0.70, alpha=0.48))
    if len(adjacent):
        axis.add_collection(LineCollection(adjacent, colors="#ffb347", linewidths=0.95, alpha=0.72))
    if len(skipped):
        axis.add_collection(
            LineCollection(skipped, colors="#e879f9", linewidths=1.0, alpha=0.78, linestyles="dashed")
        )
    if len(absent_endpoints):
        axis.scatter(
            absent_endpoints[:, 0], absent_endpoints[:, 1], s=4, c="#b8bec4",
            marker="x", linewidths=0.35, alpha=0.30,
        )
    direct = present[:, frame] & ~interpolated[:, frame]
    interp = present[:, frame] & interpolated[:, frame]
    axis.scatter(xy[direct, frame, 0], xy[direct, frame, 1], s=4, c="#72fff4", linewidths=0, alpha=0.82)
    axis.scatter(
        xy[interp, frame, 0], xy[interp, frame, 1], s=10, facecolors="none",
        edgecolors="#ffe08a", linewidths=0.55, alpha=0.9,
    )

    current = int(present[:, frame].sum())
    first_cohort = int((present[:, frame] & present[:, 0]).sum())
    first_fraction = first_cohort / max(int(present[:, 0].sum()), 1)
    elapsed_hours = (timestamp - start_time).total_seconds() / 3600.0
    fig.text(
        0.10, 0.965, "Production ORB material-point trajectories",
        fontsize=16, fontweight="bold", color="#f4f7f8", ha="left", va="top",
    )
    fig.text(
        0.10, 0.935,
        f"Sentinel-1 {timestamp:%Y-%m-%d %H:%M UTC}  ·  observation {frame + 1}/{present.shape[1]}",
        color="#d7dee3", fontsize=10, ha="left", va="top",
    )
    axis.text(
        0.985,
        0.975,
        f"{current:,} persisted positions\n"
        f"{first_cohort:,}/{present[:, 0].sum():,} first-frame IDs linked ({first_fraction:.1%})\n"
        f"{int(interp.sum()):,} interpolated · {len(skipped):,} skip links\n"
        f"{elapsed_hours:.1f} h elapsed",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color="white",
        bbox={"facecolor": "#101820", "edgecolor": "#7b8790", "alpha": 0.84, "boxstyle": "round,pad=0.55"},
    )
    axis.plot([], [], color="#ffb347", linewidth=2, label="adjacent-image link")
    axis.plot([], [], color="#e879f9", linewidth=2, linestyle="--", label="link across skipped image(s)")
    axis.plot([], [], color="#36d6d0", linewidth=2, label="persisted trajectory history")
    axis.scatter([], [], s=16, facecolors="none", edgecolors="#ffe08a", label="interpolated position")
    legend = axis.legend(loc="lower left", frameon=True, fontsize=8.3, ncol=2)
    legend.get_frame().set_facecolor("#101820")
    legend.get_frame().set_edgecolor("#7b8790")
    for text in legend.get_texts():
        text.set_color("white")

    xmin, ymin, xmax, ymax = bounds
    axis.plot([xmax - 72_000, xmax - 22_000], [ymin + 22_000] * 2, color="white", linewidth=3)
    axis.text(xmax - 47_000, ymin + 29_000, "50 km", color="white", ha="center", fontsize=9)
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal")
    axis.set_xlabel("EPSG:3413 easting (m)", color="#d7dee3")
    axis.set_ylabel("EPSG:3413 northing (m)", color="#d7dee3")
    axis.tick_params(colors="#d7dee3")
    for spine in axis.spines.values():
        spine.set_color("#7b8790")
    fig.savefig(path, dpi=125, facecolor="#0d1720")
    plt.close(fig)


def make_animation(
    output_dir: Path,
    images: pd.DataFrame,
    xy: np.ndarray,
    present: np.ndarray,
    interpolated: np.ndarray,
) -> tuple[Path, Path, Path, tuple[float, float, float, float]]:
    valid_xy = xy[present]
    padding = 12_000.0
    bounds = (
        float(valid_xy[:, 0].min() - padding),
        float(valid_xy[:, 1].min() - padding),
        float(valid_xy[:, 0].max() + padding),
        float(valid_xy[:, 1].max() + padding),
    )
    backgrounds = [
        warp_background(Path(row.standard_vae_output_path), bounds)
        for row in images.itertuples(index=False)
    ]
    gif_path = output_dir / "production_orb_march_trajectory_sequence.gif"
    mp4_path = output_dir / "production_orb_march_trajectory_sequence.mp4"
    final_path = output_dir / "production_orb_march_trajectory_final_frame.png"
    with tempfile.TemporaryDirectory(prefix="orb_frames_") as temporary:
        temporary_dir = Path(temporary)
        frame_paths = []
        for frame, row in enumerate(images.itertuples(index=False)):
            frame_path = temporary_dir / f"frame_{frame:02d}.png"
            draw_orb_frame(
                frame_path, backgrounds[frame], bounds, xy, present, interpolated,
                frame, row.image_time, images.image_time.iloc[0],
            )
            frame_paths.append(frame_path)
        with Image.open(frame_paths[-1]) as final_frame:
            final_frame.save(final_path)
        gif_frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as frame:
                gif_frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=160))
        gif_frames[0].save(
            gif_path, save_all=True, append_images=gif_frames[1:],
            duration=1250, loop=0, optimize=False,
        )
        environment_ffmpeg = Path(sys.executable).with_name("ffmpeg")
        ffmpeg = environment_ffmpeg if environment_ffmpeg.exists() else Path("ffmpeg")
        subprocess.run(
            [
                str(ffmpeg), "-loglevel", "error", "-y", "-framerate", "0.8",
                "-i", str(temporary_dir / "frame_%02d.png"),
                "-vf", "fps=12,pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
                "-c:v", "libx264", "-crf", "20", str(mp4_path),
            ],
            check=True,
        )
    return gif_path, mp4_path, final_path, bounds


def make_deformation(
    output_dir: Path,
    images: pd.DataFrame,
    xy: np.ndarray,
    present: np.ndarray,
    minimum_days: float,
    maximum_days: float,
    spacing_m: float,
    maximum_edge_m: float,
    minimum_quality: float,
    grid_origin_m: tuple[float, float],
    map_bounds_m: tuple[float, float, float, float],
) -> tuple[Path, Path, dict]:
    verify_affine_calculation()
    pair_stats = []
    samples = []
    pairs = valid_time_pairs(images.image_time, minimum_days, maximum_days)
    for pair_id, (source_index, target_index, elapsed_days) in enumerate(pairs):
        pair_samples, stats = pair_triangle_gradients(
            xy,
            present,
            source_index,
            target_index,
            elapsed_days,
            maximum_edge_m,
            minimum_quality,
        )
        pair_samples["pair_id"] = pair_id
        samples.append(pair_samples)
        stats.update(
            pair_id=pair_id,
            source_image_id=int(images.iloc[source_index].image_id),
            target_image_id=int(images.iloc[target_index].image_id),
        )
        pair_stats.append(stats)
    all_samples = pd.concat(samples, ignore_index=True)
    grid, grid_metadata = area_weighted_grid(all_samples, spacing_m, grid_origin_m)
    report = {
        "minimum_days": minimum_days,
        "maximum_days": maximum_days,
        "valid_interval_count": len(pairs),
        "maximum_triangle_edge_m": maximum_edge_m,
        "minimum_triangle_quality": minimum_quality,
        "synthetic_affine_check": "passed",
        "pair_statistics": pair_stats,
        "grid": grid_metadata,
        "total_deformation_per_day": {
            "median": float(grid.total_per_day.median()),
            "p90": float(grid.total_per_day.quantile(0.90)),
            "p99": float(grid.total_per_day.quantile(0.99)),
        },
        "interval_support": {
            "median": float(grid.interval_support.median()),
            "p10": float(grid.interval_support.quantile(0.10)),
            "maximum": int(grid.interval_support.max()),
        },
        **grid_metadata,
    }
    csv_path = output_dir / "production_orb_deformation_0p5_2p5day_grid.csv"
    png_path = output_dir / "production_orb_deformation_0p5_2p5day_composite.png"
    grid.to_csv(csv_path, index=False)
    plot_deformation(grid, report, png_path, "Production ORB", map_bounds_m)
    return png_path, csv_path, report


def dense_field(
    grid: pd.DataFrame,
    name: str,
    row_min: int,
    row_max: int,
    column_min: int,
    column_max: int,
) -> np.ma.MaskedArray:
    array = np.full((row_max - row_min + 1, column_max - column_min + 1), np.nan)
    array[
        grid.row.to_numpy(int) - row_min,
        grid.column.to_numpy(int) - column_min,
    ] = grid[name].to_numpy(float)
    return np.ma.masked_invalid(array)


def make_comparison_plot(
    orb_grid_path: Path,
    reference_grid_path: Path,
    reference_manifest: dict,
    output_path: Path,
) -> dict:
    orb = pd.read_csv(orb_grid_path)
    learned = pd.read_csv(reference_grid_path)
    origin_x = float(reference_manifest["deformation"]["grid"]["x_origin_m"])
    origin_y = float(reference_manifest["deformation"]["grid"]["y_origin_m"])
    spacing = float(reference_manifest["deformation"]["grid"]["spacing_m"])
    row_min = int(min(orb.row.min(), learned.row.min()))
    row_max = int(max(orb.row.max(), learned.row.max()))
    column_min = int(min(orb.column.min(), learned.column.min()))
    column_max = int(max(orb.column.max(), learned.column.max()))
    x_edges = origin_x + np.arange(column_min, column_max + 2) * spacing
    y_edges = origin_y + np.arange(row_min, row_max + 2) * spacing
    orb_field = dense_field(orb, "total_per_day", row_min, row_max, column_min, column_max)
    learned_field = dense_field(learned, "total_per_day", row_min, row_max, column_min, column_max)
    difference = np.ma.masked_invalid(orb_field - learned_field)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), constrained_layout=True, sharex=True, sharey=True)
    orb_artist = axes[0].pcolormesh(x_edges, y_edges, orb_field, cmap="plasma_r", norm=Normalize(0, 0.1), shading="flat")
    axes[1].pcolormesh(x_edges, y_edges, learned_field, cmap="plasma_r", norm=Normalize(0, 0.1), shading="flat")
    difference_artist = axes[2].pcolormesh(
        x_edges, y_edges, difference, cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-0.05, vcenter=0.0, vmax=0.05), shading="flat",
    )
    axes[0].set_title("Production ORB")
    axes[1].set_title("EfficientLoFTR")
    axes[2].set_title("ORB − EfficientLoFTR\ncommon cells only")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("EPSG:3413 easting (m)")
    axes[0].set_ylabel("EPSG:3413 northing (m)")
    fig.colorbar(
        orb_artist, ax=axes[:2], orientation="horizontal", pad=0.08,
        fraction=0.055, label="total deformation (day⁻¹; clipped at 0.10)",
    )
    fig.colorbar(
        difference_artist, ax=axes[2], orientation="horizontal", pad=0.08,
        fraction=0.055, label="difference (day⁻¹; clipped at ±0.05)",
    )
    fig.suptitle("Matched 0.5–2.5-day trajectory deformation comparison", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=210, facecolor="white")
    plt.close(fig)

    common = orb.merge(
        learned[["row", "column", "total_per_day"]],
        on=["row", "column"],
        suffixes=("_orb", "_efficientloftr"),
    )
    delta = common.total_per_day_orb - common.total_per_day_efficientloftr
    return {
        "orb_cells": int(len(orb)),
        "efficientloftr_cells": int(len(learned)),
        "common_cells": int(len(common)),
        "orb_only_cells": int(len(orb) - len(common)),
        "efficientloftr_only_cells": int(len(learned) - len(common)),
        "common_cell_median_absolute_difference_per_day": float(np.median(np.abs(delta))),
        "common_cell_spearman": float(common.total_per_day_orb.corr(common.total_per_day_efficientloftr, method="spearman")),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_manifest = json.loads(args.reference_manifest.read_text())
    rows, selected_timings, run_manifest, database, table = load_orb_rows(
        args.run_dir, args.catalog_image_ids
    )
    images = image_metadata(args.observations, args.catalog_image_ids)
    _, xy, present, interpolated, _ = sparse_trajectory_arrays(rows, args.catalog_image_ids)

    gif_path, mp4_path, final_path, map_bounds = make_animation(
        args.output_dir, images, xy, present, interpolated
    )
    reference_grid_metadata = reference_manifest["deformation"]["grid"]
    grid_origin = (
        float(reference_grid_metadata["x_origin_m"]),
        float(reference_grid_metadata["y_origin_m"]),
    )
    deformation_path, deformation_grid_path, deformation_report = make_deformation(
        args.output_dir,
        images,
        xy,
        present,
        args.minimum_days,
        args.maximum_days,
        args.grid_spacing_m,
        args.maximum_triangle_edge_m,
        args.minimum_triangle_quality,
        grid_origin,
        map_bounds,
    )
    comparison_path = args.output_dir / "production_orb_vs_efficientloftr_deformation_comparison.png"
    comparison = make_comparison_plot(
        deformation_grid_path,
        args.reference_grid,
        reference_manifest,
        comparison_path,
    )

    first = present[:, 0]
    final = present[:, -1]
    reference_bounds = np.asarray(reference_manifest["map_bounds_m"], dtype=float)
    within_reference = (
        (rows.x_m >= reference_bounds[0])
        & (rows.y_m >= reference_bounds[1])
        & (rows.x_m <= reference_bounds[2])
        & (rows.y_m <= reference_bounds[3])
    )
    manifest = {
        "production_run": str(args.run_dir),
        "production_run_manifest_sha256": sha256(args.run_dir / "run_manifest.json"),
        "database": str(database),
        "table": table,
        "configuration": "production ORB Q24/I48 quadratic subpixel pattern matching with bilinear template sampling",
        "trajectory_policy": "all persisted direct and interpolated positions; trajectory IDs may enter, leave, or reconnect across skipped images",
        "analysis_crs": "EPSG:3413",
        "coordinate_units": "metres",
        "catalog_image_ids": args.catalog_image_ids,
        "run_image_ids": selected_timings.run_image_id.astype(int).tolist(),
        "image_times_utc": [time.isoformat() for time in images.image_time],
        "unique_trajectories_in_window": int(len(present)),
        "positions_by_image": present.sum(axis=0).astype(int).tolist(),
        "interpolated_positions_by_image": interpolated.sum(axis=0).astype(int).tolist(),
        "present_in_every_displayed_image": int(present.all(axis=1).sum()),
        "first_frame_ids_present_at_final_frame": int((first & final).sum()),
        "first_frame_link_fraction_at_final_frame": float((first & final).sum() / first.sum()),
        "rows_inside_efficientloftr_animation_bounds_fraction": float(within_reference.mean()),
        "map_bounds_m": list(map_bounds),
        "deformation": deformation_report,
        "comparison": comparison,
        "outputs": {
            "trajectory_gif": str(gif_path),
            "trajectory_mp4": str(mp4_path),
            "trajectory_final_frame": str(final_path),
            "deformation_png": str(deformation_path),
            "deformation_grid_csv": str(deformation_grid_path),
            "comparison_png": str(comparison_path),
        },
        "reference": {
            "efficientloftr_manifest": str(args.reference_manifest),
            "efficientloftr_manifest_sha256": sha256(args.reference_manifest),
            "efficientloftr_grid": str(args.reference_grid),
            "efficientloftr_grid_sha256": sha256(args.reference_grid),
        },
        "production_run_elapsed_seconds": float(run_manifest["elapsed_seconds"]),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
