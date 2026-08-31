#!/usr/bin/env python3
"""Render a learned-drift trajectory sequence and a LiMOSAT-style deformation window."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from scipy.spatial import Delaunay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-days", type=float, default=0.5)
    parser.add_argument("--maximum-days", type=float, default=2.5)
    parser.add_argument("--grid-spacing-m", type=float, default=4000.0)
    parser.add_argument("--maximum-triangle-edge-m", type=float, default=6400.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_metadata(observations_path: Path, image_ids: list[int]) -> pd.DataFrame:
    observations = pd.read_csv(
        observations_path,
        usecols=["image_id", "image_time", "standard_vae_output_path"],
    )
    images = (
        observations.loc[observations.image_id.isin(image_ids)]
        .drop_duplicates("image_id")
        .set_index("image_id")
        .reindex(image_ids)
        .reset_index()
    )
    if images[["image_time", "standard_vae_output_path"]].isna().any().any():
        missing = images.loc[images.image_time.isna(), "image_id"].tolist()
        raise ValueError(f"Missing observation metadata for image IDs: {missing}")
    images["image_time"] = pd.to_datetime(images.image_time, utc=True)
    return images


def trajectory_arrays(
    trajectories: pd.DataFrame, image_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trajectory_ids = np.sort(trajectories.trajectory_id.unique())
    index_by_trajectory = {value: index for index, value in enumerate(trajectory_ids)}
    index_by_image = {value: index for index, value in enumerate(image_ids)}
    xy = np.full((len(trajectory_ids), len(image_ids), 2), np.nan, dtype=float)
    active = np.zeros((len(trajectory_ids), len(image_ids)), dtype=bool)
    for row in trajectories.itertuples(index=False):
        i = index_by_trajectory[row.trajectory_id]
        j = index_by_image[row.image_id]
        xy[i, j] = (row.x_m, row.y_m)
        active[i, j] = bool(row.active)
    if np.isnan(xy).any():
        raise ValueError("Trajectory table is not rectangular across the image sequence")
    if np.any(np.diff(active.astype(np.int8), axis=1) > 0):
        raise ValueError("Strict trajectories unexpectedly reactivate after ending")
    return trajectory_ids, xy, active, active.sum(axis=0)


def warp_background(path: Path, bounds: tuple[float, float, float, float], width: int = 1050) -> np.ma.MaskedArray:
    from osgeo import gdal

    gdal.UseExceptions()
    xmin, ymin, xmax, ymax = bounds
    height = max(1, round(width * (ymax - ymin) / (xmax - xmin)))
    source = gdal.Open(str(path))
    if source is None:
        raise FileNotFoundError(path)
    warped = gdal.Warp(
        "",
        source,
        format="MEM",
        srcBands=[1],
        dstAlpha=True,
        dstSRS="EPSG:3413",
        outputBounds=bounds,
        width=width,
        height=height,
        resampleAlg="bilinear",
        multithread=True,
    )
    arrays = warped.ReadAsArray()
    intensity = arrays[0].astype(float)
    valid = arrays[1] > 0
    if valid.any():
        low, high = np.quantile(intensity[valid], (0.02, 0.98))
        intensity = np.clip((intensity - low) / max(high - low, 1.0), 0.0, 1.0)
    return np.ma.array(intensity, mask=~valid)


def line_segments(xy: np.ndarray, active: np.ndarray, frame: int) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    current: list[np.ndarray] = []
    ended: list[np.ndarray] = []
    ended_points = []
    for row in range(len(xy)):
        indices = np.flatnonzero(active[row, : frame + 1])
        if not len(indices):
            continue
        path = xy[row, indices]
        if active[row, frame]:
            if len(path) > 1:
                current.append(path)
        elif len(path) > 1:
            ended.append(path)
            ended_points.append(path[-1])
    points = np.asarray(ended_points, dtype=float).reshape(-1, 2)
    return current, ended, points


def draw_trajectory_frame(
    path: Path,
    background: np.ma.MaskedArray,
    bounds: tuple[float, float, float, float],
    xy: np.ndarray,
    active: np.ndarray,
    active_counts: np.ndarray,
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
    current, ended, ended_points = line_segments(xy, active, frame)
    if ended:
        axis.add_collection(LineCollection(ended, colors="#9aa4ad", linewidths=0.35, alpha=0.22))
    if current:
        axis.add_collection(LineCollection(current, colors="#36d6d0", linewidths=0.55, alpha=0.42))
    if frame > 0:
        latest = active[:, frame] & active[:, frame - 1]
        latest_segments = np.stack((xy[latest, frame - 1], xy[latest, frame]), axis=1)
        axis.add_collection(LineCollection(latest_segments, colors="#ffb347", linewidths=0.75, alpha=0.55))
    if len(ended_points):
        axis.scatter(ended_points[:, 0], ended_points[:, 1], s=3, c="#b8bec4", marker="x", linewidths=0.3, alpha=0.32)
    current_points = xy[active[:, frame], frame]
    axis.scatter(current_points[:, 0], current_points[:, 1], s=2.2, c="#72fff4", linewidths=0, alpha=0.78)

    elapsed_hours = (timestamp - start_time).total_seconds() / 3600.0
    survival = active_counts[frame] / active_counts[0]
    fig.text(
        0.10,
        0.965,
        "EfficientLoFTR material-point trajectories",
        fontsize=16,
        fontweight="bold",
        color="#f4f7f8",
        ha="left",
        va="top",
    )
    fig.text(
        0.10,
        0.935,
        f"Sentinel-1 {timestamp:%Y-%m-%d %H:%M UTC}  ·  observation {frame + 1}/{active.shape[1]}",
        color="#d7dee3",
        fontsize=10,
        ha="left",
        va="top",
    )
    axis.text(
        0.985,
        0.975,
        f"{active_counts[frame]:,} supported\n{survival:.1%} of {active_counts[0]:,} seeds\n{elapsed_hours:.1f} h elapsed",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="white",
        bbox={"facecolor": "#101820", "edgecolor": "#7b8790", "alpha": 0.82, "boxstyle": "round,pad=0.55"},
    )
    axis.plot([], [], color="#ffb347", linewidth=2, label="latest observed displacement")
    axis.plot([], [], color="#36d6d0", linewidth=2, label="supported trajectory history")
    axis.plot([], [], color="#9aa4ad", linewidth=2, label="trajectory ended")
    legend = axis.legend(loc="lower left", frameon=True, fontsize=8.5, ncol=1)
    legend.get_frame().set_facecolor("#101820")
    legend.get_frame().set_edgecolor("#7b8790")
    for text in legend.get_texts():
        text.set_color("white")

    xmin, ymin, xmax, ymax = bounds
    axis.plot([xmin + 18_000, xmin + 68_000], [ymin + 18_000] * 2, color="white", linewidth=3)
    axis.text(xmin + 43_000, ymin + 23_000, "50 km", color="white", ha="center", fontsize=9)
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
    active: np.ndarray,
    active_counts: np.ndarray,
) -> tuple[Path, Path, Path, tuple[float, float, float, float]]:
    valid_xy = xy[active]
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
    gif_path = output_dir / "efficientloftr_march_trajectory_sequence.gif"
    mp4_path = output_dir / "efficientloftr_march_trajectory_sequence.mp4"
    overview_path = output_dir / "efficientloftr_march_trajectory_final_frame.png"
    with tempfile.TemporaryDirectory(prefix="efficientloftr_frames_") as temporary:
        temporary_dir = Path(temporary)
        frame_paths = []
        for frame, row in enumerate(images.itertuples(index=False)):
            frame_path = temporary_dir / f"frame_{frame:02d}.png"
            draw_trajectory_frame(
                frame_path,
                backgrounds[frame],
                bounds,
                xy,
                active,
                active_counts,
                frame,
                row.image_time,
                images.image_time.iloc[0],
            )
            frame_paths.append(frame_path)
        with Image.open(frame_paths[-1]) as final_frame:
            final_frame.save(overview_path)
        gif_frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as frame:
                gif_frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=160))
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=1250,
            loop=0,
            optimize=False,
        )
        environment_ffmpeg = Path(sys.executable).with_name("ffmpeg")
        ffmpeg = environment_ffmpeg if environment_ffmpeg.exists() else Path("ffmpeg")
        subprocess.run(
            [
                str(ffmpeg),
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                "0.8",
                "-i",
                str(temporary_dir / "frame_%02d.png"),
                "-vf",
                "fps=12,pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                str(mp4_path),
            ],
            check=True,
        )
    return gif_path, mp4_path, overview_path, bounds


def deformation_components(velocity_gradient: np.ndarray) -> dict[str, np.ndarray]:
    divergence = velocity_gradient[..., 0, 0] + velocity_gradient[..., 1, 1]
    shear = np.hypot(
        velocity_gradient[..., 0, 0] - velocity_gradient[..., 1, 1],
        velocity_gradient[..., 0, 1] + velocity_gradient[..., 1, 0],
    )
    return {
        "divergence_per_day": divergence,
        "shear_per_day": shear,
        "total_per_day": np.hypot(divergence, shear),
        "vorticity_per_day": velocity_gradient[..., 1, 0] - velocity_gradient[..., 0, 1],
    }


def valid_time_pairs(times: pd.Series, minimum_days: float, maximum_days: float) -> list[tuple[int, int, float]]:
    pairs = []
    for source_index in range(len(times)):
        for target_index in range(source_index + 1, len(times)):
            elapsed_days = (times.iloc[target_index] - times.iloc[source_index]).total_seconds() / 86_400.0
            if minimum_days <= elapsed_days <= maximum_days:
                pairs.append((source_index, target_index, elapsed_days))
    return pairs


def pair_triangle_gradients(
    xy: np.ndarray,
    active: np.ndarray,
    source_index: int,
    target_index: int,
    elapsed_days: float,
    maximum_edge_m: float,
    minimum_quality: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    available = active[:, source_index] & active[:, target_index]
    source = xy[available, source_index]
    target = xy[available, target_index]
    triangulation = Delaunay(source)
    triangles = triangulation.simplices
    source_triangles = source[triangles]
    target_triangles = target[triangles]
    source_edges = np.stack(
        (source_triangles[:, 1] - source_triangles[:, 0], source_triangles[:, 2] - source_triangles[:, 0]),
        axis=2,
    )
    target_edges = np.stack(
        (target_triangles[:, 1] - target_triangles[:, 0], target_triangles[:, 2] - target_triangles[:, 0]),
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
    valid = (
        np.isfinite(source_cross)
        & (np.abs(source_cross) > 1.0)
        & (edge_lengths.max(axis=1) <= maximum_edge_m)
        & (quality >= minimum_quality)
        & (source_cross * target_cross > 0.0)
    )
    deformation_gradient = target_edges[valid] @ np.linalg.inv(source_edges[valid])
    velocity_gradient = (deformation_gradient - np.eye(2)) / elapsed_days
    centres = source_triangles[valid].mean(axis=1)
    areas = np.abs(source_cross[valid]) / 2.0
    records = pd.DataFrame(
        {
            "x_m": centres[:, 0],
            "y_m": centres[:, 1],
            "area_m2": areas,
            "ux_per_day": velocity_gradient[:, 0, 0],
            "uy_per_day": velocity_gradient[:, 0, 1],
            "vx_per_day": velocity_gradient[:, 1, 0],
            "vy_per_day": velocity_gradient[:, 1, 1],
        }
    )
    return records, {
        "source_index": source_index,
        "target_index": target_index,
        "elapsed_days": elapsed_days,
        "shared_trajectories": int(available.sum()),
        "triangles": int(len(triangles)),
        "valid_triangles": int(valid.sum()),
        "folded_triangles_excluded": int((source_cross * target_cross <= 0.0).sum()),
        "long_triangles_excluded": int((edge_lengths.max(axis=1) > maximum_edge_m).sum()),
        "low_quality_triangles_excluded": int((quality < minimum_quality).sum()),
    }


def area_weighted_grid(
    samples: pd.DataFrame,
    spacing_m: float,
    origin_m: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, dict]:
    maximum_sample_area = (1.5 * spacing_m) ** 2
    samples_before_area_filter = len(samples)
    samples = samples.loc[samples.area_m2 < maximum_sample_area].copy()
    if origin_m is None:
        x_origin = float(np.floor(samples.x_m.min() / spacing_m) * spacing_m)
        y_origin = float(np.floor(samples.y_m.min() / spacing_m) * spacing_m)
    else:
        x_origin, y_origin = map(float, origin_m)
    samples = samples.copy()
    samples["column"] = np.floor((samples.x_m - x_origin) / spacing_m).astype(int)
    samples["row"] = np.floor((samples.y_m - y_origin) / spacing_m).astype(int)
    gradient_names = ("ux_per_day", "uy_per_day", "vx_per_day", "vy_per_day")
    for name in gradient_names:
        samples[f"weighted_{name}"] = samples[name] * samples.area_m2
    aggregations = {
        "area_sum_m2": ("area_m2", "sum"),
        "interval_support": ("pair_id", "nunique"),
        "triangle_samples": ("pair_id", "size"),
    }
    aggregations.update({f"weighted_{name}": (f"weighted_{name}", "sum") for name in gradient_names})
    gridded = samples.groupby(["row", "column"], as_index=False).agg(**aggregations)
    for name in gradient_names:
        gridded[name] = gridded[f"weighted_{name}"] / gridded.area_sum_m2
    velocity_gradient = np.empty((len(gridded), 2, 2), dtype=float)
    velocity_gradient[:, 0, 0] = gridded.ux_per_day
    velocity_gradient[:, 0, 1] = gridded.uy_per_day
    velocity_gradient[:, 1, 0] = gridded.vx_per_day
    velocity_gradient[:, 1, 1] = gridded.vy_per_day
    for name, values in deformation_components(velocity_gradient).items():
        gridded[name] = values
    gridded["x_m"] = x_origin + (gridded.column + 0.5) * spacing_m
    gridded["y_m"] = y_origin + (gridded.row + 0.5) * spacing_m
    minimum_area = (0.75 * spacing_m) ** 2
    before = len(gridded)
    gridded = gridded.loc[gridded.area_sum_m2 >= minimum_area].copy()
    keep_columns = [
        "row",
        "column",
        "x_m",
        "y_m",
        "interval_support",
        "triangle_samples",
        "area_sum_m2",
        *gradient_names,
        "divergence_per_day",
        "shear_per_day",
        "total_per_day",
        "vorticity_per_day",
    ]
    return gridded[keep_columns], {
        "x_origin_m": x_origin,
        "y_origin_m": y_origin,
        "spacing_m": spacing_m,
        "minimum_aggregate_area_m2": minimum_area,
        "maximum_triangle_area_m2": maximum_sample_area,
        "triangle_samples_before_area_filter": samples_before_area_filter,
        "triangle_samples_after_area_filter": int(len(samples)),
        "cells_before_area_filter": before,
        "cells_after_area_filter": int(len(gridded)),
    }


def plot_deformation(
    grid: pd.DataFrame,
    metadata: dict,
    output_path: Path,
    method_label: str = "EfficientLoFTR",
    map_bounds_m: tuple[float, float, float, float] | None = None,
) -> None:
    rows = np.arange(grid.row.min(), grid.row.max() + 1)
    columns = np.arange(grid.column.min(), grid.column.max() + 1)
    shape = (len(rows), len(columns))
    fields = {}
    for name in ("total_per_day", "divergence_per_day", "interval_support"):
        array = np.full(shape, np.nan)
        array[grid.row.to_numpy() - rows[0], grid.column.to_numpy() - columns[0]] = grid[name]
        fields[name] = np.ma.masked_invalid(array)
    spacing = metadata["spacing_m"]
    x_edges = metadata["x_origin_m"] + np.arange(columns[0], columns[-1] + 2) * spacing
    y_edges = metadata["y_origin_m"] + np.arange(rows[0], rows[-1] + 2) * spacing

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), constrained_layout=True, sharex=True, sharey=True)
    total = axes[0].pcolormesh(x_edges, y_edges, fields["total_per_day"], cmap="plasma_r", norm=Normalize(0.0, 0.1), shading="flat")
    divergence = axes[1].pcolormesh(
        x_edges,
        y_edges,
        fields["divergence_per_day"],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-0.05, vcenter=0.0, vmax=0.05),
        shading="flat",
    )
    support = axes[2].pcolormesh(
        x_edges,
        y_edges,
        fields["interval_support"],
        cmap="viridis",
        norm=Normalize(1, metadata["valid_interval_count"]),
        shading="flat",
    )
    colorbars = [
        (total, axes[0], "total deformation (day⁻¹; clipped at 0.10)"),
        (divergence, axes[1], "divergence (day⁻¹)"),
        (support, axes[2], "contributing image-pair intervals"),
    ]
    for artist, axis, label in colorbars:
        fig.colorbar(artist, ax=axis, orientation="horizontal", pad=0.08, fraction=0.055, label=label)
    axes[0].set_title("Total deformation")
    axes[1].set_title("Divergence\nblue: convergence · red: opening")
    axes[2].set_title("Temporal support")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("EPSG:3413 easting (m)")
        axis.grid(color="white", alpha=0.12, linewidth=0.35)
        if map_bounds_m is not None:
            xmin, ymin, xmax, ymax = map_bounds_m
            axis.set_xlim(xmin, xmax)
            axis.set_ylim(ymin, ymax)
    axes[0].set_ylabel("EPSG:3413 northing (m)")
    fig.suptitle(
        f"{method_label} deformation · {metadata['valid_interval_count']} trajectory intervals from 0.5–2.5 days",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=210, facecolor="white")
    plt.close(fig)


def verify_affine_calculation() -> None:
    source = np.array([[0.0, 0.0], [4000.0, 0.0], [0.0, 4000.0]])
    expected = np.array([[0.02, 0.01], [-0.005, -0.01]])
    target = source @ (np.eye(2) + expected).T + np.array([1000.0, -500.0])
    source_edges = np.stack((source[1] - source[0], source[2] - source[0]), axis=1)
    target_edges = np.stack((target[1] - target[0], target[2] - target[0]), axis=1)
    recovered = target_edges @ np.linalg.inv(source_edges) - np.eye(2)
    if not np.allclose(recovered, expected, atol=1.0e-12):
        raise AssertionError("Synthetic affine deformation check failed")


def make_deformation(
    output_dir: Path,
    images: pd.DataFrame,
    xy: np.ndarray,
    active: np.ndarray,
    minimum_days: float,
    maximum_days: float,
    spacing_m: float,
    maximum_edge_m: float,
) -> tuple[Path, Path, dict]:
    verify_affine_calculation()
    pair_stats = []
    samples = []
    pairs = valid_time_pairs(images.image_time, minimum_days, maximum_days)
    for pair_id, (source_index, target_index, elapsed_days) in enumerate(pairs):
        pair_samples, stats = pair_triangle_gradients(
            xy,
            active,
            source_index,
            target_index,
            elapsed_days,
            maximum_edge_m,
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
    grid, grid_metadata = area_weighted_grid(all_samples, spacing_m)
    report = {
        "minimum_days": minimum_days,
        "maximum_days": maximum_days,
        "valid_interval_count": len(pairs),
        "maximum_triangle_edge_m": maximum_edge_m,
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
    }
    report.update(grid_metadata)
    csv_path = output_dir / "efficientloftr_deformation_0p5_2p5day_grid.csv"
    png_path = output_dir / "efficientloftr_deformation_0p5_2p5day_composite.png"
    grid.to_csv(csv_path, index=False)
    plot_deformation(grid, report, png_path)
    return png_path, csv_path, report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = args.sequence_dir / "trajectories_4km.csv"
    trajectories = pd.read_csv(trajectories_path)
    image_ids = trajectories.sort_values("image_index").image_id.drop_duplicates().astype(int).tolist()
    images = image_metadata(args.observations, image_ids)
    _, xy, active, active_counts = trajectory_arrays(trajectories, image_ids)

    gif_path, mp4_path, overview_path, bounds = make_animation(
        args.output_dir, images, xy, active, active_counts
    )
    deformation_path, grid_path, deformation_report = make_deformation(
        args.output_dir,
        images,
        xy,
        active,
        args.minimum_days,
        args.maximum_days,
        args.grid_spacing_m,
        args.maximum_triangle_edge_m,
    )
    manifest = {
        "sequence_directory": str(args.sequence_dir),
        "trajectory_source": str(trajectories_path),
        "trajectory_source_sha256": sha256(trajectories_path),
        "trajectory_policy": "strict observed trajectories; no predicted gap continuation",
        "analysis_crs": "EPSG:3413",
        "coordinate_units": "metres",
        "image_ids": image_ids,
        "image_times_utc": [timestamp.isoformat() for timestamp in images.image_time],
        "seeded_trajectories": int(active_counts[0]),
        "active_by_image": active_counts.astype(int).tolist(),
        "complete_trajectories": int(active_counts[-1]),
        "complete_fraction": float(active_counts[-1] / active_counts[0]),
        "map_bounds_m": list(bounds),
        "deformation": deformation_report,
        "outputs": {
            "trajectory_gif": str(gif_path),
            "trajectory_mp4": str(mp4_path),
            "trajectory_final_frame": str(overview_path),
            "deformation_png": str(deformation_path),
            "deformation_grid_csv": str(grid_path),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
