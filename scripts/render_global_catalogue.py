#!/usr/bin/env python3
"""Render field-replay diagnostics and a pan-Arctic trajectory trail animation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from limosat.store import file_sha256


DEFAULT_ICE = Path(
    "/Users/seachu/data/shared/NSIDC/SIE/"
    "extent_N_202004_polygon_v4.0.shp"
)
EXTENT = (-4_000_000.0, 4_000_000.0, -4_000_000.0, 4_000_000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sea-ice-shapefile", type=Path, default=DEFAULT_ICE)
    parser.add_argument("--maximum-trajectories", type=int, default=30_000)
    parser.add_argument("--spatial-bin-km", type=float, default=250.0)
    parser.add_argument("--frame-hours", type=float, default=6.0)
    parser.add_argument("--trail-hours", type=float, default=48.0)
    parser.add_argument("--fps", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    database = args.database.resolve()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "field_coverage": output / "field-coverage.png",
        "trajectory_lifetime": output / "trajectory-lifetime.png",
        "observation_count": output / "observation-count.png",
        "mp4": output / "pan-arctic-global-trajectories.mp4",
        "gif": output / "pan-arctic-global-trajectories.gif",
        "report": output / "render-report-v1.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"render outputs already exist: {existing}")
    if not database.is_file():
        raise FileNotFoundError(database)
    if not args.sea_ice_shapefile.is_file():
        raise FileNotFoundError(args.sea_ice_shapefile)

    frame_times = _frame_times(database, args.frame_hours)
    global_support, frame_support, field_counts, node_counts = _field_support(
        source, frame_times
    )
    sea_ice = gpd.read_file(args.sea_ice_shapefile).to_crs(3413)
    _plot_field_coverage(global_support, sea_ice, paths["field_coverage"])
    observations, lifetimes = _trajectory_distributions(database)
    _plot_lifetimes(lifetimes, paths["trajectory_lifetime"])
    _plot_observations(observations, paths["observation_count"])

    selected = _balanced_subset(
        database,
        args.maximum_trajectories,
        args.spatial_bin_km * 1_000.0,
    )
    segments, segment_times = _load_segments(database, selected)
    ffmpeg_executable = _render_animation(
        segments,
        segment_times,
        frame_times,
        frame_support,
        field_counts,
        node_counts,
        sea_ice,
        paths["mp4"],
        paths["gif"],
        trail_hours=args.trail_hours,
        fps=args.fps,
    )
    provenance_path = database.parent / "field-replay-provenance-v1.json"
    provenance = (
        json.loads(provenance_path.read_text())
        if provenance_path.is_file()
        else None
    )
    outputs = {
        key: {
            "path": str(path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for key, path in paths.items()
        if key != "report"
    }
    report = {
        "schema_version": "limosat_global_render_report_v1",
        "label": "FIELD REPLAY — no local EfficientLoFTR inference",
        "style_reference": (
            "IICWG-DA pan-Arctic presentation; thin measured trajectory "
            "segments replace the diagnostic point-marker style"
        ),
        "database": {
            "path": str(database),
            "sha256": file_sha256(database),
        },
        "source_field_set_sha256": (
            provenance["source"]["field_set_sha256"] if provenance else None
        ),
        "selection": {
            "method": (
                "deterministic SHA256 ordering, round-robin across EPSG:3413 "
                f"{args.spatial_bin_km:g} km seed bins"
            ),
            "eligible_minimum_observations": 2,
            "maximum_trajectories": args.maximum_trajectories,
            "selected_trajectories": len(selected),
            "singleton_trajectories": 0,
        },
        "animation": {
            "frame_count": len(frame_times),
            "frame_interval_hours": args.frame_hours,
            "trail_window_hours": args.trail_hours,
            "fps": args.fps,
            "duration_seconds": len(frame_times) / args.fps,
            "first_frame_utc": frame_times[0].isoformat(),
            "last_frame_utc": frame_times[-1].isoformat(),
            "trajectory_rendering": "thin source-to-target measured segments",
            "dormant_interval_policy": "no segment across a dormant interval",
            "context_panel": "field-support density synchronized by target time",
            "ffmpeg_executable": ffmpeg_executable,
        },
        "outputs": outputs,
    }
    paths["report"].write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"report": str(paths["report"]), "outputs": outputs}))
    return 0


def _frame_times(database: Path, interval_hours: float) -> list[datetime]:
    with sqlite3.connect(database) as connection:
        first, last = connection.execute(
            "SELECT MIN(time_utc),MAX(time_utc) FROM images"
        ).fetchone()
    start = _utc(first)
    end = _utc(last)
    step = timedelta(hours=interval_hours)
    frames = []
    current = start
    while current <= end:
        frames.append(current)
        current += step
    if frames[-1] < end:
        frames.append(end)
    return frames


def _field_support(source: Path, frame_times: list[datetime]):
    bins = 180
    global_hist = np.zeros((bins, bins), dtype=np.float64)
    frame_hist = np.zeros((len(frame_times), bins, bins), dtype=np.float32)
    field_counts = np.zeros(len(frame_times), dtype=np.int32)
    node_counts = np.zeros(len(frame_times), dtype=np.int64)
    with sqlite3.connect(source / "control" / "state.sqlite") as connection:
        rows = connection.execute(
            """
            SELECT a.product_relative_path,s.time_utc
            FROM edges e
            JOIN edge_attempts a ON e.edge_id=a.edge_id
            JOIN scenes s ON e.target_scene_id=s.scene_id
            WHERE e.role='primary' AND a.status='completed'
            ORDER BY e.edge_id
            """
        ).fetchall()
    start = frame_times[0]
    step = (frame_times[1] - start).total_seconds() if len(frame_times) > 1 else 1
    xmin, xmax, ymin, ymax = EXTENT
    for index, (relative, target_time) in enumerate(rows, start=1):
        frame = pd.read_csv(
            source / relative,
            usecols=["source_x", "source_y", "available"],
        )
        available = frame["available"].astype(str).str.lower().eq("true")
        x = frame.loc[available, "source_x"].to_numpy()
        y = frame.loc[available, "source_y"].to_numpy()
        hist, _, _ = np.histogram2d(
            y, x, bins=bins, range=((ymin, ymax), (xmin, xmax))
        )
        global_hist += hist
        frame_index = int(
            np.clip(
                ((_utc(target_time) - start).total_seconds()) // step,
                0,
                len(frame_times) - 1,
            )
        )
        frame_hist[frame_index] += hist.astype(np.float32)
        field_counts[frame_index] += 1
        node_counts[frame_index] += len(x)
        if index % 100 == 0:
            print(f"summarized field support {index}/{len(rows)}")
    return global_hist, frame_hist, field_counts, node_counts


def _trajectory_distributions(database: Path):
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT observation_count,
                   (julianday(last_observed_utc)-
                    julianday(first_observed_utc))*24.0
            FROM trajectory_statistics
            """
        ).fetchall()
    values = np.asarray(rows, dtype=np.float64)
    return values[:, 0], values[:, 1]


def _base_map(ax, sea_ice, title: str) -> None:
    xmin, xmax, ymin, ymax = EXTENT
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal", title=title)
    ax.set_facecolor("#f4f7f9")
    sea_ice.plot(
        ax=ax,
        facecolor="#dbeaf2",
        edgecolor="#91aab8",
        linewidth=0.25,
        alpha=0.6,
        zorder=0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#b8c2c8")


def _plot_field_coverage(hist, sea_ice, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    _base_map(ax, sea_ice, "Completed primary pair field support")
    masked = np.ma.masked_less_equal(hist, 0)
    image = ax.imshow(
        masked,
        origin="lower",
        extent=EXTENT,
        cmap="magma",
        norm=LogNorm(vmin=1, vmax=max(2, float(hist.max()))),
        alpha=0.82,
        zorder=1,
    )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.72)
    colorbar.set_label("Available field-node occurrences")
    ax.text(
        0.02,
        0.02,
        "April 1–7, 2020 · FIELD REPLAY",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_lifetimes(lifetimes, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.hist(lifetimes / 24.0, bins=np.arange(0, 7.25, 0.25), color="#2d718e")
    ax.set(
        xlabel="Measured trajectory lifetime (days)",
        ylabel="Trajectories",
        title="Global trajectory lifetime distribution",
        yscale="log",
    )
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_observations(observations, path: Path) -> None:
    maximum = max(2, int(observations.max()))
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.hist(
        observations,
        bins=np.arange(0.5, maximum + 1.5),
        color="#d65f36",
    )
    ax.set(
        xlabel="Measured observations per trajectory",
        ylabel="Trajectories",
        title="Global observation-count distribution",
        yscale="log",
    )
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _balanced_subset(database: Path, maximum: int, bin_size_m: float) -> list[str]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT trajectory_id,seed_x_m,seed_y_m
            FROM trajectory_statistics
            WHERE observation_count>=2
              AND seed_x_m IS NOT NULL AND seed_y_m IS NOT NULL
            """
        ).fetchall()
    buckets = defaultdict(list)
    for identity, x_m, y_m in rows:
        cell = (int(np.floor(x_m / bin_size_m)), int(np.floor(y_m / bin_size_m)))
        buckets[cell].append(identity)
    for values in buckets.values():
        values.sort(key=lambda identity: hashlib.sha256(identity.encode()).digest())
    selected = []
    cells = sorted(buckets)
    rank = 0
    while len(selected) < maximum:
        added = False
        for cell in cells:
            if rank < len(buckets[cell]):
                selected.append(buckets[cell][rank])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        rank += 1
    return selected


def _load_segments(database: Path, selected: list[str]):
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TEMP TABLE selected(trajectory_id TEXT PRIMARY KEY)"
        )
        connection.executemany(
            "INSERT INTO selected VALUES (?)",
            [(identity,) for identity in selected],
        )
        rows = connection.execute(
            """
            SELECT target.time_utc,source.x_m,source.y_m,target.x_m,target.y_m
            FROM trajectory_points target
            JOIN selected USING(trajectory_id)
            JOIN pairs ON target.run_id=pairs.run_id
                      AND target.source_pair_id=pairs.pair_id
            JOIN trajectory_points source
              ON source.run_id=target.run_id
             AND source.trajectory_id=target.trajectory_id
             AND source.image_id=pairs.source_image_id
            WHERE target.state='observed'
              AND source.x_m IS NOT NULL AND target.x_m IS NOT NULL
            ORDER BY target.time_utc,target.trajectory_id
            """
        ).fetchall()
    segments = np.asarray(
        [[[row[1], row[2]], [row[3], row[4]]] for row in rows],
        dtype=np.float64,
    )
    times = np.asarray([_utc(row[0]).timestamp() for row in rows])
    return segments, times


def _render_animation(
    segments,
    segment_times,
    frame_times,
    frame_support,
    field_counts,
    node_counts,
    sea_ice,
    mp4,
    gif,
    *,
    trail_hours,
    fps,
) -> str:
    environment_ffmpeg = Path(sys.executable).with_name("ffmpeg")
    ffmpeg_executable = (
        str(environment_ffmpeg)
        if environment_ffmpeg.is_file()
        else (shutil.which("ffmpeg") or "ffmpeg")
    )
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_executable
    fig, (trail_ax, support_ax) = plt.subplots(
        1,
        2,
        figsize=(12.8, 6.8),
        gridspec_kw={"width_ratios": (1.55, 1.0)},
        constrained_layout=True,
    )
    _base_map(trail_ax, sea_ice, "Measured Lagrangian trajectory trails")
    _base_map(support_ax, sea_ice, "Primary pair field support")
    trails = LineCollection([], linewidths=0.45, zorder=3)
    trail_ax.add_collection(trails)
    support = support_ax.imshow(
        np.zeros_like(frame_support[0]),
        origin="lower",
        extent=EXTENT,
        cmap="viridis",
        vmin=0,
        vmax=max(1.0, float(np.log1p(frame_support).max())),
        alpha=0.82,
        zorder=2,
    )
    timestamp = trail_ax.text(
        0.02, 0.02, "", transform=trail_ax.transAxes, fontsize=10, weight="bold"
    )
    context = support_ax.text(
        0.02, 0.02, "", transform=support_ax.transAxes, fontsize=9
    )
    fig.suptitle(
        "LiMOSAT EfficientLoFTR · April 2020 pan-Arctic FIELD REPLAY",
        fontsize=14,
        weight="bold",
    )
    window_seconds = trail_hours * 3_600.0
    base_color = np.array([0.91, 0.28, 0.12, 1.0])

    def update(index):
        current = frame_times[index]
        seconds = current.timestamp()
        active = (
            (segment_times <= seconds)
            & (segment_times > seconds - window_seconds)
        )
        current_segments = segments[active]
        trails.set_segments(current_segments)
        ages = (seconds - segment_times[active]) / max(window_seconds, 1)
        colors = np.tile(base_color, (len(current_segments), 1))
        colors[:, 3] = 0.12 + 0.72 * (1.0 - ages)
        trails.set_color(colors)
        support.set_data(np.log1p(frame_support[index]))
        timestamp.set_text(current.strftime("%Y-%m-%d %H:%M UTC"))
        context.set_text(
            f"{int(field_counts[index])} fields\n"
            f"{int(node_counts[index]):,} available nodes"
        )
        return trails, support, timestamp, context

    movie = animation.FuncAnimation(
        fig, update, frames=len(frame_times), interval=1_000 / fps, blit=False
    )
    movie.save(mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2_800), dpi=120)
    movie.save(gif, writer=animation.PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return ffmpeg_executable


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())
