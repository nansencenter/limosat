#!/usr/bin/env python3
"""Compose a global SQLite catalogue from immutable completed pair fields."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from limosat import (
    FieldConfig,
    RunConfig,
    RunStore,
    TrajectoryConfig,
    iter_global_trajectory_points,
    load_production_field_replay,
)
from limosat.replay import replay_field_set_sha256
from limosat.store import file_sha256


DEFAULT_SOURCE = Path(
    "/Volumes/KINGSTON/arktalas-nrt/method-neutral-benchmark/"
    "efficientloftr-production/april2020-week01-primary-v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="april2020-week01-global-field-replay-v1")
    parser.add_argument(
        "--skip-checksum-verification",
        action="store_true",
        help="Trust recorded field checksums instead of hashing each CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    database = output / "global-trajectories.sqlite"
    report_path = output / "field-replay-provenance-v1.json"
    if database.exists() or report_path.exists():
        raise FileExistsError(
            f"replay output already exists; choose a new --output path: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    replay = load_production_field_replay(
        args.source,
        verify_checksums=not args.skip_checksum_verification,
    )
    config = RunConfig(
        run_id=args.run_id,
        catalogue=str(replay.root / "control" / "state.sqlite"),
        database=str(database),
        output_directory=str(output),
    )
    store = RunStore(config)
    store.start_run()
    _register_source_inventory(store, replay)
    state_counts = Counter()
    point_count = 0

    def counted_batches():
        nonlocal point_count
        batches = iter_global_trajectory_points(
            replay.edges,
            replay.images,
            FieldConfig(),
            TrajectoryConfig(),
        )
        for image_index, batch in enumerate(batches, start=1):
            state_counts.update(point.state for point in batch)
            point_count += len(batch)
            if image_index % 50 == 0:
                print(
                    f"composed {image_index}/{len(replay.images)} images; "
                    f"{point_count:,} trajectory rows"
                )
            yield batch

    store.replace_global_trajectory_batches(counted_batches())
    _build_analysis_tables(database)
    runtime_seconds = time.perf_counter() - clock
    completed = datetime.now(timezone.utc)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE runs SET status='complete',completed_utc=?,runtime_seconds=?,
                            error=NULL
            WHERE run_id=?
            """,
            (completed.isoformat(), runtime_seconds, args.run_id),
        )
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    statistics = _statistics(database)
    if statistics["state_counts"] != {
        key: int(value) for key, value in state_counts.items()
    }:
        raise AssertionError("streamed and persisted trajectory counts differ")
    prior = _prior_sharded_statistics(replay.root)
    report = {
        "schema_version": "limosat_global_field_replay_v1",
        "label": "FIELD REPLAY — no local EfficientLoFTR inference",
        "run_id": args.run_id,
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "runtime_seconds": runtime_seconds,
        "coordinates": {
            "crs": "EPSG:3413",
            "dtype": "float64",
            "distance_unit": "metre",
            "time": "timezone-aware UTC",
        },
        "product_schemas": {
            "sqlite": 4,
            "lagrangian_trajectory": 4,
            "field_replay_provenance": 1,
        },
        "source": {
            "root": str(replay.root),
            **replay.source_metadata,
            "completed_primary_fields": len(replay.fields),
            "field_set_sha256": replay_field_set_sha256(replay.fields),
            "field_rows": int(sum(field.node_count for field in replay.fields)),
            "available_field_rows": int(
                sum(field.available_node_count for field in replay.fields)
            ),
        },
        "composition": {
            "component_id_policy": "compute-planning label only",
            "seed_exclusion_radius_m": 2_000.0,
            "incoming_selection": (
                "most recent measured source time; selected matches, residual, "
                "support radius, and pair identity break equal-time ties"
            ),
            "dormant_coordinates": "SQL NULL",
            "recovery_fields_present": 0,
            "deformation_replayed": False,
        },
        "global_statistics": statistics,
        "prior_sharded_statistics": prior,
        "global_vs_prior": {
            "trajectory_count_difference": (
                statistics["trajectory_count"] - prior["trajectory_count"]
            ),
            "trajectory_row_count_difference": (
                statistics["trajectory_point_count"] - prior["total_rows"]
            ),
        },
        "outputs": {
            "sqlite": {
                "path": str(database),
                "sha256": file_sha256(database),
                "size_bytes": database.stat().st_size,
            }
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"database": str(database), "report": str(report_path)}))
    return 0


def _register_source_inventory(store, replay) -> None:
    run_id = store.config.run_id
    image_by_id = {image.image_id: image for image in replay.images}
    inventory = {row["scene_id"]: row for row in replay.scene_inventory}
    image_index = {
        image.image_id: index for index, image in enumerate(replay.images)
    }
    with sqlite3.connect(store.path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executemany(
            """
            INSERT INTO images
            (run_id,image_id,component_id,platform,absolute_orbit,path,time_utc,
             size_bytes,sha256)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    run_id,
                    image.image_id,
                    image.component_id,
                    image.platform,
                    image.absolute_orbit,
                    inventory[image.image_id]["relative_path"],
                    image.time_utc.isoformat(),
                    inventory[image.image_id]["size_bytes"],
                    inventory[image.image_id]["sha256"],
                )
                for image in replay.images
            ],
        )
        for ordinal, (edge, source) in enumerate(
            zip(replay.edges, replay.fields, strict=True)
        ):
            field = edge.field
            source_image = image_by_id[field.source_image_id]
            target_image = image_by_id[field.target_image_id]
            elapsed = (
                target_image.time_utc - source_image.time_utc
            ).total_seconds()
            connection.execute(
                """
                INSERT INTO candidate_pairs
                (run_id,pair_id,ordinal,selection,planning_component_id,
                 source_component_id,target_component_id,source_image_id,
                 target_image_id,source_time_utc,target_time_utc,
                 elapsed_seconds,overlap_fraction,overlap_area_m2,skipped_images)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    field.pair_id,
                    ordinal,
                    "primary",
                    source.shard_id,
                    source_image.component_id,
                    target_image.component_id,
                    field.source_image_id,
                    field.target_image_id,
                    field.source_time_utc.isoformat(),
                    field.target_time_utc.isoformat(),
                    elapsed,
                    source.overlap_fraction,
                    None,
                    image_index[field.target_image_id]
                    - image_index[field.source_image_id]
                    - 1,
                ),
            )
            connection.execute(
                """
                INSERT INTO pairs
                (run_id,pair_id,component_id,source_image_id,target_image_id,
                 source_time_utc,target_time_utc,elapsed_seconds,kind,targeted,
                 status,started_utc,completed_utc,field_sha256,node_count,
                 available_node_count)
                VALUES (?,?,?,?,?,?,?,?,?,'0','complete',?,?,?,?,?)
                """,
                (
                    run_id,
                    field.pair_id,
                    source.shard_id,
                    field.source_image_id,
                    field.target_image_id,
                    field.source_time_utc.isoformat(),
                    field.target_time_utc.isoformat(),
                    elapsed,
                    "primary",
                    source.started_utc,
                    source.completed_utc,
                    source.product_sha256,
                    source.node_count,
                    source.available_node_count,
                ),
            )


def _build_analysis_tables(database: Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE INDEX trajectory_points_run_image
              ON trajectory_points(run_id,image_id);
            CREATE INDEX trajectory_points_run_pair
              ON trajectory_points(run_id,source_pair_id);
            CREATE TABLE trajectory_statistics AS
            SELECT run_id,trajectory_id,
                   SUM(CASE WHEN x_m IS NOT NULL THEN 1 ELSE 0 END)
                     AS observation_count,
                   MIN(CASE WHEN x_m IS NOT NULL THEN time_utc END)
                     AS first_observed_utc,
                   MAX(CASE WHEN x_m IS NOT NULL THEN time_utc END)
                     AS last_observed_utc,
                   MAX(CASE WHEN state='created' THEN x_m END) AS seed_x_m,
                   MAX(CASE WHEN state='created' THEN y_m END) AS seed_y_m
            FROM trajectory_points
            GROUP BY run_id,trajectory_id;
            CREATE UNIQUE INDEX trajectory_statistics_identity
              ON trajectory_statistics(run_id,trajectory_id);
            """
        )


def _statistics(database: Path) -> dict:
    with sqlite3.connect(database) as connection:
        trajectory_count = connection.execute(
            "SELECT COUNT(*) FROM trajectories"
        ).fetchone()[0]
        point_count = connection.execute(
            "SELECT COUNT(*) FROM trajectory_points"
        ).fetchone()[0]
        states = dict(
            connection.execute(
                "SELECT state,COUNT(*) FROM trajectory_points GROUP BY state"
            )
        )
        observations = np.asarray(
            [
                row[0]
                for row in connection.execute(
                    "SELECT observation_count FROM trajectory_statistics"
                )
            ],
            dtype=np.int32,
        )
        lifetimes = np.asarray(
            [
                row[0]
                for row in connection.execute(
                    """
                    SELECT (julianday(last_observed_utc)-
                            julianday(first_observed_utc))*24.0
                    FROM trajectory_statistics
                    """
                )
            ],
            dtype=np.float64,
        )
    return {
        "trajectory_count": int(trajectory_count),
        "trajectory_point_count": int(point_count),
        "state_counts": {key: int(value) for key, value in states.items()},
        "multi_observation_trajectories": int((observations >= 2).sum()),
        "observation_count": _distribution(observations),
        "lifetime_hours": _distribution(lifetimes),
    }


def _distribution(values: np.ndarray) -> dict:
    if not len(values):
        return {"minimum": None, "median": None, "mean": None, "maximum": None}
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
    }


def _prior_sharded_statistics(root: Path) -> dict:
    totals = Counter(
        trajectory_count=0,
        initial_trajectories=0,
        new_trajectories=0,
        dormant_rows=0,
        total_rows=0,
        observed_rows=0,
    )
    for path in root.glob("shards/*/raw/learned_output/run_manifest.json"):
        values = json.loads(path.read_text())["trajectories"][
            "adjacent_observed_graph_with_new_points"
        ]
        for key in (
            "trajectory_count",
            "initial_trajectories",
            "new_trajectories",
            "dormant_rows",
        ):
            totals[key] += int(values[key])
        rows = sum(int(value) for value in values["trajectory_count_by_image"])
        totals["total_rows"] += rows
        totals["observed_rows"] += rows - int(values["dormant_rows"])
    return dict(totals)


if __name__ == "__main__":
    raise SystemExit(main())
