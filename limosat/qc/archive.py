#!/usr/bin/env python3
"""Finalize a large LiMOSAT SQLite archive with streaming trajectory QC.

The scan reads descriptor-free rows once in time/image/rowid order, preserves
vectors across calendar boundaries, and checkpoints only at complete image
boundaries.  It stores every rejected/review vector, a deterministic 1/1000 vector
sample, pair/image summaries, and full marginal distributions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sqlite3
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import AuditWriter, DistributionAccumulator, VECTOR_COLUMNS, quote_identifier
from .core import (
    PROTOCOL_ID,
    PROTOCOL_PATH,
    QCConfig,
    load_protocol,
    score_vectors,
    validate_frozen_protocol,
)

REQUIRED_COLUMNS = {
    "image_id",
    "is_last",
    "trajectory_id",
    "geometry",
    "time",
    "corr",
    "interpolated",
}


def infer_limosat_table(connection: sqlite3.Connection, requested: str | None = None) -> str:
    if requested is not None:
        candidates = [requested]
    else:
        candidates = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            if not row[0].startswith("qc_") and not row[0].endswith("__qc")
        ]
    valid = []
    for name in candidates:
        columns = {
            row[1]
            for row in connection.execute(f"PRAGMA table_info({quote_identifier(name)})")
        }
        if REQUIRED_COLUMNS.issubset(columns):
            valid.append(name)
    if len(valid) != 1:
        raise ValueError(
            f"Expected exactly one LiMOSAT table, found {valid}; pass --table explicitly"
        )
    return valid[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_point_wkt(value: str) -> tuple[float, float]:
    if not value.startswith("POINT"):
        raise ValueError(f"Expected POINT WKT, got {value[:40]!r}")
    coordinates = value[value.find("(") + 1 : value.rfind(")")].strip().split()
    if len(coordinates) < 2:
        raise ValueError(f"Malformed point WKT: {value!r}")
    return float(coordinates[0]), float(coordinates[1])


def seconds_since_epoch(value: str, cache: dict[str, float]) -> float:
    if value in cache:
        return cache[value]
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        seconds = (parsed - datetime(1970, 1, 1)).total_seconds()
    else:
        seconds = (parsed.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()
    cache[value] = seconds
    return seconds


def make_vectors(records: list[tuple]) -> pd.DataFrame:
    vectors = pd.DataFrame.from_records(records, columns=VECTOR_COLUMNS)
    vectors["u_m"] = vectors["x1_m"] - vectors["x0_m"]
    vectors["v_m"] = vectors["y1_m"] - vectors["y0_m"]
    vectors["magnitude_m"] = np.hypot(vectors["u_m"], vectors["v_m"])
    vectors["speed_m_per_day"] = vectors["magnitude_m"] / vectors["elapsed_days"]
    vectors["image_gap"] = vectors["target_image_id"] - vectors["source_image_id"]
    return vectors


def save_checkpoint(path: Path, state: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> dict:
    with path.open("rb") as stream:
        return pickle.load(stream)


def verify_source(args) -> None:
    """Verify a closed SQLite file against the supplied source identity."""
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    wal = Path(str(args.input) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Source SQLite has an uncheckpointed WAL; close/checkpoint it first")
    if sha256(args.input) != args.input_sha256:
        raise ValueError("Source SQLite checksum does not match --input-sha256")


def build_compact_database(args) -> dict:
    verify_source(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.compact_database
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    uri = f"file:{args.input.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as source:
        source_table = infer_limosat_table(source, args.table)
    started = time.perf_counter()
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-400000")
        connection.execute("ATTACH DATABASE ? AS src", (uri,))
        connection.execute(
            "CREATE TABLE compact_points (source_rowid INTEGER PRIMARY KEY, image_id INTEGER NOT NULL, "
            "is_last INTEGER NOT NULL, trajectory_id INTEGER NOT NULL, geometry TEXT NOT NULL, "
            "time TEXT NOT NULL, corr REAL, interpolated INTEGER)"
        )
        connection.execute(
            "INSERT INTO compact_points "
            f"SELECT rowid, image_id, is_last, trajectory_id, geometry, time, corr, interpolated "
            f"FROM src.{quote_identifier(source_table)}"
        )
        connection.execute(
            "CREATE INDEX idx_compact_time_image_rowid ON compact_points "
            "(time, image_id, source_rowid)"
        )
        connection.execute(
            "CREATE INDEX idx_compact_trajectory ON compact_points (trajectory_id, source_rowid)"
        )
        row = connection.execute(
            "SELECT COUNT(*), MIN(time), MAX(time), MIN(image_id), MAX(image_id), "
            "MIN(trajectory_id), MAX(trajectory_id) FROM compact_points"
        ).fetchone()
        manifest = {
            "status": "complete",
            "source_database": str(args.input.resolve()),
            "source_table": source_table,
            "source_sha256": args.input_sha256,
            "compact_database": str(target.resolve()),
            "rows": int(row[0]),
            "minimum_time": row[1],
            "maximum_time": row[2],
            "minimum_image_id": int(row[3]),
            "maximum_image_id": int(row[4]),
            "minimum_trajectory_id": int(row[5]),
            "maximum_trajectory_id": int(row[6]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        connection.execute(
            "CREATE TABLE compact_manifest (manifest_json TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO compact_manifest VALUES (?)",
            (json.dumps(manifest, sort_keys=True),),
        )
        connection.commit()
        connection.close()
        verify_source(args)
        os.replace(temporary, target)
    except Exception:
        connection.close()
        raise
    (args.output_dir / "compact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def scan_archive(args, config: QCConfig) -> dict:
    implementation_hashes = {
        "streaming_implementation_sha256": sha256(Path(__file__)),
        "scoring_implementation_sha256": sha256(
            Path(__file__).with_name("core.py")
        ),
        "audit_implementation_sha256": sha256(Path(__file__).with_name("audit.py")),
        "protocol_id": PROTOCOL_ID,
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sidecar = args.output_dir / "qc_analysis.sqlite"
    checkpoint_path = args.output_dir / "scan_checkpoint.pkl"
    progress_path = args.output_dir / "scan_progress.json"
    resume = bool(args.resume and checkpoint_path.exists())
    if args.resume and not resume:
        raise FileNotFoundError(checkpoint_path)
    if not args.compact_database.exists():
        raise FileNotFoundError(args.compact_database)
    uri = f"file:{args.compact_database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as compact:
        compact_manifest = json.loads(compact.execute(
            "SELECT manifest_json FROM compact_manifest"
        ).fetchone()[0])
    if compact_manifest.get("source_sha256") != args.input_sha256:
        raise ValueError("Compact database source checksum does not match --input-sha256")
    if args.table is not None and compact_manifest.get("source_table") != args.table:
        raise ValueError("Compact database source table does not match --table")
    run_identity = {
        "input_sha256": args.input_sha256,
        "compact_manifest": compact_manifest,
        "config": asdict(config),
        "prescreen_residual_m": args.prescreen_residual_m,
        **implementation_hashes,
    }
    state = None
    if resume:
        state = load_checkpoint(checkpoint_path)
        if state.get("run_identity") != run_identity:
            raise ValueError("Checkpoint source, protocol, code, or configuration changed")
    writer = AuditWriter(sidecar, resume=resume)
    if resume:
        active = state["active"]
        distributions = state["distributions"]
        stats = state["stats"]
        last_rowid = int(state["last_rowid"])
        last_image_id = int(state["last_image_id"])
        last_time_text = str(state["last_time_text"])
        writer.truncate_after(last_time_text, last_image_id)
    else:
        active = {}
        distributions = DistributionAccumulator()
        stats = {
            "point_rows": 0,
            "links": 0,
            "images": 0,
            "trajectories": 0,
            "rejected": 0,
            "review": 0,
            "hard_speed": 0,
            "topology_incident": 0,
            "local_evaluated": 0,
            "local_supported": 0,
            "duplicate_points": 0,
        }
        last_rowid = 0
        last_image_id = -1
        last_time_text = ""
    stats.setdefault("duplicate_points", 0)

    started = time.perf_counter()
    time_cache: dict[str, float] = {}
    vector_records: list[tuple] = []
    current_image_id = None
    current_time_text = None
    current_image_rows = 0
    current_image_seen: dict[int, tuple] = {}
    previous_image_id = last_image_id
    previous_time_text = last_time_text
    rows_since_start = 0

    def progress(status: str):
        elapsed = max(time.perf_counter() - started, 1e-9)
        payload = {
            "status": status,
            **stats,
            "last_rowid": last_rowid,
            "last_image_id": previous_image_id,
            "last_time": previous_time_text,
            "active_trajectories": len(active),
            "elapsed_seconds_this_invocation": elapsed,
            "rows_per_second_this_invocation": rows_since_start / elapsed,
            "config": asdict(config),
            "prescreen_residual_m": args.prescreen_residual_m,
        }
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, progress_path)
        return payload

    def finish_image():
        nonlocal vector_records, previous_image_id, previous_time_text, current_image_seen
        if current_image_id is None:
            return
        if vector_records:
            scored = score_vectors(
                make_vectors(vector_records),
                config,
                prescreen_residual_m=args.prescreen_residual_m,
            )
            distributions.update(scored)
            writer.add_scored(scored, current_image_rows)
            stats["links"] += len(scored)
            stats["rejected"] += int(scored["reject"].sum())
            stats["review"] += int(scored["review"].sum())
            stats["hard_speed"] += int(scored["hard_speed_reject"].sum())
            stats["topology_incident"] += int(scored["topology_flip_incident"].sum())
            stats["local_evaluated"] += int(scored["local_evaluated"].sum())
            stats["local_supported"] += int(scored["local_supported"].sum())
        else:
            writer.add_empty_image(current_image_id, current_time_text, current_image_rows)
        stats["images"] += 1
        previous_image_id = int(current_image_id)
        previous_time_text = str(current_time_text)
        vector_records = []
        current_image_seen = {}

    try:
        with sqlite3.connect(uri, uri=True) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA cache_size=-200000")
            selected_table = compact_manifest["source_table"]
            maximum_original_id = compact_manifest["maximum_trajectory_id"]
            if resume:
                query = (
                    "SELECT source_rowid, image_id, is_last, trajectory_id, geometry, time, corr, interpolated "
                    "FROM compact_points WHERE time > ? OR "
                    "(time = ? AND image_id > ?) OR "
                    "(time = ? AND image_id = ? AND source_rowid > ?) "
                    "ORDER BY time, image_id, source_rowid"
                )
                parameters = (
                    last_time_text,
                    last_time_text,
                    last_image_id,
                    last_time_text,
                    last_image_id,
                    last_rowid,
                )
            else:
                query = (
                    "SELECT source_rowid, image_id, is_last, trajectory_id, geometry, time, corr, interpolated "
                    "FROM compact_points ORDER BY time, image_id, source_rowid"
                )
                parameters = ()
            cursor = connection.execute(query, parameters)
            stop_requested = False
            while not stop_requested:
                rows = cursor.fetchmany(args.fetch_rows)
                if not rows:
                    break
                for row in rows:
                    rowid, image_id, is_last, trajectory_id, geometry, time_text, corr, interpolated = row
                    rowid = int(rowid)
                    image_id = int(image_id)
                    trajectory_id = int(trajectory_id)
                    if current_image_id is not None and (
                        image_id != current_image_id or str(time_text) != current_time_text
                    ):
                        finish_image()
                        if stats["images"] % args.flush_images == 0:
                            writer.flush()
                        if stats["images"] % args.progress_images == 0:
                            payload = progress("running")
                            print(json.dumps(payload, sort_keys=True), flush=True)
                        if stats["images"] % args.checkpoint_images == 0:
                            writer.flush()
                            save_checkpoint(
                                checkpoint_path,
                                {
                                    "run_identity": run_identity,
                                    "active": active,
                                    "distributions": distributions,
                                    "stats": stats,
                                    "last_rowid": last_rowid,
                                    "last_image_id": previous_image_id,
                                    "last_time_text": previous_time_text,
                                },
                            )
                        if args.maximum_rows and rows_since_start >= args.maximum_rows:
                            stop_requested = True
                            break
                        current_image_id = image_id
                        current_time_text = str(time_text)
                        current_image_rows = 0
                    elif current_image_id is None:
                        current_image_id = image_id
                        current_time_text = str(time_text)

                    x_m, y_m = parse_point_wkt(str(geometry))
                    time_text = str(time_text)
                    time_seconds = seconds_since_epoch(time_text, time_cache)
                    retained = current_image_seen.get(trajectory_id)
                    if retained is not None:
                        (
                            retained_rowid,
                            retained_geometry,
                            retained_x,
                            retained_y,
                            retained_corr,
                            retained_interpolated,
                            retained_is_last,
                        ) = retained
                        corr_equal = (
                            corr is None and retained_corr is None
                        ) or (
                            corr is not None
                            and retained_corr is not None
                            and float(corr) == float(retained_corr)
                        )
                        duplicate_interpolated = 0 if interpolated is None else int(interpolated)
                        separation_m = float(np.hypot(x_m - retained_x, y_m - retained_y))
                        exact = (
                            str(geometry) == retained_geometry
                            and corr_equal
                            and duplicate_interpolated == retained_interpolated
                            and int(is_last) == retained_is_last
                        )
                        writer.add_duplicate(
                            (
                                rowid,
                                retained_rowid,
                                trajectory_id,
                                image_id,
                                time_text,
                                separation_m,
                                None if retained_corr is None else float(retained_corr),
                                None if corr is None else float(corr),
                                retained_interpolated,
                                duplicate_interpolated,
                                retained_is_last,
                                int(is_last),
                                int(exact),
                            )
                        )
                        stats["duplicate_points"] += 1
                        if int(is_last):
                            active.pop(trajectory_id, None)
                        stats["point_rows"] += 1
                        rows_since_start += 1
                        current_image_rows += 1
                        last_rowid = rowid
                        continue
                    current_image_seen[trajectory_id] = (
                        rowid,
                        str(geometry),
                        x_m,
                        y_m,
                        corr,
                        0 if interpolated is None else int(interpolated),
                        int(is_last),
                    )
                    previous = active.get(trajectory_id)
                    if previous is None:
                        stats["trajectories"] += 1
                    else:
                        (
                            source_rowid,
                            source_image_id,
                            source_time_text,
                            source_time_seconds,
                            source_x,
                            source_y,
                        ) = previous
                        if source_image_id == image_id:
                            raise ValueError(
                                f"trajectory {trajectory_id} repeats image {image_id}"
                            )
                        elapsed_days = (time_seconds - source_time_seconds) / 86_400.0
                        if elapsed_days <= 0:
                            raise ValueError(
                                f"non-positive elapsed time for trajectory {trajectory_id} at rowid {rowid}"
                            )
                        vector_records.append(
                            (
                                trajectory_id,
                                source_rowid,
                                rowid,
                                source_image_id,
                                image_id,
                                source_time_text,
                                time_text,
                                elapsed_days,
                                source_x,
                                source_y,
                                x_m,
                                y_m,
                                np.nan if corr is None else float(corr),
                                0 if interpolated is None else int(interpolated),
                            )
                        )
                    if int(is_last):
                        active.pop(trajectory_id, None)
                    else:
                        active[trajectory_id] = (
                            rowid,
                            image_id,
                            time_text,
                            time_seconds,
                            x_m,
                            y_m,
                        )
                    stats["point_rows"] += 1
                    rows_since_start += 1
                    current_image_rows += 1
                    last_rowid = rowid
            if not stop_requested:
                finish_image()
            writer.flush()
            final_status = "partial" if stop_requested else "complete"
            save_checkpoint(
                checkpoint_path,
                {
                    "run_identity": run_identity,
                    "active": active,
                    "distributions": distributions,
                    "stats": stats,
                    "last_rowid": last_rowid,
                    "last_image_id": previous_image_id,
                    "last_time_text": previous_time_text,
                },
            )
            metadata = {
                "status": final_status,
                "input_database": str(args.input.resolve()),
                "input_sha256": args.input_sha256 or sha256(args.input),
                **implementation_hashes,
                "source_table": selected_table,
                "maximum_original_trajectory_id": maximum_original_id,
                "coordinate_reference_system": "EPSG:3413",
                "coordinate_units": "metres",
                "config": asdict(config),
                "prescreen_residual_m": args.prescreen_residual_m,
                "sample_rule": "target_rowid modulo 1000 equals zero",
                "stats": stats,
                "active_trajectories_at_end": len(active),
            }
            if final_status == "complete":
                writer.finish(distributions, metadata)
            payload = progress(final_status)
            print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
            return metadata
    finally:
        writer.close()


def source_schema(connection: sqlite3.Connection, source_table: str):
    rows = list(
        connection.execute(f"PRAGMA src.table_info({quote_identifier(source_table)})")
    )
    declarations = []
    for _, name, declared_type, not_null, default, primary_key in rows:
        declaration = f"{quote_identifier(name)} {declared_type or ''}".rstrip()
        if not_null:
            declaration += " NOT NULL"
        if default is not None:
            declaration += f" DEFAULT {default}"
        if primary_key:
            declaration += " PRIMARY KEY"
        declarations.append(declaration)
    return [row[1] for row in rows], declarations


def materialize(args) -> dict:
    verify_source(args)
    sidecar = args.output_dir / "qc_analysis.sqlite"
    if not sidecar.exists():
        raise FileNotFoundError(sidecar)
    with sqlite3.connect(sidecar) as audit:
        metadata = {
            key: json.loads(value)
            for key, value in audit.execute("SELECT key, value_json FROM qc_metadata")
        }
    if metadata.get("status") != "complete":
        raise ValueError("Scan must be complete before materialization")
    if metadata.get("input_sha256") != args.input_sha256:
        raise ValueError("QC scan source checksum does not match --input-sha256")
    source_table = metadata["source_table"]
    cleaned_table = source_table + "__qc"
    target = args.cleaned_output
    if target is None:
        target = args.output_dir / f"{args.input.stem}_qc.sqlite"
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-400000")
        source_uri = f"file:{args.input.resolve()}?mode=ro"
        connection.execute("ATTACH DATABASE ? AS src", (source_uri,))
        connection.execute("ATTACH DATABASE ? AS audit", (str(sidecar.resolve()),))
        columns, declarations = source_schema(connection, source_table)
        selected = ",".join(quote_identifier(column) for column in columns)
        connection.execute(
            f"CREATE TABLE {quote_identifier(cleaned_table)} ({', '.join(declarations)})"
        )
        connection.execute(
            f"INSERT INTO {quote_identifier(cleaned_table)} (rowid,{selected}) "
            f"SELECT rowid,{selected} FROM src.{quote_identifier(source_table)} ORDER BY rowid"
        )
        maximum_id = int(metadata["maximum_original_trajectory_id"])
        connection.execute(
            "CREATE TABLE qc_break_assignment AS "
            "WITH ranked AS ("
            " SELECT trajectory_id AS original_trajectory_id, source_rowid, target_rowid, "
            " source_image_id, target_image_id, source_time, target_time, decision_reason, "
            " ROW_NUMBER() OVER (PARTITION BY trajectory_id "
            "  ORDER BY target_time, target_image_id, target_rowid) AS segment_index, "
            " ROW_NUMBER() OVER (ORDER BY trajectory_id, target_time, target_image_id, target_rowid) "
            "  AS global_break_index "
            " FROM audit.qc_flagged_edges WHERE reject = 1) "
            "SELECT *, ? + global_break_index AS new_trajectory_id FROM ranked",
            (maximum_id,),
        )
        connection.execute(
            "CREATE INDEX idx_qc_assignment_original_start ON qc_break_assignment "
            "(original_trajectory_id, target_time, target_image_id, target_rowid)"
        )
        break_count = connection.execute(
            "SELECT COUNT(*) FROM qc_break_assignment"
        ).fetchone()[0]
        connection.execute(
            "CREATE TABLE qc_duplicate_assignment AS "
            "SELECT duplicate_rowid, retained_rowid, trajectory_id AS original_trajectory_id, "
            " image_id, time, exact_compact_duplicate, "
            " ? + ? + ROW_NUMBER() OVER (ORDER BY trajectory_id, time, image_id, duplicate_rowid) "
            " AS new_trajectory_id "
            "FROM audit.qc_duplicate_points",
            (maximum_id, break_count),
        )
        connection.execute(
            f"UPDATE {quote_identifier(cleaned_table)} AS clean SET trajectory_id = COALESCE(("
            " SELECT assignment.new_trajectory_id FROM qc_break_assignment AS assignment "
            " WHERE assignment.original_trajectory_id = clean.trajectory_id "
            " AND (assignment.target_time < clean.time "
            "  OR (assignment.target_time = clean.time AND assignment.target_image_id < clean.image_id) "
            "  OR (assignment.target_time = clean.time AND assignment.target_image_id = clean.image_id "
            "   AND assignment.target_rowid <= clean.rowid)) "
            " ORDER BY assignment.target_time DESC, assignment.target_image_id DESC, "
            "  assignment.target_rowid DESC LIMIT 1), clean.trajectory_id) "
            "WHERE clean.trajectory_id IN "
            "(SELECT DISTINCT original_trajectory_id FROM qc_break_assignment)"
        )
        connection.execute(
            f"UPDATE {quote_identifier(cleaned_table)} SET is_last = 1 WHERE rowid IN "
            "(SELECT source_rowid FROM qc_break_assignment)"
        )
        connection.execute(
            f"UPDATE {quote_identifier(cleaned_table)} SET is_last = 1 WHERE rowid IN "
            "(SELECT retained_rowid FROM audit.qc_duplicate_points WHERE duplicate_is_last = 1)"
        )
        connection.execute(
            f"UPDATE {quote_identifier(cleaned_table)} AS clean SET "
            "trajectory_id = (SELECT new_trajectory_id FROM qc_duplicate_assignment "
            " WHERE duplicate_rowid = clean.rowid), is_last = 1 "
            "WHERE rowid IN (SELECT duplicate_rowid FROM qc_duplicate_assignment)"
        )
        if "converged_to" in columns:
            connection.execute(
                f"UPDATE {quote_identifier(cleaned_table)} AS clean SET converged_to = COALESCE(("
                " SELECT assignment.new_trajectory_id FROM qc_break_assignment AS assignment "
                " WHERE assignment.original_trajectory_id = clean.converged_to "
                " AND (assignment.target_time < clean.time "
                "  OR (assignment.target_time = clean.time AND assignment.target_image_id <= clean.image_id)) "
                " ORDER BY assignment.target_time DESC, assignment.target_image_id DESC, "
                "  assignment.target_rowid DESC LIMIT 1), clean.converged_to) "
                "WHERE clean.converged_to IN "
                "(SELECT DISTINCT original_trajectory_id FROM qc_break_assignment)"
            )
        connection.execute(
            f"CREATE INDEX {quote_identifier('idx_' + cleaned_table + '_traj_last')} "
            f"ON {quote_identifier(cleaned_table)} (trajectory_id, is_last)"
        )
        for table in (
            "qc_flagged_edges",
            "qc_pair_summary",
            "qc_image_summary",
            "qc_distribution_counts",
            "qc_metadata",
            "qc_duplicate_points",
        ):
            connection.execute(f"CREATE TABLE {table} AS SELECT * FROM audit.{table}")
        source_rows = connection.execute(
            f"SELECT COUNT(*) FROM src.{quote_identifier(source_table)}"
        ).fetchone()[0]
        cleaned_rows = connection.execute(
            f"SELECT COUNT(*) FROM {quote_identifier(cleaned_table)}"
        ).fetchone()[0]
        duplicate_count = connection.execute(
            "SELECT COUNT(*) FROM qc_duplicate_assignment"
        ).fetchone()[0]
        expected_break_count = connection.execute(
            "SELECT COUNT(*) FROM audit.qc_flagged_edges WHERE reject = 1"
        ).fetchone()[0]
        expected_duplicate_count = connection.execute(
            "SELECT COUNT(*) FROM audit.qc_duplicate_points"
        ).fetchone()[0]
        invalid_last = connection.execute(
            f"SELECT COUNT(*) FROM (SELECT trajectory_id FROM {quote_identifier(cleaned_table)} "
            "GROUP BY trajectory_id HAVING SUM(is_last) != 1)"
        ).fetchone()[0]
        repeated_image = connection.execute(
            f"SELECT COUNT(*) FROM (SELECT trajectory_id, image_id FROM {quote_identifier(cleaned_table)} "
            "GROUP BY trajectory_id, image_id HAVING COUNT(*) != 1)"
        ).fetchone()[0]
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        validation = {
            "status": "complete",
            "protocol_id": metadata["protocol_id"],
            "protocol_sha256": metadata["protocol_sha256"],
            "input_database": str(args.input.resolve()),
            "input_sha256": args.input_sha256,
            "source_table": source_table,
            "cleaned_table": cleaned_table,
            "source_rows": int(source_rows),
            "cleaned_rows": int(cleaned_rows),
            "break_assignments": int(break_count),
            "expected_break_assignments": int(expected_break_count),
            "duplicate_point_assignments": int(duplicate_count),
            "expected_duplicate_point_assignments": int(expected_duplicate_count),
            "invalid_is_last_trajectories": int(invalid_last),
            "repeated_trajectory_image_groups": int(repeated_image),
            "quick_check": quick_check,
            "elapsed_seconds": time.perf_counter() - started,
        }
        failures = []
        if source_rows != cleaned_rows:
            failures.append("source and cleaned row counts differ")
        if source_rows != metadata["stats"]["point_rows"]:
            failures.append("source row count differs from the completed scan")
        if break_count != expected_break_count:
            failures.append("break assignment count differs from rejected-vector count")
        if duplicate_count != expected_duplicate_count:
            failures.append("duplicate assignment count differs from duplicate audit")
        if invalid_last:
            failures.append(f"{invalid_last} trajectories have invalid is_last counts")
        if repeated_image:
            failures.append(f"{repeated_image} trajectory/image groups are repeated")
        if quick_check != "ok":
            failures.append(f"SQLite quick_check returned {quick_check!r}")
        if failures:
            raise ValueError("QC materialization validation failed: " + "; ".join(failures))
        connection.execute(
            "CREATE TABLE qc_materialization_manifest (manifest_json TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO qc_materialization_manifest VALUES (?)",
            (json.dumps(validation, sort_keys=True),),
        )
        connection.commit()
        connection.close()
        verify_source(args)
        os.replace(temporary, target)
    except Exception:
        connection.close()
        raise
    validation["output_database"] = str(target.resolve())
    validation["output_sha256"] = sha256(target)
    (args.output_dir / "materialization_manifest.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return validation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--table")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "scan", "materialize", "all"), default="all"
    )
    parser.add_argument("--compact-database", type=Path)
    parser.add_argument("--cleaned-output", type=Path)
    parser.add_argument(
        "--input-sha256",
        required=True,
        help="verified lowercase SHA-256 of the closed source SQLite",
    )
    parser.add_argument(
        "--configured-speed-m-per-day",
        type=float,
        required=True,
        help="actual max_speed_m_per_day used by the source tracking run",
    )
    parser.add_argument("--fetch-rows", type=int, default=100_000)
    parser.add_argument("--flush-images", type=int, default=100)
    parser.add_argument("--progress-images", type=int, default=250)
    parser.add_argument("--checkpoint-images", type=int, default=2_000)
    parser.add_argument("--maximum-rows", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.input_sha256):
        raise ValueError("--input-sha256 must be 64 lowercase hexadecimal characters")
    protocol = load_protocol()
    args.prescreen_residual_m = float(
        protocol["parameters"]["archive_prescreen_residual_m"]
    )
    if args.compact_database is None:
        args.compact_database = args.output_dir / "compact_points.sqlite"
    config = QCConfig(
        configured_speed_m_per_day=args.configured_speed_m_per_day,
    )
    config.validate()
    validate_frozen_protocol(config)
    for name in ("fetch_rows", "flush_images", "progress_images", "checkpoint_images"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.maximum_rows is not None and args.maximum_rows <= 0:
        raise ValueError("maximum_rows must be positive")
    if args.resume and args.stage == "prepare":
        raise ValueError("--resume applies to scan or all, not prepare")
    if args.prescreen_residual_m <= 0:
        raise ValueError("prescreen residual must be positive")
    if args.stage in {"prepare", "all"} and not args.resume:
        build_compact_database(args)
    if args.stage in {"scan", "all"}:
        scan_archive(args, config)
    if args.stage in {"materialize", "all"}:
        materialize(args)


if __name__ == "__main__":
    main()
