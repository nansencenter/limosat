"""Finalize a completed native run into compact assessment products."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .config import RunConfig
from .store import DATABASE_SCHEMA_VERSION, PAIR_MATCH_ENCODING, file_sha256


TRAJECTORY_CATALOGUE_SCHEMA_VERSION = 1
ASSESSMENT_SUMMARY_SCHEMA_VERSION = 1


def finalize_products(
    config: RunConfig,
    *,
    export_parquet: bool = True,
    batch_size: int = 100_000,
) -> dict:
    """Validate and package a completed run without changing scientific rows."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    database = Path(config.database)
    run = _validate_completed_run(database, config)
    manifest_path = Path(run["manifest_path"])
    manifest_sha256 = file_sha256(manifest_path)
    if manifest_sha256 != run["manifest_sha256"]:
        raise ValueError("native run manifest failed its recorded checksum")
    integrity = _checkpoint_and_check(database, config.run_id)
    output = Path(config.output_directory)
    output.mkdir(parents=True, exist_ok=True)

    parquet_path = output / "global-trajectory-catalogue-v1.parquet"
    if export_parquet:
        _write_trajectory_parquet(
            database, config.run_id, parquet_path, batch_size
        )
    counts = _product_counts(database, config.run_id)
    products = {
        "sqlite": _file_record(database),
        "native_manifest": _file_record(manifest_path),
        "trajectory_parquet": (
            _file_record(parquet_path) if export_parquet else None
        ),
    }
    report = {
        "assessment_summary_schema_version": ASSESSMENT_SUMMARY_SCHEMA_VERSION,
        "trajectory_catalogue_schema_version": (
            TRAJECTORY_CATALOGUE_SCHEMA_VERSION if export_parquet else None
        ),
        "run_id": config.run_id,
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
        "sqlite_schema_version": DATABASE_SCHEMA_VERSION,
        "coordinates": {
            "crs": "EPSG:3413",
            "dtype": "float64",
            "distance_unit": "metre",
            "time": "timezone-aware UTC",
        },
        "integrity": integrity,
        "counts": counts,
        "raw_match_retention": {
            "enabled": config.retain_pair_matches,
            "archived_pairs": counts["retained_pair_match_archives"],
            "retained_matches": counts["retained_pair_matches"],
            "compressed_bytes": counts[
                "retained_pair_match_compressed_bytes"
            ],
        },
        "products": products,
    }
    report_path = output / "assessment-summary-v1.json"
    temporary = output / ".assessment-summary-v1.json.writing"
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    return {
        "assessment_summary": str(report_path),
        "assessment_summary_sha256": file_sha256(report_path),
        "trajectory_parquet": str(parquet_path) if export_parquet else None,
        "counts": counts,
    }


def _validate_completed_run(database: Path, config: RunConfig) -> dict:
    if not database.is_file():
        raise FileNotFoundError(database)
    with closing(sqlite3.connect(database)) as connection:
        connection.row_factory = sqlite3.Row
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version != DATABASE_SCHEMA_VERSION:
            raise ValueError(
                f"finalization requires SQLite schema {DATABASE_SCHEMA_VERSION}; "
                f"found {version or 'legacy'}"
            )
        row = connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (config.run_id,)
        ).fetchone()
    if row is None:
        raise ValueError(f"run not found in database: {config.run_id}")
    run = dict(row)
    if run["config_sha256"] != config.sha256:
        raise ValueError("finalization config differs from the completed run")
    if run["status"] != "complete":
        raise ValueError("only a complete run can be finalized")
    if not run["manifest_path"] or not run["manifest_sha256"]:
        raise ValueError("completed run has no native manifest identity")
    return run


def _checkpoint_and_check(database: Path, run_id: str) -> dict[str, str | int]:
    with closing(sqlite3.connect(database, timeout=60.0)) as connection:
        connection.row_factory = sqlite3.Row
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        foreign_key_errors = len(
            connection.execute("PRAGMA foreign_key_check").fetchall()
        )
        coordinate_errors = connection.execute(
            """
            SELECT COUNT(*) FROM trajectory_points WHERE run_id=? AND (
              (x_m IS NULL) != (y_m IS NULL) OR
              (state='dormant' AND x_m IS NOT NULL) OR
              (state!='dormant' AND x_m IS NULL)
            )
            """,
            (run_id,),
        ).fetchone()[0]
        if checkpoint[0]:
            raise RuntimeError(
                "SQLite WAL checkpoint is busy; stop run writers before finalization"
            )
        if quick_check != "ok" or foreign_key_errors or coordinate_errors:
            raise ValueError(
                "SQLite integrity check failed: "
                f"quick_check={quick_check}, "
                f"foreign_key_errors={foreign_key_errors}, "
                f"trajectory_coordinate_errors={coordinate_errors}"
            )
        archive_count = 0
        for row in connection.execute(
            """
            SELECT archive.*,pairs.match_count pair_match_count
            FROM pair_match_archives archive JOIN pairs USING(run_id,pair_id)
            WHERE archive.run_id=? ORDER BY archive.pair_id
            """,
            (run_id,),
        ):
            archive_count += 1
            payload = bytes(row["payload"])
            if row["encoding"] != PAIR_MATCH_ENCODING:
                raise ValueError(
                    f"unsupported retained pair-match encoding: {row['encoding']}"
                )
            if hashlib.sha256(payload).hexdigest() != row["payload_sha256"]:
                raise ValueError(
                    f"retained pair matches failed checksum: {row['pair_id']}"
                )
            raw = zlib.decompress(payload)
            expected_bytes = int(row["match_count"]) * 48
            if (
                int(row["match_count"]) != int(row["pair_match_count"])
                or len(raw) != int(row["uncompressed_bytes"])
                or len(raw) != expected_bytes
            ):
                raise ValueError(
                    "retained pair-match metadata is inconsistent: "
                    f"{row['pair_id']}"
                )
    return {
        "quick_check": quick_check,
        "foreign_key_errors": foreign_key_errors,
        "trajectory_coordinate_errors": int(coordinate_errors),
        "verified_pair_match_archives": archive_count,
        "wal_checkpoint_busy": int(checkpoint[0]),
        "wal_frames": int(checkpoint[1]),
        "wal_frames_checkpointed": int(checkpoint[2]),
    }


def _write_trajectory_parquet(
    database: Path, run_id: str, destination: Path, batch_size: int
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyArrow is required only for Parquet finalization; use the GPU "
            "overlay that already provides it or pass --skip-parquet"
        ) from error

    schema = pa.schema(
        [
            ("trajectory_id", pa.string()),
            ("image_id", pa.string()),
            ("time_utc", pa.timestamp("us", tz="UTC")),
            ("state", pa.string()),
            ("position_basis", pa.string()),
            ("x_m", pa.float64()),
            ("y_m", pa.float64()),
            ("source_pair_id", pa.string()),
            ("selected_matches", pa.int32()),
            ("support_radius_m", pa.float64()),
            ("maximum_residual_m", pa.float64()),
        ],
        metadata={
            b"limosat_schema": b"global_trajectory_catalogue_v1",
            b"crs": b"EPSG:3413",
            b"coordinate_unit": b"metre",
            b"time_zone": b"UTC",
        },
    )
    query = """
        SELECT trajectory_id,image_id,time_utc,state,position_basis,x_m,y_m,
               source_pair_id,selected_matches,support_radius_m,
               maximum_residual_m
        FROM trajectory_points WHERE run_id=?
        ORDER BY trajectory_id,time_utc,image_id
    """
    temporary = destination.with_name(f".{destination.name}.writing")
    writer = None
    try:
        with closing(sqlite3.connect(database)) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.execute(query, (run_id,))
            writer = pq.ParquetWriter(temporary, schema, compression="zstd")
            while rows := cursor.fetchmany(batch_size):
                columns = {name: [] for name in schema.names}
                for row in rows:
                    for name in schema.names:
                        value = row[name]
                        if name == "time_utc":
                            value = datetime.fromisoformat(value)
                            if value.tzinfo is None or value.utcoffset() is None:
                                raise ValueError(
                                    "trajectory time is not timezone-aware"
                                )
                            value = value.astimezone(timezone.utc)
                        elif name == "selected_matches" and value is not None:
                            value = int(value)
                        columns[name].append(value)
                writer.write_table(pa.table(columns, schema=schema))
        writer.close()
        writer = None
        temporary.replace(destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def _product_counts(database: Path, run_id: str) -> dict[str, int | dict]:
    with closing(sqlite3.connect(database)) as connection:
        state_counts = dict(
            connection.execute(
                """
                SELECT state,COUNT(*) FROM trajectory_points
                WHERE run_id=? GROUP BY state ORDER BY state
                """,
                (run_id,),
            )
        )
        pair_counts = dict(
            connection.execute(
                """
                SELECT kind,COUNT(*) FROM pairs
                WHERE run_id=? AND status='complete' GROUP BY kind ORDER BY kind
                """,
                (run_id,),
            )
        )
        values = connection.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM images WHERE run_id=?) images,
              (SELECT COUNT(*) FROM candidate_pairs WHERE run_id=?) candidate_pairs,
              (SELECT COUNT(*) FROM trajectories WHERE run_id=?) trajectories,
              (SELECT COUNT(*) FROM trajectory_points WHERE run_id=?) trajectory_points,
              (SELECT COUNT(*) FROM field_nodes WHERE run_id=?) field_nodes,
              (SELECT COUNT(*) FROM field_nodes
                 WHERE run_id=? AND available=1) available_field_nodes,
              (SELECT COUNT(*) FROM deformation_cells
                 WHERE run_id=?) deformation_cells,
              (SELECT COUNT(*) FROM pair_match_archives
                 WHERE run_id=?) retained_pair_match_archives,
              (SELECT COALESCE(SUM(match_count),0) FROM pair_match_archives
                 WHERE run_id=?) retained_pair_matches,
              (SELECT COALESCE(SUM(compressed_bytes),0) FROM pair_match_archives
                 WHERE run_id=?) retained_pair_match_compressed_bytes
            """,
            (run_id,) * 10,
        ).fetchone()
    counts = dict(zip((
        "images",
        "candidate_pairs",
        "trajectories",
        "trajectory_points",
        "field_nodes",
        "available_field_nodes",
        "deformation_cells",
        "retained_pair_match_archives",
        "retained_pair_matches",
        "retained_pair_match_compressed_bytes",
    ), values, strict=True))
    counts["trajectory_states"] = state_counts
    counts["completed_pairs"] = pair_counts
    return counts


def _file_record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }
