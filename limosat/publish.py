"""Materialize a concise scientific Parquet dataset from a frozen QC release.

This module deliberately consumes the *materialized* QC SQLite output.  It does
not score trajectories or revisit QC decisions; post-QC trajectory IDs and
segment boundaries are the source of truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"
CRS = "EPSG:3413"
PUBLIC_COLUMNS = (
    "trajectory_id",
    "source_image_id",
    "target_image_id",
    "start_time_utc",
    "end_time_utc",
    "elapsed_seconds",
    "x0_m",
    "y0_m",
    "x1_m",
    "y1_m",
    "dx_m",
    "dy_m",
    "distance_m",
    "speed_m_s",
    "speed_m_per_day",
    "source_position_type",
    "target_position_type",
)


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - dependent on environment
        raise RuntimeError(
            "Publishing requires the optional PyArrow dependency. Install it with "
            "`conda install -c conda-forge pyarrow` or `pip install pyarrow`."
        ) from error
    return pa, pq


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a closed file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def parse_point_wkt(value: str) -> tuple[float, float]:
    if not value.startswith("POINT"):
        raise ValueError(f"Expected POINT WKT, got {value[:40]!r}")
    values = value[value.find("(") + 1 : value.rfind(")")].strip().split()
    if len(values) < 2:
        raise ValueError(f"Malformed POINT WKT: {value!r}")
    return float(values[0]), float(values[1])


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _position_type(interpolated: Any) -> str:
    return "interpolated" if interpolated else "observed"


@dataclass(frozen=True)
class ReleaseSource:
    path: Path
    table: str
    release_id: str
    window: str
    source_sha256: str
    expected_edges: int
    break_pairs: frozenset[tuple[int, int]]


def _connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    wal = Path(str(path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Source SQLite has an uncheckpointed WAL; close/checkpoint it first")
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def _manifest(connection: sqlite3.Connection) -> dict[str, Any]:
    try:
        row = connection.execute(
            "SELECT manifest_json FROM qc_materialization_manifest"
        ).fetchone()
    except sqlite3.OperationalError as error:
        raise ValueError("Source is not a materialized LiMOSAT QC SQLite release") from error
    if row is None:
        raise ValueError("Source QC SQLite has no materialization manifest")
    manifest = json.loads(row[0])
    required = {
        "status": "complete",
        "quick_check": "ok",
        "invalid_is_last_trajectories": 0,
        "repeated_trajectory_image_groups": 0,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"Source QC materialization manifest has {field}={manifest.get(field)!r}; "
                f"expected {expected!r}"
            )
    return manifest


def _cleaned_table(connection: sqlite3.Connection) -> str:
    tables = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name LIKE '%__qc' ORDER BY name"
        )
    ]
    if len(tables) != 1:
        raise ValueError(f"Expected exactly one materialized '*__qc' table, found {tables}")
    required = {
        "image_id",
        "is_last",
        "trajectory_id",
        "geometry",
        "time",
        "interpolated",
    }
    columns = {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({quote_identifier(tables[0])})")
    }
    missing = required - columns
    if missing:
        raise ValueError(f"Cleaned QC table is missing required columns: {sorted(missing)}")
    return tables[0]


def inspect_source(path: Path, release_id: str | None = None) -> ReleaseSource:
    """Check a frozen QC SQLite release and return immutable source details."""
    path = path.resolve()
    with _connection(path) as connection:
        _manifest(connection)
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Source SQLite quick_check did not return 'ok'")
        table = _cleaned_table(connection)
        summary = connection.execute(
            "SELECT COALESCE(SUM(links), 0), COALESCE(SUM(rejected), 0) "
            "FROM qc_pair_summary"
        ).fetchone()
        expected_edges = int(summary[0] - summary[1])
        break_pairs = frozenset(
            (int(source_rowid), int(target_rowid))
            for source_rowid, target_rowid in connection.execute(
                "SELECT source_rowid, target_rowid FROM qc_break_assignment"
            )
        )
    window = path.parent.name
    return ReleaseSource(
        path=path,
        table=table,
        release_id=release_id or path.parent.parent.name,
        window=window,
        source_sha256=sha256(path),
        expected_edges=expected_edges,
        break_pairs=break_pairs,
    )


def _edge_query(table: str) -> str:
    quoted = quote_identifier(table)
    return f"""
        WITH ordered AS (
            SELECT
                rowid AS source_rowid,
                trajectory_id,
                image_id AS source_image_id,
                time AS source_time,
                geometry AS source_geometry,
                interpolated AS source_interpolated,
                is_last AS source_is_last,
                LEAD(rowid) OVER trajectory AS target_rowid,
                LEAD(image_id) OVER trajectory AS target_image_id,
                LEAD(time) OVER trajectory AS target_time,
                LEAD(geometry) OVER trajectory AS target_geometry,
                LEAD(interpolated) OVER trajectory AS target_interpolated
            FROM {quoted}
            WINDOW trajectory AS (
                PARTITION BY trajectory_id ORDER BY time, image_id, rowid
            )
        )
        SELECT trajectory_id, source_rowid, target_rowid, source_image_id,
               target_image_id, source_time, target_time, source_geometry,
               target_geometry, source_interpolated, target_interpolated
        FROM ordered
        WHERE target_rowid IS NOT NULL AND source_is_last = 0
        ORDER BY trajectory_id, source_time, source_image_id, source_rowid
    """


def _public_edge(row: tuple[Any, ...]) -> tuple[dict[str, Any], tuple[int, int]]:
    (
        trajectory_id,
        source_rowid,
        target_rowid,
        source_image_id,
        target_image_id,
        source_time,
        target_time,
        source_geometry,
        target_geometry,
        source_interpolated,
        target_interpolated,
    ) = row
    start = parse_utc(source_time)
    end = parse_utc(target_time)
    elapsed_seconds = (end - start).total_seconds()
    if elapsed_seconds <= 0:
        raise ValueError(
            f"Non-positive elapsed time for trajectory {trajectory_id}: "
            f"{source_time!r} -> {target_time!r}"
        )
    x0, y0 = parse_point_wkt(source_geometry)
    x1, y1 = parse_point_wkt(target_geometry)
    dx = x1 - x0
    dy = y1 - y0
    distance = math.hypot(dx, dy)
    public = {
        "trajectory_id": int(trajectory_id),
        "source_image_id": int(source_image_id),
        "target_image_id": int(target_image_id),
        "start_time_utc": start,
        "end_time_utc": end,
        "elapsed_seconds": elapsed_seconds,
        "x0_m": x0,
        "y0_m": y0,
        "x1_m": x1,
        "y1_m": y1,
        "dx_m": dx,
        "dy_m": dy,
        "distance_m": distance,
        "speed_m_s": distance / elapsed_seconds,
        "speed_m_per_day": distance / elapsed_seconds * 86_400.0,
        "source_position_type": _position_type(source_interpolated),
        "target_position_type": _position_type(target_interpolated),
    }
    return public, (int(source_rowid), int(target_rowid))


def public_schema(source: ReleaseSource):
    pa, _ = _pyarrow()
    fields = [
        pa.field("trajectory_id", pa.int64()),
        pa.field("source_image_id", pa.int64()),
        pa.field("target_image_id", pa.int64()),
        pa.field("start_time_utc", pa.timestamp("us", tz="UTC")),
        pa.field("end_time_utc", pa.timestamp("us", tz="UTC")),
        pa.field("elapsed_seconds", pa.float64()),
        pa.field("x0_m", pa.float64()),
        pa.field("y0_m", pa.float64()),
        pa.field("x1_m", pa.float64()),
        pa.field("y1_m", pa.float64()),
        pa.field("dx_m", pa.float64()),
        pa.field("dy_m", pa.float64()),
        pa.field("distance_m", pa.float64()),
        pa.field("speed_m_s", pa.float64()),
        pa.field("speed_m_per_day", pa.float64()),
        pa.field("source_position_type", pa.string()),
        pa.field("target_position_type", pa.string()),
    ]
    metadata = {
        b"limosat.schema_version": SCHEMA_VERSION.encode(),
        b"limosat.release_id": source.release_id.encode(),
        b"limosat.source_window": source.window.encode(),
        b"limosat.crs": CRS.encode(),
        b"limosat.coordinate_units": b"metres",
        b"limosat.time_units": b"UTC timestamps; elapsed_seconds in seconds",
        b"limosat.source_sqlite_sha256": source.source_sha256.encode(),
    }
    return pa.schema(fields, metadata=metadata)


def _default_output(path: Path) -> Path:
    window = path.parent.name
    return path.parent / f"{window}_qc_publish.parquet"


def _default_manifest(output: Path) -> Path:
    return output.with_suffix(".manifest.json")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _bounds_update(bounds: dict[str, float | None], public: dict[str, Any]) -> None:
    for key, values in {
        "min_x_m": (public["x0_m"], public["x1_m"]),
        "max_x_m": (public["x0_m"], public["x1_m"]),
        "min_y_m": (public["y0_m"], public["y1_m"]),
        "max_y_m": (public["y0_m"], public["y1_m"]),
    }.items():
        value = min(values) if key.startswith("min") else max(values)
        bounds[key] = value if bounds[key] is None else (
            min(bounds[key], value) if key.startswith("min") else max(bounds[key], value)
        )


def _manifest_payload(
    source: ReleaseSource,
    output: Path,
    row_count: int,
    trajectory_ids: set[int],
    start_time: datetime | None,
    end_time: datetime | None,
    bounds: dict[str, float | None],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": source.release_id,
        "source_window": source.window,
        "input_sqlite": str(source.path),
        "input_sqlite_sha256": source.source_sha256,
        "output_parquet": str(output),
        "output_parquet_sha256": sha256(output),
        "row_count": row_count,
        "trajectory_count": len(trajectory_ids),
        "time_bounds_utc": {
            "start": _utc_text(start_time) if start_time else None,
            "end": _utc_text(end_time) if end_time else None,
        },
        "spatial_bounds_epsg_3413_m": bounds,
        "validation": {
            "status": "passed",
            "expected_accepted_edge_count": source.expected_edges,
            "actual_accepted_edge_count": row_count,
            "no_invalid_trajectory_continuity": True,
            "no_edge_across_qc_break": True,
        },
    }


def publish(
    input_sqlite: Path,
    output: Path | None = None,
    manifest: Path | None = None,
    release_id: str | None = None,
    batch_size: int = 100_000,
) -> dict[str, Any]:
    """Write one public edge dataset and its adjacent checked manifest."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    source = inspect_source(input_sqlite, release_id)
    output = (output or _default_output(source.path)).resolve()
    manifest = (manifest or _default_manifest(output)).resolve()
    if output.exists():
        raise FileExistsError(output)
    if manifest.exists():
        raise FileExistsError(manifest)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    output.parent.mkdir(parents=True, exist_ok=True)
    pa, pq = _pyarrow()
    schema = public_schema(source)
    row_count = 0
    trajectory_ids: set[int] = set()
    start_time: datetime | None = None
    end_time: datetime | None = None
    bounds: dict[str, float | None] = {
        "min_x_m": None, "max_x_m": None, "min_y_m": None, "max_y_m": None,
    }
    current_trajectory: int | None = None
    previous_target: tuple[Any, ...] | None = None
    try:
        with _connection(source.path) as connection, pq.ParquetWriter(
            temporary, schema, compression="zstd"
        ) as writer:
            cursor = connection.execute(_edge_query(source.table))
            while rows := cursor.fetchmany(batch_size):
                public_rows: list[dict[str, Any]] = []
                for row in rows:
                    public, pair = _public_edge(row)
                    if pair in source.break_pairs:
                        raise ValueError("QC break was selected for publication")
                    trajectory_id = public["trajectory_id"]
                    source_endpoint = (
                        public["source_image_id"], public["start_time_utc"],
                        public["x0_m"], public["y0_m"], public["source_position_type"],
                    )
                    if trajectory_id == current_trajectory and source_endpoint != previous_target:
                        raise ValueError(
                            f"Invalid trajectory continuity for post-QC trajectory {trajectory_id}"
                        )
                    current_trajectory = trajectory_id
                    previous_target = (
                        public["target_image_id"], public["end_time_utc"],
                        public["x1_m"], public["y1_m"], public["target_position_type"],
                    )
                    public_rows.append(public)
                    row_count += 1
                    trajectory_ids.add(trajectory_id)
                    start_time = public["start_time_utc"] if start_time is None else min(
                        start_time, public["start_time_utc"]
                    )
                    end_time = public["end_time_utc"] if end_time is None else max(
                        end_time, public["end_time_utc"]
                    )
                    _bounds_update(bounds, public)
                writer.write_table(pa.Table.from_pylist(public_rows, schema=schema))
        if row_count != source.expected_edges:
            raise ValueError(
                f"Published {row_count} edges but QC input expects {source.expected_edges}"
            )
        if sha256(source.path) != source.source_sha256:
            raise ValueError("Source SQLite checksum changed while publishing")
        os.replace(temporary, output)
        result = _manifest_payload(
            source, output, row_count, trajectory_ids, start_time, end_time, bounds
        )
        _atomic_write_json(manifest, result)
        return result
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _parquet_rows(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    _, pq = _pyarrow()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(PUBLIC_COLUMNS)):
        yield from batch.to_pylist()


def validate(
    input_sqlite: Path,
    output: Path | None = None,
    release_id: str | None = None,
    batch_size: int = 100_000,
) -> dict[str, Any]:
    """Validate an existing Parquet dataset row-for-row against its QC SQLite input."""
    source = inspect_source(input_sqlite, release_id)
    output = (output or _default_output(source.path)).resolve()
    if not output.is_file():
        raise FileNotFoundError(output)
    pa, pq = _pyarrow()
    schema = pq.ParquetFile(output).schema_arrow
    if tuple(schema.names) != PUBLIC_COLUMNS:
        raise ValueError("Parquet public column order does not match the publishing schema")
    metadata = schema.metadata or {}
    expected_metadata = public_schema(source).metadata or {}
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"Parquet schema metadata mismatch for {key.decode()}")

    parquet_rows = _parquet_rows(output, batch_size)
    row_count = 0
    trajectory_ids: set[int] = set()
    current_trajectory: int | None = None
    previous_target: tuple[Any, ...] | None = None
    with _connection(source.path) as connection:
        for source_row in connection.execute(_edge_query(source.table)):
            expected, pair = _public_edge(source_row)
            if pair in source.break_pairs:
                raise ValueError("QC break was selected for publication")
            try:
                actual = next(parquet_rows)
            except StopIteration as error:
                raise ValueError("Parquet has fewer rows than the QC input expects") from error
            if actual != expected:
                raise ValueError(f"Parquet row {row_count} does not match its QC SQLite edge")
            trajectory_id = expected["trajectory_id"]
            source_endpoint = (
                expected["source_image_id"], expected["start_time_utc"], expected["x0_m"],
                expected["y0_m"], expected["source_position_type"],
            )
            if trajectory_id == current_trajectory and source_endpoint != previous_target:
                raise ValueError(f"Invalid trajectory continuity for {trajectory_id}")
            current_trajectory = trajectory_id
            previous_target = (
                expected["target_image_id"], expected["end_time_utc"], expected["x1_m"],
                expected["y1_m"], expected["target_position_type"],
            )
            row_count += 1
            trajectory_ids.add(trajectory_id)
    try:
        next(parquet_rows)
    except StopIteration:
        pass
    else:
        raise ValueError("Parquet has more rows than the QC input expects")
    if row_count != source.expected_edges:
        raise ValueError(
            f"Validated {row_count} edges but QC input expects {source.expected_edges}"
        )
    return {
        "status": "passed",
        "input_sqlite": str(source.path),
        "output_parquet": str(output),
        "row_count": row_count,
        "trajectory_count": len(trajectory_ids),
        "expected_accepted_edge_count": source.expected_edges,
        "no_invalid_trajectory_continuity": True,
        "no_edge_across_qc_break": True,
    }


def discover(release_root: Path) -> list[Path]:
    """Find frozen QC SQLite releases, excluding macOS sidecar files."""
    return sorted(
        path for path in release_root.rglob("*_qc.sqlite")
        if not path.name.startswith("._") and path.is_file()
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("one", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path)
        command.add_argument("--release-id")
        command.add_argument("--batch-size", type=int, default=100_000)
        if name == "one":
            command.add_argument("--manifest", type=Path)
    for name in ("all", "validate-all"):
        command = commands.add_parser(name)
        command.add_argument("--release-root", type=Path, required=True)
        command.add_argument("--release-id")
        command.add_argument("--batch-size", type=int, default=100_000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "one":
        result = publish(args.input, args.output, args.manifest, args.release_id, args.batch_size)
    elif args.command == "validate":
        result = validate(args.input, args.output, args.release_id, args.batch_size)
    else:
        inputs = discover(args.release_root)
        if not inputs:
            raise ValueError(f"No eligible '*_qc.sqlite' files under {args.release_root}")
        operation = publish if args.command == "all" else validate
        result = [
            operation(path, release_id=args.release_id or args.release_root.name, batch_size=args.batch_size)
            for path in inputs
        ]
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
