"""Durable SQLite run state: the registered catalogue and image-pair plan.

``limosat prepare`` writes it once; pair workers read the plan from it (read-only), so
every worker and batch agrees on the pairs. Pair results live in pair products, and
trajectories are composed from those into Parquet (compose.py).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .catalog import ImageCatalogue, ImagePair
from .config import RunConfig
from .planning import PlannedPair


DATABASE_SCHEMA_VERSION = 6


class RunStore:
    """SQLite run state for one run: catalogue, candidate pairs and planning counts."""

    def __init__(self, config: RunConfig, *, read_only: bool = False) -> None:
        self.config = config
        self.path = Path(config.database)
        self.read_only = read_only
        self.implementation_sha256 = implementation_sha256()
        self.model_sha256 = (
            file_sha256(config.matcher.checkpoint)
            if config.matcher.checkpoint
            and Path(config.matcher.checkpoint).is_file()
            else None
        )
        if read_only:
            self._validate_existing()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_schema()
            self._ensure_run()


    def register_catalogue(self, catalogue: ImageCatalogue) -> None:
        rows = []
        for image in catalogue.records:
            if not image.path.is_file():
                raise FileNotFoundError(image.path)
            rows.append(
                (
                    self.config.run_id,
                    image.image_id,
                    image.component_id,
                    image.platform,
                    image.absolute_orbit,
                    str(image.path),
                    image.time_utc.isoformat(),
                    image.path.stat().st_size,
                    file_sha256(image.path),
                )
            )
        with closing(self._connect()) as connection, connection:
            for row in rows:
                existing = connection.execute(
                    """
                    SELECT platform,absolute_orbit,path,time_utc,sha256
                    FROM images WHERE run_id=? AND image_id=?
                    """,
                    row[:2],
                ).fetchone()
                if existing is not None and tuple(existing) != (
                    row[3], row[4], row[5], row[6], row[8]
                ):
                    raise ValueError(f"image identity changed during resume: {row[1]}")
                connection.execute(
                    """
                    INSERT OR IGNORE INTO images
                    (run_id,image_id,component_id,platform,absolute_orbit,path,
                     time_utc,size_bytes,sha256)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    row,
                )


    def start_run(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE runs SET status='running', started_utc=COALESCE(started_utc, ?),
                    completed_utc=NULL, error=NULL WHERE run_id=?
                """,
                (_utc_now(), self.config.run_id),
            )


    def register_candidate_pairs(
        self, planned: Iterable[PlannedPair]
    ) -> None:
        values = tuple(planned)
        rows = [
            (
                self.config.run_id,
                item.pair.pair_id,
                item.ordinal,
                item.selection,
                item.planning_component_id,
                item.pair.source.component_id,
                item.pair.target.component_id,
                item.pair.source.image_id,
                item.pair.target.image_id,
                item.pair.source.time_utc.isoformat(),
                item.pair.target.time_utc.isoformat(),
                item.pair.elapsed_seconds,
                item.overlap_fraction,
                item.overlap_area_m2,
                item.skipped_images,
            )
            for item in values
        ]
        with closing(self._connect()) as connection, connection:
            existing = [
                tuple(row)
                for row in connection.execute(
                    """
                    SELECT run_id,pair_id,ordinal,selection,planning_component_id,
                           source_component_id,target_component_id,source_image_id,
                           target_image_id,source_time_utc,target_time_utc,
                           elapsed_seconds,overlap_fraction,overlap_area_m2,
                           skipped_images
                    FROM candidate_pairs WHERE run_id=? ORDER BY ordinal
                    """,
                    (self.config.run_id,),
                )
            ]
            if existing and existing != rows:
                raise ValueError(
                    "candidate image-pair plan changed during resume; "
                    "use a new database path and run_id"
                )
            connection.executemany(
                """
                INSERT OR IGNORE INTO candidate_pairs
                (run_id,pair_id,ordinal,selection,planning_component_id,
                 source_component_id,target_component_id,source_image_id,
                 target_image_id,source_time_utc,target_time_utc,
                 elapsed_seconds,overlap_fraction,overlap_area_m2,skipped_images)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )


    def register_planning_counts(self, counts: dict[str, int]) -> None:
        rows = [
            (self.config.run_id, name, int(count))
            for name, count in sorted(counts.items())
        ]
        with closing(self._connect()) as connection, connection:
            existing = [
                tuple(row)
                for row in connection.execute(
                    """
                    SELECT run_id,reason,count FROM planning_counts
                    WHERE run_id=? ORDER BY reason
                    """,
                    (self.config.run_id,),
                )
            ]
            if existing and existing != rows:
                raise ValueError(
                    "candidate image-pair planning counts changed during resume; "
                    "use a new database path and run_id"
                )
            connection.executemany(
                """
                INSERT OR IGNORE INTO planning_counts(run_id,reason,count)
                VALUES (?,?,?)
                """,
                rows,
            )


    def fail_run(self, error: Exception) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE runs SET status='failed', completed_utc=?, error=? WHERE run_id=?",
                (_utc_now(), str(error)[:2_000], self.config.run_id),
            )


    def planned_pairs(self, catalogue: ImageCatalogue) -> tuple[PlannedPair, ...]:
        """Reconstruct the registered immutable plan without replanning imagery."""
        images = {image.image_id: image for image in catalogue.records}
        with closing(self._connect()) as connection:
            registered_images = {
                row["image_id"]: row
                for row in connection.execute(
                    """
                    SELECT image_id,component_id,platform,absolute_orbit,path,
                           time_utc,size_bytes
                    FROM images WHERE run_id=? ORDER BY image_id
                    """,
                    (self.config.run_id,),
                )
            }
            registered_count = connection.execute(
                """
                SELECT count FROM planning_counts
                WHERE run_id=? AND reason='accepted_candidate_pairs'
                """,
                (self.config.run_id,),
            ).fetchone()
            if registered_count is None:
                raise RuntimeError(
                    "candidate image-pair plan is not registered; run prepare first"
                )
            rows = connection.execute(
                """
                SELECT * FROM candidate_pairs WHERE run_id=? ORDER BY ordinal
                """,
                (self.config.run_id,),
            ).fetchall()
        if registered_images.keys() != images.keys():
            raise ValueError("registered catalogue image identities changed")
        for image_id, image in images.items():
            registered = registered_images[image_id]
            if not image.path.is_file():
                raise FileNotFoundError(image.path)
            current = (
                image.component_id,
                image.platform,
                image.absolute_orbit,
                str(image.path),
                image.time_utc.isoformat(),
                image.path.stat().st_size,
            )
            if tuple(registered)[1:] != current:
                raise ValueError(
                    f"registered catalogue image metadata changed: {image_id}"
                )
        if int(registered_count[0]) != len(rows):
            raise ValueError("registered candidate image-pair plan is incomplete")
        planned = []
        for row in rows:
            if int(row["ordinal"]) != len(planned):
                raise ValueError(
                    "registered candidate image-pair ordinals are not contiguous"
                )
            try:
                pair = ImagePair(
                    images[row["source_image_id"]],
                    images[row["target_image_id"]],
                )
            except KeyError as error:
                raise ValueError(
                    f"registered image pair refers to an unknown image: {row['pair_id']}"
                ) from error
            if pair.pair_id != row["pair_id"]:
                raise ValueError(f"registered image-pair identity changed: {row['pair_id']}")
            planned.append(
                PlannedPair(
                    pair=pair,
                    ordinal=int(row["ordinal"]),
                    selection=row["selection"],
                    overlap_fraction=row["overlap_fraction"],
                    overlap_area_m2=row["overlap_area_m2"],
                    skipped_images=int(row["skipped_images"]),
                    planning_component_id=row["planning_component_id"],
                )
            )
        return tuple(planned)


    def status(self) -> dict:
        with closing(self._connect()) as connection:
            run = connection.execute(
                "SELECT * FROM runs WHERE run_id=?", (self.config.run_id,)
            ).fetchone()
            counts = connection.execute(
                "SELECT selection,COUNT(*) count FROM candidate_pairs WHERE run_id=? GROUP BY selection",
                (self.config.run_id,),
            ).fetchall()
        return {"run": dict(run), "candidate_pairs": {row[0]: row[1] for row in counts}}

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            uri = self.path.resolve().as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, timeout=60.0, uri=True)
        else:
            connection = sqlite3.connect(self.path, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        if self.read_only:
            connection.execute("PRAGMA query_only=ON")
        else:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
        return connection


    def _validate_existing(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(
                f"run database does not exist; run prepare first: {self.path}"
            )
        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version != DATABASE_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported LiMOSAT SQLite schema {version or 'legacy'}; "
                    f"expected schema {DATABASE_SCHEMA_VERSION}"
                )
            row = connection.execute(
                """
                SELECT config_sha256,implementation_sha256,model_sha256
                FROM runs WHERE run_id=?
                """,
                (self.config.run_id,),
            ).fetchone()
        expected = (
            self.config.sha256,
            self.implementation_sha256,
            self.model_sha256,
        )
        if row is None:
            raise ValueError(f"run not found: {self.config.run_id}")
        if tuple(row) != expected:
            raise ValueError(
                f"run {self.config.run_id!r} uses different code, model, or config"
            )


    def _ensure_schema(self) -> None:
        connection = sqlite3.connect(self.path, timeout=60.0)
        try:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if tables and version != DATABASE_SCHEMA_VERSION:
                raise ValueError(
                    "unsupported LiMOSAT SQLite schema "
                    f"{version or 'legacy'}; global catalogue runs require a new "
                    f"schema-v{DATABASE_SCHEMA_VERSION} database path and run_id"
                )
            if not tables:
                connection.executescript(_SCHEMA)
                connection.execute(
                    f"PRAGMA user_version={DATABASE_SCHEMA_VERSION}"
                )
                connection.commit()
            else:
                connection.executescript(_SCHEMA)
                connection.commit()
        finally:
            connection.close()


    def _ensure_run(self) -> None:
        config_json = json.dumps(
            self.config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT schema_version,config_sha256,implementation_sha256,model_sha256
                FROM runs WHERE run_id=?
                """,
                (self.config.run_id,),
            ).fetchone()
            expected = (
                DATABASE_SCHEMA_VERSION,
                self.config.sha256,
                self.implementation_sha256,
                self.model_sha256,
            )
            if row is not None and tuple(row) != expected:
                raise ValueError(
                    f"run {self.config.run_id!r} uses different code, model, or config"
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO runs
                (run_id,schema_version,config_sha256,config_json,
                 implementation_sha256,model_sha256,status,created_utc)
                VALUES (?,?,?,?,?,?, 'created', ?)
                """,
                (
                    self.config.run_id,
                    DATABASE_SCHEMA_VERSION,
                    self.config.sha256,
                    config_json,
                    self.implementation_sha256,
                    self.model_sha256,
                    _utc_now(),
                ),
            )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def implementation_sha256() -> str:
    """Hash package source so resume cannot mix different local code."""
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()




def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  schema_version INTEGER NOT NULL,
  config_sha256 TEXT NOT NULL,
  config_json TEXT NOT NULL,
  implementation_sha256 TEXT NOT NULL,
  model_sha256 TEXT,
  status TEXT NOT NULL CHECK(status IN ('created','running','complete','failed')),
  created_utc TEXT NOT NULL,
  started_utc TEXT,
  completed_utc TEXT,
  error TEXT
);
CREATE TABLE IF NOT EXISTS images (
  run_id TEXT NOT NULL, image_id TEXT NOT NULL, component_id TEXT NOT NULL,
  platform TEXT, absolute_orbit INTEGER, path TEXT NOT NULL,
  time_utc TEXT NOT NULL, size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
  PRIMARY KEY(run_id,image_id),
  FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS candidate_pairs (
  run_id TEXT NOT NULL, pair_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
  selection TEXT NOT NULL CHECK(selection IN ('candidate','primary')),
  planning_component_id TEXT NOT NULL,
  source_component_id TEXT NOT NULL, target_component_id TEXT NOT NULL,
  source_image_id TEXT NOT NULL, target_image_id TEXT NOT NULL,
  source_time_utc TEXT NOT NULL, target_time_utc TEXT NOT NULL,
  elapsed_seconds REAL NOT NULL, overlap_fraction REAL, overlap_area_m2 REAL,
  skipped_images INTEGER NOT NULL,
  PRIMARY KEY(run_id,pair_id), UNIQUE(run_id,ordinal),
  FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS planning_counts (
  run_id TEXT NOT NULL, reason TEXT NOT NULL, count INTEGER NOT NULL,
  PRIMARY KEY(run_id,reason), FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
"""
