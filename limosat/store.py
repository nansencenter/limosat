"""Durable, deterministic SQLite run state."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .catalog import ImageCatalogue, ImagePair
from .config import RunConfig
from .deformation import DeformationCell
from .models import DisplacementField, PairResult
from .planning import PlannedPair
from .trajectory import TrajectoryPoint


DATABASE_SCHEMA_VERSION = 2


class RunStore:
    """SQLite-backed state whose complete pair products are immutable."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.path = Path(config.database)
        self.implementation_sha256 = implementation_sha256()
        self.model_sha256 = (
            file_sha256(config.matcher.checkpoint)
            if config.matcher.checkpoint
            and Path(config.matcher.checkpoint).is_file()
            else None
        )
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
                    str(image.path),
                    image.time_utc.isoformat(),
                    image.path.stat().st_size,
                    file_sha256(image.path),
                )
            )
        with closing(self._connect()) as connection, connection:
            for row in rows:
                existing = connection.execute(
                    "SELECT path, time_utc, sha256 FROM images WHERE run_id=? AND image_id=?",
                    row[:2],
                ).fetchone()
                if existing is not None and tuple(existing) != (row[3], row[4], row[6]):
                    raise ValueError(f"image identity changed during resume: {row[1]}")
                connection.execute(
                    """
                    INSERT OR IGNORE INTO images
                    (run_id,image_id,component_id,path,time_utc,size_bytes,sha256)
                    VALUES (?,?,?,?,?,?,?)
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
                           elapsed_seconds,overlap_fraction
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
                 elapsed_seconds,overlap_fraction)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )

    def finish_run(
        self, runtime_seconds: float, manifest_path: Path, manifest_sha256: str
    ) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE runs SET status='complete', completed_utc=?, runtime_seconds=?,
                    manifest_path=?, manifest_sha256=?, error=NULL WHERE run_id=?
                """,
                (
                    _utc_now(),
                    float(runtime_seconds),
                    str(manifest_path),
                    manifest_sha256,
                    self.config.run_id,
                ),
            )

    def fail_run(self, error: Exception) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE runs SET status='failed', completed_utc=?, error=? WHERE run_id=?",
                (_utc_now(), str(error)[:2_000], self.config.run_id),
            )

    def claim_pair(
        self,
        pair: ImagePair,
        component_id: str,
        kind: str,
        targeted: bool,
    ) -> bool:
        """Mark an incomplete pair running; return False for immutable completion."""
        if kind not in {"primary", "recovery"}:
            raise ValueError("pair kind must be primary or recovery")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM pairs WHERE run_id=? AND pair_id=?",
                (self.config.run_id, pair.pair_id),
            ).fetchone()
            if row is not None and row[0] == "complete":
                connection.rollback()
                return False
            connection.execute(
                """
                INSERT INTO pairs
                (run_id,pair_id,component_id,source_image_id,target_image_id,
                 source_time_utc,target_time_utc,elapsed_seconds,kind,targeted,status,started_utc)
                VALUES (?,?,?,?,?,?,?,?,?,?, 'running', ?)
                ON CONFLICT(run_id,pair_id) DO UPDATE SET
                    status='running', started_utc=excluded.started_utc,
                    completed_utc=NULL, error=NULL
                """,
                (
                    self.config.run_id,
                    pair.pair_id,
                    component_id,
                    pair.source.image_id,
                    pair.target.image_id,
                    pair.source.time_utc.isoformat(),
                    pair.target.time_utc.isoformat(),
                    pair.elapsed_seconds,
                    kind,
                    int(targeted),
                    _utc_now(),
                ),
            )
            connection.commit()
        return True

    def save_pair(self, pair: ImagePair, result: PairResult) -> bool:
        """Atomically replace only incomplete state and mark completion last."""
        if pair.pair_id != result.field.pair_id:
            raise ValueError("pair identity and field identity differ")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM pairs WHERE run_id=? AND pair_id=?",
                (self.config.run_id, pair.pair_id),
            ).fetchone()
            if row is not None and row[0] == "complete":
                connection.rollback()
                return False
            if row is None:
                connection.rollback()
                raise ValueError("pair must be claimed before it is saved")
            connection.execute(
                "DELETE FROM field_nodes WHERE run_id=? AND pair_id=?",
                (self.config.run_id, pair.pair_id),
            )
            connection.executemany(
                """
                INSERT INTO field_nodes
                (run_id,pair_id,node_index,grid_row,grid_column,x_m,y_m,available,
                 dx_m,dy_m,selected_matches,candidate_matches,support_radius_m,
                 maximum_residual_m)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                self._field_rows(result.field),
            )
            runtime = result.runtime_seconds
            connection.execute(
                """
                UPDATE pairs SET status='complete', completed_utc=?, field_sha256=?,
                    match_count=?, node_count=?, available_node_count=?,
                    fold_rejected_count=?, matcher_calls=?, sampling_seconds=?,
                    matching_seconds=?, field_seconds=?, total_seconds=?, error=NULL
                WHERE run_id=? AND pair_id=?
                """,
                (
                    _utc_now(),
                    result.field.checksum,
                    len(result.matches),
                    len(result.field),
                    int(result.field.available.sum()),
                    len(result.fold_rejected_indices),
                    result.matcher_calls,
                    float(runtime.get("sampling", 0.0)),
                    float(runtime.get("matching", 0.0)),
                    float(runtime.get("field", 0.0)),
                    float(runtime.get("total", 0.0)),
                    self.config.run_id,
                    pair.pair_id,
                ),
            )
            connection.commit()
        return True

    def fail_pair(self, pair_id: str, error: Exception) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE pairs SET status='failed', completed_utc=?, error=?
                WHERE run_id=? AND pair_id=? AND status!='complete'
                """,
                (_utc_now(), str(error)[:2_000], self.config.run_id, pair_id),
            )

    def load_field(self, pair_id: str) -> DisplacementField | None:
        with closing(self._connect()) as connection:
            pair = connection.execute(
                """
                SELECT * FROM pairs WHERE run_id=? AND pair_id=? AND status='complete'
                """,
                (self.config.run_id, pair_id),
            ).fetchone()
            if pair is None:
                return None
            nodes = connection.execute(
                """
                SELECT * FROM field_nodes WHERE run_id=? AND pair_id=? ORDER BY node_index
                """,
                (self.config.run_id, pair_id),
            ).fetchall()
        available = np.asarray([row["available"] for row in nodes], dtype=bool)
        displacement = np.asarray(
            [
                [
                    np.nan if row["dx_m"] is None else row["dx_m"],
                    np.nan if row["dy_m"] is None else row["dy_m"],
                ]
                for row in nodes
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        return DisplacementField(
            pair_id=pair["pair_id"],
            source_image_id=pair["source_image_id"],
            target_image_id=pair["target_image_id"],
            source_time_utc=datetime.fromisoformat(pair["source_time_utc"]),
            target_time_utc=datetime.fromisoformat(pair["target_time_utc"]),
            grid_row=np.asarray([row["grid_row"] for row in nodes]),
            grid_column=np.asarray([row["grid_column"] for row in nodes]),
            source_xy_m=np.asarray(
                [[row["x_m"], row["y_m"]] for row in nodes], dtype=np.float64
            ).reshape(-1, 2),
            displacement_m=displacement,
            available=available,
            selected_matches=np.asarray([row["selected_matches"] for row in nodes]),
            candidate_matches=np.asarray([row["candidate_matches"] for row in nodes]),
            support_radius_m=np.asarray(
                [np.nan if row["support_radius_m"] is None else row["support_radius_m"] for row in nodes]
            ),
            maximum_residual_m=np.asarray(
                [np.nan if row["maximum_residual_m"] is None else row["maximum_residual_m"] for row in nodes]
            ),
        )

    def replace_global_trajectories(
        self, points: Iterable[TrajectoryPoint]
    ) -> None:
        self.replace_global_trajectory_batches((points,))

    def replace_global_trajectory_batches(
        self, batches: Iterable[Iterable[TrajectoryPoint]]
    ) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM trajectory_points WHERE run_id=?",
                (self.config.run_id,),
            )
            connection.execute(
                "DELETE FROM trajectories WHERE run_id=?",
                (self.config.run_id,),
            )
            for batch in batches:
                values = tuple(batch)
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO trajectories
                    (run_id,trajectory_id,seed_image_id) VALUES (?,?,?)
                    """,
                    [
                        (
                            self.config.run_id,
                            point.trajectory_id,
                            point.image_id,
                        )
                        for point in values
                        if point.state == "created"
                    ],
                )
                connection.executemany(
                    """
                    INSERT INTO trajectory_points
                    (run_id,trajectory_id,image_id,time_utc,state,
                     position_basis,x_m,y_m,source_pair_id,selected_matches,
                     support_radius_m,maximum_residual_m)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    [
                        (
                            self.config.run_id,
                            point.trajectory_id,
                            point.image_id,
                            point.time_utc.isoformat(),
                            point.state,
                            point.position_basis,
                            point.x_m,
                            point.y_m,
                            point.source_pair_id,
                            _finite_or_none(point.selected_matches),
                            _finite_or_none(point.support_radius_m),
                            _finite_or_none(point.maximum_residual_m),
                        )
                        for point in values
                    ],
                )

    def replace_deformation(
        self, pair_id: str, cells: Iterable[DeformationCell]
    ) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM deformation_cells WHERE run_id=? AND pair_id=?",
                (self.config.run_id, pair_id),
            )
            connection.executemany(
                """
                INSERT INTO deformation_cells
                (run_id,pair_id,triangle_index,centroid_x_m,centroid_y_m,
                 source_area_m2,divergence_s_1,shear_s_1,total_deformation_s_1,
                 vorticity_s_1)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        self.config.run_id,
                        pair_id,
                        cell.triangle_index,
                        cell.centroid_x_m,
                        cell.centroid_y_m,
                        cell.source_area_m2,
                        cell.divergence_s_1,
                        cell.shear_s_1,
                        cell.total_deformation_s_1,
                        cell.vorticity_s_1,
                    )
                    for cell in cells
                ],
            )

    def manifest_rows(self) -> dict[str, list[dict]]:
        with closing(self._connect()) as connection:
            images = [dict(row) for row in connection.execute(
                "SELECT * FROM images WHERE run_id=? ORDER BY time_utc,image_id",
                (self.config.run_id,),
            )]
            pairs = [dict(row) for row in connection.execute(
                "SELECT * FROM pairs WHERE run_id=? ORDER BY component_id,source_time_utc,target_time_utc",
                (self.config.run_id,),
            )]
            candidate_pairs = [dict(row) for row in connection.execute(
                "SELECT * FROM candidate_pairs WHERE run_id=? ORDER BY ordinal",
                (self.config.run_id,),
            )]
            counts = dict(connection.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM candidate_pairs WHERE run_id=?) candidate_pairs,
                  (SELECT COUNT(*) FROM candidate_pairs
                     WHERE run_id=? AND selection='primary') primary_pairs,
                  (SELECT COUNT(*) FROM trajectories WHERE run_id=?) trajectories,
                  (SELECT COUNT(*) FROM trajectory_points WHERE run_id=?) trajectory_points,
                  (SELECT COUNT(*) FROM deformation_cells WHERE run_id=?) deformation_cells
                """,
                (self.config.run_id,) * 5,
            ).fetchone())
        return {
            "images": images,
            "candidate_pairs": candidate_pairs,
            "pairs": pairs,
            "counts": counts,
        }

    def status(self) -> dict:
        with closing(self._connect()) as connection:
            run = connection.execute(
                "SELECT * FROM runs WHERE run_id=?", (self.config.run_id,)
            ).fetchone()
            pairs = connection.execute(
                "SELECT status,COUNT(*) count FROM pairs WHERE run_id=? GROUP BY status",
                (self.config.run_id,),
            ).fetchall()
        return {"run": dict(run), "pairs": {row[0]: row[1] for row in pairs}}

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

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
                    "schema-v2 database path and run_id"
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

    def _field_rows(self, field: DisplacementField):
        for index in range(len(field)):
            yield (
                self.config.run_id,
                field.pair_id,
                index,
                int(field.grid_row[index]),
                int(field.grid_column[index]),
                float(field.source_xy_m[index, 0]),
                float(field.source_xy_m[index, 1]),
                int(field.available[index]),
                _finite_or_none(field.displacement_m[index, 0]),
                _finite_or_none(field.displacement_m[index, 1]),
                int(field.selected_matches[index]),
                int(field.candidate_matches[index]),
                _finite_or_none(field.support_radius_m[index]),
                _finite_or_none(field.maximum_residual_m[index]),
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


def _finite_or_none(value):
    return None if value is None or not np.isfinite(value) else float(value)


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
  runtime_seconds REAL,
  manifest_path TEXT,
  manifest_sha256 TEXT,
  error TEXT
);
CREATE TABLE IF NOT EXISTS images (
  run_id TEXT NOT NULL, image_id TEXT NOT NULL, component_id TEXT NOT NULL,
  path TEXT NOT NULL, time_utc TEXT NOT NULL, size_bytes INTEGER NOT NULL,
  sha256 TEXT NOT NULL, PRIMARY KEY(run_id,image_id),
  FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS pairs (
  run_id TEXT NOT NULL, pair_id TEXT NOT NULL, component_id TEXT NOT NULL,
  source_image_id TEXT NOT NULL, target_image_id TEXT NOT NULL,
  source_time_utc TEXT NOT NULL, target_time_utc TEXT NOT NULL,
  elapsed_seconds REAL NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('primary','recovery')),
  targeted INTEGER NOT NULL, status TEXT NOT NULL CHECK(status IN ('running','complete','failed')),
  started_utc TEXT NOT NULL, completed_utc TEXT, field_sha256 TEXT,
  match_count INTEGER, node_count INTEGER, available_node_count INTEGER,
  fold_rejected_count INTEGER, matcher_calls INTEGER, sampling_seconds REAL,
  matching_seconds REAL, field_seconds REAL, total_seconds REAL, error TEXT,
  PRIMARY KEY(run_id,pair_id), FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS pairs_run_status ON pairs(run_id,status);
CREATE TABLE IF NOT EXISTS candidate_pairs (
  run_id TEXT NOT NULL, pair_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
  selection TEXT NOT NULL CHECK(selection IN ('candidate','primary')),
  planning_component_id TEXT NOT NULL,
  source_component_id TEXT NOT NULL, target_component_id TEXT NOT NULL,
  source_image_id TEXT NOT NULL, target_image_id TEXT NOT NULL,
  source_time_utc TEXT NOT NULL, target_time_utc TEXT NOT NULL,
  elapsed_seconds REAL NOT NULL, overlap_fraction REAL,
  PRIMARY KEY(run_id,pair_id), UNIQUE(run_id,ordinal),
  FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS field_nodes (
  run_id TEXT NOT NULL, pair_id TEXT NOT NULL, node_index INTEGER NOT NULL,
  grid_row INTEGER NOT NULL, grid_column INTEGER NOT NULL, x_m REAL NOT NULL,
  y_m REAL NOT NULL, available INTEGER NOT NULL, dx_m REAL, dy_m REAL,
  selected_matches INTEGER NOT NULL, candidate_matches INTEGER NOT NULL,
  support_radius_m REAL, maximum_residual_m REAL,
  PRIMARY KEY(run_id,pair_id,node_index),
  FOREIGN KEY(run_id,pair_id) REFERENCES pairs(run_id,pair_id)
);
CREATE TABLE IF NOT EXISTS trajectories (
  run_id TEXT NOT NULL, trajectory_id TEXT NOT NULL,
  seed_image_id TEXT NOT NULL, PRIMARY KEY(run_id,trajectory_id),
  FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS trajectory_points (
  run_id TEXT NOT NULL, trajectory_id TEXT NOT NULL,
  image_id TEXT NOT NULL, time_utc TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('created','observed','dormant','reappeared')),
  position_basis TEXT NOT NULL, x_m REAL, y_m REAL, source_pair_id TEXT,
  selected_matches REAL, support_radius_m REAL, maximum_residual_m REAL,
  PRIMARY KEY(run_id,trajectory_id,image_id),
  FOREIGN KEY(run_id,trajectory_id)
    REFERENCES trajectories(run_id,trajectory_id)
);
CREATE TABLE IF NOT EXISTS deformation_cells (
  run_id TEXT NOT NULL, pair_id TEXT NOT NULL, triangle_index INTEGER NOT NULL,
  centroid_x_m REAL NOT NULL, centroid_y_m REAL NOT NULL, source_area_m2 REAL NOT NULL,
  divergence_s_1 REAL NOT NULL, shear_s_1 REAL NOT NULL,
  total_deformation_s_1 REAL NOT NULL, vorticity_s_1 REAL NOT NULL,
  PRIMARY KEY(run_id,pair_id,triangle_index),
  FOREIGN KEY(run_id,pair_id) REFERENCES pairs(run_id,pair_id)
);
"""
