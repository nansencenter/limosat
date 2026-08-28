"""Resumable SQLite and Zarr storage for learned drift image pairs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

from .config import ALIKEDConfig
from .types import DriftField, MotionMatches, PairResult


STORE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ImagePair:
    """Stable image identity and scientific inputs for one drift estimate."""

    source_image_id: str | int
    target_image_id: str | int
    source_path: str
    target_path: str
    elapsed_hours: float
    source_time_utc: str | None = None
    target_time_utc: str | None = None
    prior_displacement_m: tuple[float, float] | None = None
    prior_uncertainty_m: float | None = None

    def __post_init__(self) -> None:
        if not str(self.source_image_id) or not str(self.target_image_id):
            raise ValueError("image IDs cannot be empty")
        if not self.source_path or not self.target_path:
            raise ValueError("image paths cannot be empty")
        if not np.isfinite(self.elapsed_hours) or self.elapsed_hours <= 0:
            raise ValueError("elapsed hours must be positive")
        if self.prior_displacement_m is not None and not np.isfinite(
            self.prior_displacement_m
        ).all():
            raise ValueError("prior displacement must be finite")
        if self.prior_uncertainty_m is not None and (
            not np.isfinite(self.prior_uncertainty_m)
            or self.prior_uncertainty_m <= 0
        ):
            raise ValueError("prior uncertainty must be finite and positive")


class LearnedDriftStore:
    """Persist completed pair observations without the ORB template schema."""

    def __init__(
        self,
        database_path: str | Path,
        zarr_path: str | Path,
        run_name: str,
        config: ALIKEDConfig,
    ) -> None:
        if not run_name:
            raise ValueError("run name cannot be empty")
        self.database_path = Path(database_path)
        self.zarr_path = Path(zarr_path)
        self.run_name = run_name
        self.config = config
        self.config_json = json.dumps(
            asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self.config_sha256 = hashlib.sha256(
            self.config_json.encode("utf-8")
        ).hexdigest()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.zarr_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_database()
        self._ensure_zarr()

    def status(self, pair: ImagePair) -> str | None:
        """Return ``complete``, ``writing``, ``failed``, or None."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT status FROM learned_drift_pairs WHERE pair_key = ?",
                (self._pair_key(pair),),
            ).fetchone()
        return None if row is None else str(row[0])

    def save_pair(
        self,
        pair: ImagePair,
        result: PairResult,
        *,
        overwrite: bool = False,
    ) -> str:
        """Write arrays first and mark the SQLite row complete last."""
        pair_key = self._pair_key(pair)
        if result.prior_displacement_m != pair.prior_displacement_m:
            raise ValueError("pair and result prior displacements differ")
        if self.status(pair) == "complete" and not overwrite:
            return pair_key

        now = _utc_now()
        prior_dx, prior_dy = (
            pair.prior_displacement_m
            if pair.prior_displacement_m is not None
            else (None, None)
        )
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO learned_drift_pairs (
                    pair_key, run_name, source_image_id, target_image_id,
                    source_path, target_path, source_time_utc, target_time_utc,
                    elapsed_hours, prior_dx_m, prior_dy_m, prior_uncertainty_m,
                    status, zarr_group, updated_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'writing', ?, ?)
                ON CONFLICT(pair_key) DO UPDATE SET
                    source_path = excluded.source_path,
                    target_path = excluded.target_path,
                    source_time_utc = excluded.source_time_utc,
                    target_time_utc = excluded.target_time_utc,
                    status = 'writing',
                    updated_utc = excluded.updated_utc,
                    error = NULL
                """,
                (
                    pair_key,
                    self.run_name,
                    str(pair.source_image_id),
                    str(pair.target_image_id),
                    pair.source_path,
                    pair.target_path,
                    pair.source_time_utc,
                    pair.target_time_utc,
                    float(pair.elapsed_hours),
                    prior_dx,
                    prior_dy,
                    pair.prior_uncertainty_m,
                    f"pairs/{pair_key}.zarr.zip",
                    now,
                ),
            )

        try:
            self._write_result(pair_key, result)
        except Exception as error:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    UPDATE learned_drift_pairs
                    SET status = 'failed', updated_utc = ?, error = ?
                    WHERE pair_key = ?
                    """,
                    (_utc_now(), str(error)[:2000], pair_key),
                )
            raise

        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE learned_drift_pairs SET
                    status = 'complete', matching_seconds = ?, field_seconds = ?,
                    match_count = ?, grid_node_count = ?, available_node_count = ?,
                    fold_rejected_count = ?, updated_utc = ?, error = NULL
                WHERE pair_key = ?
                """,
                (
                    float(result.matching_seconds),
                    float(result.field_seconds),
                    len(result.matches),
                    len(result.field),
                    int(result.field.available.sum()),
                    len(result.fold_rejected_indices),
                    _utc_now(),
                    pair_key,
                ),
            )
        return pair_key

    def load_pair(self, pair: ImagePair) -> PairResult | None:
        """Load an exact completed pair, or return None for resume/retry."""
        pair_key = self._pair_key(pair)
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT status, zarr_group, matching_seconds, field_seconds,
                       prior_dx_m, prior_dy_m
                FROM learned_drift_pairs WHERE pair_key = ?
                """,
                (pair_key,),
            ).fetchone()
        if row is None or row["status"] != "complete":
            return None

        archive_path = self.zarr_path / row["zarr_group"]
        with zarr.storage.ZipStore(str(archive_path), mode="r") as archive:
            group = zarr.open_group(store=archive, mode="r")
            matches = group["matches"]
            field = group["field"]
            prior = (
                None
                if row["prior_dx_m"] is None
                else (float(row["prior_dx_m"]), float(row["prior_dy_m"]))
            )
            return PairResult(
                matches=MotionMatches(
                    source_feature_id=np.asarray(matches["source_feature_id"]),
                    source_tile_id=np.asarray(matches["source_tile_id"]),
                    target_tile_id=np.asarray(matches["target_tile_id"]),
                    source_xy_m=np.asarray(matches["source_xy_m"]),
                    target_xy_m=np.asarray(matches["target_xy_m"]),
                    score=np.asarray(matches["score"]),
                ),
                field=DriftField(
                    grid_row=np.asarray(field["grid_row"]),
                    grid_column=np.asarray(field["grid_column"]),
                    source_xy_m=np.asarray(field["source_xy_m"]),
                    displacement_m=np.asarray(field["displacement_m"]),
                    available=np.asarray(field["available"]),
                    selected_matches=np.asarray(field["selected_matches"]),
                    candidate_matches=np.asarray(field["candidate_matches"]),
                    support_radius_m=np.asarray(field["support_radius_m"]),
                    maximum_residual_m=np.asarray(field["maximum_residual_m"]),
                ),
                fold_rejected_indices=np.asarray(
                    group["fold_rejected_indices"]
                ),
                matching_seconds=float(row["matching_seconds"]),
                field_seconds=float(row["field_seconds"]),
                prior_displacement_m=prior,
            )

    def incomplete_pair_keys(self) -> tuple[str, ...]:
        """List pairs that can be safely retried after a stopped run."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT pair_key FROM learned_drift_pairs
                WHERE run_name = ? AND status != 'complete' ORDER BY updated_utc
                """,
                (self.run_name,),
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _ensure_database(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS learned_drift_runs (
                    run_name TEXT PRIMARY KEY,
                    config_sha256 TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    analysis_epsg INTEGER NOT NULL,
                    created_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS learned_drift_pairs (
                    pair_key TEXT PRIMARY KEY,
                    run_name TEXT NOT NULL,
                    source_image_id TEXT NOT NULL,
                    target_image_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    target_path TEXT NOT NULL,
                    source_time_utc TEXT,
                    target_time_utc TEXT,
                    elapsed_hours REAL NOT NULL,
                    prior_dx_m REAL,
                    prior_dy_m REAL,
                    prior_uncertainty_m REAL,
                    status TEXT NOT NULL
                        CHECK(status IN ('writing', 'complete', 'failed')),
                    zarr_group TEXT NOT NULL,
                    matching_seconds REAL,
                    field_seconds REAL,
                    match_count INTEGER,
                    grid_node_count INTEGER,
                    available_node_count INTEGER,
                    fold_rejected_count INTEGER,
                    updated_utc TEXT NOT NULL,
                    error TEXT,
                    FOREIGN KEY(run_name) REFERENCES learned_drift_runs(run_name)
                );
                CREATE INDEX IF NOT EXISTS learned_drift_pairs_run_status
                    ON learned_drift_pairs(run_name, status);
                """
            )
            existing = connection.execute(
                """
                SELECT config_sha256 FROM learned_drift_runs WHERE run_name = ?
                """,
                (self.run_name,),
            ).fetchone()
            if existing is not None and existing[0] != self.config_sha256:
                raise ValueError(
                    f"run {self.run_name!r} already uses a different config"
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO learned_drift_runs
                    (run_name, config_sha256, config_json, analysis_epsg, created_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    self.run_name,
                    self.config_sha256,
                    self.config_json,
                    self.config.analysis_epsg,
                    _utc_now(),
                ),
            )

    def _ensure_zarr(self) -> None:
        root = zarr.open_group(str(self.zarr_path), mode="a")
        existing_run = root.attrs.get("run_name")
        existing_config = root.attrs.get("config_sha256")
        if existing_run is not None and existing_run != self.run_name:
            raise ValueError("Zarr store already belongs to a different run")
        if existing_config is not None and existing_config != self.config_sha256:
            raise ValueError("Zarr store already uses a different config")
        root.attrs.update(
            {
                "format": "limosat.learned_drift",
                "schema_version": STORE_SCHEMA_VERSION,
                "run_name": self.run_name,
                "config_sha256": self.config_sha256,
                "analysis_epsg": self.config.analysis_epsg,
                "coordinate_units": "metres",
            }
        )
        root.require_group("pairs")

    def _pair_key(self, pair: ImagePair) -> str:
        identity = {
            "run_name": self.run_name,
            "source_image_id": str(pair.source_image_id),
            "target_image_id": str(pair.target_image_id),
            "elapsed_hours": float(pair.elapsed_hours),
            "prior_displacement_m": pair.prior_displacement_m,
            "prior_uncertainty_m": pair.prior_uncertainty_m,
        }
        encoded = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _write_result(self, pair_key: str, result: PairResult) -> None:
        pairs_path = self.zarr_path / "pairs"
        final_path = pairs_path / f"{pair_key}.zarr.zip"
        temporary_path = pairs_path / f".{pair_key}.writing.zarr.zip"
        try:
            with zarr.storage.ZipStore(str(temporary_path), mode="w") as archive:
                group = zarr.group(store=archive)
                matches = group.create_group("matches")
                field = group.create_group("field")
                for name in (
                    "source_feature_id",
                    "source_tile_id",
                    "target_tile_id",
                    "source_xy_m",
                    "target_xy_m",
                    "score",
                ):
                    _create_array(matches, name, getattr(result.matches, name))
                for name in (
                    "grid_row",
                    "grid_column",
                    "source_xy_m",
                    "displacement_m",
                    "available",
                    "selected_matches",
                    "candidate_matches",
                    "support_radius_m",
                    "maximum_residual_m",
                ):
                    _create_array(field, name, getattr(result.field, name))
                _create_array(
                    group, "fold_rejected_indices", result.fold_rejected_indices
                )
            temporary_path.replace(final_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def _create_array(group, name: str, values: np.ndarray) -> None:
    array = np.asarray(values)
    first_chunk = min(max(len(array), 1), 65_536)
    chunks = (first_chunk, *array.shape[1:])
    if hasattr(group, "create_array"):
        group.create_array(name, data=array, chunks=chunks, overwrite=True)
    else:  # Zarr 2 compatibility on existing HPC environments.
        group.create_dataset(name, data=array, chunks=chunks, overwrite=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
