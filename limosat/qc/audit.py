"""Bounded audit buffers and aggregate diagnostics for trajectory QC.

Persisted ``qc_flagged_edges``, ``qc_link_sample``, and ``links`` names are
retained for existing SQLite report consumers; they all describe drift vectors.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


VECTOR_COLUMNS = [
    "trajectory_id",
    "source_rowid",
    "target_rowid",
    "source_image_id",
    "target_image_id",
    "source_time",
    "target_time",
    "elapsed_days",
    "x0_m",
    "y0_m",
    "x1_m",
    "y1_m",
    "corr",
    "interpolated",
]

AUDIT_COLUMNS = [
    *VECTOR_COLUMNS,
    "u_m",
    "v_m",
    "magnitude_m",
    "speed_m_per_day",
    "image_gap",
    "coarse_local_residual_m",
    "local_u_m",
    "local_v_m",
    "local_neighbor_count",
    "local_inlier_count",
    "local_inlier_fraction",
    "local_support_radius_m",
    "local_typical_residual_m",
    "local_residual_scale_m",
    "local_residual_m",
    "local_evaluated",
    "local_supported",
    "topology_flip_count",
    "topology_flip_incident",
    "hard_speed_reject",
    "ultra_local_reject",
    "topology_local_reject",
    "speed_local_reject",
    "reject",
    "review",
    "decision_reason",
]

SQL_TYPES = {
    "trajectory_id": "INTEGER",
    "source_rowid": "INTEGER",
    "target_rowid": "INTEGER PRIMARY KEY",
    "source_image_id": "INTEGER",
    "target_image_id": "INTEGER",
    "source_time": "TEXT",
    "target_time": "TEXT",
    "elapsed_days": "REAL",
    "x0_m": "REAL",
    "y0_m": "REAL",
    "x1_m": "REAL",
    "y1_m": "REAL",
    "corr": "REAL",
    "interpolated": "INTEGER",
    "u_m": "REAL",
    "v_m": "REAL",
    "magnitude_m": "REAL",
    "speed_m_per_day": "REAL",
    "image_gap": "INTEGER",
    "coarse_local_residual_m": "REAL",
    "local_u_m": "REAL",
    "local_v_m": "REAL",
    "local_neighbor_count": "INTEGER",
    "local_inlier_count": "INTEGER",
    "local_inlier_fraction": "REAL",
    "local_support_radius_m": "REAL",
    "local_typical_residual_m": "REAL",
    "local_residual_scale_m": "REAL",
    "local_residual_m": "REAL",
    "local_evaluated": "INTEGER",
    "local_supported": "INTEGER",
    "topology_flip_count": "INTEGER",
    "topology_flip_incident": "INTEGER",
    "hard_speed_reject": "INTEGER",
    "ultra_local_reject": "INTEGER",
    "topology_local_reject": "INTEGER",
    "speed_local_reject": "INTEGER",
    "reject": "INTEGER",
    "review": "INTEGER",
    "decision_reason": "TEXT",
}

NUMERIC_BINS = {
    "elapsed_hours": (
        np.array([-np.inf, 0.5, 1, 2, 3, 6, 12, 24, 48, 96, 168, np.inf]),
        ["<0.5", "0.5-1", "1-2", "2-3", "3-6", "6-12", "12-24", "24-48", "48-96", "96-168", ">=168"],
    ),
    "speed_km_per_day": (
        np.array([-np.inf, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 500, np.inf]),
        ["<5", "5-10", "10-15", "15-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80-100", "100-150", "150-200", "200-500", ">=500"],
    ),
    "magnitude_km": (
        np.array([-np.inf, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, np.inf]),
        ["<0.5", "0.5-1", "1-2", "2-3", "3-5", "5-10", "10-20", "20-50", "50-100", ">=100"],
    ),
    "correlation": (
        np.array([-np.inf, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, np.inf]),
        ["<0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0", ">=1.0"],
    ),
    "image_gap": (
        np.array([-np.inf, 1, 2, 3, 4, 5, 6, 11, 31, 101, np.inf]),
        ["<1", "1", "2", "3", "4", "5", "6-10", "11-30", "31-100", ">100"],
    ),
}


def native(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


class DistributionAccumulator:
    def __init__(self):
        self.counts: dict[str, dict[str, list[int]]] = {}
        self.spatial: dict[tuple[int, int], list[int]] = {}

    def update_values(
        self,
        dimension: str,
        values: np.ndarray,
        rejected: np.ndarray,
        review: np.ndarray,
    ) -> None:
        values = np.asarray(values).astype(str)
        unique, inverse = np.unique(values, return_inverse=True)
        totals = np.bincount(inverse)
        rejected_counts = np.bincount(inverse, weights=rejected.astype(int))
        review_counts = np.bincount(inverse, weights=review.astype(int))
        dimension_counts = self.counts.setdefault(dimension, {})
        for index, value in enumerate(unique):
            target = dimension_counts.setdefault(str(value), [0, 0, 0])
            target[0] += int(totals[index])
            target[1] += int(rejected_counts[index])
            target[2] += int(review_counts[index])

    def update_numeric(
        self,
        dimension: str,
        values: np.ndarray,
        rejected: np.ndarray,
        review: np.ndarray,
    ) -> None:
        bin_boundaries, labels = NUMERIC_BINS[dimension]
        values = np.asarray(values, dtype=float)
        codes = np.searchsorted(bin_boundaries[1:-1], values, side="right")
        text = np.asarray(labels, dtype=object)[codes]
        text[~np.isfinite(values)] = "missing"
        self.update_values(dimension, text, rejected, review)

    def update(self, scored: pd.DataFrame) -> None:
        rejected = scored["reject"].to_numpy(bool)
        review = scored["review"].to_numpy(bool)
        count = len(scored)
        if not count:
            return
        self.update_values(
            "target_month",
            scored["target_time"].astype(str).str.slice(0, 7).to_numpy(),
            rejected,
            review,
        )
        self.update_values(
            "target_date",
            scored["target_time"].astype(str).str.slice(0, 10).to_numpy(),
            rejected,
            review,
        )
        self.update_numeric(
            "elapsed_hours", scored["elapsed_days"].to_numpy(float) * 24, rejected, review
        )
        self.update_numeric(
            "speed_km_per_day", scored["speed_m_per_day"].to_numpy(float) / 1_000, rejected, review
        )
        self.update_numeric(
            "magnitude_km", scored["magnitude_m"].to_numpy(float) / 1_000, rejected, review
        )
        self.update_numeric(
            "correlation", scored["corr"].to_numpy(float), rejected, review
        )
        self.update_numeric(
            "image_gap", scored["image_gap"].to_numpy(float), rejected, review
        )
        self.update_values(
            "interpolated",
            np.where(scored["interpolated"].to_numpy(int) == 0, "direct", "interpolated"),
            rejected,
            review,
        )
        self.update_values(
            "topology_flip_incident",
            scored["topology_flip_incident"].astype(str).to_numpy(),
            rejected,
            review,
        )
        self.update_values(
            "local_evaluated",
            scored["local_evaluated"].astype(str).to_numpy(),
            rejected,
            review,
        )
        self.update_values(
            "decision_reason", scored["decision_reason"].to_numpy(), rejected, review
        )
        grid_x = np.floor(scored["x0_m"].to_numpy(float) / 100_000).astype(int)
        grid_y = np.floor(scored["y0_m"].to_numpy(float) / 100_000).astype(int)
        cells, inverse = np.unique(np.column_stack([grid_x, grid_y]), axis=0, return_inverse=True)
        totals = np.bincount(inverse)
        rejected_counts = np.bincount(inverse, weights=rejected.astype(int))
        review_counts = np.bincount(inverse, weights=review.astype(int))
        for index, cell in enumerate(cells):
            target = self.spatial.setdefault(
                (int(cell[0]), int(cell[1])), [0, 0, 0]
            )
            target[0] += int(totals[index])
            target[1] += int(rejected_counts[index])
            target[2] += int(review_counts[index])

    def rows(self):
        for dimension in sorted(self.counts):
            for value, counts in sorted(self.counts[dimension].items()):
                yield (dimension, value, *counts)
        for (grid_x, grid_y), counts in sorted(self.spatial.items()):
            yield ("source_grid_100km", f"{grid_x},{grid_y}", *counts)


class AuditWriter:
    def __init__(self, path: Path, *, resume: bool):
        if path.exists() and not resume:
            raise FileExistsError(path)
        if resume and not path.is_file():
            raise FileNotFoundError(path)
        self.connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=rw", uri=True
        ) if resume else sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.flagged = []
        self.sample = []
        self.pairs = []
        self.images = []
        self.duplicates = []
        if not resume:
            self._create_schema()

    def bind_run(self, identity: dict, *, resume: bool):
        if resume:
            row = self.connection.execute(
                "SELECT value_json FROM qc_metadata WHERE key = 'run_identity'"
            ).fetchone()
            if row is None or json.loads(row[0]) != identity:
                raise ValueError("Audit database does not match the checkpoint run")
        else:
            self.connection.execute(
                "INSERT INTO qc_metadata VALUES ('run_identity', ?)",
                (json.dumps(identity, sort_keys=True),),
            )
            self.connection.commit()

    def validate_checkpoint(self, target_time: str, target_image_id: int, stats: dict):
        """Check persisted coverage before deleting any post-checkpoint rows."""
        boundary = (target_time, target_image_id)
        images, points, links = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(point_rows), 0), COALESCE(SUM(links), 0) "
            "FROM qc_image_summary WHERE (target_time, target_image_id) <= (?, ?)",
            boundary,
        ).fetchone()
        rejected, review = self.connection.execute(
            "SELECT COALESCE(SUM(reject), 0), COALESCE(SUM(review), 0) "
            "FROM qc_flagged_edges WHERE (target_time, target_image_id) <= (?, ?)",
            boundary,
        ).fetchone()
        duplicates = self.connection.execute(
            "SELECT COUNT(*) FROM qc_duplicate_points WHERE (time, image_id) <= (?, ?)",
            boundary,
        ).fetchone()[0]
        observed = dict(images=images, point_rows=points, links=links,
                        rejected=rejected, review=review, duplicate_points=duplicates)
        if any(value != stats[key] for key, value in observed.items()):
            raise ValueError("Audit database coverage does not match the checkpoint")

    def _create_schema(self):
        audit_schema = ", ".join(
            f"{quote_identifier(column)} {SQL_TYPES[column]}" for column in AUDIT_COLUMNS
        )
        self.connection.execute(f"CREATE TABLE IF NOT EXISTS qc_flagged_edges ({audit_schema})")
        self.connection.execute(
            f"CREATE TABLE IF NOT EXISTS qc_link_sample ({audit_schema})"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS qc_pair_summary ("
            "source_image_id INTEGER, target_image_id INTEGER, source_time TEXT, target_time TEXT, "
            "elapsed_days REAL, links INTEGER, supported INTEGER, evaluated INTEGER, rejected INTEGER, "
            "review INTEGER, direct INTEGER, interpolated INTEGER, hard_speed INTEGER, topology_incident INTEGER, "
            "speed_p50 REAL, speed_p95 REAL, speed_max REAL, residual_p95 REAL, "
            "PRIMARY KEY (source_image_id, target_image_id))"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS qc_image_summary ("
            "target_image_id INTEGER PRIMARY KEY, target_time TEXT, point_rows INTEGER, links INTEGER, "
            "source_pairs INTEGER, evaluated INTEGER, supported INTEGER, rejected INTEGER, review INTEGER, "
            "hard_speed INTEGER, topology_incident INTEGER)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS qc_distribution_counts ("
            "dimension TEXT, value TEXT, links INTEGER, rejected INTEGER, review INTEGER, "
            "PRIMARY KEY (dimension, value))"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS qc_metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS qc_duplicate_points ("
            "duplicate_rowid INTEGER PRIMARY KEY, retained_rowid INTEGER, trajectory_id INTEGER, "
            "image_id INTEGER, time TEXT, separation_m REAL, retained_corr REAL, duplicate_corr REAL, "
            "retained_interpolated INTEGER, duplicate_interpolated INTEGER, "
            "retained_is_last INTEGER, duplicate_is_last INTEGER, exact_compact_duplicate INTEGER)"
        )
        self.connection.commit()

    def truncate_after(self, target_time: str, target_image_id: int):
        for table in ("qc_flagged_edges", "qc_link_sample", "qc_pair_summary"):
            self.connection.execute(
                f"DELETE FROM {table} WHERE target_time > ? OR "
                f"(target_time = ? AND target_image_id > ?)",
                (target_time, target_time, target_image_id),
            )
        self.connection.execute(
            "DELETE FROM qc_image_summary WHERE target_time > ? OR "
            "(target_time = ? AND target_image_id > ?)",
            (target_time, target_time, target_image_id),
        )
        self.connection.execute(
            "DELETE FROM qc_duplicate_points WHERE time > ? OR "
            "(time = ? AND image_id > ?)",
            (target_time, target_time, target_image_id),
        )
        # A previously completed scan may be ahead of the restored checkpoint.
        # It must not remain publishable while the replay is partial.
        self.connection.execute("DELETE FROM qc_metadata WHERE key != 'run_identity'")
        self.connection.execute("DELETE FROM qc_distribution_counts")
        self.connection.commit()

    def add_duplicate(self, record: tuple):
        self.duplicates.append(record)

    def add_scored(self, scored: pd.DataFrame, point_rows: int):
        flagged = scored.loc[scored["reject"] | scored["review"], AUDIT_COLUMNS]
        sample = scored.loc[scored["target_rowid"].mod(1_000).eq(0), AUDIT_COLUMNS]
        self.flagged.extend(flagged.itertuples(index=False, name=None))
        self.sample.extend(sample.itertuples(index=False, name=None))
        pairs = pair_summary(scored)
        for row in pairs.itertuples(index=False):
            self.pairs.append(
                (
                    row.source_image_id, row.target_image_id, row.source_time, row.target_time,
                    row.elapsed_days, row.links, row.supported, row.evaluated, row.rejected,
                    row.review, row.direct, row.interpolated, row.hard_speed,
                    row.topology_incident, row.speed_p50, row.speed_p95, row.speed_max,
                    row.residual_p95,
                )
            )
        self.images.append(
            (
                int(scored["target_image_id"].iloc[0]),
                str(scored["target_time"].iloc[0]),
                int(point_rows),
                int(len(scored)),
                int(scored["source_image_id"].nunique()),
                int(scored["local_evaluated"].sum()),
                int(scored["local_supported"].sum()),
                int(scored["reject"].sum()),
                int(scored["review"].sum()),
                int(scored["hard_speed_reject"].sum()),
                int(scored["topology_flip_incident"].sum()),
            )
        )

    def add_empty_image(self, image_id: int, time_text: str, point_rows: int):
        self.images.append((image_id, time_text, point_rows, 0, 0, 0, 0, 0, 0, 0, 0))

    def flush(self):
        placeholders = ",".join("?" for _ in AUDIT_COLUMNS)
        for table, buffer in (
            ("qc_flagged_edges", self.flagged),
            ("qc_link_sample", self.sample),
        ):
            if buffer:
                self.connection.executemany(
                    f"INSERT OR REPLACE INTO {table} VALUES ({placeholders})",
                    (tuple(native(value) for value in row) for row in buffer),
                )
                buffer.clear()
        if self.pairs:
            self.connection.executemany(
                "INSERT OR REPLACE INTO qc_pair_summary VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tuple(native(value) for value in row) for row in self.pairs),
            )
            self.pairs.clear()
        if self.images:
            self.connection.executemany(
                "INSERT OR REPLACE INTO qc_image_summary VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                self.images,
            )
            self.images.clear()
        if self.duplicates:
            self.connection.executemany(
                "INSERT OR REPLACE INTO qc_duplicate_points VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                self.duplicates,
            )
            self.duplicates.clear()
        self.connection.commit()

    def finish(self, distributions: DistributionAccumulator, metadata: dict):
        self.flush()
        self.connection.execute("DELETE FROM qc_distribution_counts")
        self.connection.executemany(
            "INSERT INTO qc_distribution_counts VALUES (?,?,?,?,?)",
            distributions.rows(),
        )
        self.connection.execute("DELETE FROM qc_metadata")
        self.connection.executemany(
            "INSERT INTO qc_metadata VALUES (?,?)",
            [(key, json.dumps(value, sort_keys=True)) for key, value in metadata.items()],
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_qc_flagged_reason ON qc_flagged_edges (reject, decision_reason)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_qc_flagged_pair ON qc_flagged_edges (source_image_id, target_image_id)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_qc_flagged_trajectory ON qc_flagged_edges (trajectory_id, target_rowid)"
        )
        self.connection.commit()

    def close(self):
        self.connection.close()


def pair_summary(vectors: pd.DataFrame) -> pd.DataFrame:
    def finite_quantile(values, quantile):
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        return float(np.quantile(array, quantile)) if len(array) else math.nan

    return (
        vectors.groupby(["source_image_id", "target_image_id"], sort=True)
        .agg(
            source_time=("source_time", "first"),
            target_time=("target_time", "first"),
            elapsed_days=("elapsed_days", "first"),
            links=("target_rowid", "size"),
            evaluated=("local_evaluated", "sum"),
            review=("review", "sum"),
            hard_speed=("hard_speed_reject", "sum"),
            topology_incident=("topology_flip_incident", "sum"),
            supported=("local_supported", "sum"),
            rejected=("reject", "sum"),
            direct=("interpolated", lambda values: int(np.sum(np.asarray(values) == 0))),
            interpolated=("interpolated", "sum"),
            speed_p50=("speed_m_per_day", "median"),
            speed_p95=("speed_m_per_day", lambda values: float(np.quantile(values, 0.95))),
            speed_max=("speed_m_per_day", "max"),
            residual_p95=(
                "local_residual_m", lambda values: finite_quantile(values, 0.95)
            ),
        )
        .reset_index()
    )
