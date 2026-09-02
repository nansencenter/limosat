import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from limosat import load_production_field_replay
from limosat.replay import replay_field_set_sha256


def _production_fixture(root: Path) -> tuple[Path, Path]:
    control = root / "control"
    field_directory = root / "shards" / "chain-000001" / "raw" / "learned_output"
    control.mkdir(parents=True)
    field_directory.mkdir(parents=True)
    field_path = field_directory / "field_4km.csv"
    field_path.write_text(
        "grid_row,grid_column,source_x,source_y,available,selected_vectors,"
        "candidate_count,support_radius_m,proposal_dx_m,proposal_dy_m,"
        "maximum_vector_residual_m\n"
        "0,0,0,0,true,8,12,500,100,0,20\n"
        "0,1,1000,0,false,0,0,500,999,999,30\n"
        "1,0,0,1000,true,8,12,500,100,0,20\n"
        "1,1,1000,1000,true,8,12,500,100,0,20\n"
    )
    checksum = hashlib.sha256(field_path.read_bytes()).hexdigest()
    state_path = control / "state.sqlite"
    with sqlite3.connect(state_path) as connection:
        connection.executescript(
            """
            CREATE TABLE scenes (
              scene_id TEXT PRIMARY KEY, image_id INTEGER NOT NULL UNIQUE,
              time_utc TEXT NOT NULL, relative_path TEXT NOT NULL,
              size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
              orbit_number INTEGER
            );
            CREATE TABLE edges (
              edge_id TEXT PRIMARY KEY, source_scene_id TEXT NOT NULL,
              target_scene_id TEXT NOT NULL, elapsed_seconds REAL NOT NULL,
              overlap_fraction REAL NOT NULL, role TEXT NOT NULL,
              shard_id TEXT NOT NULL, ordinal INTEGER NOT NULL
            );
            CREATE TABLE edge_attempts (
              attempt_id TEXT PRIMARY KEY, edge_id TEXT NOT NULL,
              status TEXT NOT NULL, worker_id TEXT NOT NULL,
              started_utc TEXT NOT NULL, finished_utc TEXT,
              product_relative_path TEXT, product_sha256 TEXT, message TEXT
            );
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        connection.executemany(
            "INSERT INTO scenes VALUES (?,?,?,?,?,?,?)",
            [
                ("scene-a", 1, "2020-04-01T00:00:00Z", "a.tif", 1, "a", 1),
                ("scene-b", 2, "2020-04-02T00:00:00Z", "b.tif", 1, "b", 2),
            ],
        )
        connection.execute(
            "INSERT INTO edges VALUES (?,?,?,?,?,?,?,?)",
            (
                "primary-000001",
                "scene-a",
                "scene-b",
                86_400.0,
                0.5,
                "primary",
                "chain-000001",
                0,
            ),
        )
        connection.execute(
            "INSERT INTO edge_attempts VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "attempt-1",
                "primary-000001",
                "completed",
                "worker",
                "2020-04-03T00:00:00Z",
                "2020-04-03T00:01:00Z",
                str(field_path.relative_to(root)),
                checksum,
                None,
            ),
        )
        connection.execute(
            "INSERT INTO metadata VALUES (?,?)",
            ("source_run_plan_sha256", json.dumps("plan-sha")),
        )
    return field_path, state_path


def test_read_only_production_field_replay_preserves_field_semantics(tmp_path):
    field_path, state_path = _production_fixture(tmp_path)
    before = state_path.stat().st_mtime_ns

    replay = load_production_field_replay(tmp_path)

    assert state_path.stat().st_mtime_ns == before
    assert len(replay.images) == len(replay.edges) + 1 == 2
    assert replay.images[0].time_utc.utcoffset().total_seconds() == 0
    field = replay.edges[0].field
    assert field.source_xy_m.dtype == np.float64
    assert field.displacement_m.dtype == np.float64
    assert field.crs_epsg == 3413
    assert field.available.tolist() == [True, False, True, True]
    np.testing.assert_allclose(field.displacement_m[0], [100.0, 0.0])
    assert np.isnan(field.displacement_m[1]).all()
    assert replay.source_metadata["verified_field_checksums"] is True
    expected = hashlib.sha256(
        b"primary-000001" + hashlib.sha256(field_path.read_bytes()).hexdigest().encode()
    ).hexdigest()
    assert replay_field_set_sha256(replay.fields) == expected


def test_production_field_replay_rejects_changed_completed_field(tmp_path):
    field_path, _ = _production_fixture(tmp_path)
    field_path.write_text(field_path.read_text() + "\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_production_field_replay(tmp_path)


def test_replay_script_creates_new_global_sqlite_product(tmp_path):
    source = tmp_path / "source"
    _production_fixture(source)
    output = tmp_path / "output"
    script = Path(__file__).parents[2] / "scripts" / "replay_global_catalogue_fields.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--source", str(source), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "global-trajectories.sqlite" in completed.stdout
    report = json.loads((output / "field-replay-provenance-v1.json").read_text())
    assert report["label"].startswith("FIELD REPLAY")
    assert report["product_schemas"]["sqlite"] == 3
    assert report["source"]["completed_primary_fields"] == 1
    assert report["global_statistics"]["trajectory_count"] == 3
    assert report["global_statistics"]["trajectory_point_count"] == 6
    with sqlite3.connect(output / "global-trajectories.sqlite") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute(
            "SELECT COUNT(*) FROM trajectory_points WHERE x_m IS NULL OR y_m IS NULL"
        ).fetchone()[0] == 0
