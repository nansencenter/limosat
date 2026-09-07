import sqlite3
import json
import subprocess
import sys
from argparse import Namespace

import pytest

from limosat.qc import archive as streaming


pytestmark = pytest.mark.unit


def fixture_database(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE fixture (image_id INTEGER, is_last INTEGER, trajectory_id INTEGER, "
            "geometry TEXT, descriptors TEXT, corr REAL, time DATETIME, interpolated INTEGER)"
        )
        rows = []
        for image_id, date in enumerate(("2024-01-01", "2024-01-02", "2024-01-03"), start=1):
            for trajectory_id in range(10):
                source_x = trajectory_id * 1_000.0
                if trajectory_id == 0 and image_id >= 2:
                    x = 70_000.0 + (image_id - 2) * 100.0
                else:
                    x = source_x + (image_id - 1) * 100.0
                rows.append(
                    (
                        image_id,
                        int(image_id == 3),
                        trajectory_id,
                        f"POINT ({x} 0)",
                        "descriptor",
                        0.7,
                        date,
                        0,
                    )
                )
        connection.executemany("INSERT INTO fixture VALUES (?,?,?,?,?,?,?,?)", rows)
        connection.execute(
            "CREATE INDEX idx_fixture_traj_last ON fixture (trajectory_id, is_last)"
        )


def nonchronological_rowid_database(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE fixture (image_id INTEGER, is_last INTEGER, trajectory_id INTEGER, "
            "geometry TEXT, descriptors TEXT, corr REAL, time DATETIME, interpolated INTEGER)"
        )
        connection.executemany(
            "INSERT INTO fixture VALUES (?,?,?,?,?,?,?,?)",
            [
                (2, 0, 0, "POINT (70000 0)", "descriptor", 0.7, "2024-01-02", 0),
                (3, 1, 0, "POINT (70100 0)", "descriptor", 0.7, "2024-01-03", 0),
                (1, 0, 0, "POINT (0 0)", "descriptor", 0.7, "2024-01-01", 0),
                (2, 0, 1, "POINT (1100 0)", "descriptor", 0.7, "2024-01-02", 0),
                (3, 1, 1, "POINT (1200 0)", "descriptor", 0.7, "2024-01-03", 0),
                (1, 0, 1, "POINT (1000 0)", "descriptor", 0.7, "2024-01-01", 0),
            ],
        )


def duplicate_point_database(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE fixture (image_id INTEGER, is_last INTEGER, trajectory_id INTEGER, "
            "geometry TEXT, descriptors TEXT, corr REAL, time DATETIME, interpolated INTEGER)"
        )
        connection.executemany(
            "INSERT INTO fixture VALUES (?,?,?,?,?,?,?,?)",
            [
                (1, 0, 0, "POINT (0 0)", "descriptor", 0.7, "2024-01-01", 0),
                (2, 0, 0, "POINT (100 0)", "descriptor", 0.7, "2024-01-02", 0),
                (2, 0, 0, "POINT (100 0)", "descriptor", 0.7, "2024-01-02", 0),
                (3, 1, 0, "POINT (200 0)", "descriptor", 0.7, "2024-01-03", 0),
            ],
        )


def arguments(source, output_dir):
    return Namespace(
        input=source,
        table="fixture",
        output_dir=output_dir,
        resume=False,
        input_sha256=streaming.sha256(source),
        prescreen_residual_m=1_000.0,
        fetch_rows=10,
        flush_images=1,
        progress_images=100,
        checkpoint_images=2,
        maximum_rows=None,
        cleaned_output=output_dir / "cleaned.sqlite",
        compact_database=output_dir / "compact.sqlite",
    )


def test_stream_scan_and_materialization_preserve_suffix_after_break(tmp_path):
    source = tmp_path / "source.sqlite"
    output_dir = tmp_path / "qc"
    fixture_database(source)
    args = arguments(source, output_dir)
    config = streaming.QCConfig(
        configured_speed_m_per_day=50_000.0,
        hard_speed_m_per_day=60_000.0,
    )

    streaming.build_compact_database(args)
    metadata = streaming.scan_archive(args, config)
    materialized = streaming.materialize(args)

    assert metadata["status"] == "complete"
    assert metadata["stats"]["point_rows"] == 30
    assert metadata["stats"]["links"] == 20
    assert metadata["stats"]["rejected"] == 1
    assert metadata["active_trajectories_at_end"] == 0
    assert materialized["status"] == "complete"
    assert materialized["protocol_id"] == streaming.PROTOCOL_ID
    assert materialized["quick_check"] == "ok"
    assert materialized["invalid_is_last_trajectories"] == 0
    with sqlite3.connect(args.cleaned_output) as connection:
        split = connection.execute(
            'SELECT trajectory_id, is_last FROM "fixture__qc" '
            "WHERE rowid IN (1, 11, 21) ORDER BY rowid"
        ).fetchall()
        assignments = connection.execute(
            "SELECT original_trajectory_id, new_trajectory_id FROM qc_break_assignment"
        ).fetchall()
    assert split == [(0, 1), (10, 0), (10, 1)]
    assert assignments == [(0, 10)]
    with sqlite3.connect(args.compact_database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(compact_points)")}
    assert "descriptors" not in columns
    with sqlite3.connect(output_dir / "qc_analysis.sqlite") as connection:
        counts = connection.execute(
            "SELECT SUM(links), SUM(rejected), SUM(review) FROM qc_pair_summary"
        ).fetchone()
    assert counts == (20, 1, 0)


def test_multiple_breaks_preserve_points_and_remap_convergence(tmp_path):
    source = tmp_path / "source.sqlite"
    fixture_database(source)
    with sqlite3.connect(source) as connection:
        connection.execute("ALTER TABLE fixture ADD COLUMN converged_to INTEGER")
        connection.execute(
            "UPDATE fixture SET geometry = 'POINT (140000 0)' "
            "WHERE trajectory_id = 0 AND image_id = 3"
        )
        connection.execute(
            "UPDATE fixture SET converged_to = 0 WHERE trajectory_id = 1"
        )
    args = arguments(source, tmp_path / "qc")
    streaming.build_compact_database(args)
    metadata = streaming.scan_archive(args, streaming.QCConfig())
    streaming.materialize(args)

    assert metadata["stats"]["rejected"] == 2
    assert streaming.sha256(source) == args.input_sha256
    with sqlite3.connect(args.cleaned_output) as connection:
        assert connection.execute(
            'SELECT trajectory_id, is_last FROM "fixture__qc" '
            'WHERE rowid IN (1, 11, 21) ORDER BY time'
        ).fetchall() == [(0, 1), (10, 1), (11, 1)]
        assert connection.execute(
            'SELECT converged_to FROM "fixture__qc" '
            'WHERE trajectory_id = 1 ORDER BY time'
        ).fetchall() == [(0,), (10,), (11,)]
        clean_points = connection.execute(
            'SELECT rowid, image_id, geometry, descriptors, corr, time, interpolated '
            'FROM "fixture__qc" ORDER BY rowid'
        ).fetchall()
    with sqlite3.connect(source) as connection:
        raw_points = connection.execute(
            'SELECT rowid, image_id, geometry, descriptors, corr, time, interpolated '
            'FROM fixture ORDER BY rowid'
        ).fetchall()
    assert clean_points == raw_points


def test_resume_maximum_rows_is_per_invocation(tmp_path):
    source = tmp_path / "source.sqlite"
    output_dir = tmp_path / "qc"
    fixture_database(source)
    args = arguments(source, output_dir)
    args.maximum_rows = 10
    config = streaming.QCConfig(
        configured_speed_m_per_day=50_000.0,
        hard_speed_m_per_day=60_000.0,
    )

    streaming.build_compact_database(args)
    first = streaming.scan_archive(args, config)
    args.resume = True
    second = streaming.scan_archive(args, config)

    assert first["status"] == "partial"
    assert first["stats"]["point_rows"] == 10
    assert second["status"] == "partial"
    assert second["stats"]["point_rows"] == 20


def test_scan_refuses_compact_database_from_another_source_checksum(tmp_path):
    source = tmp_path / "source.sqlite"
    output_dir = tmp_path / "qc"
    fixture_database(source)
    args = arguments(source, output_dir)
    streaming.build_compact_database(args)
    args.input_sha256 = "1" * 64

    with pytest.raises(ValueError, match="Compact database source checksum"):
        streaming.scan_archive(
            args, streaming.QCConfig(configured_speed_m_per_day=50_000.0)
        )


def test_materialization_assigns_suffix_by_time_not_rowid(tmp_path):
    source = tmp_path / "source.sqlite"
    output_dir = tmp_path / "qc"
    nonchronological_rowid_database(source)
    args = arguments(source, output_dir)
    config = streaming.QCConfig(
        configured_speed_m_per_day=50_000.0,
        hard_speed_m_per_day=60_000.0,
    )

    streaming.build_compact_database(args)
    metadata = streaming.scan_archive(args, config)
    materialized = streaming.materialize(args)

    assert metadata["stats"]["rejected"] == 1
    assert materialized["invalid_is_last_trajectories"] == 0
    with sqlite3.connect(args.cleaned_output) as connection:
        split = connection.execute(
            'SELECT image_id, trajectory_id, is_last FROM "fixture__qc" '
            "WHERE rowid IN (1, 2, 3) ORDER BY image_id"
        ).fetchall()
    assert split == [(1, 0, 1), (2, 2, 0), (3, 2, 1)]


def test_exact_duplicate_point_is_preserved_as_singleton(tmp_path):
    source = tmp_path / "source.sqlite"
    output_dir = tmp_path / "qc"
    duplicate_point_database(source)
    args = arguments(source, output_dir)
    config = streaming.QCConfig(
        configured_speed_m_per_day=50_000.0,
        hard_speed_m_per_day=60_000.0,
    )

    streaming.build_compact_database(args)
    metadata = streaming.scan_archive(args, config)
    materialized = streaming.materialize(args)

    assert metadata["stats"]["point_rows"] == 4
    assert metadata["stats"]["links"] == 2
    assert metadata["stats"]["duplicate_points"] == 1
    assert materialized["duplicate_point_assignments"] == 1
    assert materialized["invalid_is_last_trajectories"] == 0
    assert materialized["repeated_trajectory_image_groups"] == 0
    with sqlite3.connect(args.cleaned_output) as connection:
        rows = connection.execute(
            'SELECT rowid, trajectory_id, is_last FROM "fixture__qc" ORDER BY rowid'
        ).fetchall()
    assert rows == [(1, 0, 0), (2, 0, 0), (3, 1, 1), (4, 0, 1)]


def test_resume_refuses_changed_config_without_modifying_audit(tmp_path):
    source = tmp_path / "source.sqlite"
    fixture_database(source)
    args = arguments(source, tmp_path / "qc")
    args.maximum_rows = 10
    streaming.build_compact_database(args)
    streaming.scan_archive(args, streaming.QCConfig(configured_speed_m_per_day=50_000.0))
    audit = args.output_dir / "qc_analysis.sqlite"
    before = streaming.sha256(audit)
    args.resume = True

    with pytest.raises(ValueError, match="Checkpoint source, protocol, code, or configuration"):
        streaming.scan_archive(args, streaming.QCConfig(configured_speed_m_per_day=35_000.0))
    assert streaming.sha256(audit) == before


def test_source_checksum_is_verified_before_preparation(tmp_path):
    source = tmp_path / "source.sqlite"
    fixture_database(source)
    args = arguments(source, tmp_path / "qc")
    args.input_sha256 = "0" * 64
    with pytest.raises(ValueError, match="Source SQLite checksum"):
        streaming.build_compact_database(args)
    assert not args.compact_database.exists()


def test_invalid_terminal_markers_prevent_publication(tmp_path):
    source = tmp_path / "source.sqlite"
    fixture_database(source)
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE fixture SET is_last = 0 WHERE trajectory_id = 1")
    args = arguments(source, tmp_path / "qc")
    streaming.build_compact_database(args)
    streaming.scan_archive(args, streaming.QCConfig(configured_speed_m_per_day=50_000.0))

    with pytest.raises(ValueError, match="invalid is_last"):
        streaming.materialize(args)
    assert not args.cleaned_output.exists()
    assert not (args.output_dir / "materialization_manifest.json").exists()


def test_package_cli_completes_with_preserved_source_and_verifiable_manifest(tmp_path):
    source = tmp_path / "source.sqlite"
    fixture_database(source)
    before = streaming.sha256(source)
    output_dir = tmp_path / "qc"
    result = subprocess.run(
        [sys.executable, "-m", "limosat.qc", "--input", str(source),
         "--table", "fixture", "--input-sha256", before,
         "--configured-speed-m-per-day", "50000", "--output-dir", str(output_dir)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "materialization_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["cleaned_rows"] == 30
    assert manifest["break_assignments"] == 1
    assert streaming.sha256(source) == before
    assert streaming.sha256(output_dir / "source_qc.sqlite") == manifest["output_sha256"]


def test_cli_resume_all_matches_uninterrupted_output(tmp_path):
    source = tmp_path / "source.sqlite"
    fixture_database(source)
    args = arguments(source, tmp_path / "resumed")
    config = streaming.QCConfig(configured_speed_m_per_day=50_000.0)
    args.maximum_rows = 10
    streaming.build_compact_database(args)
    streaming.scan_archive(args, config)

    result = subprocess.run(
        [sys.executable, "-m", "limosat.qc", "--input", str(source),
         "--table", "fixture", "--input-sha256", args.input_sha256,
         "--configured-speed-m-per-day", "50000", "--output-dir", str(args.output_dir),
         "--compact-database", str(args.compact_database),
         "--cleaned-output", str(args.cleaned_output), "--resume"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr

    full = arguments(source, tmp_path / "full")
    streaming.build_compact_database(full)
    streaming.scan_archive(full, config)
    streaming.materialize(full)
    with sqlite3.connect(full.cleaned_output) as connection:
        expected = connection.execute('SELECT rowid, * FROM "fixture__qc" ORDER BY rowid').fetchall()
    with sqlite3.connect(args.cleaned_output) as connection:
        actual = connection.execute('SELECT rowid, * FROM "fixture__qc" ORDER BY rowid').fetchall()
    assert actual == expected


def test_materialization_refuses_source_changed_after_scan(tmp_path):
    source = tmp_path / "source.sqlite"
    fixture_database(source)
    args = arguments(source, tmp_path / "qc")
    streaming.build_compact_database(args)
    streaming.scan_archive(args, streaming.QCConfig(configured_speed_m_per_day=50_000.0))
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE fixture SET corr = 0.9 WHERE rowid = 1")
    with pytest.raises(ValueError, match="Source SQLite checksum"):
        streaming.materialize(args)
    assert not args.cleaned_output.exists()
