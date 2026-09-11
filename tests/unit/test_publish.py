import json
import sqlite3

import pytest

from limosat import publish


pytestmark = pytest.mark.unit


def frozen_qc_database(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            'CREATE TABLE "window__qc" ('
            'image_id INTEGER, is_last INTEGER, trajectory_id INTEGER, geometry TEXT, '
            'time TEXT, interpolated INTEGER)'
        )
        connection.executemany(
            'INSERT INTO "window__qc" VALUES (?,?,?,?,?,?)',
            [
                # Direct accepted link.
                (1, 0, 10, 'POINT (0 0)', '2024-01-01T00:00:00Z', 0),
                (2, 1, 10, 'POINT (100 0)', '2024-01-02T00:00:00Z', 0),
                # Accepted link with an interpolated source and target endpoint.
                (1, 0, 11, 'POINT (0 10)', '2024-01-01T00:00:00Z', 1),
                (2, 1, 11, 'POINT (0 110)', '2024-01-02T00:00:00Z', 1),
                # A rejected original link is split into post-QC trajectories.
                (1, 1, 12, 'POINT (0 20)', '2024-01-01T00:00:00Z', 0),
                (2, 1, 99, 'POINT (50000 20)', '2024-01-02T00:00:00Z', 0),
                # A second accepted segment, proving the stable post-QC ID is used.
                (3, 0, 99, 'POINT (50100 20)', '2024-01-03T00:00:00Z', 0),
                (4, 1, 99, 'POINT (50200 20)', '2024-01-04T00:00:00Z', 0),
            ],
        )
        connection.execute(
            'CREATE TABLE qc_materialization_manifest (manifest_json TEXT NOT NULL)'
        )
        connection.execute(
            'INSERT INTO qc_materialization_manifest VALUES (?)',
            (json.dumps({
                'status': 'complete', 'quick_check': 'ok',
                'invalid_is_last_trajectories': 0,
                'repeated_trajectory_image_groups': 0,
            }),),
        )
        connection.execute('CREATE TABLE qc_pair_summary (links INTEGER, rejected INTEGER)')
        connection.execute('INSERT INTO qc_pair_summary VALUES (4, 1)')
        connection.execute(
            'CREATE TABLE qc_break_assignment (source_rowid INTEGER, target_rowid INTEGER)'
        )
        connection.execute('INSERT INTO qc_break_assignment VALUES (5, 6)')


def test_publish_materializes_accepted_direct_and_interpolated_edges(tmp_path):
    pytest.importorskip('pyarrow')
    source = tmp_path / 'release' / 'window' / 'window_qc.sqlite'
    source.parent.mkdir(parents=True)
    frozen_qc_database(source)

    result = publish.publish(source, release_id='test-release', batch_size=1)
    output = source.parent / 'window_qc_publish.parquet'
    manifest = output.with_suffix('.manifest.json')

    assert result['row_count'] == 3
    assert result['trajectory_count'] == 3
    assert result['validation']['no_edge_across_qc_break'] is True
    assert output.is_file()
    payload = json.loads(manifest.read_text())
    assert payload['input_sqlite_sha256'] == publish.sha256(source)
    assert payload['output_parquet_sha256'] == publish.sha256(output)
    validated = publish.validate(source, release_id='test-release', batch_size=1)
    assert validated['row_count'] == 3

    _, pq = publish._pyarrow()
    rows = pq.read_table(output).to_pylist()
    assert [row['trajectory_id'] for row in rows] == [10, 11, 99]
    assert rows[0]['distance_m'] == 100.0
    assert rows[0]['speed_m_per_day'] == 100.0
    assert rows[1]['source_position_type'] == 'interpolated'
    assert rows[1]['target_position_type'] == 'interpolated'
    assert all(row['trajectory_id'] != 12 or row['target_image_id'] != 2 for row in rows)
    metadata = pq.ParquetFile(output).schema_arrow.metadata
    assert metadata[b'limosat.crs'] == b'EPSG:3413'
    assert metadata[b'limosat.release_id'] == b'test-release'


def test_publish_rejects_output_when_qc_break_is_not_split(tmp_path):
    pytest.importorskip('pyarrow')
    source = tmp_path / 'window_qc.sqlite'
    frozen_qc_database(source)
    with sqlite3.connect(source) as connection:
        connection.execute('UPDATE "window__qc" SET is_last = 0 WHERE rowid = 5')
        connection.execute('UPDATE "window__qc" SET trajectory_id = 12 WHERE rowid = 6')

    with pytest.raises(ValueError, match='QC break was selected'):
        publish.publish(source)
