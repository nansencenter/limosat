import json
import math
from datetime import datetime, timezone

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from limosat import publish  # noqa: E402

pytestmark = pytest.mark.unit

PERIOD = "202305_202407"  # primary window [2023-08-01, 2024-08-01)

# image id -> acquisition time (UTC)
IMAGES = {
    1: "2023-07-30T00:00:00Z", 2: "2023-07-31T12:00:00Z", 3: "2023-08-02T00:00:00Z", 4: "2023-08-04T00:00:00Z",
    5: "2023-09-10T00:00:00Z", 6: "2023-09-11T00:00:00Z", 7: "2024-01-01T00:00:00Z", 8: "2024-01-03T00:00:00Z",
    9: "2024-02-01T00:00:00Z",  # in the catalogue, in no link
}


def scene(image_id):
    stamp = IMAGES[image_id].replace("-", "").replace(":", "").replace("Z", "")
    return f"S1A_EW_GRDM_1SDH_{stamp}_{stamp}_{image_id:06d}_05D023_AB{image_id:02d}"


def utc(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


# (trajectory, link index, source image, target image, x0, y0, x1, y1, interpolated, corr, share, flags)
LINKS = [
    # Trajectory 10 crosses the start of the primary window (1 Aug 2023): its first two steps start before it.
    (10, 0, 1, 2, 0.0, 0.0, 100.0, 0.0, 0, 0.5, 0.9, 0),
    (10, 1, 2, 3, 100.0, 0.0, 200.0, 50.0, 1, 0.6, 0.3, 2),
    (10, 2, 3, 4, 200.0, 50.0, 300.0, 50.0, 0, 0.7, 0.8, 4),
    # Trajectory 11 continues an original trajectory: its first link index is 5.
    (11, 5, 5, 6, 0.0, 10.0, 50.0, 10.0, 0, 0.4, 0.6, 1),
    (11, 6, 6, 7, 50.0, 10.0, 50.0, 60.0, 0, 0.8, 0.7, 0),
    (12, 0, 7, 8, 5.0, 5.0, 25.0, 5.0, 0, 0.9, 0.5, 0),
]


def write_source(path, links=LINKS):
    rows = list(links)
    # The internal file is only loosely ordered, so scramble it and use tiny row groups.
    order = [3, 0, 5, 1, 4, 2][: len(rows)]
    rows = [rows[i] for i in order if i < len(rows)] + [r for i, r in enumerate(rows) if i not in order]
    columns = {name: [] for name in publish.SOURCE_COLUMNS}
    for traj, index, src, tgt, x0, y0, x1, y1, interp, corr, share, flags in rows:
        start, end = utc(IMAGES[src]), utc(IMAGES[tgt])
        distance = math.hypot(x1 - x0, y1 - y0)
        days = (end - start).total_seconds() / 86400
        for name, value in dict(
            qc_trajectory_id=traj, trajectory_link_index=index, source_image_id=src, target_image_id=tgt,
            start_time_utc=start, end_time_utc=end, x0_m=x0, y0_m=y0, x1_m=x1, y1_m=y1, dx_m=x1 - x0, dy_m=y1 - y0,
            speed_m_per_day=distance / days, target_interpolated=interp, target_corr=corr, pair_direct_share=share,
            qc_flags=flags,
        ).items():
            columns[name].append(value)
    kinds = {
        "qc_trajectory_id": pa.int64(), "trajectory_link_index": pa.int32(), "source_image_id": pa.int64(),
        "target_image_id": pa.int64(), "start_time_utc": pa.timestamp("ms", tz="UTC"),
        "end_time_utc": pa.timestamp("ms", tz="UTC"), "target_interpolated": pa.int8(), "target_corr": pa.float64(),
        "pair_direct_share": pa.float32(), "qc_flags": pa.uint16(),
    }
    table = pa.table({name: pa.array(values, type=kinds.get(name, pa.float64())) for name, values in columns.items()})
    pq.write_table(table, path, row_group_size=2)


def write_manifest(folder, **changes):
    parquet = folder / f"{PERIOD}_qc.parquet"
    manifest = {
        "status": "complete", "period": PERIOD, "continuity_violations": 0, "input_links": 8, "kept_links": 6,
        "protocol_id": "internal_protocol", "protocol_sha256": "0" * 64,
        "outputs": {parquet.name: publish.sha256(parquet)}, **changes,
    }
    (folder / f"{PERIOD}_qc.manifest.json").write_text(json.dumps(manifest))


def write_catalog(directory, drop=(), shift=None):
    directory.mkdir(parents=True, exist_ok=True)
    features = []
    for image_id, when in IMAGES.items():
        if image_id in drop:
            continue
        if shift == image_id:
            when = "2023-08-02T00:00:01Z"
        features.append({"properties": {
            "image_id": image_id, "scene_id": scene(image_id), "datetime": when, "orbit_num": 48000 + image_id,
        }})
    (directory / f"{PERIOD}_image_catalog_test.json").write_text(json.dumps({"features": features}))


def make_release(tmp_path, links=LINKS, manifest=None, drop=(), shift=None):
    folder = tmp_path / "source" / PERIOD
    folder.mkdir(parents=True)
    write_source(folder / f"{PERIOD}_qc.parquet", links)
    write_manifest(folder, **(manifest or {}))
    write_catalog(tmp_path / "catalogs", drop, shift)
    return tmp_path / "source", tmp_path / "catalogs", tmp_path / "out"


def build(tmp_path, **kwargs):
    source, catalogs, out = make_release(tmp_path, **kwargs)
    return publish.build(source, catalogs, out, "test-release"), source, catalogs, out


def test_primary_windows_tile_the_ten_periods():
    expected = {
        "201505_201607": ("2015-05-01", "2016-08-01"), "201605_201707": ("2016-08-01", "2017-08-01"),
        "202305_202407": ("2023-08-01", "2024-08-01"), "202405_202507": ("2024-08-01", "2025-08-01"),
    }
    for period, (start, stop) in expected.items():
        window = publish.ownership_window(period)
        assert window[0].strftime("%Y-%m-%d") == start and window[1].strftime("%Y-%m-%d") == stop
    periods = [f"{year}05_{year + 1}07" for year in range(2015, 2025)]
    windows = [publish.ownership_window(p) for p in periods]
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))  # no gaps, no overlaps
    with pytest.raises(ValueError):
        publish.ownership_window("202305_202507")
    with pytest.raises(ValueError):
        publish.ownership_window("2023")


def test_point_table_rows_flags_and_order(tmp_path):
    manifest, _, _, out = build(tmp_path)
    table = pq.read_table(out / f"limosat_{PERIOD}.parquet").to_pylist()
    keys = [(r["time"].strftime("%Y-%m-%dT%H:%M"), r["trajectory_id"], r["point_index"], r["is_primary"]) for r in table]
    assert keys == [
        ("2023-07-30T00:00", 10, 0, False), ("2023-07-31T12:00", 10, 1, False), ("2023-08-02T00:00", 10, 2, False),
        ("2023-08-04T00:00", 10, 3, True), ("2023-09-10T00:00", 11, 0, True), ("2023-09-11T00:00", 11, 1, True),
        ("2024-01-01T00:00", 11, 2, True), ("2024-01-01T00:00", 12, 0, True), ("2024-01-03T00:00", 12, 1, True),
    ]
    assert all(r["period"] == PERIOD for r in table)
    by = {(r["trajectory_id"], r["point_index"]): r for r in table}
    assert by[(10, 0)]["scene"] == scene(1) and by[(10, 3)]["scene"] == scene(4) and by[(12, 1)]["scene"] == scene(8)
    # first observations have no step values
    for key in ((10, 0), (11, 0), (12, 0)):
        first = by[key]
        assert all(first[name] is None for name in
                   ("dx_m", "dy_m", "speed_m_per_day", "interpolated", "corr", "direct_match_share", "flags"))
    # steps carry the arriving link's values (end point attributes)
    step = by[(10, 2)]
    assert (step["x_m"], step["y_m"], step["dx_m"], step["dy_m"]) == (200.0, 50.0, 100.0, 50.0)
    assert step["interpolated"] is True and step["flags"] == 2
    assert step["corr"] == pytest.approx(0.6) and step["direct_match_share"] == pytest.approx(0.3)
    assert by[(10, 1)]["speed_m_per_day"] == pytest.approx(100.0 / 1.5)
    assert by[(10, 3)]["flags"] == 4
    # point_index counts from the trajectory's first kept observation, not from the original link index
    assert [by[(11, i)]["point_index"] for i in range(3)] == [0, 1, 2]
    assert manifest["files"][0]["rows"] == 9 and manifest["files"][0]["trajectories"] == 3
    assert manifest["files"][0]["steps"] == 6 and manifest["files"][0]["primary_steps"] == 4
    assert manifest["files"][0]["primary_rows"] == 6 and manifest["files"][0]["scenes"] == 8


def test_row_groups_follow_time_and_schema_metadata(tmp_path):
    _, _, _, out = build(tmp_path)
    parquet = pq.ParquetFile(out / f"limosat_{PERIOD}.parquet")
    index = parquet.schema_arrow.names.index("time")
    ranges = []
    for group in range(parquet.num_row_groups):
        stats = parquet.metadata.row_group(group).column(index).statistics
        ranges.append((stats.min, stats.max))
    assert len(ranges) >= 3 and all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
    metadata = parquet.schema_arrow.metadata
    assert metadata[b"limosat.schema_version"] == b"2.0.0" and metadata[b"limosat.crs"] == b"EPSG:3413"
    assert json.loads(metadata[b"limosat.flag_bits"]) == publish.FLAG_BITS
    assert json.loads(metadata[b"limosat.primary_window"]) == {
        "start": "2023-08-01T00:00:00Z", "stop_exclusive": "2024-08-01T00:00:00Z"}
    assert metadata[b"limosat.release_id"] == b"test-release" and metadata[b"limosat.period"] == PERIOD.encode()
    assert [f.name for f in parquet.schema_arrow] == list(publish.COLUMN_NAMES)


def test_public_outputs_carry_no_local_paths_or_internal_detail(tmp_path):
    manifest, source, _, out = build(tmp_path)
    public = (out / "manifest.json").read_text() + (out / "DATA_DICTIONARY.md").read_text()
    assert str(tmp_path) not in public and "internal_protocol" not in public and "qc_" not in public
    assert (out / "checksums.sha256").read_text() == f"{manifest['files'][0]['sha256']}  limosat_{PERIOD}.parquet\n"
    sidecar = json.loads((tmp_path / "out.provenance.json").read_text())
    assert sidecar["periods"][PERIOD]["protocol_id"] == "internal_protocol"
    assert str(source) in sidecar["source_root"] and not (out / "out.provenance.json").exists()
    assert "`is_primary`" in (out / "DATA_DICTIONARY.md").read_text()
    assert manifest["totals"] == {"rows": 9, "trajectories": 3, "steps": 6, "primary_rows": 6, "primary_steps": 4}


def test_validate_checks_every_trajectory_against_the_source(tmp_path):
    _, source, catalogs, out = build(tmp_path)
    result = publish.validate(source, catalogs, out, sample_modulus=1)
    assert result[0]["status"] == "passed" and result[0]["sampled_steps"] == 6
    # a source link that no longer matches the published file is detected
    parquet = source / PERIOD / f"{PERIOD}_qc.parquet"
    table = pq.read_table(parquet)
    x1 = table["x1_m"].to_pylist()
    x1[0] += 1.0
    pq.write_table(table.set_column(table.schema.get_field_index("x1_m"), "x1_m", pa.array(x1)), parquet)
    with pytest.raises(ValueError, match="differ from the source"):
        publish.validate(source, catalogs, out, sample_modulus=1)


def test_validate_detects_a_modified_output(tmp_path):
    _, source, catalogs, out = build(tmp_path)
    path = out / f"limosat_{PERIOD}.parquet"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checksum"):
        publish.validate(source, catalogs, out)


def test_catalogue_problems_are_rejected(tmp_path):
    source, catalogs, out = make_release(tmp_path / "a", drop=(4,))
    with pytest.raises(ValueError, match="not in catalogue"):
        publish.build(source, catalogs, out, "test-release")
    assert not list(out.glob("*.parquet*"))  # nothing partial is left behind
    source, catalogs, out = make_release(tmp_path / "b", shift=3)
    with pytest.raises(ValueError, match="different from their catalogue"):
        publish.build(source, catalogs, out, "test-release")


def test_source_problems_are_rejected(tmp_path):
    source, catalogs, out = make_release(tmp_path / "flag", links=[LINKS[0][:-1] + (8,)] + LINKS[1:])
    with pytest.raises(ValueError, match="bits other than"):
        publish.build(source, catalogs, out, "test-release")
    gap = [LINKS[0], LINKS[1][:1] + (3,) + LINKS[1][2:]] + LINKS[2:]
    source, catalogs, out = make_release(tmp_path / "gap", links=gap)
    with pytest.raises(ValueError, match="not consecutive"):
        publish.build(source, catalogs, out, "test-release")
    source, catalogs, out = make_release(tmp_path / "running", manifest={"status": "running"})
    with pytest.raises(ValueError, match="not complete"):
        publish.build(source, catalogs, out, "test-release")
    source, catalogs, out = make_release(tmp_path / "sum", manifest={"outputs": {f"{PERIOD}_qc.parquet": "0" * 64}})
    with pytest.raises(ValueError, match="checksum"):
        publish.build(source, catalogs, out, "test-release")
    publish.build(source, catalogs, out, "test-release", verify_source=False)


def test_existing_output_is_not_overwritten_unless_requested(tmp_path):
    _, source, catalogs, out = build(tmp_path)
    with pytest.raises(FileExistsError):
        publish.build(source, catalogs, out, "test-release")
    again = publish.build(source, catalogs, out, "test-release", overwrite=True)
    assert again["files"][0]["rows"] == 9


def test_data_dictionary_is_generated_from_the_schema():
    text = publish.data_dictionary()
    for column in publish.COLUMN_NAMES:
        assert f"`{column}`" in text
    for name in publish.FLAG_BITS:
        assert name in text
    assert np.array([c.nullable for c in publish.COLUMNS]).any()
