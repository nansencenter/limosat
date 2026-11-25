import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("pyproj")

from limosat import publish  # noqa: E402

T0 = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
SCENES = [f"S1A_EW_GRDM_1SDH_2020010{d}T000000_2020010{d}T000100_000{d}_00_{d}ABC" for d in range(1, 8)]


def tid(n: int) -> str:
    return "trj_" + f"{n:016x}" + "abcd"


def link(trajectory, source_scene, target_scene, source_hours, target_hours, x0, y0, x1, y1, *, accepted=True,
         exceeds=False, local=False, matches=12.0):
    elapsed = int((target_hours - source_hours) * 3600)
    dx, dy = x1 - x0, y1 - y0
    return {
        "trajectory_id": tid(trajectory), "pair_id": "p", "pair_kind": "primary",
        "source_image_id": SCENES[source_scene], "target_image_id": SCENES[target_scene],
        "start_time_utc": T0 + timedelta(hours=source_hours), "end_time_utc": T0 + timedelta(hours=target_hours),
        "elapsed_seconds": float(elapsed), "x0_m": x0, "y0_m": y0, "x1_m": x1, "y1_m": y1, "dx_m": dx, "dy_m": dy,
        "distance_m": float(np.hypot(dx, dy)), "speed_m_per_day": float(np.hypot(dx, dy) / elapsed * 86400),
        "target_state": "observed", "target_position_basis": "primary_pair_field", "selected_matches": matches,
        "support_radius_m": 1200.5, "maximum_residual_m": 80.25, "local_neighbour_count": 24, "local_support_radius_m": 1.0,
        "local_median_dx_m": 0.0, "local_median_dy_m": 0.0, "local_residual_m": 0.0, "local_robust_z": 0.0,
        "exceeds_configured_speed": exceeds, "hard_speed_reject": not accepted, "local_neighbour_review": local,
        "qc_status": "accept" if accepted and not (exceeds or local) else ("review" if accepted else "reject"),
        "qc_reasons": "", "accepted": accepted,
    }


def rows():
    return [
        # trajectory 5: chain 0 -> 1 -> 2 and a second link from the first observation (a later scene reached from the older one)
        link(5, 0, 1, 0, 2, 1000.0, 2000.0, 1500.0, 2100.0),
        link(5, 1, 2, 2, 5, 1500.0, 2100.0, 1900.0, 2300.0, exceeds=True),
        link(5, 0, 3, 0, 8, 1000.0, 2000.0, 2200.0, 2500.0),
        # trajectory 2: starts earlier, one review link
        link(2, 5, 2, -3, 5, -3000.0, -4000.0, -2800.0, -4200.0, local=True),
        # trajectory 9: starts later, two links
        link(9, 2, 3, 5, 8, 500.0, -500.0, 600.0, -600.0),
        link(9, 3, 4, 8, 30, 600.0, -600.0, 700.0, -900.0),
        # trajectory 7: only a rejected link: not published; trajectory 8: one accepted, then a rejected link (dropped)
        link(7, 1, 2, 2, 5, 0.0, 0.0, 90000.0, 0.0, accepted=False),
        link(8, 1, 2, 2, 5, 3000.0, 3000.0, 3100.0, 3050.0),
        link(8, 2, 3, 5, 8, 3100.0, 3050.0, 99000.0, 3050.0, accepted=False),
    ]


def write_composition(directory, records):
    directory.mkdir(parents=True, exist_ok=True)
    columns = {key: [r[key] for r in records] for key in records[0]}
    schema = {k: (pa.timestamp("us", tz="UTC") if k.endswith("_utc") and k.startswith(("start", "end")) else None) for k in columns}
    arrays = {k: pa.array(v, type=schema[k]) if schema[k] else pa.array(v) for k, v in columns.items()}
    table = pa.table(arrays)
    pq.write_table(table, directory / publish.EXTENSIONS_FILE, row_group_size=3)
    manifest = {"trajectory_extension_schema_version": 1, "counts": {"trajectory_extensions": len(records)},
                "qc": {"configured_speed_m_per_day": 43200.0, "hard_speed_m_per_day": 60000.0, "neighbour_search_radius_m": 20000.0},
                "run_id": "test", "config_sha256": "0" * 64}
    (directory / publish.COMPOSITION_MANIFEST).write_text(json.dumps(manifest))
    return directory


def build(tmp_path, records=None, **kwargs):
    source = write_composition(tmp_path / "composition", records or rows())
    out = tmp_path / "release"
    manifest = publish.build(source, out, "test_v1", overwrite=True, tmp_dir=tmp_path / "spill", sample_modulus=1,
                             bucket_observations=kwargs.pop("bucket_observations", 4), batch_rows=4, **kwargs)
    return source, out, manifest


def read(out):
    manifest = json.loads((out / "manifest.json").read_text())
    ds = netCDF4.Dataset(out / manifest["files"][0]["file"])
    ds.set_auto_maskandscale(False)
    return manifest, ds


def test_build_writes_a_valid_cf_trajectory_file(tmp_path):
    source, out, manifest = build(tmp_path)
    manifest, ds = read(out)
    try:
        assert ds.featureType == "trajectory" and "CF-1.8" in ds.Conventions
        # trajectories 2, 5, 8, 9 are published; 7 (only a rejected link) is not; observations = links + trajectories
        assert list(ds["rowsize"][:]) == [2, 4, 2, 3]
        assert ds.dimensions["obs"].size == 11
        assert manifest["totals"] == {"rows": 11, "trajectories": 4, "displacements": 7}
        # ordered by start time: 2 (-3 h), 5 (0 h), 8 (2 h), 9 (5 h)
        assert list(ds["time_start"][:]) == [(T0 + timedelta(hours=h)).timestamp() for h in (-3, 0, 2, 5)]
        assert list(ds["trajectory_id"][:]) == [0, 1, 2, 3]
        assert list(ds["point_index"][:]) == [0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2]
    finally:
        ds.close()


def test_displacements_start_at_the_recorded_source_observation(tmp_path):
    _, out, _ = build(tmp_path)
    _, ds = read(out)
    try:
        x, y = ds["x_m"][:], ds["y_m"][:]
        source = ds["source_point_index"][:]
        offset = np.r_[0, np.cumsum(ds["rowsize"][:])]
        # trajectory 5 (second in the file): observations at 0 h, 2 h, 5 h and 8 h; the 8 h one starts at the first observation
        a = offset[1]
        assert list(source[a:a + 4]) == [-1, 0, 1, 0]
        assert x[a + 3] == 2200.0 and ds["dx_m"][:][a + 3] == pytest.approx(2200.0 - 1000.0)
        assert ds["dy_m"][:][a + 3] == pytest.approx(500.0)
        assert np.isnan(ds["dx_m"][:][a]) and np.isnan(ds["selected_matches"][:][a])
    finally:
        ds.close()


def test_flags_scenes_and_missing_values(tmp_path):
    _, out, _ = build(tmp_path)
    _, ds = read(out)
    try:
        flags = ds["quality_flags"][:]
        offset = np.r_[0, np.cumsum(ds["rowsize"][:])]
        assert list(ds["quality_flags"].flag_masks) == [1, 2]
        assert flags[offset[0]] == 255 and flags[offset[0] + 1] == 2        # trajectory 2: local review
        assert flags[offset[1] + 2] == 1                                      # trajectory 5: exceeds configured speed
        assert flags[offset[1] + 1] == 0
        ds.set_auto_chartostring(False)
        chars = np.ascontiguousarray(ds["scene_name"][:])
        names = [row.rstrip(b"\x00").decode() for row in chars.view(f"S{chars.shape[1]}").reshape(-1)]
        assert names == [SCENES[i] for i in (5, 0, 1, 2, 3, 4)]               # scenes used, ordered by acquisition time
        assert names[ds["scene_index"][:][offset[3] + 2]] == SCENES[4]
        assert ds["lat"][:].min() > 0 and (ds["lon"][:] >= -180).all()
        assert np.isclose(ds["selected_matches"][:][offset[0] + 1], 12.0)
    finally:
        ds.close()


def test_rejected_displacements_are_not_published_but_counted(tmp_path):
    _, out, manifest = build(tmp_path)
    assert manifest["files"][0]["links_rejected_not_published"] == 2
    provenance = json.loads((tmp_path / "release.provenance.json").read_text())
    assert provenance["run_id"] == "test" and "source_directory" in provenance
    assert "composition" not in json.dumps(manifest["files"][0]["bounds_epsg3413_m"])


def test_validation_detects_a_modified_value(tmp_path):
    source, out, manifest = build(tmp_path)
    entry = manifest["files"][0]
    path = out / entry["file"]
    ds = netCDF4.Dataset(path, "a")
    ds["dx_m"][3] = 123456.0
    ds.close()
    with pytest.raises(ValueError, match="checksum"):
        publish.validate(source, out, 1)
    manifest["files"][0]["sha256"] = publish.sha256(path)
    manifest["files"][0]["bytes"] = path.stat().st_size
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differ from the source"):
        publish.validate(source, out, 1)


def test_a_displacement_that_does_not_start_at_an_observation_stops_the_build(tmp_path):
    records = rows()
    records[1] = link(5, 1, 2, 2, 5, 1501.0, 2100.0, 1900.0, 2300.0)   # starts 1 m away from the previous target
    source = write_composition(tmp_path / "composition", records)
    with pytest.raises(ValueError, match="does not start at an earlier observation"):
        publish.build(source, tmp_path / "release", "test_v1", tmp_dir=tmp_path / "spill", validate_output=False,
                      bucket_observations=4, batch_rows=4)
    assert not (tmp_path / "spill").exists()                                 # scratch files are removed on failure
    assert not list((tmp_path / "release").glob("*.tmp"))


def test_ids_must_have_the_composition_form(tmp_path):
    records = rows()
    records[0]["trajectory_id"] = "trajectory-1"
    source = write_composition(tmp_path / "composition", records)
    with pytest.raises(ValueError, match="trajectory ids"):
        publish.build(source, tmp_path / "release", "test_v1", tmp_dir=tmp_path / "spill", validate_output=False)


def test_single_bucket_and_many_buckets_give_the_same_file(tmp_path):
    _, out_a, _ = build(tmp_path / "a", bucket_observations=1000)
    _, out_b, _ = build(tmp_path / "b", bucket_observations=2)
    _, a = read(out_a)
    _, b = read(out_b)
    try:
        for name in ("x_m", "y_m", "dx_m", "source_point_index", "quality_flags", "scene_index", "time", "rowsize"):
            assert np.array_equal(a[name][:], b[name][:], equal_nan=a[name][:].dtype.kind == "f")
    finally:
        a.close()
        b.close()
