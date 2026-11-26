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
        # trajectory 5 is a tree. From the first observation R (0 h, scene 0) one branch runs R -> a1 (2 h) -> a2 (5 h) -> a3 (8 h), the
        # other R -> b1 (30 h). The latest observation is b1, the deepest one is a3.
        link(5, 0, 1, 0, 2, 1000.0, 2000.0, 1500.0, 2100.0),
        link(5, 1, 2, 2, 5, 1500.0, 2100.0, 1900.0, 2300.0, exceeds=True),
        link(5, 2, 3, 5, 8, 1900.0, 2300.0, 2200.0, 2500.0),
        link(5, 0, 4, 0, 30, 1000.0, 2000.0, 5000.0, 2600.0, local=True),
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


def test_default_publishes_the_whole_tree_with_source_point_index(tmp_path):
    source, out, manifest = build(tmp_path)
    manifest, ds = read(out)
    try:
        assert ds.featureType == "trajectory" and "CF-1.8" in ds.Conventions
        # trajectories 2, 5, 8, 9 are published (7 has only a rejected link); trajectory 5 is a tree of four links from R
        assert list(ds["rowsize"][:]) == [2, 5, 2, 3]
        assert manifest["totals"] == {"rows": 12, "trajectories": 4, "displacements": 8}
        entry = manifest["files"][0]
        assert entry["links_rejected_not_published"] == 2 and entry["links_off_chain_not_published"] == 0
        assert manifest["chain_rule"]["rule"] == "none" and ds.chain_rule.startswith("none")
        # ordered by start time: 2 (-3 h), 5 (0 h), 8 (2 h), 9 (5 h)
        assert list(ds["time_start"][:]) == [(T0 + timedelta(hours=h)).timestamp() for h in (-3, 0, 2, 5)]
        assert list(ds["trajectory_id"][:]) == [0, 1, 2, 3]
        assert list(ds["point_index"][:]) == [0, 1, 0, 1, 2, 3, 4, 0, 1, 0, 1, 2]
        # trajectory 5: R (0 h), a1 (2 h), a2 (5 h), a3 (8 h), b1 (30 h); b1 starts at R, not at the previous row a3
        assert list(ds["source_point_index"][:]) == [-1, 0, -1, 0, 1, 2, 0, -1, 0, -1, 0, 1]
        offset = np.r_[0, np.cumsum(ds["rowsize"][:])]
        x, dx = ds["x_m"][:], ds["dx_m"][:]
        assert x[offset[1] + 4] == 5000.0 and dx[offset[1] + 4] == pytest.approx(4000.0) and np.isnan(dx[offset[1]])
    finally:
        ds.close()


@pytest.mark.parametrize("rule, rowsize, off_chain", [("latest", [2, 2, 2, 3], 3), ("deepest", [2, 4, 2, 3], 1)])
def test_chain_rules_publish_one_chain_per_trajectory(tmp_path, rule, rowsize, off_chain):
    _, out, manifest = build(tmp_path, chain_rule=rule)
    manifest, ds = read(out)
    try:
        assert list(ds["rowsize"][:]) == rowsize
        assert manifest["files"][0]["links_off_chain_not_published"] == off_chain
        assert manifest["chain_rule"]["rule"] == rule
        pidx, source = ds["point_index"][:], ds["source_point_index"][:]
        assert np.array_equal(source[pidx > 0], pidx[pidx > 0] - 1)           # every displacement starts at the previous row
        if rule == "deepest":
            offset = np.r_[0, np.cumsum(ds["rowsize"][:])]
            assert list(ds["x_m"][:][offset[1]:offset[1] + 4]) == [1000.0, 1500.0, 1900.0, 2200.0]
            assert ds["quality_flags"][:][offset[1] + 2] == 1
        else:
            assert 1500.0 not in ds["x_m"][:]                                  # a1 lies off the chain R -> b1
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
        assert flags[offset[1] + 2] == 1                                      # trajectory 5, link to a2: exceeds configured speed
        assert flags[offset[1] + 4] == 2                                      # link to b1: local review
        assert flags[offset[2] + 1] == 0
        ds.set_auto_chartostring(False)
        chars = np.ascontiguousarray(ds["scene_name"][:])
        names = [row.rstrip(b"\x00").decode() for row in chars.view(f"S{chars.shape[1]}").reshape(-1)]
        assert names == [SCENES[i] for i in (5, 0, 1, 2, 3, 4)]               # scenes used, ordered by acquisition time
        assert names[ds["scene_index"][:][offset[3] + 2]] == SCENES[4]
        assert ds["lat"][:].min() > 0 and (ds["lon"][:] >= -180).all()
        assert np.isclose(ds["selected_matches"][:][offset[0] + 1], 12.0)
    finally:
        ds.close()


def test_rejected_links_are_not_published_but_counted(tmp_path):
    _, out, manifest = build(tmp_path)
    provenance = json.loads((tmp_path / "release.provenance.json").read_text())
    assert provenance["run_id"] == "test" and "source_directory" in provenance
    assert manifest["files"][0]["links_rejected_not_published"] == 2


def _brute_force_chain(links, rule):
    """links: (trajectory, start, end). Walk the tree of every trajectory in plain Python."""
    kept = np.zeros(len(links), dtype=bool)
    for traj in {t for t, _, _ in links}:
        mine = [i for i, (t, _, _) in enumerate(links) if t == traj]
        by_end = {links[i][2]: i for i in mine}
        depth = {}

        def depth_of(i):
            if i not in depth:
                parent = by_end.get(links[i][1])
                depth[i] = 1 + (depth_of(parent) if parent is not None else 0)
            return depth[i]
        children = {links[i][1] for i in mine}
        leaves = [i for i in mine if links[i][2] not in children]
        pick = max(leaves, key=(lambda i: links[i][2]) if rule == "latest" else (lambda i: (depth_of(i), links[i][2])))
        while pick is not None:
            kept[pick] = True
            pick = by_end.get(links[pick][1])
    return kept


@pytest.mark.parametrize("rule", ["latest", "deepest"])
def test_select_chain_agrees_with_a_plain_walk_of_random_trees(rule):
    rng = np.random.default_rng(3)
    links = []
    for traj in range(1, 40):
        times = [0]
        for _ in range(int(rng.integers(1, 14))):
            source = int(rng.choice(times))                                    # a link starts at any earlier observation
            end = int(times[-1] + rng.integers(1, 9))
            links.append((traj, source, end))
            times.append(end)
    order = rng.permutation(len(links))
    links = [links[i] for i in order]
    tk = np.array([t for t, _, _ in links], dtype=np.uint64)
    mask, stats = publish.select_chain(tk, np.array([s for _, s, _ in links]), np.array([e for _, _, e in links]), rule)
    assert np.array_equal(mask, _brute_force_chain(links, rule))
    assert stats["trajectories"] == 39 and stats["kept"] == int(mask.sum())
    # every kept link starts at the end of another kept link of its trajectory or is the first of the chain
    ends = {(t, e) for (t, _, e), k in zip(links, mask) if k}
    firsts = [(t, s) for (t, s, e), k in zip(links, mask) if k and (t, s) not in ends]
    assert len(firsts) == 39


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
    records[1] = link(5, 1, 2, 2, 5, 1501.0, 2100.0, 1900.0, 2300.0, exceeds=True)   # starts 1 m away from a1
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
