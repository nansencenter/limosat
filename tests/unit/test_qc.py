import numpy as np
import pandas as pd
import pytest

from limosat.qc import qc as limosat_qc
from test_qc_v3 import pair_links, shift


pytestmark = pytest.mark.unit


class NoLand:
    def lookup(self, x, y):
        return np.zeros(len(x), dtype="float32")


class LandAt:
    """Land wherever x is beyond a limit (inland distance 5 km there)."""

    def __init__(self, x_min):
        self.x_min = x_min

    def lookup(self, x, y):
        return np.where(np.asarray(x) >= self.x_min, 5.0, 0.0).astype("float32")


def run(links, land=None):
    return limosat_qc.decide(links, limosat_qc.LimosatQCConfig(), land or NoLand(), None)


def test_protocol_matches_code():
    cfg = limosat_qc.LimosatQCConfig()
    cfg.validate()
    limosat_qc.validate_protocol(cfg)


def test_each_step_removes_its_own_case():
    links = shift(pair_links(), [40], 3_500.0, -2_000.0)  # rogue
    # 2.5 km off, interpolated, weak correlation; its direct neighbour 69 moves the same way, so the
    # pair corroborate each other and neither is a rogue vector
    links = shift(links, [69, 70], 2_500.0, 0.0)
    links.loc[70, "target_interpolated"] = 1
    links.loc[70, "target_corr"] = 0.45
    links.loc[5, "speed_m_per_day"] = 70_000.0             # impossible speed
    d = run(links)
    assert d.loc[5, "qc_exclusion"] == "SPEED"
    assert d.loc[40, "qc_exclusion"] == "ROGUE_VECTOR"
    assert d.loc[70, "qc_exclusion"] == "INTERP_UNSUPPORTED"
    assert set(d.index[d.qc_exclusion != ""]) == {5, 40, 70}


def test_2_to_3_km_needs_correlation_0_50_and_1_to_2_km_needs_0_40():
    links = shift(pair_links(), [69, 70], 2_500.0, 0.0)  # 69 (direct) corroborates 70, so no rogue vector
    links = shift(links, [30], 1_500.0, 0.0)
    links.loc[[30, 70], "target_interpolated"] = 1
    links.loc[[30, 70], "target_corr"] = 0.55
    assert (run(links).qc_exclusion == "").all()
    links.loc[70, "target_corr"] = 0.45
    links.loc[30, "target_corr"] = 0.35
    d = run(links)
    assert d.loc[70, "qc_exclusion"] == d.loc[30, "qc_exclusion"] == "INTERP_UNSUPPORTED"


def test_a_link_removed_by_an_earlier_step_gets_only_that_code_and_leaves_the_neighbour_set():
    links = pair_links()
    links = shift(links, [40], 3_500.0, -2_000.0)
    d = run(links, land=LandAt(links.x0_m.max() - 1))        # last column of the grid is on land
    on_land = links.x0_m >= links.x0_m.max() - 1
    assert (d.qc_exclusion[on_land] == "LAND").all()
    assert d.loc[40, "qc_exclusion"] == "ROGUE_VECTOR"
    assert d.neighbours_30km[on_land].isna().all()


def test_isolated_long_gap_link_is_removed_before_the_neighbour_tests():
    links = pd.concat([pair_links(), pair_links(size=1, id0=500).assign(x0_m=500_000.0, x1_m=506_000.0)],
                      ignore_index=True)
    links.loc[len(links) - 1, "elapsed_seconds"] = 60 * 3600.0
    d = run(links)
    assert d.iloc[-1].qc_exclusion == "ISOLATED_UNVERIFIED"


def test_pair_direct_share_and_flags():
    links = pair_links()
    links.loc[: len(links) * 3 // 4, "target_interpolated"] = 1
    d = run(links)
    kept = d[d.qc_exclusion == ""]
    assert np.allclose(d.pair_direct_share, (links.target_interpolated == 0).mean())
    assert ((kept.qc_flags.to_numpy() >> limosat_qc.FLAG_BITS["LOW_PAIR_DIRECT_SHARE"]) & 1).all()
    assert kept.qc_flag_reasons.str.contains("LOW_PAIR_DIRECT_SHARE").all()


def test_gross_outliers_are_rogue_even_with_two_agreeing_neighbours():
    links = shift(pair_links(), [40, 41, 52], 7_000.0, 0.0)   # three neighbours, all 7 km off together
    links = shift(links, [20, 21], 3_000.0, 0.0)              # 3 km off with one agreeing neighbour
    d = run(links)
    assert (d.loc[[40, 41, 52], "qc_exclusion"] == "ROGUE_VECTOR").all()
    assert (d.loc[[20, 21], "qc_exclusion"] == "").all()


def compact_store(path, points):
    import sqlite3

    con = sqlite3.connect(path)
    con.execute("CREATE TABLE compact_points (source_rowid INTEGER PRIMARY KEY, image_id INTEGER NOT NULL, "
                "is_last INTEGER NOT NULL, trajectory_id INTEGER NOT NULL, geometry TEXT NOT NULL, "
                "time TEXT NOT NULL, corr REAL, interpolated INTEGER)")
    con.executemany("INSERT INTO compact_points VALUES (?,?,?,?,?,?,?,?)", points)
    con.commit()
    con.close()


def test_links_join_consecutive_points_of_each_trajectory_across_batches(tmp_path):
    days = [("2024-03-01 10:00:00", 1), ("2024-03-02 10:00:00", 2), ("2024-04-01 10:00:00", 3)]
    pts, rowid = [], 0
    for k, (day, image) in enumerate(days):
        for traj in (7, 9):
            rowid += 1
            pts.append((rowid, image, int(k == 2), traj, f"POINT ({1000.0 * traj + 500 * k} -2000.0)", day, 0.5,
                        int(traj == 9 and k == 1)))
    compact_store(tmp_path / "c.sqlite", pts)
    counts = limosat_qc.build_links(tmp_path / "c.sqlite", None, tmp_path / "links", batch_rows=3)
    assert counts == {"points": 6, "links": 4}
    files = sorted((tmp_path / "links").glob("*.links.parquet"))
    assert [p.name for p in files] == ["2024-03.links.parquet"]  # a link belongs to the month it starts in
    links = pd.read_parquet(files[0])
    t9 = links[links.original_trajectory_id == 9].sort_values("start_time_utc")
    assert list(t9.dx_m) == [500.0, 500.0]
    assert list(t9.target_interpolated) == [1, 0] and list(t9.source_interpolated) == [0, 1]
    assert np.allclose(links.speed_m_per_day[links.elapsed_seconds == 86400], 500.0)
    assert sorted(t9.trajectory_link_index) == [0, 1]


def test_period_run_splits_at_a_zero_gap_link_and_keeps_continuity(tmp_path):
    # trajectory 1 has two points in image 2 (same time): the zero-gap link between them is impossible
    # (SPEED) and must split the trajectory although its neighbours share its start or end time
    pts = [(1, 1, 0, 1, "POINT (0 0)", "2024-03-01 10:00:00", 0.5, 0),
           (2, 2, 0, 1, "POINT (1000 0)", "2024-03-02 10:00:00", 0.5, 0),
           (3, 2, 0, 1, "POINT (1200 0)", "2024-03-02 10:00:00", 0.5, 0),
           (4, 3, 1, 1, "POINT (2200 0)", "2024-03-03 10:00:00", 0.5, 0)]
    compact_store(tmp_path / "c.sqlite", pts)
    import sqlite3

    raw = sqlite3.connect(tmp_path / "raw.sqlite")
    raw.execute('CREATE TABLE "test" (image_id BIGINT, is_last BIGINT, trajectory_id BIGINT, geometry TEXT, corr FLOAT, '
                'time DATETIME, interpolated BIGINT, orbit_num BIGINT, stopped BOOLEAN, converged_to BIGINT)')
    raw.executemany('INSERT INTO "test" (rowid, image_id, is_last, trajectory_id, geometry, corr, time, interpolated, '
                    'orbit_num, stopped, converged_to) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                    [(r, i, last, t, g, c, tm, ip, 100 + i, int(r == 4), 9 if r == 4 else None)
                     for r, i, last, t, g, tm, c, ip in pts])
    raw.commit()
    raw.close()
    np.savez(tmp_path / "land.npz", inland_km=np.zeros((10, 10), "float32"), x0=-1e6, y1=1e6, res=2e5)
    m = limosat_qc.run(tmp_path / "c.sqlite", tmp_path / "raw.sqlite", "test", tmp_path / "out", tmp_path / "land.npz", None)
    assert m["status"] == "complete" and m["continuity_violations"] == 0
    assert m["removed_by_step"] == {"SPEED": 1} and m["kept_links"] == 2
    kept = pd.read_parquet(tmp_path / "out" / "test_qc.parquet").sort_values("trajectory_link_index")
    assert kept.qc_trajectory_id.nunique() == 2 and list(kept.qc_segment) == [0, 1]
    assert list(kept.target_stopped) == [0, 1] and kept.target_converged_to.iloc[-1] == 9
    assert kept.source_converged_to.isna().all() and list(kept.target_orbit_num) == [102, 103]
