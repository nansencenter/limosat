import numpy as np
import pandas as pd
import pytest

from limosat.qc import v3
from limosat.qc.core import QCConfig


pytestmark = pytest.mark.unit


def pair_links(size=12, seed=3, noise=40.0, pair=(1, 2), id0=0):
    rng = np.random.default_rng(seed)
    axis = np.arange(size, dtype=float) * 5_000.0
    x, y = (a.ravel() for a in np.meshgrid(axis, axis))
    u = 6_000.0 + 0.004 * x + rng.normal(0, noise, len(x))
    v = 2_000.0 - 0.002 * y + rng.normal(0, noise, len(x))
    n = len(x)
    return pd.DataFrame({
        "edge_uid": [f"e{id0 + i}" for i in range(n)],
        "qc_trajectory_id": id0 + np.arange(n, dtype=np.int64),
        "source_rowid": 10 * (id0 + np.arange(n)) + 1,
        "target_rowid": 10 * (id0 + np.arange(n)) + 2,
        "source_image_id": pair[0], "target_image_id": pair[1],
        "start_time_utc": pd.Timestamp("2024-03-01", tz="UTC"),
        "end_time_utc": pd.Timestamp("2024-03-02", tz="UTC"),
        "x0_m": x, "y0_m": y, "dx_m": u, "dy_m": v, "x1_m": x + u, "y1_m": y + v,
        "source_interpolated": np.int8(0), "target_interpolated": np.int8(0),
    })


def shift(links, rows, du, dv):
    links.loc[rows, "dx_m"] += du
    links.loc[rows, "dy_m"] += dv
    links["x1_m"] = links.x0_m + links.dx_m
    links["y1_m"] = links.y0_m + links.dy_m
    return links


def test_isolated_direct_outlier_is_r1_and_interpolated_one_is_only_flagged():
    links = shift(pair_links(), [40, 90], 3_500.0, -2_000.0)
    links.loc[90, "target_interpolated"] = 1
    scored = v3.score_links(links, QCConfig(), v3.V3Config())
    assert set(scored.index[scored.r1]) == {40}
    assert set(scored.index[scored.r1_interpolated]) == {90}
    assert scored.loc[90, "interp_residual"]


def test_residual_in_incoherent_field_is_not_r1():
    links = shift(pair_links(noise=2_500.0, seed=5), [40], 3_500.0, -2_000.0)
    scored = v3.score_links(links, QCConfig(), v3.V3Config())
    assert not scored.loc[40, "r1"]


def test_corroborated_outlier_is_not_r1():
    links = shift(pair_links(), [40, 41], 3_500.0, -2_000.0)  # two neighbours moving together
    scored = v3.score_links(links, QCConfig(), v3.V3Config())
    assert not scored.loc[[40, 41], "r1"].any()


def test_land_exclusion_coastal_flag_and_bitmask(tmp_path):
    links = pair_links()
    inland = np.zeros((100, 100), dtype="float32")
    inland[:, :10] = 5.0  # x < 5 km: deep land
    inland[:, 10:12] = 1.0  # 5 km <= x < 7 km: coastal band
    np.savez(tmp_path / "land.npz", inland_km=inland, x0=-5_000.0, y1=60_000.0, res=1_000.0)
    land = v3.LandRaster(tmp_path / "land.npz")
    decided = v3.decide(v3.score_links(links, QCConfig(), v3.V3Config()), v3.V3Config(), land, None)
    assert decided.land.equals(decided.x0_m < 5_000)  # the x = 0 column starts on land
    assert decided.coastal.equals((decided.x0_m >= 5_000) & (decided.x0_m < 7_000))
    assert (decided.loc[decided.land, "qc3_exclusion"] == "LAND").all()
    bit = 1 << v3.FLAG_BITS["COASTAL"]
    assert ((decided.qc3_flags & bit) > 0).equals(decided.coastal)


def test_assign_segments_splits_at_excluded_links_deterministically():
    t = pd.Timestamp("2024-01-01", tz="UTC")
    links = pd.DataFrame({
        "edge_uid": [f"l{i}" for i in range(6)],
        "qc_trajectory_id": [7] * 6,
        "start_time_utc": [t + pd.Timedelta(days=i) for i in range(6)],
        "source_rowid": [1, 2, 3, 4, 5, 6], "target_rowid": [2, 3, 4, 5, 6, 7],
    })
    excluded = links.iloc[[2, 4]]
    kept = v3.assign_segments(links.drop(index=[2, 4]), excluded, base_id=100)
    assert kept.qc3_segment.tolist() == [0, 0, 1, 2]
    assert kept.qc3_trajectory_id.tolist() == [7, 7, 100, 101]
    assert v3.continuity_violations(kept) == 0
    unsplit = links.assign(qc3_trajectory_id=7)
    assert v3.continuity_violations(unsplit.drop(index=[2])) == 1


def test_bundled_v3_protocol_matches_code():
    v3.validate_v3_protocol(v3.V3Config())
