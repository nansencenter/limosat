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
