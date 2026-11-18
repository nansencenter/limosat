import pandas as pd
import pytest

from experiments.summarize_osisaf_routing_confirmations import (
    agreement_statistics,
    confirmation_row,
    exact_two_sided_sign_test,
    long_gap_statistics,
)


def test_confirmation_row_computes_coverage_change_and_match_ratio():
    summary = {
        "coverage_after_fold_rejection": 0.1,
        "physics_valid_matches": 20,
        "routing": {"source_counts": {"phase": 2}},
        "buoys": {
            "available": 1,
            "correct_within_2km": 1,
            "median_error_m": 100.0,
        },
        "timing_seconds": {"pair_total": 5.0},
    }
    manifest = {
        "case_id": "case",
        "role": "control",
        "elapsed_hours": 48.0,
        "device": "mps",
        "fallback": "same_center",
        "osi455_available_tile_fraction": 0.5,
        "osi455_available_tiles": 1,
        "source_tiles": 2,
        "physics_clipped_tiles": 0,
        "baseline": summary,
        "osisaf_assisted": {
            **summary,
            "coverage_after_fold_rejection": 0.25,
            "physics_valid_matches": 50,
        },
    }

    row = confirmation_row(manifest)

    assert row["coverage_change_percentage_points"] == pytest.approx(15.0)
    assert row["match_ratio"] == pytest.approx(2.5)
    assert not row["baseline_is_direct_phase"]


def test_long_gap_statistics_uses_direct_phase_and_balances_clusters():
    rows = pd.DataFrame(
        {
            "case_id": ["a", "b", "c", "old", "short"],
            "cluster": ["one", "one", "two", "three", "four"],
            "elapsed_hours": [48.0, 72.0, 96.0, 72.0, 24.0],
            "baseline_is_direct_phase": [True, True, True, False, True],
            "coverage_change_percentage_points": [2.0, 4.0, -1.0, 99.0, 99.0],
            "match_ratio": [1.0] * 5,
            "osi455_available_tile_fraction": [1.0] * 5,
            "baseline_buoys_available": [1, 1, 1, 1, 1],
            "assisted_buoys_available": [1, 2, 1, 1, 1],
            "baseline_buoys_within_2km": [1, 1, 1, 1, 1],
            "assisted_buoys_within_2km": [1, 2, 1, 1, 1],
            "baseline_buoy_median_error_m": [20.0, 30.0, 40.0, 1.0, 1.0],
            "assisted_buoy_median_error_m": [10.0, 35.0, 30.0, 1.0, 1.0],
        }
    )

    cases, clusters, statistics = long_gap_statistics(rows)

    assert list(cases["case_id"]) == ["a", "b", "c"]
    assert len(clusters) == 2
    assert statistics["cluster_balanced_mean_coverage_change_percentage_points"] == pytest.approx(1.0)
    assert statistics["assisted_lower_buoy_median_error_cases"] == 2
    assert exact_two_sided_sign_test(9, 2) == pytest.approx(0.0654296875)


def test_agreement_statistics_groups_observable_disagreement():
    cases = pd.DataFrame(
        {
            "case_id": ["agree", "differ", "missing"],
            "coverage_change_percentage_points": [-1.0, 5.0, 0.0],
            "baseline_buoys_available": [2, 0, 0],
            "assisted_buoys_available": [1, 3, 0],
        }
    )
    pair_summary = pd.DataFrame(
        {
            "case_id": ["agree", "differ", "missing"],
            "phase_response": [0.5, 0.01, 0.0],
            "phase_osi455_median_distance_km": [2.0, 25.0, float("nan")],
        }
    )

    grouped = agreement_statistics(cases, pair_summary).set_index(
        "routing_agreement"
    )

    assert grouped.loc["agree_le_5km", "coverage_increased_cases"] == 0
    assert grouped.loc["differ_gt_20km", "assisted_case_buoys_available"] == 3
