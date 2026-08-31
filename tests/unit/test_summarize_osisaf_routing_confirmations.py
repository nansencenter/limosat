import pytest

from experiments.summarize_osisaf_routing_confirmations import confirmation_row


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
