import numpy as np

from experiments.summarize_fair_orb_aliked_benchmark import (
    error_summary,
    relative_error_summary,
)


def test_error_summary_respects_availability():
    result = error_summary(
        np.array([100.0, 3000.0, 50.0]),
        np.array([True, True, False]),
    )
    assert result["available"] == 2
    assert result["median_error_m"] == 1550.0
    assert result["correct_within_2km"] == 1


def test_relative_error_summary_uses_buoy_pairs_not_individual_points():
    truth = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 200.0]])
    prediction = truth + np.array([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]])
    result = relative_error_summary(
        prediction, truth, np.array([True, True, True])
    )
    assert result["pairs"] == 3
    assert result["median_error_m"] == 0.0
    assert result["p90_error_m"] == 0.0
