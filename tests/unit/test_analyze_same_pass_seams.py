import numpy as np

from experiments.analyze_same_pass_seams import (
    edge_pixels,
    move_inward,
    pair_statistics,
)


def test_edge_pixels_and_inward_offset_are_explicit():
    bottom = edge_pixels((100, 200), "bottom", 5)
    moved = move_inward(bottom, (100, 200), "bottom", 16)

    assert np.all(bottom[:, 1] == 99)
    assert np.all(moved[:, 1] == 83)


def test_pair_statistics_preserve_native_uint8_difference_units():
    result = pair_statistics(
        np.array([0.0, 10.0, 20.0]),
        np.array([2.0, 12.0, 22.0]),
    )

    assert result["valid_pairs"] == 3
    assert result["median_bias_dn"] == 2.0
    assert result["mean_absolute_difference_dn"] == 2.0
    assert result["correlation"] == 1.0
