import numpy as np
import pandas as pd

from experiments.analyze_multisensor_alignment_sensitivity import (
    apply_track_offset,
    direct_registration,
    piecewise_source_coordinates,
)
from experiments.validate_icesat2_deformation import TriangleDisplacementField


def constant_field(dx_m, dy_m):
    source = np.array(
        [[0.0, 0.0], [10_000.0, 0.0], [0.0, 10_000.0], [10_000.0, 10_000.0]]
    )
    vectors = pd.DataFrame(
        {
            "source_x": source[:, 0],
            "source_y": source[:, 1],
            "dx_m": dx_m,
            "dy_m": dy_m,
        }
    )
    return TriangleDisplacementField.build(vectors, maximum_edge_m=15_000.0)


def test_track_offsets_use_symmetric_along_and_cross_directions():
    observations = pd.DataFrame(
        {
            "beam": ["track"] * 3,
            "along_track_m": [0.0, 1000.0, 2000.0],
            "laser_x": [0.0, 1000.0, 2000.0],
            "laser_y": [50.0, 50.0, 50.0],
        }
    )
    positive = apply_track_offset(observations, "beam", 100.0, 200.0)
    negative = apply_track_offset(observations, "beam", -100.0, -200.0)
    np.testing.assert_allclose(positive["laser_x"], observations["laser_x"] + 100)
    np.testing.assert_allclose(positive["laser_y"], observations["laser_y"] + 200)
    np.testing.assert_allclose(negative["laser_x"], observations["laser_x"] - 100)
    np.testing.assert_allclose(negative["laser_y"], observations["laser_y"] - 200)


def test_direct_registration_recovers_constant_translation_source_support():
    start = pd.Timestamp("2020-01-01T00:00:00Z")
    end = pd.Timestamp("2020-01-02T00:00:00Z")
    alpha = 0.5
    source = np.array([4000.0, 6000.0])
    displacement = np.array([1000.0, -500.0])
    observations = pd.DataFrame(
        {
            "time_utc": [start + alpha * (end - start)],
            "laser_x": [source[0] + alpha * displacement[0]],
            "laser_y": [source[1] + alpha * displacement[1]],
        }
    )
    field = constant_field(*displacement)
    result = direct_registration(observations, {"test": field}, start, end)
    assert result.loc[0, "test_available"]
    assert np.isclose(result.loc[0, "test_shear_per_day"], 0.0)
    assert np.isclose(result.loc[0, "test_divergence_per_day"], 0.0)


def test_piecewise_registration_reverses_both_constant_motion_segments():
    start = pd.Timestamp("2020-01-01T00:00:00Z")
    middle = pd.Timestamp("2020-01-02T00:00:00Z")
    end = pd.Timestamp("2020-01-03T00:00:00Z")
    first = constant_field(1000.0, 0.0)
    second = constant_field(500.0, 0.0)
    observations = pd.DataFrame(
        {
            "time_utc": [end],
            "laser_x": [5500.0],
            "laser_y": [6000.0],
        }
    )
    source, available = piecewise_source_coordinates(
        observations, first, second, start, middle, end
    )
    assert available.all()
    np.testing.assert_allclose(source, [[4000.0, 6000.0]])
