import numpy as np
import pandas as pd

from experiments.validate_icesat2_deformation import (
    TriangleDisplacementField,
    add_atl07_relative_topography,
    atlas_utc,
    circular_shift_test,
    colocate_method,
    compare_common_bins,
    invert_to_source_time,
    json_safe,
    top_fraction_mask,
)


def test_atlas_epoch_conversion_is_utc():
    result = atlas_utc(np.array([0.0, 60.0]))
    assert result[0] == pd.Timestamp("2018-01-01T00:00:00Z")
    assert result[1] == pd.Timestamp("2018-01-01T00:01:00Z")


def test_partial_advection_recovers_source_point_for_affine_field():
    source = np.array(
        [[0.0, 0.0], [10_000.0, 0.0], [0.0, 10_000.0], [10_000.0, 10_000.0]]
    )
    # Translation plus a small, spatially varying displacement.
    displacement = np.column_stack(
        (1000.0 + 0.01 * source[:, 0], -500.0 + 0.02 * source[:, 1])
    )
    vectors = pd.DataFrame(
        {
            "source_x": source[:, 0],
            "source_y": source[:, 1],
            "dx_m": displacement[:, 0],
            "dy_m": displacement[:, 1],
        }
    )
    field = TriangleDisplacementField.build(vectors, maximum_edge_m=15_000.0)
    expected = np.array([[4000.0, 6000.0]])
    alpha = np.array([0.75])
    sampled, available = field.sample_displacement(expected)
    laser = expected + alpha[:, None] * sampled
    recovered, valid, residual = invert_to_source_time(field, laser, alpha)
    assert available.all() and valid.all()
    np.testing.assert_allclose(recovered, expected, atol=1e-6)
    assert residual[0] < 1e-6


def test_deformation_reports_principal_compression_and_extension():
    source = np.array(
        [[0.0, 0.0], [10_000.0, 0.0], [0.0, 10_000.0], [10_000.0, 10_000.0]]
    )
    vectors = pd.DataFrame(
        {
            "source_x": source[:, 0],
            "source_y": source[:, 1],
            "dx_m": 0.10 * source[:, 0],
            "dy_m": -0.20 * source[:, 1],
        }
    )
    field = TriangleDisplacementField.build(vectors, maximum_edge_m=15_000.0)

    deformation, available = field.sample_deformation(
        np.array([[4000.0, 6000.0]]), elapsed_days=1.0
    )

    assert available.all()
    np.testing.assert_allclose(deformation["maximum_compression_per_day"], [0.20])
    np.testing.assert_allclose(deformation["maximum_extension_per_day"], [0.10])


def test_colocation_records_material_shift_uncertainty_and_static_control():
    source = np.array(
        [[0.0, 0.0], [10_000.0, 0.0], [0.0, 10_000.0], [10_000.0, 10_000.0]]
    )
    vectors = pd.DataFrame(
        {
            "source_x": source[:, 0],
            "source_y": source[:, 1],
            "dx_m": 1000.0,
            "dy_m": -500.0,
        }
    )
    field = TriangleDisplacementField.build(vectors, maximum_edge_m=15_000.0)
    pair_start = pd.Timestamp("2020-01-01T00:00:00Z")
    pair_end = pd.Timestamp("2020-01-02T00:00:00Z")
    expected_source = np.array([4000.0, 6000.0])
    alpha = 0.75
    laser = expected_source + alpha * np.array([1000.0, -500.0])
    observations = pd.DataFrame(
        {
            "time_utc": [pair_start + alpha * (pair_end - pair_start)],
            "laser_x": [laser[0]],
            "laser_y": [laser[1]],
        }
    )

    result = colocate_method(
        observations,
        field,
        pair_start,
        pair_end,
        "test",
        endpoint_error_p90_m=1200.0,
    )

    assert result.loc[0, "test_available"]
    assert result.loc[0, "test_static_available"]
    np.testing.assert_allclose(
        result.loc[0, ["test_source_x", "test_source_y"]].to_numpy(float),
        expected_source,
    )
    assert np.isclose(
        result.loc[0, "test_drift_correction_m"],
        alpha * np.hypot(1000.0, 500.0),
    )
    assert np.isclose(result.loc[0, "test_position_error_p90_m"], 900.0)


def test_atl07_ridge_prior_detects_local_peak_after_detrending():
    count = 601
    data = pd.DataFrame(
        {
            "beam": "gt1r",
            "beam_type": "strong",
            "along_track_m": np.arange(count) * 10.0,
            "height_quality": 1,
            "fit_quality": 1,
            "ssh_flag": 0,
            "surface_height_m": np.zeros(count),
        }
    )
    data.loc[count // 2, "surface_height_m"] = 0.8
    result = add_atl07_relative_topography(data)
    assert result.loc[count // 2, "ridge_event"]
    assert result.loc[count // 2, "relative_height_m"] > 0.6


def test_json_safe_represents_undefined_scientific_statistics_as_null():
    assert json_safe({"statistic": np.nan, "count": np.int64(2)}) == {
        "statistic": None,
        "count": 2,
    }


def test_top_fraction_selects_exact_count_when_threshold_values_tie():
    selected = top_fraction_mask(np.zeros(10), fraction=0.2)
    assert selected.sum() == 2


def test_circular_shift_reports_short_beam_as_insufficient():
    bins = pd.DataFrame(
        {
            "beam": ["gt1r"] * 9,
            "track_bin": np.arange(9),
            "predictor": np.arange(9, dtype=float),
            "response": np.arange(9, dtype=float),
        }
    )

    result = circular_shift_test(bins, "predictor", "response")

    assert result["repetitions"] == 0
    assert result["one_sided_p"] is None


def test_circular_shift_uses_physical_distance_at_nondefault_bin_size():
    bins = pd.DataFrame(
        {
            "beam": ["gt1r"] * 50,
            "track_bin": np.arange(50),
            "predictor": np.arange(50, dtype=float),
            "response": np.arange(50, dtype=float),
        }
    )

    result = circular_shift_test(
        bins,
        "predictor",
        "response",
        repetitions=5,
        bin_size_m=1000.0,
        minimum_shift_m=20_000.0,
    )

    assert result["repetitions"] == 5
    assert result["minimum_shift_km"] == 20.0


def test_common_bin_comparison_handles_empty_support():
    orb = pd.DataFrame(
        columns=["beam", "track_bin", "orb_divergence_per_day"]
    )
    aliked = pd.DataFrame(
        columns=["beam", "track_bin", "aliked_divergence_per_day"]
    )

    result = compare_common_bins(orb, aliked)

    assert result == {
        "bins": 0,
        "divergence_spearman": None,
        "median_absolute_divergence_difference_per_day": None,
    }
