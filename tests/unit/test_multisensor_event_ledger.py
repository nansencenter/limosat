import numpy as np
import pandas as pd
from pyproj import Transformer

from experiments.multisensor_event_ledger import (
    assign_along_track_bins,
    deterministic_checkpoints,
    exact_common_support,
    interval_fraction,
    reverse_displacement_vectors,
    selection_flow_table,
)
from experiments.validate_icesat2_deformation import (
    TriangleDisplacementField,
    colocate_method,
)


PAIR_START = pd.Timestamp("2020-03-28T12:13:29Z")
PAIR_END = pd.Timestamp("2020-03-29T11:16:05Z")


def constant_field(dx_m=0.0, dy_m=0.0):
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
    return TriangleDisplacementField.build(vectors, maximum_edge_m=15_000.0), vectors


def observation_at(x, y, fraction=0.5):
    return pd.DataFrame(
        {
            "time_utc": [PAIR_START + fraction * (PAIR_END - PAIR_START)],
            "laser_x": [x],
            "laser_y": [y],
        }
    )


def test_zero_motion_registration_preserves_observed_coordinate():
    field, _ = constant_field()
    result = colocate_method(
        observation_at(4000.0, 6000.0), field, PAIR_START, PAIR_END, "test"
    )
    assert result.loc[0, "test_available"]
    np.testing.assert_allclose(
        result.loc[0, ["test_source_x", "test_source_y"]], [4000, 6000]
    )
    np.testing.assert_allclose(
        result.loc[0, ["test_target_x", "test_target_y"]], [4000, 6000]
    )


def test_known_constant_translation_recovers_source_and_advection_fraction():
    field, _ = constant_field(1200.0, -400.0)
    alpha = 0.75
    observed = np.array([4000.0, 6000.0]) + alpha * np.array([1200.0, -400.0])
    result = colocate_method(
        observation_at(*observed, fraction=alpha), field, PAIR_START, PAIR_END, "test"
    )
    np.testing.assert_allclose(
        result.loc[0, ["test_source_x", "test_source_y"]], [4000, 6000]
    )
    np.testing.assert_allclose(
        result.loc[0, ["test_drift_to_laser_dx_m", "test_drift_to_laser_dy_m"]],
        alpha * np.array([1200.0, -400.0]),
    )
    np.testing.assert_allclose(
        interval_fraction(
            observation_at(*observed, fraction=alpha)["time_utc"],
            PAIR_START,
            PAIR_END,
        ),
        [alpha],
    )


def test_reverse_time_vectors_negate_translation_and_swap_endpoints():
    _, vectors = constant_field(1200.0, -400.0)
    reversed_vectors = reverse_displacement_vectors(vectors)
    np.testing.assert_allclose(
        reversed_vectors[["source_x", "source_y"]].to_numpy(),
        vectors[["source_x", "source_y"]].to_numpy() + np.array([1200.0, -400.0]),
    )
    np.testing.assert_allclose(
        reversed_vectors[["dx_m", "dy_m"]].to_numpy(),
        -vectors[["dx_m", "dy_m"]].to_numpy(),
    )
    np.testing.assert_allclose(
        reversed_vectors[["source_x", "source_y"]].to_numpy()
        + reversed_vectors[["dx_m", "dy_m"]].to_numpy(),
        vectors[["source_x", "source_y"]].to_numpy(),
    )


def test_epsg3413_round_trip_is_consistent_in_always_xy_order():
    forward = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    inverse = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
    longitude = np.array([-45.0, 15.0, 120.0])
    latitude = np.array([70.0, 80.0, 85.0])
    x, y = forward.transform(longitude, latitude)
    recovered_longitude, recovered_latitude = inverse.transform(x, y)
    np.testing.assert_allclose(recovered_longitude, longitude, atol=1e-10)
    np.testing.assert_allclose(recovered_latitude, latitude, atol=1e-10)


def test_exact_common_support_returns_identical_row_identity_for_both_methods():
    observations = pd.DataFrame(
        {
            "row_id": [10, 11, 12, 13],
            "qc": [True, True, False, True],
            "orb_available": [True, True, True, False],
            "aliked_available": [True, False, True, True],
        }
    )
    common = exact_common_support(
        observations, ["orb_available", "aliked_available"], observations["qc"]
    )
    orb_rows = observations.loc[common, "row_id"].to_numpy()
    aliked_rows = observations.loc[common, "row_id"].to_numpy()
    np.testing.assert_array_equal(orb_rows, [10])
    np.testing.assert_array_equal(orb_rows, aliked_rows)


def test_along_track_bins_are_unique_left_closed_and_preserve_missing_support():
    result = assign_along_track_bins(
        [0.0, 3999.999, 4000.0, 8000.0, np.nan], 4000.0
    )
    assert result.tolist() == [0, 0, 1, 2, pd.NA]
    assigned = pd.DataFrame({"row": np.arange(4), "bin": result.iloc[:4]})
    assert assigned["row"].nunique() == len(assigned)


def test_boundary_and_missing_support_are_excluded_without_extrapolation():
    field, _ = constant_field(100.0, 50.0)
    observations = pd.concat(
        [observation_at(5000.0, 5000.0), observation_at(20_000.0, 20_000.0)],
        ignore_index=True,
    )
    result = colocate_method(observations, field, PAIR_START, PAIR_END, "test")
    assert result["test_available"].tolist() == [True, False]
    assert np.isnan(result.loc[1, "test_pair_dx_m"])
    assert np.isnan(result.loc[1, "test_shear_per_day"])


def test_selection_flow_and_checkpoints_are_deterministic():
    flow = selection_flow_table(
        {
            "candidate_observations": 20,
            "temporally_eligible_observations": 18,
            "product_qc_survivors": 12,
            "spatially_supported_observations": 9,
            "common_method_observations": 7,
            "final_bins": 3,
        },
        "event",
    )
    assert flow["count"].tolist() == [20, 18, 12, 9, 7, 3]

    observations = pd.DataFrame(
        {
            "beam": ["gt1r"] * 5,
            "along_track_m": [0, 1, 2, 3, 4],
            "laser_x": [10, 11, 12, 13, 14],
            "laser_y": [20, 21, 22, 23, 24],
        }
    )
    first = deterministic_checkpoints(observations, "beam", [True] * 5)
    second = deterministic_checkpoints(observations, "beam", [True] * 5)
    pd.testing.assert_frame_equal(first, second)
    assert first["along_track_m"].tolist() == [0, 2, 4]
