import numpy as np

from limosat.learned_drift import DriftField, advect_trajectories, sample_field


def affine_field(
    matrix: np.ndarray,
    offset_m: np.ndarray,
    *,
    size: int = 5,
    spacing_m: float = 4000.0,
    available: np.ndarray | None = None,
) -> DriftField:
    source = np.array(
        [(x * spacing_m, y * spacing_m) for y in range(size) for x in range(size)],
        dtype=np.float64,
    )
    if available is None:
        available = np.ones(len(source), dtype=bool)
    displacement = source @ matrix.T + offset_m
    return DriftField(
        grid_row=np.repeat(np.arange(size), size),
        grid_column=np.tile(np.arange(size), size),
        source_xy_m=source,
        displacement_m=displacement,
        available=available,
        selected_matches=np.full(len(source), 8, dtype=np.int32),
        candidate_matches=np.full(len(source), 12, dtype=np.int32),
        support_radius_m=np.full(len(source), 5000.0),
        maximum_residual_m=np.full(len(source), 80.0),
    )


def test_sample_field_exactly_interpolates_affine_motion():
    matrix = np.array([[0.02, -0.01], [0.015, 0.005]])
    offset = np.array([800.0, -300.0])
    field = affine_field(matrix, offset)
    query = np.array([[1500.0, 2500.0], [7600.0, 8100.0], [13_200.0, 4200.0]])

    sampled = sample_field(field, query, maximum_triangle_edge_m=6400.0)

    assert sampled.available.all()
    np.testing.assert_allclose(sampled.displacement_m, query @ matrix.T + offset)
    np.testing.assert_allclose(sampled.selected_matches, 8.0)
    np.testing.assert_allclose(sampled.support_radius_m, 5000.0)
    np.testing.assert_allclose(sampled.maximum_residual_m, 80.0)


def test_trajectories_follow_current_positions_through_multiple_fields():
    first_matrix = np.array([[0.01, 0.0], [0.0, -0.005]])
    second_matrix = np.array([[0.0, 0.015], [-0.01, 0.0]])
    first_offset = np.array([500.0, 200.0])
    second_offset = np.array([-100.0, 300.0])
    fields = [
        affine_field(first_matrix, first_offset, size=7),
        affine_field(second_matrix, second_offset, size=7),
    ]
    seed = np.array([[4000.0, 4000.0], [10_000.0, 12_000.0]])

    trajectories = advect_trajectories(
        fields, ["a", "b", "c"], 4000.0, seed_xy_m=seed
    )

    first_target = seed + seed @ first_matrix.T + first_offset
    final_target = (
        first_target + first_target @ second_matrix.T + second_offset
    )
    np.testing.assert_allclose(
        trajectories.loc[trajectories.image_index == 1, ["x_m", "y_m"]],
        first_target,
    )
    np.testing.assert_allclose(
        trajectories.loc[trajectories.image_index == 2, ["x_m", "y_m"]],
        final_target,
    )
    assert trajectories.active.all()


def test_unsupported_gap_deactivates_trajectory_without_reappearing():
    available = np.ones(25, dtype=bool)
    available[12] = False
    gapped = affine_field(np.zeros((2, 2)), np.array([100.0, 0.0]), available=available)
    complete = affine_field(np.zeros((2, 2)), np.array([100.0, 0.0]))
    seed = np.array([[8000.0, 8000.0], [2000.0, 2000.0]])

    trajectories = advect_trajectories(
        [gapped, complete], [0, 1, 2], 4000.0, seed_xy_m=seed
    )

    failed = trajectories[trajectories.trajectory_id == 0].reset_index(drop=True)
    assert failed.active.tolist() == [True, False, False]
    assert failed.failure_reason.tolist() == [
        "",
        "outside_supported_field",
        "inactive_previous_step",
    ]
    np.testing.assert_allclose(failed[["x_m", "y_m"]], np.tile(seed[0], (3, 1)))
    survived = trajectories[trajectories.trajectory_id == 1].reset_index(drop=True)
    assert survived.active.all()
    np.testing.assert_allclose(survived.iloc[-1][["x_m", "y_m"]].astype(float), [2200.0, 2000.0])


def test_default_seeds_preserve_first_field_material_point_ids():
    field = affine_field(np.zeros((2, 2)), np.array([250.0, -50.0]), size=3)

    trajectories = advect_trajectories([field], ["source", "target"], 4000.0)

    source = trajectories[trajectories.image_index == 0]
    target = trajectories[trajectories.image_index == 1]
    assert len(source) == 9
    assert source.trajectory_id.tolist() == target.trajectory_id.tolist()
    np.testing.assert_allclose(
        target[["x_m", "y_m"]].to_numpy(),
        source[["x_m", "y_m"]].to_numpy() + [250.0, -50.0],
    )


def test_gap_prediction_is_explicit_and_can_regain_field_support():
    complete = affine_field(np.zeros((2, 2)), np.array([100.0, 0.0]))
    available = np.ones(25, dtype=bool)
    available[12] = False
    gapped = affine_field(
        np.zeros((2, 2)), np.array([100.0, 0.0]), available=available
    )

    trajectories = advect_trajectories(
        [complete, gapped, complete],
        [0, 1, 2, 3],
        4000.0,
        seed_xy_m=np.array([[7900.0, 8000.0]]),
        elapsed_hours=[24.0, 24.0, 24.0],
        maximum_prediction_gap_hours=24.0,
    )

    assert trajectories.active.all()
    assert trajectories.trajectory_state.tolist() == [
        "seed",
        "observed",
        "predicted",
        "field_resupported",
    ]
    assert trajectories.field_observed.tolist() == [False, True, False, True]
    assert trajectories.prediction_gap_hours.tolist() == [0.0, 0.0, 24.0, 0.0]
    np.testing.assert_allclose(
        trajectories[["x_m", "y_m"]],
        [[7900.0, 8000.0], [8000.0, 8000.0], [8100.0, 8000.0], [8200.0, 8000.0]],
    )
