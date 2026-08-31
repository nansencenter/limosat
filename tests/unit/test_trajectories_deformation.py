from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from limosat import (
    DisplacementField,
    FieldConfig,
    FieldEdge,
    ImageRecord,
    TrajectoryConfig,
    build_trajectories,
)
from limosat.deformation import deformation_from_field
from limosat.trajectory import targeted_recovery_positions


START = datetime(2020, 1, 1, tzinfo=timezone.utc)
GRID = np.array([[0.0, 0.0], [1_000.0, 0.0], [0.0, 1_000.0], [1_000.0, 1_000.0]])


def _field(source, target, step, displacement, available=None, points=GRID):
    available = np.ones(len(points), dtype=bool) if available is None else np.asarray(available)
    values = np.tile(displacement, (len(points), 1)).astype(float)
    values[~available] = np.nan
    return DisplacementField(
        pair_id=f"{source}__{target}",
        source_image_id=source,
        target_image_id=target,
        source_time_utc=START + timedelta(days=step),
        target_time_utc=START + timedelta(days=step + 1),
        grid_row=np.arange(len(points)),
        grid_column=np.zeros(len(points)),
        source_xy_m=points,
        displacement_m=values,
        available=available,
        selected_matches=np.full(len(points), 8),
        candidate_matches=np.full(len(points), 12),
        support_radius_m=np.full(len(points), 500.0),
        maximum_residual_m=np.full(len(points), 20.0),
    )


def _images(names=("a", "b", "c")):
    return [
        ImageRecord(name, Path("/tmp/unused.tif"), START + timedelta(days=index), "c")
        for index, name in enumerate(names)
    ]


def _settings():
    return FieldConfig(
        grid_spacing_m=1_000.0,
        neighbour_count=4,
        minimum_agreeing_matches=3,
        maximum_neighbour_distance_m=2_000.0,
        agreement_distance_m=100.0,
        maximum_triangle_edge_m=1_500.0,
    )


def test_dormant_trajectory_reappears_only_on_observed_skip_edge():
    adjacent = _field("a", "b", 0, [100.0, 0.0])
    empty = _field("b", "c", 1, [0.0, 0.0], available=[False] * 4, points=GRID + [100.0, 0.0])
    skip = _field("b", "d", 1, [200.0, 0.0], points=GRID + [100.0, 0.0])
    edges = [FieldEdge(adjacent), FieldEdge(empty), FieldEdge(skip, skipped_images=1)]

    images = _images(("a", "b", "c", "d"))
    points = build_trajectories(edges, images, _settings(), TrajectoryConfig())
    at_c = [point for point in points if point.image_id == "c"]
    at_d = [point for point in points if point.image_id == "d"]

    assert at_c
    assert {point.state for point in at_c} == {"dormant"}
    assert {point.state for point in at_d} == {"reappeared"}
    assert {point.position_basis for point in at_d} == {"field_advected_skip"}
    first_run_ids = [point.trajectory_id for point in points]
    second_run_ids = [
        point.trajectory_id
        for point in build_trajectories(edges, images, _settings(), TrajectoryConfig())
    ]
    assert first_run_ids == second_run_ids


def test_targeted_recovery_selects_last_measured_source_positions():
    adjacent = _field("a", "b", 0, [100.0, 0.0])
    empty = _field("b", "c", 1, [0.0, 0.0], available=[False] * 4, points=GRID + [100.0, 0.0])
    points = build_trajectories(
        [FieldEdge(adjacent), FieldEdge(empty)],
        _images(),
        _settings(),
        TrajectoryConfig(),
    )

    selected = targeted_recovery_positions(points, "a", "c")

    assert selected.shape == (4, 2)


def test_new_trajectory_is_created_when_outgoing_coverage_enters():
    first = _field("a", "b", 0, [100.0, 0.0])
    expanded_grid = np.vstack((GRID + [100.0, 0.0], GRID + [5_000.0, 0.0]))
    second = _field("b", "c", 1, [100.0, 0.0], points=expanded_grid)

    points = build_trajectories(
        [FieldEdge(first), FieldEdge(second)],
        _images(),
        _settings(),
        TrajectoryConfig(new_point_exclusion_radius_m=500.0),
    )

    created_at_b = [point for point in points if point.image_id == "b" and point.state == "created"]
    assert len(created_at_b) == 4


def test_deformation_uses_inverse_seconds_and_known_affine_gradient():
    points = GRID
    displacement = np.column_stack((0.01 * points[:, 0], 0.02 * points[:, 1]))
    field = _field("a", "b", 0, [0.0, 0.0])
    values = dict(field.__dict__)
    values["displacement_m"] = displacement
    field = DisplacementField(**values)

    cells = deformation_from_field(field, 1_500.0)

    assert len(cells) == 2
    np.testing.assert_allclose([cell.divergence_s_1 for cell in cells], 0.03 / 86_400.0)
    np.testing.assert_allclose([cell.shear_s_1 for cell in cells], 0.01 / 86_400.0)
