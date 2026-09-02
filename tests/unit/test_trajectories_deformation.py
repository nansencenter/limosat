from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from limosat import (
    DisplacementField,
    FieldConfig,
    FieldEdge,
    ImageRecord,
    TrajectoryConfig,
    TrajectoryPoint,
    audit_trajectory_convergence,
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
    edges = [
        FieldEdge(adjacent),
        FieldEdge(empty),
        FieldEdge(skip, pair_kind="recovery", skipped_images=1),
    ]

    images = _images(("a", "b", "c", "d"))
    points = build_trajectories(edges, images, _settings(), TrajectoryConfig())
    at_c = [point for point in points if point.image_id == "c"]
    at_d = [point for point in points if point.image_id == "d"]

    assert at_c
    assert {point.state for point in at_c} == {"dormant"}
    assert {point.state for point in at_d} == {"reappeared"}
    assert {point.position_basis for point in at_d} == {"recovery_pair_field"}
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


def test_global_continuation_crosses_former_component_boundaries():
    images = [
        ImageRecord(
            name,
            Path("/tmp/unused.tif"),
            START + timedelta(days=index),
            f"former-{index}",
        )
        for index, name in enumerate(("a", "b", "c"))
    ]
    first = _field("a", "b", 0, [100.0, 0.0])
    second = _field(
        "b", "c", 1, [50.0, 0.0], points=GRID + [100.0, 0.0]
    )

    points = build_trajectories(
        [FieldEdge(first), FieldEdge(second)],
        images,
        _settings(),
        TrajectoryConfig(),
    )

    created = {
        point.trajectory_id for point in points if point.state == "created"
    }
    at_c = {
        point.trajectory_id
        for point in points
        if point.image_id == "c" and point.state == "observed"
    }
    assert len(created) == 4
    assert at_c == created


def test_global_seed_suppression_prevents_duplicate_parcels():
    first = _field("a", "b", 0, [100.0, 0.0])
    second = _field(
        "b", "c", 1, [50.0, 0.0], points=GRID + [100.0, 0.0]
    )

    points = build_trajectories(
        [FieldEdge(first), FieldEdge(second)],
        _images(),
        _settings(),
        TrajectoryConfig(new_point_exclusion_radius_m=500.0),
    )

    assert len([point for point in points if point.state == "created"]) == 4
    assert not [
        point
        for point in points
        if point.image_id == "b" and point.state == "created"
    ]


def test_equal_time_competing_fields_use_quality_then_pair_id_deterministically():
    images = [
        ImageRecord("s", Path("/tmp/unused.tif"), START, "seed"),
        ImageRecord("a", Path("/tmp/unused.tif"), START + timedelta(days=1), "left"),
        ImageRecord("b", Path("/tmp/unused.tif"), START + timedelta(days=1), "right"),
        ImageRecord("t", Path("/tmp/unused.tif"), START + timedelta(days=2), "target"),
    ]
    to_a = _field("s", "a", 0, [0.0, 0.0])
    to_b = _field("s", "b", 0, [0.0, 0.0])
    low = _field("a", "t", 1, [100.0, 0.0])
    high = _field("b", "t", 1, [200.0, 0.0])
    values = dict(low.__dict__)
    values["selected_matches"] = np.full(4, 5)
    low = DisplacementField(**values)
    values = dict(high.__dict__)
    values["selected_matches"] = np.full(4, 10)
    high = DisplacementField(**values)
    edges = [FieldEdge(low), FieldEdge(to_b), FieldEdge(high), FieldEdge(to_a)]

    first = build_trajectories(
        edges, images, _settings(), TrajectoryConfig()
    )
    second = build_trajectories(
        tuple(reversed(edges)), images, _settings(), TrajectoryConfig()
    )
    first_t = [
        point for point in first if point.image_id == "t" and point.available
    ]
    second_t = [
        point for point in second if point.image_id == "t" and point.available
    ]

    assert {point.source_pair_id for point in first_t} == {"b__t"}
    assert [(point.trajectory_id, point.x_m) for point in first_t] == [
        (point.trajectory_id, point.x_m) for point in second_t
    ]


def test_dormant_coordinates_are_explicitly_null():
    empty = _field(
        "b",
        "c",
        1,
        [0.0, 0.0],
        available=[False] * 4,
        points=GRID + [100.0, 0.0],
    )
    points = build_trajectories(
        [FieldEdge(_field("a", "b", 0, [100.0, 0.0])), FieldEdge(empty)],
        _images(),
        _settings(),
        TrajectoryConfig(),
    )

    dormant = [point for point in points if point.image_id == "c"]
    assert dormant and {point.state for point in dormant} == {"dormant"}
    assert all(point.x_m is None and point.y_m is None for point in dormant)


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


def test_convergence_inherits_ranking_but_does_not_merge_trajectories():
    points = (
        TrajectoryPoint("long", "a", START, "created", "seed_grid", 0.0, 0.0, None),
        TrajectoryPoint("long", "b", START + timedelta(days=1), "observed", "primary_pair_field", 100.0, 0.0, "a__b", selected_matches=8),
        TrajectoryPoint("short", "b", START + timedelta(days=1), "created", "seed_grid", 150.0, 0.0, None),
    )

    events = audit_trajectory_convergence(points, 100.0)

    assert len(events) == 1
    assert events[0].winner_trajectory_id == "long"
    assert events[0].candidate_trajectory_id == "short"
    assert events[0].separation_m == 50.0
    assert {point.trajectory_id for point in points} == {"long", "short"}
