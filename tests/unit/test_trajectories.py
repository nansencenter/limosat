from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from limosat import (
    DisplacementField,
    FieldConfig,
    FieldEdge,
    ImageRecord,
    TrajectoryConfig,
    iter_global_trajectory_batches,
)
from limosat.trajectory import _supported_continuations


START = datetime(2020, 1, 1, tzinfo=timezone.utc)
GRID = np.array([[0.0, 0.0], [1_000.0, 0.0], [0.0, 1_000.0], [1_000.0, 1_000.0]])


def _field(
    source,
    target,
    step,
    displacement,
    available=None,
    points=GRID,
    target_step=None,
):
    available = np.ones(len(points), dtype=bool) if available is None else np.asarray(available)
    values = np.tile(displacement, (len(points), 1)).astype(float)
    values[~available] = np.nan
    return DisplacementField(
        pair_id=f"{source}__{target}",
        source_image_id=source,
        target_image_id=target,
        source_time_utc=START + timedelta(days=step),
        target_time_utc=START
        + timedelta(days=step + 1 if target_step is None else target_step),
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


def build_trajectories(edges, images, field_config, trajectory_config, **options):
    """Every composed trajectory point, in image order."""
    return tuple(
        point
        for batch in iter_global_trajectory_batches(
            edges, images, field_config, trajectory_config, **options
        )
        for point in batch.points
    )


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



def test_supported_boundary_seed_gets_a_primary_measurement():
    geometry = np.array([[0., 0.], [4000., 0.], [8000., 0.], [0., 4000.]])
    edge = FieldEdge(_field("a", "b", 0, [100., 0.], points=geometry))
    points = build_trajectories([edge], _images(("a", "b")),
                                FieldConfig(grid_spacing_m=4_000.0,
                                            maximum_triangle_edge_m=6_400.0),
                                TrajectoryConfig())
    observed = [p for p in points if p.state == "observed"]
    assert len(observed) == 3
    assert any(p.x_m == 100. and p.y_m == 4000. for p in observed)
    # The fourth seed has no supporting triangle: a gap, which is not stored.
    assert sum(p.image_id == "b" for p in points) == 3



def test_competing_continuations_prefer_the_most_recent_source():
    geometry = np.array([[0., 0.], [4000., 0.], [8000., 0.], [0., 4000.]])
    ids = ["parcel-a", "parcel-b", "parcel-c", "parcel-d"]
    positions = [dict(zip(ids, geometry)), dict(zip(ids, geometry))]
    older = FieldEdge(_field("a", "c", 0, [100., 0.], points=geometry,
                             target_step=2))
    newer = FieldEdge(_field("b", "c", 1, [200., 0.], points=geometry))
    for incoming in [((0, 2, older), (1, 2, newer)),
                     ((1, 2, newer), (0, 2, older))]:
        chosen = _supported_continuations(
            incoming, positions, ids,
            FieldConfig(grid_spacing_m=4_000.0, maximum_triangle_edge_m=6_400.0),
        )
        assert chosen["parcel-d"].edge is newer
        np.testing.assert_array_equal(chosen["parcel-d"].displacement_m, [200., 0.])
        assert "parcel-c" not in chosen



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
        point for point in first if point.image_id == "t"
    ]
    second_t = [
        point for point in second if point.image_id == "t"
    ]

    assert {point.source_pair_id for point in first_t} == {"b__t"}
    assert [(point.trajectory_id, point.x_m) for point in first_t] == [
        (point.trajectory_id, point.x_m) for point in second_t
    ]
