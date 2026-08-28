import numpy as np

from experiments.evaluate_closure_trajectory_graph import (
    advect_closure_fused_graph,
)
from limosat.learned_drift import (
    DriftField,
    FieldEdge,
    advect_trajectory_graph,
)


def field(displacement, *, centre_available=True, available=None):
    source = np.array(
        [(x * 4000.0, y * 4000.0) for y in range(5) for x in range(5)]
    )
    if available is None:
        available = np.ones(len(source), dtype=bool)
        available[12] = centre_available
    return DriftField(
        np.repeat(np.arange(5), 5),
        np.tile(np.arange(5), 5),
        source,
        np.tile(displacement, (len(source), 1)),
        available,
        np.full(len(source), 8),
        np.full(len(source), 12),
        np.full(len(source), 5000.0),
        np.full(len(source), 80.0),
    )


def test_skip_edge_recovers_after_adjacent_field_gap():
    edges = [
        FieldEdge("a", "b", 24.0, field([100.0, 0.0])),
        FieldEdge("b", "c", 24.0, field([100.0, 0.0], centre_available=False)),
        FieldEdge("a", "c", 48.0, field([200.0, 0.0])),
    ]

    result = advect_trajectory_graph(
        edges, ["a", "b", "c"], 4000.0, np.array([[8000.0, 8000.0]])
    )

    assert result.active.tolist() == [True, True, True]
    assert result.trajectory_state.tolist() == [
        "seed",
        "observed_adjacent",
        "observed_skip_edge",
    ]
    assert result.edge_source_image_id.tolist() == ["", "a", "a"]
    np.testing.assert_allclose(
        result[["x_m", "y_m"]],
        [[8000.0, 8000.0], [8100.0, 8000.0], [8200.0, 8000.0]],
    )


def test_adjacent_observation_is_preferred_when_skip_edge_also_exists():
    edges = [
        FieldEdge(0, 1, 24.0, field([100.0, 0.0])),
        FieldEdge(1, 2, 24.0, field([110.0, 0.0])),
        FieldEdge(0, 2, 48.0, field([500.0, 0.0])),
    ]

    result = advect_trajectory_graph(
        edges, [0, 1, 2], 4000.0, np.array([[8000.0, 8000.0]])
    )

    final = result.iloc[-1]
    assert final.trajectory_state == "observed_adjacent"
    assert final.edge_source_image_id == "1"
    np.testing.assert_allclose([final.x_m, final.y_m], [8210.0, 8000.0])


def test_new_trajectories_are_added_only_in_newly_supported_coverage():
    left_columns = np.tile(np.arange(5), 5) <= 2
    edges = [
        FieldEdge("a", "b", 24.0, field([100.0, 0.0], available=left_columns)),
        FieldEdge("b", "c", 24.0, field([100.0, 0.0])),
    ]

    result = advect_trajectory_graph(
        edges,
        ["a", "b", "c"],
        4000.0,
        maximum_triangle_edge_m=6400.0,
        add_new_trajectories=True,
        new_point_exclusion_radius_m=2000.0,
    )

    assert result.loc[result.image_index == 0, "active"].sum() == 15
    assert result.loc[result.image_index == 1, "active"].sum() == 25
    assert result.loc[result.image_index == 2, "active"].sum() == 25
    points_new = result[result.trajectory_state == "new_trajectory"]
    assert len(points_new) == 10
    assert (points_new.seed_image_id == "b").all()


def test_new_points_do_not_duplicate_current_trajectories():
    edges = [
        FieldEdge("a", "b", 24.0, field([100.0, 0.0])),
        FieldEdge("b", "c", 24.0, field([100.0, 0.0])),
    ]

    result = advect_trajectory_graph(
        edges,
        ["a", "b", "c"],
        4000.0,
        add_new_trajectories=True,
        new_point_exclusion_radius_m=2000.0,
    )

    assert len(result.trajectory_id.unique()) == 25
    assert not (result.trajectory_state == "new_trajectory").any()


def test_dormant_trajectory_reconnects_without_a_duplicate_new_point():
    edges = [
        FieldEdge("a", "b", 24.0, field([100.0, 0.0], centre_available=False)),
        FieldEdge("b", "c", 24.0, field([100.0, 0.0])),
        FieldEdge("a", "c", 48.0, field([200.0, 0.0])),
    ]

    result = advect_trajectory_graph(
        edges,
        ["a", "b", "c"],
        4000.0,
        add_new_trajectories=True,
        new_point_exclusion_radius_m=2000.0,
    )
    centre_id = int(
        result.loc[
            result.image_index.eq(0)
            & result.x_m.eq(8000.0)
            & result.y_m.eq(8000.0),
            "trajectory_id",
        ].iloc[0]
    )
    centre = result[result.trajectory_id == centre_id].sort_values("image_index")

    assert centre.active.tolist() == [True, False, True]
    assert centre.trajectory_state.tolist() == [
        "seed",
        "dormant",
        "observed_skip_edge",
    ]
    assert centre.reconnected_after_gap.tolist() == [False, False, True]
    assert not (result.trajectory_state == "new_trajectory").any()


def test_closure_graph_fuses_consistent_adjacent_and_skip_endpoints():
    edges = [
        FieldEdge("a", "b", 24.0, field([100.0, 0.0])),
        FieldEdge("b", "c", 24.0, field([100.0, 0.0])),
        FieldEdge("a", "c", 48.0, field([220.0, 0.0])),
    ]

    result, candidates = advect_closure_fused_graph(
        edges,
        ["a", "b", "c"],
        np.array([[8000.0, 8000.0]]),
        6400.0,
        1000.0,
        80.0,
    )

    final = result.iloc[-1]
    assert final.trajectory_state == "observed_adjacent"
    assert final.candidate_count == 2
    assert final.consistent_candidate_count == 2
    assert final.closure_fused
    np.testing.assert_allclose([final.x_m, final.y_m], [8210.0, 8000.0])
    assert candidates.accepted_for_fusion.all()


def test_closure_graph_keeps_conflicting_skip_candidate_out_of_fusion():
    edges = [
        FieldEdge("a", "b", 24.0, field([100.0, 0.0])),
        FieldEdge("b", "c", 24.0, field([100.0, 0.0])),
        FieldEdge("a", "c", 48.0, field([2000.0, 0.0])),
    ]

    result, candidates = advect_closure_fused_graph(
        edges,
        ["a", "b", "c"],
        np.array([[8000.0, 8000.0]]),
        6400.0,
        1000.0,
        80.0,
    )

    final = result.iloc[-1]
    assert final.conflicting_candidate_count == 1
    assert not final.closure_fused
    np.testing.assert_allclose([final.x_m, final.y_m], [8200.0, 8000.0])
    final_candidates = candidates[candidates.target_image_id.eq("c")]
    assert final_candidates.accepted_for_fusion.tolist() == [True, False]


def test_closure_graph_uses_skip_edge_to_reconnect_missing_adjacent_support():
    edges = [
        FieldEdge("a", "b", 24.0, field([100.0, 0.0], centre_available=False)),
        FieldEdge("b", "c", 24.0, field([100.0, 0.0])),
        FieldEdge("a", "c", 48.0, field([200.0, 0.0])),
    ]

    result, _ = advect_closure_fused_graph(
        edges,
        ["a", "b", "c"],
        np.array([[8000.0, 8000.0]]),
        6400.0,
        1000.0,
        80.0,
    )

    final = result.iloc[-1]
    assert final.trajectory_state == "observed_skip_edge"
    assert final.reconnected_after_gap
    np.testing.assert_allclose([final.x_m, final.y_m], [8200.0, 8000.0])
