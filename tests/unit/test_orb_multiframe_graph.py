import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_graph_module():
    experiments = Path(__file__).resolve().parents[2] / "experiments"
    benchmark_path = experiments / "buoy_descriptor_benchmark.py"
    benchmark_spec = importlib.util.spec_from_file_location(
        "buoy_descriptor_benchmark", benchmark_path
    )
    benchmark = importlib.util.module_from_spec(benchmark_spec)
    sys.modules[benchmark_spec.name] = benchmark
    benchmark_spec.loader.exec_module(benchmark)

    graph_path = experiments / "orb_multiframe_graph.py"
    graph_spec = importlib.util.spec_from_file_location("orb_multiframe_graph", graph_path)
    graph = importlib.util.module_from_spec(graph_spec)
    sys.modules[graph_spec.name] = graph
    graph_spec.loader.exec_module(graph)
    return graph, benchmark


def test_majority_binary_descriptor_resolves_ties_with_latest():
    graph, _ = load_graph_module()

    tied = graph.majority_binary_descriptor(
        (np.array([0], dtype=np.uint8), np.array([255], dtype=np.uint8))
    )
    majority = graph.majority_binary_descriptor(
        (
            np.array([0], dtype=np.uint8),
            np.array([255], dtype=np.uint8),
            np.array([255], dtype=np.uint8),
        )
    )

    assert tied.tolist() == [255]
    assert majority.tolist() == [255]


def test_explicit_experiment_trajectory_is_preferred_over_buoy_id():
    graph, _ = load_graph_module()
    fixture = pd.DataFrame(
        {
            "buoy_id": ["same", "same"],
            "experiment_trajectory_id": ["segment_1", "segment_2"],
        }
    )

    assert graph.trajectory_column(fixture) == "experiment_trajectory_id"
    assert graph.trajectory_column(fixture[["buoy_id"]]) == "buoy_id"


def test_summary_keeps_failed_path_transitions_in_denominator():
    graph, _ = load_graph_module()
    records = pd.DataFrame(
        [
            {
                "config": "greedy_rolling",
                "trajectory_id": "tracked",
                "buoy_id": "a",
                "status": "ok",
                "path_observations": 3,
                "observation_index": 0,
                "endpoint_error_m": 0.0,
            },
            {
                "config": "greedy_rolling",
                "trajectory_id": "tracked",
                "buoy_id": "a",
                "status": "ok",
                "path_observations": 3,
                "observation_index": 1,
                "endpoint_error_m": 1000.0,
            },
            {
                "config": "greedy_rolling",
                "trajectory_id": "tracked",
                "buoy_id": "a",
                "status": "ok",
                "path_observations": 3,
                "observation_index": 2,
                "endpoint_error_m": 3000.0,
            },
            {
                "config": "greedy_rolling",
                "trajectory_id": "failed",
                "buoy_id": "b",
                "status": "seed_unavailable",
                "path_observations": 2,
                "observation_index": np.nan,
                "endpoint_error_m": np.nan,
            },
        ]
    )

    summary = graph.summarize(records).iloc[0]

    assert summary.eligible_transitions == 3
    assert summary.tracked_transitions == 2
    assert summary.within_2km_fraction_all == 1 / 3


def test_beam_retains_a_physically_viable_non_greedy_branch():
    graph, benchmark = load_graph_module()
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    empty_grid = benchmark.CandidateGrid(
        pixel_xy=np.empty((0, 2)), map_xy=np.empty((0, 2))
    )
    layers = [
        graph.DescriptorLayer(0, "seed", t0, empty_grid, np.empty((0, 1), dtype=np.uint8)),
        graph.DescriptorLayer(
            1,
            "first",
            t0 + pd.Timedelta(days=1),
            benchmark.CandidateGrid(
                pixel_xy=np.array([[0.0, 0.0], [1.0, 0.0]]),
                map_xy=np.array([[0.0, 0.0], [10.0, 0.0]]),
            ),
            np.array([[0], [1]], dtype=np.uint8),
        ),
        graph.DescriptorLayer(
            2,
            "second",
            t0 + pd.Timedelta(days=2),
            benchmark.CandidateGrid(
                pixel_xy=np.array([[2.0, 0.0]]),
                map_xy=np.array([[20.0, 0.0]]),
            ),
            np.array([[1]], dtype=np.uint8),
        ),
    ]

    greedy = graph.search_layered_graph(
        layers,
        seed_xy=np.array([0.0, 0.0]),
        seed_descriptor=np.array([0], dtype=np.uint8),
        max_speed_m_per_day=15.0,
        config=graph.GraphSearchConfig("greedy", "rolling", beam_width=1, branching=1),
    )
    beam = graph.search_layered_graph(
        layers,
        seed_xy=np.array([0.0, 0.0]),
        seed_descriptor=np.array([0], dtype=np.uint8),
        max_speed_m_per_day=15.0,
        config=graph.GraphSearchConfig("beam", "rolling", beam_width=2, branching=2),
    )

    assert greedy is None
    assert beam is not None
    assert [position.tolist() for position in beam.path_xy] == [
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 0.0],
    ]


def test_soft_displacement_cost_prefers_nearby_equal_appearance_candidate():
    graph, benchmark = load_graph_module()
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    state = graph.PathState(
        score=0.0,
        position_xy=np.array([0.0, 0.0]),
        velocity_xy_per_second=None,
        descriptors=(np.array([0], dtype=np.uint8),),
        path_xy=(np.array([0.0, 0.0]),),
        node_indices=(-1,),
        edge_costs=(),
        candidate_counts=(),
        descriptor_updates=(),
    )
    layer = graph.DescriptorLayer(
        1,
        "target",
        t0 + pd.Timedelta(days=1),
        benchmark.CandidateGrid(
            pixel_xy=np.array([[0.0, 0.0], [1.0, 0.0]]),
            map_xy=np.array([[1000.0, 0.0], [40000.0, 0.0]]),
        ),
        np.array([[0], [0]], dtype=np.uint8),
    )

    result = graph.expand_graph_layer(
        [state],
        layer,
        dt_seconds=86400.0,
        max_speed_m_per_day=50000.0,
        config=graph.GraphSearchConfig(
            "soft_speed", "anchor", 1, 2, displacement_weight=0.05
        ),
    )

    assert result[0].position_xy.tolist() == [1000.0, 0.0]


def test_soft_40km_day_preference_retains_but_penalizes_fast_candidate():
    graph, benchmark = load_graph_module()
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    state = graph.PathState(
        score=0.0,
        position_xy=np.array([0.0, 0.0]),
        velocity_xy_per_second=None,
        descriptors=(np.array([0], dtype=np.uint8),),
        path_xy=(np.array([0.0, 0.0]),),
        node_indices=(-1,),
        edge_costs=(),
        candidate_counts=(),
        descriptor_updates=(),
    )
    layer = graph.DescriptorLayer(
        1,
        "target",
        t0 + pd.Timedelta(days=1),
        benchmark.CandidateGrid(
            pixel_xy=np.array([[0.0, 0.0], [1.0, 0.0]]),
            map_xy=np.array([[39000.0, 0.0], [49000.0, 0.0]]),
        ),
        np.array([[1], [0]], dtype=np.uint8),
    )

    result = graph.expand_graph_layer(
        [state],
        layer,
        dt_seconds=86400.0,
        max_speed_m_per_day=50000.0,
        config=graph.GraphSearchConfig(
            "soft_preference",
            "anchor",
            1,
            2,
            preferred_speed_m_per_day=40000.0,
            excess_speed_weight=0.20,
        ),
    )

    assert result[0].position_xy.tolist() == [39000.0, 0.0]


def test_confidence_update_rejects_tied_appearance_candidates():
    graph, benchmark = load_graph_module()
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    state = graph.PathState(
        score=0.0,
        position_xy=np.array([0.0, 0.0]),
        velocity_xy_per_second=None,
        descriptors=(np.array([0], dtype=np.uint8),),
        path_xy=(np.array([0.0, 0.0]),),
        node_indices=(-1,),
        edge_costs=(),
        candidate_counts=(),
        descriptor_updates=(),
    )
    layer = graph.DescriptorLayer(
        1,
        "target",
        t0 + pd.Timedelta(days=1),
        benchmark.CandidateGrid(
            pixel_xy=np.array([[0.0, 0.0], [1.0, 0.0]]),
            map_xy=np.array([[1000.0, 0.0], [2000.0, 0.0]]),
        ),
        np.array([[0], [0]], dtype=np.uint8),
    )

    result = graph.expand_graph_layer(
        [state],
        layer,
        dt_seconds=86400.0,
        max_speed_m_per_day=50000.0,
        config=graph.GraphSearchConfig(
            "confidence", "confidence_rolling", 1, 2, update_min_margin=0.01
        ),
    )

    assert result[0].descriptor_updates == (False,)
    assert len(result[0].descriptors) == 1


def test_one_frame_skip_recovers_with_elapsed_physics_gate():
    graph, benchmark = load_graph_module()
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    layers = [
        graph.DescriptorLayer(
            0,
            "seed",
            t0,
            benchmark.CandidateGrid(np.empty((0, 2)), np.empty((0, 2))),
            np.empty((0, 1), dtype=np.uint8),
        ),
        graph.DescriptorLayer(
            1,
            "missing",
            t0 + pd.Timedelta(days=1),
            benchmark.CandidateGrid(np.empty((0, 2)), np.empty((0, 2))),
            np.empty((0, 1), dtype=np.uint8),
        ),
        graph.DescriptorLayer(
            2,
            "recovery",
            t0 + pd.Timedelta(days=2),
            benchmark.CandidateGrid(
                np.array([[1.0, 0.0]]),
                np.array([[20.0, 0.0]]),
            ),
            np.array([[0]], dtype=np.uint8),
        ),
    ]

    result = graph.search_layered_graph(
        layers,
        seed_xy=np.array([0.0, 0.0]),
        seed_descriptor=np.array([0], dtype=np.uint8),
        max_speed_m_per_day=15.0,
        config=graph.GraphSearchConfig(
            "skip", "anchor", 2, 1, max_consecutive_skips=1
        ),
    )

    assert result is not None
    assert result.path_xy[1] is None
    assert result.path_xy[2].tolist() == [20.0, 0.0]
