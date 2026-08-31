import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_module():
    experiments = Path(__file__).resolve().parents[2] / "experiments"
    for name in (
        "buoy_descriptor_benchmark",
        "orb_multiframe_graph",
        "xfeat_buoy_graph",
        "buoy_patch_evolution",
        "orb_candidate_forensics",
    ):
        path = experiments / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["orb_candidate_forensics"], sys.modules["orb_multiframe_graph"], sys.modules[
        "buoy_descriptor_benchmark"
    ]


def failure_row(**updates):
    row = {
        "path_status": "ok",
        "selected_error_m": 3000.0,
        "truth_grid_distance_m": 500.0,
        "truth_accessible_from_any_parent": True,
        "truth_in_selected_parent_gate": True,
        "truth_edge_rank": 2.0,
        "truth_appearance_hard": False,
        "expanded_min_truth_error_m": 500.0,
        "retained_min_truth_error_m": 500.0,
    }
    row.update(updates)
    return row


def test_failure_classifier_keeps_mechanisms_distinct():
    module, _, _ = load_module()

    assert module.classify_failure(
        failure_row(selected_error_m=900.0), branching=8, error_threshold_m=2000.0
    ) == "success"
    assert module.classify_failure(
        failure_row(truth_grid_distance_m=2500.0), branching=8, error_threshold_m=2000.0
    ) == "candidate_descriptor_coverage_failure"
    assert module.classify_failure(
        failure_row(
            truth_grid_distance_m=2500.0,
            raster_border_distance_px=40.0,
            candidate_grid_border_px=128,
        ),
        branching=8,
        error_threshold_m=2000.0,
    ) == "candidate_border_exclusion"
    assert module.classify_failure(
        failure_row(truth_accessible_from_any_parent=False),
        branching=8,
        error_threshold_m=2000.0,
    ) == "state_gate_exclusion"
    assert module.classify_failure(
        failure_row(truth_edge_rank=12, truth_appearance_hard=True),
        branching=8,
        error_threshold_m=2000.0,
    ) == "observation_appearance_ranking"
    assert module.classify_failure(
        failure_row(
            truth_edge_rank=2,
            expanded_min_truth_error_m=500.0,
            retained_min_truth_error_m=3000.0,
        ),
        branching=8,
        error_threshold_m=2000.0,
    ) == "beam_pruning_failure"


def test_trace_retains_unpruned_candidates_without_changing_selected_path():
    module, graph, benchmark = load_module()
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
            "target",
            t0 + pd.Timedelta(days=1),
            benchmark.CandidateGrid(
                np.array([[0.0, 0.0], [1.0, 0.0]]),
                np.array([[1000.0, 0.0], [2000.0, 0.0]]),
            ),
            np.array([[0], [1]], dtype=np.uint8),
        ),
    ]
    for layer in layers[1:]:
        layer.spatial_index = module.cKDTree(layer.grid.map_xy) if hasattr(module, "cKDTree") else None
    config = graph.GraphSearchConfig("greedy", "anchor", beam_width=1, branching=2)

    trace = module.search_with_trace(
        layers,
        seed_xy=np.array([0.0, 0.0]),
        descriptor=np.array([0], dtype=np.uint8),
        max_speed_m_per_day=5000.0,
        config=config,
        descriptor_norm="hamming",
    )
    regular = graph.search_layered_graph(
        layers,
        seed_xy=np.array([0.0, 0.0]),
        seed_descriptor=np.array([0], dtype=np.uint8),
        max_speed_m_per_day=5000.0,
        config=config,
    )

    assert len(trace.layers[0].expanded) == 2
    assert len(trace.layers[0].retained) == 1
    assert trace.final_state.node_indices == regular.node_indices


def test_temporal_labels_identify_update_poisoning_and_recovery():
    module, _, _ = load_module()
    records = pd.DataFrame.from_records(
        [
            {
                "transition_id": "t1",
                "sequence": "s",
                "config": "c",
                "buoy_id": "b",
                "observation_index": 1,
                "false_update": True,
                "selected_error_m": 3000.0,
                "truth_orb_anchor_hamming_norm": 0.50,
                "truth_in_selected_parent_gate": True,
                "truth_anchor_rank": 2,
                "truth_edge_rank": 2,
                "truth_accessible_from_any_parent": True,
                "best_truth_anchor_rank_any_parent": 2,
            },
            {
                "transition_id": "t2",
                "sequence": "s",
                "config": "c",
                "buoy_id": "b",
                "observation_index": 2,
                "false_update": False,
                "selected_error_m": 900.0,
                "truth_orb_anchor_hamming_norm": 0.30,
                "truth_in_selected_parent_gate": True,
                "truth_anchor_rank": 2,
                "truth_edge_rank": 12,
                "truth_accessible_from_any_parent": True,
                "best_truth_anchor_rank_any_parent": 2,
            },
        ]
    )

    labelled = module.add_temporal_labels(records, branching=8, error_threshold_m=2000.0)

    assert labelled.loc[1, "prior_false_update"]
    assert labelled.loc[1, "probable_update_poisoning"]
    assert labelled.loc[0, "recovery_opportunity_next"]
    assert labelled.loc[0, "recovered_next"]


def test_provisional_descriptor_rank_is_separate_from_confirmed_memory():
    module, graph, benchmark = load_module()
    layer = graph.DescriptorLayer(
        1,
        "target",
        pd.Timestamp("2026-01-02T00:00:00Z"),
        benchmark.CandidateGrid(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
        ),
        np.array([[0], [255]], dtype=np.uint8),
    )
    table = {"gate_indices": np.array([0, 1], dtype=int)}

    module.add_provisional_costs(
        table,
        layer,
        provisional_descriptor=np.array([255], dtype=np.uint8),
        descriptor_norm="hamming",
    )

    assert table["provisional_ranks"].tolist() == [2, 1]
