import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def load_module():
    experiments = Path(__file__).resolve().parents[2] / "experiments"
    for name in ("buoy_descriptor_benchmark", "orb_multiframe_graph", "orb_border_sweep"):
        path = experiments / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["orb_border_sweep"], sys.modules["orb_multiframe_graph"], sys.modules[
        "buoy_descriptor_benchmark"
    ]


def test_border_summary_keeps_untracked_transitions_in_denominator():
    module, graph, benchmark = load_module()
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    coincidences = pd.DataFrame.from_records(
        [
            {"buoy_id": "a", "image_time": t0, "image_filepath": "p0", "x": 0.0, "y": 0.0},
            {"buoy_id": "a", "image_time": t0 + pd.Timedelta(days=1), "image_filepath": "p1", "x": 1.0, "y": 0.0},
            {"buoy_id": "a", "image_time": t0 + pd.Timedelta(days=2), "image_filepath": "p2", "x": 2.0, "y": 0.0},
            {"buoy_id": "b", "image_time": t0, "image_filepath": "p0", "x": 0.0, "y": 0.0},
            {"buoy_id": "b", "image_time": t0 + pd.Timedelta(days=1), "image_filepath": "p1", "x": 1.0, "y": 0.0},
        ]
    )
    grid = benchmark.CandidateGrid(
        pixel_xy=np.array([[0.0, 0.0]]), map_xy=np.array([[1.0, 0.0]])
    )
    layers = {}
    for name in ("p1", "p2"):
        layer = graph.DescriptorLayer(
            1,
            name,
            t0 + pd.Timedelta(days=1),
            grid,
            np.array([[0]], dtype=np.uint8),
        )
        layer.spatial_index = cKDTree(grid.map_xy)
        layers[name] = layer
    records = pd.DataFrame.from_records(
        [
            {"buoy_id": "a", "status": "ok", "observation_index": 1, "endpoint_error_m": 1000.0},
            {"buoy_id": "a", "status": "ok", "observation_index": 2, "endpoint_error_m": 60000.0},
            {"buoy_id": "b", "status": "graph_failed", "endpoint_error_m": np.nan},
        ]
    )

    summary = module.summarize_run(
        "fixture",
        32,
        coincidences,
        records,
        layers,
        precompute_seconds=1.0,
        tracking_seconds=2.0,
    )

    assert summary["eligible_transitions"] == 3
    assert summary["tracked_transitions"] == 2
    assert summary["within_2km_fraction_all"] == 1 / 3
    assert summary["catastrophic_50km_fraction_all"] == 1 / 3
