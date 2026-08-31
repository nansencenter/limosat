import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_module():
    experiments = Path(__file__).resolve().parents[2] / "experiments"
    for name in ("buoy_descriptor_benchmark", "orb_multiframe_graph"):
        path = experiments / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    path = experiments / "buoy_keypoint_extraction_contract.py"
    spec = importlib.util.spec_from_file_location(
        "buoy_keypoint_extraction_contract", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_limosat_default_local_window_has_no_orb_detection_interior():
    module = load_module()
    model = module.build_orb(module.LIMOSAT_DEFAULT_PROFILE)
    image = np.random.default_rng(4).integers(0, 256, size=(160, 160), dtype=np.uint8)

    candidates, _, window_size = module.local_detected_candidates(
        model,
        image,
        col=80.0,
        row=80.0,
        response_threshold=0.0,
    )

    assert window_size == 47
    assert window_size < 2 * model.getEdgeThreshold()
    assert candidates == []


def test_nearest_and_strongest_are_distinct_selection_rules(monkeypatch):
    module = load_module()
    candidates = [
        cv2.KeyPoint(10.0, 10.0, 31.0, response=0.1),
        cv2.KeyPoint(12.0, 10.0, 31.0, response=0.9),
    ]
    map_xy = np.array([[100.0, 100.0], [120.0, 100.0]])
    offsets = np.array([20.0, 40.0])
    monkeypatch.setattr(
        module,
        "pixels_to_map",
        lambda path, pixel_xy, epsg: np.asarray(pixel_xy, dtype=float) * 10.0,
    )

    selected = module.select_local_keypoints(
        candidates,
        map_xy,
        offsets,
        detection_center_pixel=(10.0, 10.0),
        image_path="unused.tiff",
        analysis_epsg=3413,
        maximum_center_distance_m=300.0,
    )

    assert selected["nearest"][0] == 0
    assert selected["strongest"][0] == 1


def test_graph_summary_keeps_failed_paths_in_transition_denominator():
    module = load_module()
    coincidences = pd.DataFrame(
        {
            "buoy_id": ["a", "a", "a", "b", "b"],
        }
    )
    records = pd.DataFrame(
        [
            {
                "sequence": "test",
                "seed_method": "exact",
                "memory_method": "fixed",
                "buoy_id": "a",
                "status": "ok",
                "observation_index": 0,
                "endpoint_error_m": 0.0,
                "seed_offset_m": 0.0,
                "descriptor_updated": False,
            },
            {
                "sequence": "test",
                "seed_method": "exact",
                "memory_method": "fixed",
                "buoy_id": "a",
                "status": "ok",
                "observation_index": 1,
                "endpoint_error_m": 1000.0,
                "seed_offset_m": 0.0,
                "descriptor_updated": False,
            },
            {
                "sequence": "test",
                "seed_method": "exact",
                "memory_method": "fixed",
                "buoy_id": "a",
                "status": "ok",
                "observation_index": 2,
                "endpoint_error_m": 3000.0,
                "seed_offset_m": 0.0,
                "descriptor_updated": False,
            },
            {
                "sequence": "test",
                "seed_method": "exact",
                "memory_method": "fixed",
                "buoy_id": "b",
                "status": "seed unavailable",
                "observation_index": np.nan,
                "endpoint_error_m": np.nan,
                "seed_offset_m": np.nan,
                "descriptor_updated": False,
            },
        ]
    )

    summary = module.summarize_graph(records, coincidences).iloc[0]

    assert summary.eligible_transitions == 3
    assert summary.tracked_transitions == 2
    assert summary.within_2km_fraction_all == 1 / 3
    assert summary.seed_unavailable_paths == 1
