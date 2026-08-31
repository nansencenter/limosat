import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[2] / "experiments/buoy_supervised_update_training.py"
    spec = importlib.util.spec_from_file_location("buoy_supervised_update_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plain_policy_name_exposes_threshold_meaning():
    module = load_module()
    assert module.policy_name(0.032, 0.35) == (
        "update_when_match_clear__lead_032__difference_350"
    )


def test_policy_summary_keeps_untracked_transitions_in_denominator():
    module = load_module()
    coincidences = pd.DataFrame(
        {
            "buoy_id": ["a", "a", "a", "b", "b"],
            "image_id": [0, 1, 2, 0, 1],
        }
    )
    policy = module.GraphSearchConfig("readable", "anchor", 32, 8)
    records = pd.DataFrame(
        {
            "config": ["readable", "readable"],
            "buoy_id": ["a", "a"],
            "status": ["ok", "ok"],
            "observation_index": [1, 2],
            "endpoint_error_m": [1000.0, 60000.0],
            "descriptor_updated": [False, False],
        }
    )
    summary = module.summarize_policies(records, coincidences, (policy,)).iloc[0]
    assert summary.eligible_transitions == 3
    assert summary.tracking_fraction_all == 2 / 3
    assert summary.within_2km_fraction_all == 1 / 3
    assert summary.catastrophic_50km_fraction_all == 1 / 3


def test_policy_selection_requires_safe_memory_before_path_accuracy():
    module = load_module()
    summary = pd.DataFrame(
        {
            "policy": ["unsafe_accuracy", "safe_rule"],
            "within_2km_fraction_all": [0.8, 0.7],
            "catastrophic_50km_fraction_all": [0.05, 0.0],
            "tracking_fraction_all": [1.0, 1.0],
            "false_memory_updates": [2, 0],
            "safe_memory_updates": [10, 100],
            "memory_updates": [12, 100],
            "safe_update_precision": [10 / 12, 1.0],
            "median_error_tracked_m": [1000.0, 800.0],
            "best_match_lead": [0.03, 0.01],
            "maximum_descriptor_difference": [0.35, 0.35],
        }
    )
    assert module.select_policy(summary) == "safe_rule"


def test_binary_hamming_distance_is_normalized_to_256_bits():
    module = load_module()
    reference = np.zeros(32, dtype=np.uint8)
    candidates = np.vstack([reference, np.full(32, 255, dtype=np.uint8)])
    np.testing.assert_allclose(module.hamming_distance(reference, candidates), [0.0, 1.0])


def test_binary_hamming_distance_normalizes_brisk_length_separately():
    module = load_module()
    reference = np.zeros(64, dtype=np.uint8)
    candidates = np.full((1, 64), 255, dtype=np.uint8)
    np.testing.assert_allclose(module.hamming_distance(reference, candidates), [1.0])
