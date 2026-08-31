import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_module():
    experiments = Path(__file__).resolve().parents[2] / "experiments"
    for name in ("buoy_descriptor_benchmark", "orb_multiframe_graph", "xfeat_buoy_graph"):
        path = experiments / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    path = experiments / "buoy_patch_evolution.py"
    spec = importlib.util.spec_from_file_location("buoy_patch_evolution", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_masked_ncc_is_invariant_to_affine_brightness_change():
    module = load_module()
    left = np.arange(81, dtype=np.uint8).reshape(9, 9)
    right = (left.astype(np.float32) * 2.0 + 10.0).astype(np.uint8)
    valid = np.ones_like(left, dtype=bool)

    assert module.masked_ncc(left, right, valid) > 0.999


def test_native_patch_has_explicit_invalid_support():
    module = load_module()
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[4:5, 4:6] = 2
    mask[5:6, 4:6] = 253

    patch, valid = module.native_patch(image, mask, col=0.0, row=0.0, size=5)

    assert patch.shape == (5, 5)
    assert valid.shape == (5, 5)
    assert valid.sum() == 9
    assert np.all(patch[~valid] == 0)


def test_patch_pair_metrics_separate_radiometry_from_structure():
    module = load_module()
    left = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    right = np.clip(left.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    valid = np.ones_like(left, dtype=bool)

    metrics = module.patch_pair_metrics(left, right, valid, valid)

    assert metrics["ncc"] > 0.999
    assert metrics["rmse"] == 20.0
    assert metrics["overlap_fraction"] == 1.0


def test_binary_and_float_descriptor_distances_keep_native_units():
    module = load_module()

    assert module.hamming_normalized(
        np.array([0], dtype=np.uint8), np.array([255], dtype=np.uint8)
    ) == 1.0
    np.testing.assert_allclose(
        module.cosine_distance(
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ),
        1.0,
    )


def test_paired_update_effects_count_untrackable_transitions_in_denominator():
    module = load_module()
    identity = {
        "sequence": "holdout",
        "buoy_id": "b1",
        "source_observation_id": "s0",
        "target_observation_id": "s1",
        "backend": "ORB",
    }
    linked = pd.DataFrame(
        [
            {
                **identity,
                "config": "beam_anchor",
                "tracking_status": "graph_failed",
                "endpoint_error_m": np.nan,
            },
            {
                **identity,
                "config": "beam_confidence_update_m032",
                "tracking_status": "ok",
                "endpoint_error_m": 900.0,
            },
        ]
    )

    effects, summary = module.paired_update_effects(linked)

    assert effects.newly_trackable_with_update.iloc[0]
    assert summary.loc[summary.sequence == "holdout", "transitions"].iloc[0] == 1
    assert summary.loc[
        summary.sequence == "holdout", "update_within_2km_fraction_all"
    ].iloc[0] == 1.0


def test_clustered_intervals_resample_whole_buoys_deterministically():
    module = load_module()
    records = []
    for sequence in ("2020_02", "2015_full15"):
        for index in range(8):
            failure = index >= 4
            records.append(
                {
                    "sequence": sequence,
                    "backend": "ORB",
                    "config": "beam_confidence_update_m032",
                    "tracking_status": "ok",
                    "endpoint_error_m": 3000.0 if failure else 1000.0,
                    "buoy_id": f"{sequence}-{index}",
                    "orb_anchor_hamming_norm": 0.8 if failure else 0.2,
                    "orb_prev_hamming_norm": 0.7 if failure else 0.3,
                    "map_5000m_anchor_ncc": 0.2 if failure else 0.8,
                    "map_5000m_prev_ncc": 0.3 if failure else 0.7,
                    "map_5000m_prev_histogram_js_distance": 0.7 if failure else 0.1,
                }
            )
    linked = pd.DataFrame.from_records(records)

    first = module.clustered_association_intervals(
        linked, "map_5000m", bootstrap_replicates=50, random_seed=4
    )
    second = module.clustered_association_intervals(
        linked, "map_5000m", bootstrap_replicates=50, random_seed=4
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 10
    assert np.all(first.unique_buoys == 8)
    assert np.all(first.failure_auc == 1.0)
