from __future__ import annotations

import argparse
import copy
from types import SimpleNamespace

import cv2
import pandas as pd
import pytest

from experiments.audit_operational_buoy_probe_extraction import (
    nearest_after_response_threshold,
    summarize_results,
)
from experiments.run_operational_baseline import (
    DEFAULT_CONFIG,
    apply_storage_root,
    build_buoy_probes,
    format_storage,
    image_table,
    install_probe_seed_method,
    load_yaml,
    make_matcher,
    make_orb,
    parse_catalog_image_ids,
    select_catalog_images,
    select_image_split,
    validate_frozen_config,
)


def test_frozen_config_builds_expected_orb():
    config = load_yaml(DEFAULT_CONFIG)
    validate_frozen_config(config)

    orb = make_orb(config)
    assert orb.getMaxFeatures() == 200
    assert orb.getNLevels() == 5
    assert orb.getPatchSize() == 88
    assert orb.getWTA_K() == 2


def test_frozen_config_rejects_algorithmic_drift():
    config = load_yaml(DEFAULT_CONFIG)
    changed = copy.deepcopy(config)
    changed["image_processor_params"]["stride"] = 16

    with pytest.raises(ValueError, match="stride"):
        validate_frozen_config(changed)


def test_matcher_reads_model_coordinate_scale_from_config():
    config = load_yaml(DEFAULT_CONFIG)
    config["matcher_params"]["geometric_model"]["coordinate_scale_m"] = 1_000.0

    matcher = make_matcher(config)

    assert matcher.model_coordinate_scale_m == 1_000.0


def test_storage_root_override_keeps_dataset_state_out_of_frozen_config_path(
    tmp_path,
):
    config = load_yaml(DEFAULT_CONFIG)
    apply_storage_root(config, tmp_path / "nice_state")

    engine_url, zarr_path = format_storage(config, "nice_run")

    assert engine_url == f"sqlite:///{tmp_path}/nice_state/nice_run.sqlite"
    assert zarr_path == tmp_path / "nice_state/nice_run.zarr"


def test_image_table_and_development_probes_are_deterministic(tmp_path):
    second = tmp_path / "S1_20200102T000000_000002_test.tiff"
    first = tmp_path / "S1_20200101T000000_000001_test.tiff"
    first.touch()
    second.touch()
    observations = pd.DataFrame(
        {
            "buoy_id": ["b", "a", "c"],
            "image_id": [20, 10, 20],
            "image_time": ["2020-01-02", "2020-01-01", "2020-01-02"],
            "image_filepath": [str(second), str(first), str(second)],
            "x": [3.0, 1.0, 5.0],
            "y": [4.0, 2.0, 6.0],
            "within_dataset_split": ["development", "development", "final_holdout"],
        }
    )

    images = image_table(observations)
    probes = build_buoy_probes(observations, images, "development")

    assert images["image_id"].tolist() == [10, 20]
    assert images["run_image_id"].tolist() == [1, 2]
    assert probes["probe_id"].tolist() == ["b|20", "a|10"]
    assert probes.crs.to_epsg() == 3413
    assert probes.loc["a|10", "image_filepath"] == first.name


def test_image_split_selection_is_explicit_and_fails_closed():
    observations = pd.DataFrame(
        {
            "image_id": [1, 2, 3],
            "within_dataset_split": ["diagnostic", "buffer", "evaluation"],
        }
    )

    selected = select_image_split(observations, "evaluation")

    assert selected["image_id"].tolist() == [3]
    with pytest.raises(ValueError, match="available"):
        select_image_split(observations, "holdout")
    with pytest.raises(ValueError, match="within_dataset_split"):
        select_image_split(
            observations.drop(columns="within_dataset_split"), "evaluation"
        )


def test_explicit_catalog_image_selection_requires_every_unique_id():
    observations = pd.DataFrame({"image_id": [1, 1, 2, 3]})

    selected = select_catalog_images(observations, (1, 3))

    assert selected["image_id"].tolist() == [1, 1, 3]
    assert parse_catalog_image_ids("3,1") == (3, 1)
    with pytest.raises(ValueError, match="absent"):
        select_catalog_images(observations, (1, 4))
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        parse_catalog_image_ids("1,1")


def test_response_threshold_is_applied_before_nearest_selection():
    close_weak = cv2.KeyPoint(5.0, 5.0, 31.0, response=0.0002)
    distant_strong = cv2.KeyPoint(9.0, 9.0, 31.0, response=0.002)

    assert nearest_after_response_threshold(
        [distant_strong, close_weak], 0.0, 5.0, 5.0
    ) is close_weak
    assert nearest_after_response_threshold(
        [distant_strong, close_weak], 0.001, 5.0, 5.0
    ) is distant_strong


def test_probe_summary_keeps_missing_counterfactuals_in_denominator():
    results = pd.DataFrame(
        {
            "production_keypoint_returned": [True, False],
            "production_descriptor_available": [True, False],
            "exact_supplied_descriptor_available": [True, True],
            "failure_stage": ["production_keypoint_returned", "no_local_detection"],
            "threshold_0p0000_passes_300m_gate": [True, None],
            "threshold_0p0001_passes_300m_gate": [True, None],
            "threshold_0p0005_passes_300m_gate": [True, None],
            "threshold_0p0010_passes_300m_gate": [True, None],
        }
    )

    summary = summarize_results(results).set_index("metric")["value"]

    assert summary["threshold_0p0000_passes_300m_gate_fraction"] == 0.5


def test_exact_probe_seed_method_preserves_supplied_coordinates():
    detector = SimpleNamespace(keypoint_from_point=lambda *args, **kwargs: [])
    processor = SimpleNamespace(keypoint_detector=detector)
    image = SimpleNamespace(
        srs="EPSG:3413",
        angle=17.0,
        transform_points=lambda x, y, **kwargs: (x, y),
    )
    points = build_buoy_probes(
        pd.DataFrame(
            {
                "buoy_id": ["a"],
                "image_id": [1],
                "image_filepath": ["scene.tiff"],
                "x": [12.5],
                "y": [34.5],
                "within_dataset_split": ["development"],
            }
        ),
        pd.DataFrame({"image_id": [1]}),
        "development",
    )

    install_probe_seed_method(processor, "exact_supplied")
    extracted = detector.keypoint_from_point(points, 5, image, 0.001)

    assert len(extracted) == 1
    assert extracted[0][0].pt == (12.5, 34.5)
    assert extracted[0][1] == "a|1"
