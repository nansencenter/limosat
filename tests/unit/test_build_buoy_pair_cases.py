import pandas as pd
import pytest

from experiments.build_buoy_pair_cases import (
    build_transitions,
    map_observations,
    split_observations,
    validate_pair_plan,
)


def inputs():
    filenames = {
        image: (
            f"S1A_EW_GRDM_1SDH_2020010{image}T000000_"
            f"2020010{image}T000100_00000{image}_TEST_ABCD.tiff"
        )
        for image in (1, 2, 3, 4)
    }
    observations = pd.DataFrame(
        [
            {
                "buoy_id": buoy,
                "image_id": image,
                "image_filename": filenames[image],
                "image_filepath": f"/wrong/preprocessing/image_{image}.tiff",
                "image_time": f"2020-01-0{image}T00:00:00Z",
                "x": 1000.0 * buoy + 10.0 * image,
                "y": 2000.0 * buoy - 20.0 * image,
                "analysis_crs": "EPSG:3413",
            }
            for image in (1, 2, 3, 4)
            for buoy in (1, 2, 3)
        ]
    )
    image_map = pd.DataFrame(
        {
            "fixture_image_id": [1, 2, 3, 4],
            "operational_image_id": [101, 102, 103, 104],
            "image_time": [f"2020-01-0{i}T00:00:00Z" for i in (1, 2, 3, 4)],
            "image_filename": [filenames[i] for i in (1, 2, 3, 4)],
            "kingston_filepath": [f"/missing/image_{i}.tiff" for i in (1, 2, 3, 4)],
        }
    )
    plan = pd.DataFrame(
        {
            "source_fixture_image_id": [1, 2, 3],
            "target_fixture_image_id": [2, 3, 4],
            "within_dataset_split": ["diagnostic", "buffer", "evaluation"],
            "may_tune": [True, False, False],
            "report_primary": [False, False, True],
        }
    )
    return observations, image_map, plan


def test_builder_maps_operational_ids_and_preserves_all_pair_buoys():
    source, image_map, plan = inputs()
    plan = validate_pair_plan(plan)
    observations = map_observations(source, image_map)
    transitions = build_transitions(observations, plan)

    assert len(observations) == 12
    assert len(transitions) == 9
    assert set(observations["image_id"]) == {101, 102, 103, 104}
    assert list(observations.columns).count("image_filepath") == 1
    assert observations.loc[
        observations["image_id"].eq(101), "image_filepath"
    ].eq("/missing/image_1.tiff").all()
    assert not observations["image_exists"].any()
    first = transitions.iloc[0]
    assert first.source_image_id == 101
    assert first.target_image_id == 102
    assert first.truth_dx_m == pytest.approx(10.0)
    assert first.truth_dy_m == pytest.approx(-20.0)
    assert first.elapsed_hours == pytest.approx(24.0)

    expanded = split_observations(observations, plan)
    assert len(expanded) == 18
    assert set(expanded["within_dataset_split"]) == {
        "diagnostic",
        "buffer",
        "evaluation",
    }
    assert expanded["acquisition_pass_id"].str.startswith("S1A_orbit_").all()


def test_pair_plan_rejects_tuning_primary_image_overlap():
    _, _, plan = inputs()
    plan.loc[1, "report_primary"] = True

    with pytest.raises(ValueError, match="share fixture images"):
        validate_pair_plan(plan)
