import pandas as pd
from shapely.geometry import box

from experiments.build_full70_buoy_image_links import (
    build_buoy_transitions,
    build_level1_targets,
    build_links,
    build_repeat_publication_controls,
    build_same_pass_scene_pairs,
)


def test_builds_named_and_additional_buoy_links_with_explicit_holds(tmp_path):
    inventory = pd.DataFrame(
        {
            "image_id": [1, 2],
            "image_time": ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"],
            "acquisition_pass_id": ["S1A_orbit_1", "S1A_orbit_1"],
            "sentinel1_product_name": ["S1_A", "S1_B"],
            "resolved_product_name": ["S1_A", "S1_B"],
            "raw_zip_path": ["A.zip", "B.zip"],
            "standard_vae_output_path": ["A.tiff", "B.tiff"],
            "sequence_ids": ["sequence_1", "sequence_2"],
            "buoy_ids": ["100", "100"],
        }
    )
    exact = pd.DataFrame(
        {
            "buoy_id": ["100", "200", "100"],
            "image_id": [1, 1, 2],
            "image_time": [
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:00:00Z",
                "2020-01-02T00:00:00Z",
            ],
            "exact_position_inside_scene": [True, True, True],
            "descriptor_border_safe": [True, True, True],
            "track_qc_status": ["pass", "track_context_gap_unverified", "pass"],
            "buoy_ice_qc_status": [
                "on_ice_high_confidence_from_sic_track",
                "track_context_unverified",
                "on_ice_high_confidence_from_sic_track",
            ],
            "month": ["2020-01"] * 3,
            "spatial_block": ["0_0"] * 3,
            "sic_regime": ["pack_ice"] * 3,
            "x": [0.0, 50.0, 1000.0],
            "y": [0.0, 50.0, 0.0],
        }
    )

    links = build_links(exact, inventory)
    targets = build_level1_targets(links, tmp_path)
    transitions = build_buoy_transitions(links)
    controls = build_same_pass_scene_pairs(inventory)

    assert links["buoy_is_named_sequence_target"].tolist() == [True, True, False]
    assert set(links["current_experiment_status"]) == {
        "ready_current_catalog_qc",
        "hold_full_level1_track_context",
    }
    assert targets["buoy_id"].tolist() == ["100", "200"]
    assert targets.loc[targets["buoy_id"].eq("200"), "track_context_holds"].item() == 1
    assert len(transitions) == 1
    assert transitions.iloc[0]["truth_distance_m"] == 1000.0
    assert transitions.iloc[0]["source_month"] == "2020-01"
    assert transitions.iloc[0]["ready_for_tracking_before_level1"]
    assert len(controls) == 1
    assert controls.iloc[0]["time_separation_seconds"] == 86400.0


def test_finds_repeat_publication_of_selected_acquisition(tmp_path):
    primary = "S1B_EW_GRDM_1SDH_20200328T085700_20200328T085800_020890_0279D5_19C7"
    repeat = "S1B_EW_GRDM_1SDH_20200328T085700_20200328T085800_020890_0279D5_2E69"
    inventory = pd.DataFrame(
        {
            "image_id": [10],
            "image_time": ["2020-03-28T08:57:00Z"],
            "resolved_product_name": [primary],
            "raw_zip_path": [f"/data/{primary}.zip"],
            "standard_vae_output_path": [f"/data/{primary}.tiff"],
        }
    )
    catalog = pd.DataFrame(
        {
            "image_id": [10, 11],
            "filename": [f"{primary}.tiff", f"{repeat}.tiff"],
            "geometry": [box(0, 0, 100, 100), box(0, 0, 100, 100)],
        }
    )

    controls = build_repeat_publication_controls(inventory, catalog, tmp_path)

    assert len(controls) == 1
    assert controls.iloc[0]["repeat_product_name"] == repeat
    assert controls.iloc[0]["catalog_footprint_overlap_fraction"] == 1.0
    assert controls.iloc[0]["repeat_asf_url"].endswith(f"/SB/{repeat}.zip")
