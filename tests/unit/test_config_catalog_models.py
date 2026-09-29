from datetime import datetime, timezone
import json

import numpy as np
import pytest

from limosat import (
    DisplacementField,
    FieldConfig,
    ImageCatalogue,
    ImageRecord,
    MatcherConfig,
    RoutingConfig,
    RunConfig,
    load_catalogue,
    load_config,
)


def test_catalogue_orders_utc_components_and_preserves_identity(tmp_path):
    later = ImageRecord(
        "second", tmp_path / "b.tif", datetime(2020, 1, 2, tzinfo=timezone.utc), "c"
    )
    earlier = ImageRecord(
        "first", tmp_path / "a.tif", datetime(2020, 1, 1, tzinfo=timezone.utc), "c"
    )
    catalogue = ImageCatalogue([later, earlier])

    pair = catalogue.adjacent_pairs("c")[0]

    assert pair.pair_id == "first__second"
    assert pair.elapsed_seconds == 86_400.0
    assert pair.source.path == (tmp_path / "a.tif").resolve()


def test_catalogue_rejects_naive_time(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        ImageRecord("image", tmp_path / "a.tif", datetime(2020, 1, 1))


def test_unavailable_field_support_is_explicitly_nan():
    common = dict(
        pair_id="a__b",
        source_image_id="a",
        target_image_id="b",
        source_time_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        target_time_utc=datetime(2020, 1, 2, tzinfo=timezone.utc),
        grid_row=np.array([0]),
        grid_column=np.array([0]),
        source_xy_m=np.array([[0.0, 0.0]]),
        selected_matches=np.array([0]),
        candidate_matches=np.array([0]),
        support_radius_m=np.array([np.nan]),
        maximum_residual_m=np.array([np.nan]),
    )
    with pytest.raises(ValueError, match="explicit NaN"):
        DisplacementField(
            **common,
            displacement_m=np.array([[0.0, 0.0]]),
            available=np.array([False]),
        )

    field = DisplacementField(
        **common,
        displacement_m=np.array([[np.nan, np.nan]]),
        available=np.array([False]),
    )

    assert field.source_xy_m.dtype == np.float64
    assert not field.available.any()


def test_matcher_defaults_retain_selected_scientific_values():
    config = MatcherConfig()
    assert config.pixel_size_m == 80.0
    assert config.maximum_speed_m_per_day == 43_200.0
    assert config.tile_size_px == 512
    assert config.tile_batch_size == 4
    assert config.prefix_cuda_graph is True
    assert config.cuda_graph_warmup_batches == 3
    with pytest.raises(ValueError, match="tile_batch_size"):
        MatcherConfig(tile_batch_size=0)


def test_global_planning_defaults_are_explicit():
    routing = RoutingConfig()
    field = FieldConfig()
    assert field.grid_spacing_m == 4_000.0
    assert field.maximum_triangle_edge_m == 6_400.0
    assert field.missing_node_fallback is True
    assert routing.planning_grid_spacing_m == 4_000.0
    assert routing.candidate_minimum_elapsed_hours == 1.0
    assert routing.candidate_maximum_elapsed_hours == 96.0
    assert routing.candidate_minimum_overlap_fraction == 0.05
    assert routing.candidate_minimum_overlap_area_m2 == 1_024_000_000.0
    assert routing.maximum_recovery_elapsed_hours == 96.0
    assert routing.phase_correlation_failure == "same_center"
    assert routing.phase_correlation_minimum_response == 0.05
    assert routing.primary_maximum_pairs_per_target is None
    with pytest.raises(ValueError, match="primary_maximum_pairs_per_target"):
        RoutingConfig(primary_maximum_pairs_per_target=0)
    with pytest.raises(ValueError, match="planning_grid_spacing_m"):
        RoutingConfig(planning_grid_spacing_m=0)
    with pytest.raises(ValueError, match="cannot be combined"):
        RoutingConfig(
            primary_maximum_pairs_per_target=2,
            candidate_pair_ids=("a__b",),
        )
    assert RunConfig("run", "catalogue", "database", "output").retain_pair_matches is True
    assert RunConfig(
        "run", "catalogue", "database", "output"
    ).to_dict()["field"]["missing_node_fallback"] is True
    assert RunConfig(
        "run", "catalogue", "database", "output",
        field=FieldConfig(missing_node_fallback=False),
        retain_pair_matches=False,
    ).to_dict()["retain_pair_matches"] is False
    with pytest.raises(ValueError, match="requires retain_pair_matches"):
        RunConfig("run", "catalogue", "database", "output",
                  retain_pair_matches=False)
    with pytest.raises(ValueError, match="pair_workers"):
        RunConfig("run", "catalogue", "database", "output", pair_workers=0)

    with pytest.raises(ValueError, match="CUDA runs require pair_workers=1"):
        RunConfig(
            "run",
            "catalogue",
            "database",
            "output",
            pair_workers=2,
            matcher=MatcherConfig(device="cuda"),
        )


@pytest.mark.parametrize("retain_matches", [False, True])
def test_legacy_config_without_fallback_flag_keeps_disabled_identity(
    tmp_path, retain_matches
):
    legacy = RunConfig(
        "run", str(tmp_path / "catalogue"), str(tmp_path / "database"),
        str(tmp_path / "output"),
        field=FieldConfig(missing_node_fallback=False),
        retain_pair_matches=retain_matches,
    )
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy.to_dict()), encoding="utf-8")

    loaded = load_config(path)

    assert loaded.field.missing_node_fallback is False
    assert loaded.retain_pair_matches is retain_matches
    assert loaded.sha256 == legacy.sha256


def test_new_default_config_round_trips_with_fallback_enabled(tmp_path):
    config = RunConfig(
        "run", str(tmp_path / "catalogue"), str(tmp_path / "database"),
        str(tmp_path / "output"),
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    loaded = load_config(path)

    assert loaded.field.missing_node_fallback is True
    assert loaded.retain_pair_matches is True
    assert loaded.sha256 == config.sha256


def test_partial_field_config_uses_new_fallback_default(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "run_id": "run", "catalogue": "catalogue", "database": "database",
        "output_directory": "output", "field": {"grid_spacing_m": 4_000.0},
    }), encoding="utf-8")

    loaded = load_config(path)

    assert loaded.field.missing_node_fallback is True
    assert loaded.retain_pair_matches is True


def test_catalogue_infers_sentinel_platform_and_absolute_orbit(tmp_path):
    image_name = (
        "S1A_EW_GRDM_1SDH_20200401T010212_20200401T010317_031927_"
        "03AFA9_98E7.tiff"
    )
    catalogue_path = tmp_path / "catalogue.csv"
    catalogue_path.write_text(
        "image_id,path,time_utc\n"
        f"{image_name},{image_name},2020-04-01T01:02:12Z\n",
        encoding="utf-8",
    )

    record = load_catalogue(catalogue_path).records[0]

    assert record.platform == "S1A"
    assert record.absolute_orbit == 31_927


def test_catalogue_accepts_production_stac_properties(tmp_path):
    name = (
        "S1B_EW_GRDM_1SDH_20200402T023322_20200402T023422_020959_"
        "027C1C_7BBA"
    )
    path = tmp_path / "catalogue.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": name,
                        "geometry": None,
                        "properties": {
                            "scene_id": name,
                            "filepath": "/data/" + name + ".tiff",
                            "datetime": "2020-04-02T02:33:22Z",
                            "orbit_num": 20959,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    record = load_catalogue(path).records[0]

    assert record.image_id == name
    assert record.platform == "S1B"
    assert record.absolute_orbit == 20_959
