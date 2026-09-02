from datetime import datetime, timezone
import json

import numpy as np
import pytest

from limosat import (
    DisplacementField,
    ImageCatalogue,
    ImageRecord,
    MatcherConfig,
    RoutingConfig,
    RunConfig,
    load_catalogue,
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
    assert config.maximum_speed_m_per_day == 30_000.0
    assert config.tile_size_px == 512


def test_global_planning_defaults_are_explicit():
    routing = RoutingConfig()
    assert routing.candidate_minimum_elapsed_hours == 1.0
    assert routing.candidate_maximum_elapsed_hours == 96.0
    assert routing.candidate_minimum_overlap_fraction == 0.05
    assert routing.candidate_minimum_overlap_area_m2 == 1_024_000_000.0
    assert routing.maximum_recovery_elapsed_hours == 96.0
    assert routing.phase_correlation_failure == "same_center"
    assert routing.phase_correlation_minimum_response == 0.05
    assert RunConfig("run", "catalogue", "database", "output").retain_pair_matches is False
    with pytest.raises(ValueError, match="pair_workers"):
        RunConfig("run", "catalogue", "database", "output", pair_workers=0)


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
