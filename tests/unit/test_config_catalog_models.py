from datetime import datetime, timezone

import numpy as np
import pytest

from limosat import (
    DisplacementField,
    ImageCatalogue,
    ImageRecord,
    MatcherConfig,
    RoutingConfig,
    RunConfig,
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
    assert routing.candidate_minimum_overlap_fraction == 0.25
    with pytest.raises(ValueError, match="pair_workers"):
        RunConfig("run", "catalogue", "database", "output", pair_workers=0)
