from datetime import datetime, timedelta, timezone
from pathlib import Path

from shapely import box

from limosat import ImageCatalogue, ImageRecord, RoutingConfig, plan_candidate_pairs
from limosat.planning import recovery_candidates


START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _image(name, day, component, footprint=None):
    return ImageRecord(
        name,
        Path(f"/tmp/{name}.tif"),
        START + timedelta(days=day),
        component,
        footprint or box(0.0, 0.0, 10_000.0, 10_000.0),
    )


def test_candidate_plan_crosses_components_and_is_deterministic():
    records = [
        _image("c", 2, "former-c"),
        _image("a", 0, "former-a"),
        _image("b", 1, "former-b"),
    ]
    first = plan_candidate_pairs(ImageCatalogue(records), RoutingConfig())
    second = plan_candidate_pairs(
        ImageCatalogue(tuple(reversed(records))), RoutingConfig()
    )

    summary = [
        (item.ordinal, item.pair.pair_id, item.selection)
        for item in first
    ]
    assert summary == [
        (0, "a__b", "primary"),
        (1, "a__c", "candidate"),
        (2, "b__c", "primary"),
    ]
    assert summary == [
        (item.ordinal, item.pair.pair_id, item.selection)
        for item in second
    ]


def test_equal_time_sources_remain_competing_primary_pairs():
    catalogue = ImageCatalogue(
        [
            _image("a", 0, "left"),
            _image("b", 0, "right"),
            _image("target", 1, "target"),
        ]
    )

    planned = plan_candidate_pairs(catalogue, RoutingConfig())

    assert {
        item.pair.pair_id
        for item in planned
        if item.selection == "primary"
    } == {"a__target", "b__target"}


def test_missing_footprints_do_not_restore_component_boundaries():
    catalogue = ImageCatalogue(
        [
            ImageRecord("a", Path("/tmp/missing-a.tif"), START, "former-a"),
            ImageRecord(
                "b",
                Path("/tmp/missing-b.tif"),
                START + timedelta(days=1),
                "former-b",
            ),
        ]
    )

    planned = plan_candidate_pairs(catalogue, RoutingConfig())

    assert [item.pair.pair_id for item in planned] == ["a__b"]
    assert planned[0].overlap_fraction is None


def test_recovery_candidates_are_bounded_and_recent_first():
    catalogue = ImageCatalogue(
        [_image(name, day, name) for day, name in enumerate(("a", "b", "c", "d"))]
    )
    planned = plan_candidate_pairs(catalogue, RoutingConfig())

    recovery = recovery_candidates(planned, maximum_per_target=1)

    assert [item.pair.pair_id for item in recovery] == ["a__c", "b__d"]
