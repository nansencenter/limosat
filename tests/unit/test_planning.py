from datetime import datetime, timedelta, timezone
from pathlib import Path

from shapely import box

import pytest

from limosat import ImageCatalogue, ImageRecord, RoutingConfig, plan_candidate_pairs
from limosat.planning import (
    build_candidate_plan,
    select_overlap_probe,
)


START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _image(name, day, component, footprint=None):
    return ImageRecord(
        name,
        Path(f"/tmp/{name}.tif"),
        START + timedelta(days=day),
        component,
        footprint or box(0.0, 0.0, 100_000.0, 100_000.0),
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


def test_candidate_plan_requires_fractional_and_absolute_overlap():
    source = _image("source", 0, "a", box(0, 0, 100_000, 100_000))
    small_overlap = _image(
        "small", 1, "b", box(95_000, 0, 195_000, 100_000)
    )

    excluded = build_candidate_plan(
        ImageCatalogue([source, small_overlap]), RoutingConfig()
    )
    accepted = build_candidate_plan(
        ImageCatalogue([source, small_overlap]),
        RoutingConfig(candidate_minimum_overlap_area_m2=500_000_000.0),
    )

    assert excluded.pairs == ()
    assert excluded.exclusion_counts["below_minimum_overlap"] == 1
    assert [item.pair.pair_id for item in accepted.pairs] == ["source__small"]
    assert accepted.pairs[0].overlap_fraction == 0.05
    assert accepted.pairs[0].overlap_area_m2 == 500_000_000.0


def test_same_platform_absolute_orbit_is_excluded_and_counted():
    same_a = ImageRecord(
        "same-a", Path("/tmp/a.tif"), START,
        footprint=box(0, 0, 100_000, 100_000),
        platform="S1A", absolute_orbit=100,
    )
    same_b = ImageRecord(
        "same-b", Path("/tmp/b.tif"), START + timedelta(days=1),
        footprint=box(0, 0, 100_000, 100_000),
        platform="S1A", absolute_orbit=100,
    )
    other_platform = ImageRecord(
        "other", Path("/tmp/c.tif"), START + timedelta(days=2),
        footprint=box(0, 0, 100_000, 100_000),
        platform="S1B", absolute_orbit=100,
    )

    plan = build_candidate_plan(
        ImageCatalogue([same_a, same_b, other_platform]), RoutingConfig()
    )

    assert "same-a__same-b" not in {item.pair.pair_id for item in plan.pairs}
    assert "same-b__other" in {item.pair.pair_id for item in plan.pairs}
    assert plan.exclusion_counts["same_acquisition_pass"] == 1


def test_required_orbit_metadata_fails_before_gpu_planning():
    with pytest.raises(ValueError, match="platform and absolute orbit"):
        plan_candidate_pairs(
            ImageCatalogue([_image("a", 0, "a"), _image("b", 1, "b")]),
            RoutingConfig(require_orbit_metadata=True),
        )


def test_diagnostic_pair_allowlist_runs_every_retained_pair_as_primary():
    catalogue = ImageCatalogue(
        [_image(name, day, name) for day, name in enumerate(("a", "b", "c"))]
    )

    planned = plan_candidate_pairs(
        catalogue, RoutingConfig(candidate_pair_ids=("a__c",))
    )

    assert [(item.pair.pair_id, item.selection) for item in planned] == [
        ("a__c", "primary")
    ]


def test_overlap_probe_is_bounded_and_deterministic():
    catalogue = ImageCatalogue(
        [_image(name, day, name) for day, name in enumerate(("a", "b", "c", "d"))]
    )
    planned = plan_candidate_pairs(catalogue, RoutingConfig())

    first = select_overlap_probe(planned, ((0.9, 1.0),), maximum_per_bin=2)
    second = select_overlap_probe(
        tuple(reversed(planned)), ((0.9, 1.0),), maximum_per_bin=2
    )

    assert [item.pair.pair_id for item in first["0.900-1.000"]] == [
        item.pair.pair_id for item in second["0.900-1.000"]
    ]
    assert len(first["0.900-1.000"]) == 2


def _primary(plan):
    return {item.pair.pair_id for item in plan.pairs if item.selection == "primary"}


def test_next_observation_sends_each_cell_to_its_next_image():
    # A's west is next imaged by B (6 h), its east by C (12 h); D (18 h) re-images both
    # but is never the next observation from A, so A -> D is not kept. B -> D, C -> D are.
    a = _image("a", 0.0, "a", box(0, 0, 100_000, 40_000))
    b = _image("b", 0.25, "b", box(0, 0, 50_000, 40_000))
    c = _image("c", 0.5, "c", box(50_000, 0, 100_000, 40_000))
    d = _image("d", 0.75, "d", box(0, 0, 100_000, 40_000))
    plan = build_candidate_plan(ImageCatalogue([a, b, c, d]), RoutingConfig())
    kept = _primary(plan)
    assert {"a__b", "a__c", "b__d", "c__d"} <= kept
    assert "a__d" not in kept
    assert plan.exclusion_counts["next_observation_backup_pairs"] == 0


def test_next_observation_backup_for_long_pairs_and_minimum_cells():
    a = _image("a", 0.0, "a", box(0, 0, 100_000, 40_000))
    b = _image("b", 2.0, "b", box(0, 0, 100_000, 40_000))
    c = _image("c", 3.0, "c", box(0, 0, 100_000, 40_000))
    sliver = _image("s", 1.0, "s", box(0, 0, 8_000, 40_000))
    catalogue = ImageCatalogue([a, b, c, sliver])
    plain = build_candidate_plan(catalogue, RoutingConfig())
    assert "a__s" not in _primary(plain)            # fewer than 25 cells
    assert "a__b" in _primary(plain) and "a__c" not in _primary(plain)
    backed = build_candidate_plan(catalogue, RoutingConfig(next_observation_backup_after_hours=30.0))
    assert {"a__b", "a__c"} <= _primary(backed)      # A -> B is 48 h, so C backs it up
    assert backed.exclusion_counts["next_observation_backup_pairs"] >= 1


def test_next_observation_cell_filter_drops_land():
    a = _image("a", 0.0, "a", box(0, 0, 100_000, 40_000))
    b = _image("b", 0.25, "b", box(0, 0, 100_000, 40_000))
    land = lambda image, cells: {cell for cell in cells if cell[0] * 4_000 < 30_000}  # noqa: E731
    plan = build_candidate_plan(ImageCatalogue([a, b]), RoutingConfig(), cell_filter=land)
    assert plan.exclusion_counts["next_observation_assigned_cells"] < 8 * 11


def test_surface_cell_planning_requires_a_surface_source():
    from limosat.config import RunConfig
    with pytest.raises(ValueError, match="plan_surface_cells"):
        RunConfig("run", "catalogue", "database", "output", routing=RoutingConfig(plan_surface_cells=True))
