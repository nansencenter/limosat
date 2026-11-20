from datetime import datetime, timedelta, timezone
from pathlib import Path

from shapely import box

import pytest

from limosat import ImageCatalogue, ImageRecord, RoutingConfig, plan_candidate_pairs
from limosat.planning import (
    build_candidate_plan,
    recovery_candidates,
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


def test_recovery_candidates_include_all_elapsed_eligible_pairs_recent_first():
    catalogue = ImageCatalogue(
        [_image(name, day, name) for day, name in enumerate(("a", "b", "c", "d"))]
    )
    planned = plan_candidate_pairs(catalogue, RoutingConfig())

    recovery = recovery_candidates(planned)

    assert [item.pair.pair_id for item in recovery] == [
        "a__c",
        "b__d",
        "a__d",
    ]


def test_recovery_uses_elapsed_time_not_intervening_image_count():
    catalogue = ImageCatalogue(
        [_image(name, day, name) for day, name in enumerate(("a", "b", "c", "d"))]
    )
    planned = plan_candidate_pairs(catalogue, RoutingConfig())

    long_candidate = next(
        item for item in planned if item.pair.pair_id == "a__d"
    )

    assert long_candidate.skipped_images == 2
    assert recovery_candidates(
        (long_candidate,), maximum_elapsed_hours=72.0
    ) == (long_candidate,)
    assert recovery_candidates(
        (long_candidate,), maximum_elapsed_hours=71.9
    ) == ()


def test_primary_plan_keeps_most_recent_source_for_each_spatial_cell():
    old_full = _image("old-full", 0, "old", box(0, 0, 100_000, 100_000))
    recent_left = _image(
        "recent-left", 1, "left", box(0, 0, 60_000, 100_000)
    )
    target = _image("target", 2, "target", box(0, 0, 100_000, 100_000))

    first = plan_candidate_pairs(ImageCatalogue([old_full, recent_left, target]), RoutingConfig())
    second = plan_candidate_pairs(
        ImageCatalogue([target, recent_left, old_full]), RoutingConfig()
    )

    def target_primary_ids(planned):
        return {
            item.pair.pair_id
            for item in planned
            if item.pair.target.image_id == "target"
            and item.selection == "primary"
        }

    assert target_primary_ids(first) == {
        "old-full__target",
        "recent-left__target",
    }
    assert target_primary_ids(second) == target_primary_ids(first)


def test_diagnostic_primary_maximum_keeps_largest_spatial_contributors():
    old_full = _image("old-full", 0, "old", box(0, 0, 100_000, 100_000))
    recent_left = _image(
        "recent-left", 1, "left", box(0, 0, 60_000, 100_000)
    )
    newest_left = _image(
        "newest-left", 2, "newest", box(0, 0, 30_000, 100_000)
    )
    target = _image("target", 3, "target", box(0, 0, 100_000, 100_000))
    records = [old_full, recent_left, newest_left, target]
    routing = RoutingConfig(primary_maximum_pairs_per_target=2)

    first = build_candidate_plan(ImageCatalogue(records), routing)
    second = build_candidate_plan(ImageCatalogue(tuple(reversed(records))), routing)

    def target_primary_ids(plan):
        return {
            item.pair.pair_id
            for item in plan.pairs
            if item.pair.target.image_id == "target"
            and item.selection == "primary"
        }

    assert target_primary_ids(first) == {
        "old-full__target",
        "newest-left__target",
    }
    assert target_primary_ids(second) == target_primary_ids(first)
    assert first.exclusion_counts["primary_pairs_before_target_maximum"] == 6
    assert first.exclusion_counts["primary_pairs_excluded_by_target_maximum"] == 1
    assert first.exclusion_counts["targets_affected_by_primary_pair_maximum"] == 1


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


def test_source_coverage_completion_gives_ice_left_behind_a_pair_forward():
    # B keeps only its most useful source (cap 1): A2, the more recent and larger one.
    # A's western ice is imaged again only by B, so without completion no pair leaves A
    # over it; completion adds A -> B.
    a = _image("a", 0.0, "a", box(0, 0, 60_000, 40_000))
    a2 = _image("a2", 0.25, "a2", box(40_000, 0, 100_000, 40_000))
    b = _image("b", 0.5, "b", box(0, 0, 100_000, 40_000))
    catalogue = ImageCatalogue([a, a2, b])

    def primary(plan):
        return {item.pair.pair_id for item in plan.pairs if item.selection == "primary"}

    plain = build_candidate_plan(catalogue, RoutingConfig(primary_maximum_pairs_per_target=1))
    assert "a__b" not in primary(plain)
    assert plain.exclusion_counts["primary_pairs_added_for_source_coverage"] == 0
    completed = build_candidate_plan(catalogue, RoutingConfig(
        primary_maximum_pairs_per_target=1, source_coverage_completion=True))
    assert primary(completed) == primary(plain) | {"a__b"}
    assert completed.exclusion_counts["primary_pairs_added_for_source_coverage"] == 1


def test_source_coverage_completion_is_off_by_default_and_keeps_identity():
    from dataclasses import replace
    from limosat.config import RunConfig
    base = RunConfig("run", "catalogue", "database", "output")
    assert "source_coverage_completion" not in base.to_dict()["routing"]
    on = replace(base, routing=RoutingConfig(source_coverage_completion=True))
    assert on.sha256 != base.sha256
    with pytest.raises(ValueError, match="source_coverage_minimum_cells"):
        RoutingConfig(source_coverage_minimum_cells=0)


def _primary(plan):
    return {item.pair.pair_id for item in plan.pairs if item.selection == "primary"}


def test_next_observation_sends_each_cell_to_its_next_image():
    # A's west is next imaged by B (6 h), its east by C (12 h); D (18 h) re-images both
    # but is never the next observation from A, so A -> D is not kept. B -> D, C -> D are.
    a = _image("a", 0.0, "a", box(0, 0, 100_000, 40_000))
    b = _image("b", 0.25, "b", box(0, 0, 50_000, 40_000))
    c = _image("c", 0.5, "c", box(50_000, 0, 100_000, 40_000))
    d = _image("d", 0.75, "d", box(0, 0, 100_000, 40_000))
    plan = build_candidate_plan(ImageCatalogue([a, b, c, d]), RoutingConfig(primary_selection="next_observation"))
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
    plain = build_candidate_plan(catalogue, RoutingConfig(primary_selection="next_observation"))
    assert "a__s" not in _primary(plain)            # fewer than 25 cells
    assert "a__b" in _primary(plain) and "a__c" not in _primary(plain)
    backed = build_candidate_plan(catalogue, RoutingConfig(
        primary_selection="next_observation", next_observation_backup_after_hours=30.0))
    assert {"a__b", "a__c"} <= _primary(backed)      # A -> B is 48 h, so C backs it up
    assert backed.exclusion_counts["next_observation_backup_pairs"] >= 1


def test_next_observation_cell_filter_drops_land():
    a = _image("a", 0.0, "a", box(0, 0, 100_000, 40_000))
    b = _image("b", 0.25, "b", box(0, 0, 100_000, 40_000))
    land = lambda image, cells: {cell for cell in cells if cell[0] * 4_000 < 30_000}  # noqa: E731
    plan = build_candidate_plan(ImageCatalogue([a, b]), RoutingConfig(primary_selection="next_observation"), cell_filter=land)
    assert plan.exclusion_counts["next_observation_assigned_cells"] < 8 * 11


def test_next_observation_settings_keep_identity_at_defaults():
    from dataclasses import replace
    from limosat.config import RunConfig
    base = RunConfig("run", "catalogue", "database", "output")
    assert "primary_selection" not in base.to_dict()["routing"]
    on = replace(base, routing=RoutingConfig(primary_selection="next_observation"))
    assert on.sha256 != base.sha256
    with pytest.raises(ValueError, match="source_coverage_completion"):
        RoutingConfig(primary_selection="next_observation", source_coverage_completion=True)
    with pytest.raises(ValueError, match="plan_surface_cells"):
        RunConfig("run", "catalogue", "database", "output", routing=RoutingConfig(plan_surface_cells=True))


def test_next_observation_records_assigned_cells_for_targeted_tiles():
    from limosat.planning import assigned_positions
    a = _image("a", 0.0, "a", box(0, 0, 100_000, 40_000))
    b = _image("b", 0.25, "b", box(0, 0, 50_000, 40_000))
    c = _image("c", 0.5, "c", box(0, 0, 100_000, 40_000))
    plan = build_candidate_plan(ImageCatalogue([a, b, c]), RoutingConfig(
        primary_selection="next_observation", primary_tiles="assigned"))
    assert plan.assigned_cells is not None
    west = assigned_positions(plan, "a__b", 4_000.0)
    east = assigned_positions(plan, "a__c", 4_000.0)
    assert len(west) and len(east)
    assert west[:, 0].max() <= 50_000 and east[:, 0].min() >= 48_000   # A->C only for A's east
    assert not ({tuple(p) for p in west} & {tuple(p) for p in east})
    with pytest.raises(ValueError, match="primary_tiles"):
        RoutingConfig(primary_tiles="assigned")


def test_assigned_tiles_change_matching_identity_only_when_set():
    from dataclasses import replace
    from limosat.config import RunConfig
    from limosat.pair_identity import matching_sha256
    base = RunConfig("run", "catalogue", "database", "output")
    nextobs = replace(base, routing=RoutingConfig(primary_selection="next_observation"))
    assigned = replace(base, routing=RoutingConfig(primary_selection="next_observation", primary_tiles="assigned"))
    assert matching_sha256(nextobs) == matching_sha256(base)
    assert matching_sha256(assigned) != matching_sha256(base)
