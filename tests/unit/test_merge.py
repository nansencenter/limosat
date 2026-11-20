from dataclasses import replace

import numpy as np
import pytest

from limosat import FieldEdge, TrajectoryConfig
from limosat.config import RunConfig
from limosat.trajectory import iter_global_trajectory_batches
from test_trajectories import GRID, _field, _images, _settings


def _converging_edges():
    """Four tracks seeded on a 1 km grid; in b -> c the track at (100, 0) is carried to
    (900, 0), 200 m from the one at (1100, 0)."""
    first = _field("a", "b", 0, [100.0, 0.0])
    second = _field("b", "c", 1, [0.0, 0.0], points=GRID + [100.0, 0.0])
    displacement = second.displacement_m.copy()
    displacement[0] = [800.0, 0.0]
    second = replace(second, displacement_m=displacement)
    return [FieldEdge(first), FieldEdge(second)]


def _run(radius):
    return list(iter_global_trajectory_batches(
        _converging_edges(), _images(), _settings(),
        TrajectoryConfig(new_point_exclusion_radius_m=500.0, merge_radius_m=radius),
    ))


def test_without_merging_both_converging_tracks_continue():
    batches = _run(None)
    assert len([p for p in batches[2].points if p.state == "observed"]) == 4
    assert not any(batch.merges for batch in batches)


def test_merging_ends_one_of_two_tracks_on_the_same_ice():
    batches = _run(300.0)
    merges = batches[2].merges
    assert len(merges) == 1
    event = merges[0]
    assert event.image_id == "c" and event.separation_m == pytest.approx(200.0)
    observed = {p.trajectory_id for p in batches[2].points if p.state == "observed"}
    assert len(observed) == 3
    assert event.trajectory_id not in observed and event.merged_into in observed
    assert event.trajectory_id not in {e.trajectory_id for e in batches[2].extensions}


def test_merge_radius_configuration():
    base = RunConfig("run", "catalogue", "database", "output")
    assert base.trajectories.merge_radius_m is None
    merged = replace(base, trajectories=TrajectoryConfig(merge_radius_m=500.0))
    assert merged.sha256 != base.sha256
    with pytest.raises(ValueError, match="merge_radius_m"):
        TrajectoryConfig(merge_radius_m=3_000.0)


def _run_young(hours):
    return list(iter_global_trajectory_batches(
        _converging_edges(), _images(), _settings(),
        TrajectoryConfig(new_point_exclusion_radius_m=500.0, merge_radius_m=300.0,
                         merge_maximum_younger_age_hours=hours),
    ))


def test_young_rule_keeps_converging_tracks_with_a_shared_history():
    # Both tracks were positioned together in a and b before converging in c, so they
    # are established tracks on (possibly) different ice: never merged, whatever the age.
    for hours in (1.0, 1.0e6):
        batches = _run_young(hours)
        assert not any(batch.merges for batch in batches)
        assert len([p for p in batches[2].points if p.state == "observed"]) == 4


def test_merge_duplicates_respects_the_eligibility_check():
    from datetime import datetime, timezone
    from limosat.trajectory import _merge_duplicates
    from limosat.catalog import ImageRecord
    from pathlib import Path
    image = ImageRecord("c", Path("/tmp/c.tif"), datetime(2020, 1, 1, tzinfo=timezone.utc), "c")
    current = {"old": np.array([0.0, 0.0]), "young": np.array([100.0, 0.0])}
    observations = {"old": 10, "young": 1}

    class P:
        def __init__(self, i):
            self.trajectory_id, self.selected_matches, self.maximum_residual_m = i, 10, 50.0
    points = [P("old"), P("young")]
    kept = dict(current)
    assert _merge_duplicates(kept, points, observations, 300.0, image, eligible_loser=lambda l, w: False) == []
    assert set(kept) == {"old", "young"}
    events = _merge_duplicates(dict(current), points, observations, 300.0, image, eligible_loser=lambda l, w: l == "young")
    assert [(e.trajectory_id, e.merged_into) for e in events] == [("young", "old")]
    with pytest.raises(ValueError, match="merge_maximum_younger_age_hours"):
        TrajectoryConfig(merge_maximum_younger_age_hours=48.0)
