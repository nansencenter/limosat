from dataclasses import replace

import numpy as np
import pytest

from limosat import FieldEdge, TrajectoryConfig
from limosat.config import RunConfig
from limosat.trajectory import iter_global_trajectory_batches
from test_trajectories_deformation import GRID, _field, _images, _settings


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
        gap_rows=False,
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
    assert "merge_radius_m" not in base.to_dict()["trajectories"]
    merged = replace(base, trajectories=TrajectoryConfig(merge_radius_m=500.0))
    assert merged.sha256 != base.sha256
    with pytest.raises(ValueError, match="merge_radius_m"):
        TrajectoryConfig(merge_radius_m=3_000.0)
