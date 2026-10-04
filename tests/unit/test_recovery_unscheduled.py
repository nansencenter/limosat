from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from shapely.geometry import box

from limosat import ImagePair, ImageRecord, TrajectoryPoint
from limosat.planning import PlannedPair
from limosat.recovery import RecoveryTargetStore, _write_recovery_targets


START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _point(identity, image, step, state, xy=None):
    return TrajectoryPoint(
        identity,
        image,
        START + timedelta(days=step),
        state,
        "missing" if xy is None else "seed_grid",
        None if xy is None else float(xy[0]),
        None if xy is None else float(xy[1]),
        None,
    )


def _images(names, footprints=None):
    footprints = footprints or {}
    return tuple(
        ImageRecord(
            name, Path(f"/tmp/{name}.tif"), START + timedelta(days=step),
            footprint=footprints.get(name),
        )
        for step, name in enumerate(names)
    )


def _planned(source, target, ordinal=0):
    return PlannedPair(
        ImagePair(source, target), ordinal, "candidate", 1.0, 1.0, 1, "component"
    )


def _config(tmp_path):
    from test_staged_execution import config

    return config(tmp_path)


def test_unscheduled_losses_are_targeted_with_dormant_rows(tmp_path):
    cfg = _config(tmp_path)
    images = _images(("a", "b", "c"))
    planned = _planned(images[0], images[2])
    batches = (
        (
            _point("dormant", "a", 0, "created", (10.0, 20.0)),
            _point("absent", "a", 0, "created", (30.0, 40.0)),
            _point("later", "a", 0, "created", (50.0, 60.0)),
            _point("present", "a", 0, "created", (70.0, 80.0)),
        ),
        (_point("later", "b", 1, "observed", (55.0, 60.0)),),
        (
            _point("dormant", "c", 2, "dormant"),
            _point("present", "c", 2, "observed", (75.0, 80.0)),
        ),
    )
    store = RecoveryTargetStore(cfg, tmp_path / "targets")
    counts = {"unscheduled_loss_positions": 0}

    selected = _write_recovery_targets(
        store, (planned,), images, batches, counts=counts,
    )

    assert selected == (planned,)
    assert store.trajectory_ids(planned.pair) == frozenset({"dormant", "absent"})
    np.testing.assert_array_equal(
        store.load(planned.pair), [[30.0, 40.0], [10.0, 20.0]]
    )
    assert counts == {"unscheduled_loss_positions": 1}


def test_unscheduled_loss_is_nominated_for_first_covering_target_only(tmp_path):
    cfg = _config(tmp_path)
    covers = box(0.0, 0.0, 100.0, 100.0)
    elsewhere = box(1_000.0, 1_000.0, 2_000.0, 2_000.0)
    images = _images(
        ("a", "b", "c", "d"), {"b": elsewhere, "c": covers, "d": covers}
    )
    to_b, to_c, to_d = (
        _planned(images[0], images[1], 0),
        _planned(images[0], images[2], 1),
        _planned(images[0], images[3], 2),
    )
    batches = ((_point("lost", "a", 0, "created", (30.0, 40.0)),), (), (), ())
    store = RecoveryTargetStore(cfg, tmp_path / "targets")

    selected = _write_recovery_targets(
        store, (to_b, to_c, to_d), images, batches,
    )

    assert selected == (to_c,)
    assert store.trajectory_ids(to_c.pair) == frozenset({"lost"})


def test_explicit_requests_are_not_extended_by_unscheduled_policy(tmp_path):
    cfg = _config(tmp_path)
    images = _images(("a", "b", "c"))
    planned = _planned(images[0], images[2])
    batches = (
        (
            _point("requested", "a", 0, "created", (10.0, 20.0)),
            _point("absent", "a", 0, "created", (30.0, 40.0)),
        ),
        (),
        (),
    )
    store = RecoveryTargetStore(cfg, tmp_path / "targets")

    _write_recovery_targets(
        store, (planned,), images, batches,
        {planned.pair.pair_id: frozenset({"requested"})},
    )

    assert store.trajectory_ids(planned.pair) == frozenset({"requested"})


def test_sqlite_recovery_targets_unscheduled_losses_once(tmp_path):
    from test_store_run_manifest import _config as store_config

    from limosat.store import RunStore

    covers = box(0.0, 0.0, 100.0, 100.0)
    images = _images(("a", "b", "c", "d"), {"c": covers, "d": covers})
    store = RunStore(store_config(tmp_path))
    store.replace_global_trajectories(
        (
            _point("lost", "a", 0, "created", (30.0, 40.0)),
            _point("dormant", "a", 0, "created", (10.0, 20.0)),
            _point("later", "a", 0, "created", (50.0, 60.0)),
            _point("outside", "a", 0, "created", (500.0, 500.0)),
            _point("later", "b", 1, "observed", (55.0, 60.0)),
            _point("dormant", "c", 2, "dormant"),
        )
    )

    yielded = dict(
        (pair.target.image_id, positions)
        for pair, positions, _identities in store.iter_targeted_recovery_positions(
            (ImagePair(images[0], images[2]), ImagePair(images[0], images[3]))
        )
    )

    np.testing.assert_array_equal(yielded["c"], [[10.0, 20.0], [30.0, 40.0]])
    assert yielded["d"].shape == (0, 2)


def test_collect_mode_records_candidates_without_writing(tmp_path):
    cfg = _config(tmp_path)
    images = _images(("a", "b", "c"))
    to_b, to_c = _planned(images[0], images[1], 0), _planned(images[0], images[2], 1)
    batches = (
        (_point("lost", "a", 0, "created", (10.0, 20.0)),),
        (_point("lost", "b", 1, "dormant"),),
        (_point("lost", "c", 2, "dormant"),),
    )
    store = RecoveryTargetStore(cfg, tmp_path / "targets")
    collected = {}
    selected = _write_recovery_targets(
        store, (to_b, to_c), images, batches, collect=collected,
    )
    assert selected == (to_b, to_c)
    assert collected == {to_b.pair.pair_id: ("lost",), to_c.pair.pair_id: ("lost",)}
    assert store.count() == 0


def test_recovery_target_policy_defaults_to_loss_targeted_and_keeps_identity(tmp_path):
    from dataclasses import replace

    import pytest
    from limosat import RoutingConfig

    cfg = _config(tmp_path)
    assert cfg.routing.recovery_target_policy == "loss_targeted"
    assert "recovery_target_policy" not in cfg.to_dict()["routing"]
    legacy = replace(cfg, routing=replace(cfg.routing, recovery_target_policy="all_losses"))
    assert legacy.to_dict()["routing"]["recovery_target_policy"] == "all_losses"
    with pytest.raises(ValueError):
        RoutingConfig(recovery_target_policy="everything")
