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
    assert {k: v[0] for k, v in collected.items()} == {
        to_b.pair.pair_id: ("lost",), to_c.pair.pair_id: ("lost",)}
    np.testing.assert_array_equal(collected[to_b.pair.pair_id][1], [[10.0, 20.0]])
    # A single measurement has no velocity, so the prediction is the source position.
    np.testing.assert_array_equal(collected[to_b.pair.pair_id][2], [[10.0, 20.0]])
    assert store.count() == 0


def test_recovery_target_policy_defaults_to_loss_targeted_and_keeps_identity(tmp_path):
    from dataclasses import replace

    import pytest
    from limosat import RoutingConfig

    cfg = _config(tmp_path)
    assert cfg.routing.recovery_target_policy == "loss_targeted"
    assert "recovery_target_policy" not in cfg.to_dict()["routing"]
    legacy = replace(cfg, routing=replace(cfg.routing, recovery_target_policy="all_losses"))
    assert "recovery_target_policy" not in legacy.to_dict()["routing"]
    assert legacy.sha256 == cfg.sha256
    with pytest.raises(ValueError):
        RoutingConfig(recovery_target_policy="everything")


def test_target_validity_filter_drops_options_on_no_data(tmp_path, monkeypatch):
    from dataclasses import replace
    from limosat import imagery
    from limosat.recovery import _filter_by_target_validity

    cfg = _config(tmp_path)
    cfg = replace(cfg, routing=replace(cfg.routing, recovery_minimum_target_valid_fraction=0.5))
    images = _images(("a", "b"))
    pair = _planned(images[0], images[1])
    monkeypatch.setattr(imagery, "valid_share_around",
                        lambda path, x, y, *args, **kwargs: np.where(x > 0, 1.0, 0.2))
    kept, counts = _filter_by_target_validity(
        cfg, {pair.pair.pair_id: (("valid", "nodata"), np.zeros((2, 2)),
                                  np.array([[5.0, 0.0], [-5.0, 0.0]]))},
        {pair.pair.pair_id: pair},
    )
    assert kept == {pair.pair.pair_id: ("valid",)}
    assert counts == {"validity_options_checked": 2, "validity_options_dropped": 1,
                      "validity_pairs_without_valid_losses": 0}
    # Selection settings never change the configuration identity.
    assert "recovery_minimum_target_valid_fraction" not in cfg.to_dict()["routing"]
    assert cfg.sha256 == _config(tmp_path).sha256


def test_collected_prediction_moves_by_last_velocity(tmp_path):
    cfg = _config(tmp_path)
    images = _images(("a", "b", "c"))
    planned = _planned(images[1], images[2], 0)
    batches = (
        (_point("moving", "a", 0, "created", (0.0, 0.0)),),
        (_point("moving", "b", 1, "observed", (100.0, 0.0)),),
        (_point("moving", "c", 2, "dormant"),),
    )
    collected = {}
    _write_recovery_targets(
        RecoveryTargetStore(cfg, tmp_path / "targets"), (planned,), images, batches,
        collect=collected,
    )
    identities, frozen, predicted = collected[planned.pair.pair_id]
    lag = planned.pair.elapsed_seconds / 86_400.0
    gap = (images[1].time_utc - images[0].time_utc).total_seconds() / 86_400.0
    np.testing.assert_allclose(frozen, [[100.0, 0.0]])
    np.testing.assert_allclose(predicted, [[100.0 + 100.0 / gap * lag, 0.0]])
