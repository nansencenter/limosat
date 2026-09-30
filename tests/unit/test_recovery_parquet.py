from datetime import datetime, timedelta, timezone
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from limosat import FieldEdge, ImagePair, ImageRecord, PairProductStore, TrajectoryPoint
from limosat.planning import PlannedPair
from limosat.recovery import RecoveryTargetStore, _write_recovery_targets
from limosat.trajectory import iter_frozen_primary_augmentation_batches


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


def test_recovery_targets_are_frozen_source_positions_for_dormant_rows(tmp_path):
    from test_staged_execution import config

    cfg = config(tmp_path)
    images = tuple(
        ImageRecord(name, Path(f"/tmp/{name}.tif"), START + timedelta(days=step))
        for step, name in enumerate(("a", "b", "c"))
    )
    planned = PlannedPair(
        ImagePair(images[0], images[2]),
        ordinal=0,
        selection="candidate",
        overlap_fraction=1.0,
        overlap_area_m2=1.0,
        skipped_images=1,
        planning_component_id="component",
    )
    batches = (
        (
            _point("dormant", "a", 0, "created", (10.0, 20.0)),
            _point("observed", "a", 0, "created", (30.0, 40.0)),
        ),
        (),
        (
            _point("dormant", "c", 2, "dormant"),
            _point("observed", "c", 2, "observed", (50.0, 60.0)),
        ),
    )
    store = RecoveryTargetStore(cfg, tmp_path / "targets")

    selected = _write_recovery_targets(store, (planned,), images, batches)

    assert selected == (planned,)
    np.testing.assert_array_equal(
        store.load(planned.pair), np.array([[10.0, 20.0]])
    )
    assert store.trajectory_ids(planned.pair) == frozenset({"dormant"})
    assert store.count() == 1


def test_recovery_target_store_rejects_changed_identity_for_same_coordinates(tmp_path):
    from test_staged_execution import catalogue, config

    cfg = config(tmp_path)
    images = catalogue(tmp_path).chronological()
    pair = ImagePair(images[0], images[2])
    store = RecoveryTargetStore(cfg, tmp_path / "targets")
    coordinates = np.array([[10.0, 20.0]])
    store.save(pair, coordinates, trajectory_ids=["requested"])

    with pytest.raises(ValueError, match="identities differ"):
        store.save(pair, coordinates, trajectory_ids=["different"])
    assert store.trajectory_ids(pair) == frozenset({"requested"})


def test_explicit_recovery_request_accepts_absent_target_row(tmp_path):
    from test_staged_execution import config

    cfg = config(tmp_path)
    images = tuple(
        ImageRecord(name, Path(f"/tmp/{name}.tif"), START + timedelta(days=step))
        for step, name in enumerate(("a", "b", "c"))
    )
    planned = PlannedPair(
        ImagePair(images[0], images[2]), 0, "candidate", 1.0, 1.0, 1, "component"
    )
    batches = (
        (_point("absent", "a", 0, "created", (10.0, 20.0)),
         _point("already-measured", "a", 0, "created", (30.0, 40.0))),
        (),
        (_point("already-measured", "c", 2, "observed", (50.0, 60.0)),),
    )
    store = RecoveryTargetStore(cfg, tmp_path / "targets")

    selected = _write_recovery_targets(
        store, (planned,), images, batches,
        {planned.pair.pair_id: frozenset({"absent"})},
    )

    assert selected == (planned,)
    np.testing.assert_array_equal(store.load(planned.pair), [[10.0, 20.0]])


def test_saved_recovery_pair_fills_absent_row_then_primary_continues(tmp_path):
    from test_staged_execution import GRID, config, result

    cfg = config(tmp_path)
    images = tuple(
        ImageRecord(name, tmp_path / f"{name}.tif", START + timedelta(days=step))
        for step, name in enumerate(("a", "b", "c", "d"))
    )
    seeds = tuple(
        _point(f"parcel-{index}", "a", 0, "created", xy)
        for index, xy in enumerate(GRID)
    )
    batches = (seeds, (), (), ())
    recovery_pair = ImagePair(images[0], images[2])
    primary_pair = ImagePair(images[2], images[3])
    planned = PlannedPair(
        recovery_pair, 0, "candidate", 1.0, 1.0, 1, "component"
    )
    targets = RecoveryTargetStore(cfg, tmp_path / "targets")
    requests = {recovery_pair.pair_id: frozenset(point.trajectory_id for point in seeds)}
    assert _write_recovery_targets(
        targets, (planned,), images, batches, requests
    ) == (planned,)
    positions = targets.load(recovery_pair)
    products = PairProductStore(cfg)
    products.save(recovery_pair, "recovery", True, result(recovery_pair), positions)
    primary_result = result(primary_pair)
    shifted_field = replace(primary_result.field, source_xy_m=GRID + [200.0, 0.0])
    products.save(
        primary_pair, "primary", False,
        replace(primary_result, field=shifted_field),
    )
    edges = (
        FieldEdge(products.load(recovery_pair, "recovery", True, positions).result.field,
                  pair_kind="recovery"),
        FieldEdge(products.load(primary_pair, "primary", False).result.field),
    )

    additions = tuple(
        point
        for batch in iter_frozen_primary_augmentation_batches(
            batches, edges, images, cfg.field
        )
        for point in batch.points
    )

    assert len(additions) == 8
    assert len({(point.trajectory_id, point.image_id) for point in additions}) == 8
    assert {point.state for point in additions if point.image_id == "c"} == {
        "reappeared"
    }
    assert {point.position_basis for point in additions if point.image_id == "d"} == {
        "post_reappearance_primary_field"
    }

    blocked = tuple(
        point
        for batch in iter_frozen_primary_augmentation_batches(
            batches, edges, images, cfg.field,
            target_validity_factory=lambda _image: (
                lambda xy: np.zeros(len(xy), dtype=bool)
            ),
        )
        for point in batch.points
    )
    assert blocked == ()


def test_recovery_workers_can_use_parquet_targets_without_sqlite(tmp_path):
    from test_staged_execution import Processor, catalogue, config
    from limosat import RunStages

    cfg = config(tmp_path)
    images = catalogue(tmp_path)
    pair = ImagePair(images.chronological()[0], images.chronological()[2])
    target_root = tmp_path / "recovery-targets"
    RecoveryTargetStore(cfg, target_root / "pairs").save(
        pair, np.array([[0.0, 0.0], [1_000.0, 0.0]])
    )
    (target_root / "recovery-targets-manifest-v1.json").write_text(
        json.dumps(
            {
                "recovery_target_schema_version": 1,
                "run_id": cfg.run_id,
                "config_sha256": cfg.sha256,
                "eligible_recovery_pairs": 1,
                "targeted_recovery_pairs": 1,
                "pairs": [{"pair_id": pair.pair_id}],
            }
        )
    )
    processor = Processor()

    result = RunStages(cfg, images, processor).process_pairs(
        "recovery", recovery_target_directory=str(target_root)
    )

    assert result["planned_pairs"] == 1
    assert result["assigned_pairs"] == 1
    assert result["computed_pairs"] == 1
    assert processor.calls == [("a__c", True)]
    assert not Path(cfg.database).exists()


def test_parquet_recovery_composes_deltas_without_rewriting_primary(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    from test_staged_execution import Processor, catalogue, config
    from limosat import RunStages
    from limosat.compose import compose_primary_parquet, compose_recovery_parquet
    from limosat.recovery import prepare_recovery_targets
    from limosat.store import file_sha256

    cfg = config(tmp_path)
    images = catalogue(tmp_path)
    catalogue_path = Path(cfg.catalogue)
    catalogue_path.write_text(
        "image_id,path,time_utc,footprint_wkt,component_id\n"
        + "".join(
            f'{image.image_id},{image.path},{image.time_utc.isoformat()},'
            '"POLYGON ((0 0, 40000 0, 40000 40000, 0 40000, 0 0))",'
            "component\n"
            for image in images.chronological()
        )
    )
    processor = Processor()
    stages = RunStages(cfg, images, processor)
    stages.prepare()
    stages.process_pairs("primary")
    primary_root = tmp_path / "primary-parquet"
    primary = compose_primary_parquet(cfg, primary_root)
    primary_sha256 = file_sha256(Path(primary["trajectory_points"]))

    target_root = tmp_path / "recovery-targets"
    prepared = prepare_recovery_targets(cfg, primary_root, target_root)
    assert prepared["eligible_recovery_pairs"] == 1
    assert prepared["targeted_recovery_pairs"] == 1
    stages.process_pairs(
        "recovery", recovery_target_directory=str(target_root)
    )

    final = compose_recovery_parquet(
        cfg, primary_root, target_root, tmp_path / "recovery-parquet"
    )

    assert file_sha256(Path(primary["trajectory_points"])) == primary_sha256
    points = pq.read_table(final["trajectory_augmentations"])
    extensions = pq.read_table(final["trajectory_augmentation_extensions"])
    assert points.num_rows == 4
    assert set(points.column("state").to_pylist()) == {"reappeared"}
    assert set(points.column("position_basis").to_pylist()) == {
        "recovery_pair_field"
    }
    assert extensions.num_rows == 4
    assert set(extensions.column("pair_kind").to_pylist()) == {"recovery"}
    assert extensions.column("accepted").to_pylist() == [True] * 4


def test_parquet_request_plan_can_fill_a_missing_target_row(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from test_staged_execution import Processor, catalogue, config
    from limosat import RunStages
    from limosat.compose import compose_primary_parquet, compose_recovery_parquet
    from limosat.pair_queue import (
        PairOption, build_pair_request_plan, choose_pair_batch,
    )
    from limosat.recovery import prepare_recovery_targets
    from limosat.store import file_sha256

    cfg = config(tmp_path)
    images = catalogue(tmp_path)
    Path(cfg.catalogue).write_text(
        "image_id,path,time_utc,footprint_wkt,component_id\n"
        + "".join(
            f'{image.image_id},{image.path},{image.time_utc.isoformat()},'
            '"POLYGON ((0 0, 40000 0, 40000 40000, 0 40000, 0 0))",'
            "component\n"
            for image in images.chronological()
        )
    )
    stages = RunStages(cfg, images, Processor())
    stages.prepare()
    stages.process_pairs("primary")
    primary_root = tmp_path / "primary-parquet"
    primary = compose_primary_parquet(cfg, primary_root)
    points_path = Path(primary["trajectory_points"])
    table = pq.read_table(points_path)
    rows = table.to_pylist()
    source_ids = [row["trajectory_id"] for row in rows if row["image_id"] == "a"]
    chosen, nearby_unrequested = source_ids[:2]
    assert all(any(row["trajectory_id"] == identity and row["image_id"] == "c"
                   for row in rows) for identity in (chosen, nearby_unrequested))
    reduced = [row for row in rows
               if not (row["trajectory_id"] in {chosen, nearby_unrequested}
                       and row["image_id"] == "c")]
    with pq.ParquetWriter(points_path, table.schema) as writer:
        for image in images.chronological():
            image_rows = [row for row in reduced if row["image_id"] == image.image_id]
            if image_rows:
                writer.write_table(pa.Table.from_pylist(image_rows, schema=table.schema))
    manifest_path = primary_root / "composition-manifest-v1.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["counts"]["trajectory_points"] = len(reduced)
    manifest["products"]["trajectory_points"]["size_bytes"] = points_path.stat().st_size
    manifest["products"]["trajectory_points"]["sha256"] = file_sha256(points_path)
    manifest_path.write_text(json.dumps(manifest))
    pair = ImagePair(images.chronological()[0], images.chronological()[2])
    request_path = tmp_path / "requests.json"
    batch = choose_pair_batch(
        {chosen: (PairOption(pair.pair_id, 48.0),)}, maximum_pairs=1
    )
    request_path.write_text(json.dumps(build_pair_request_plan(
        batch,
        run_id=cfg.run_id,
        config_sha256=cfg.sha256,
        primary_composition_manifest_sha256=file_sha256(manifest_path),
    )))

    target_root = tmp_path / "recovery-targets"
    prepared = prepare_recovery_targets(
        cfg, primary_root, target_root, request_plan=request_path
    )
    assert prepared["targeted_recovery_pairs"] == 1
    stages.process_pairs("recovery", recovery_target_directory=str(target_root))
    result = compose_recovery_parquet(
        cfg, primary_root, target_root, tmp_path / "recovery-parquet"
    )
    additions = pq.read_table(result["trajectory_augmentations"]).to_pylist()
    extensions = pq.read_table(result["trajectory_augmentation_extensions"]).to_pylist()
    assert {row["trajectory_id"] for row in additions if row["image_id"] == "c"} == {chosen}
    assert all(row["state"] == "reappeared" for row in additions if row["image_id"] == "c")
    assert {row["trajectory_id"] for row in extensions} == {chosen}
