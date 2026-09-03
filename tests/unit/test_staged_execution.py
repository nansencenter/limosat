import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from limosat import (
    DisplacementField,
    FieldConfig,
    ImageCatalogue,
    ImagePair,
    ImageRecord,
    LiMOSATRun,
    MotionMatches,
    PairProductStore,
    PairResult,
    RoutingConfig,
    RunConfig,
    RunStages,
    RunStore,
    TrajectoryPoint,
)


START = datetime(2020, 1, 1, tzinfo=timezone.utc)
GRID = np.array(
    [[0.0, 0.0], [1_000.0, 0.0], [0.0, 1_000.0], [1_000.0, 1_000.0]]
)


def config(tmp_path, *, retain_matches=True):
    return RunConfig(
        run_id="staged-run",
        catalogue=str(tmp_path / "catalogue.csv"),
        database=str(tmp_path / "state.sqlite"),
        output_directory=str(tmp_path / "products"),
        pair_product_directory=str(tmp_path / "work" / "pairs"),
        retain_pair_matches=retain_matches,
        field=FieldConfig(
            grid_spacing_m=1_000.0,
            neighbour_count=4,
            minimum_agreeing_matches=3,
            maximum_neighbour_distance_m=2_000.0,
            agreement_distance_m=100.0,
            maximum_triangle_edge_m=1_500.0,
        ),
        routing=RoutingConfig(initial="same_center"),
    )


def catalogue(tmp_path):
    records = []
    for index, name in enumerate(("a", "b", "c")):
        path = tmp_path / f"{name}.tif"
        path.write_bytes(name.encode())
        records.append(
            ImageRecord(name, path, START + timedelta(days=index), "component")
        )
    return ImageCatalogue(records)


def result(pair, *, available=True, with_matches=True):
    support = np.full(4, available)
    displacement = np.tile(
        [100.0 * pair.elapsed_seconds / 86_400.0, 0.0], (4, 1)
    )
    displacement[~support] = np.nan
    field = DisplacementField(
        pair_id=pair.pair_id,
        source_image_id=pair.source.image_id,
        target_image_id=pair.target.image_id,
        source_time_utc=pair.source.time_utc,
        target_time_utc=pair.target.time_utc,
        grid_row=np.array([0, 0, 1, 1]),
        grid_column=np.array([0, 1, 0, 1]),
        source_xy_m=GRID,
        displacement_m=displacement,
        available=support,
        selected_matches=np.full(4, 8 if available else 0),
        candidate_matches=np.full(4, 12 if available else 0),
        support_radius_m=np.full(4, 500.0 if available else np.nan),
        maximum_residual_m=np.full(4, 20.0 if available else np.nan),
    )
    matches = (
        MotionMatches(
            np.array([[10.0, 20.0]]),
            np.array([[110.0, 20.0]]),
            np.array([0.9]),
            np.array([1]),
            np.array([2]),
        )
        if with_matches
        else MotionMatches.empty()
    )
    return PairResult(
        matches,
        field,
        np.array([3], dtype=np.int32),
        {"sampling": 0.1, "matching": 0.2, "field": 0.01, "total": 0.31},
        1,
        {"phase_correlation_status": "synthetic"},
        {"/fixture.nc": "0" * 64},
    )


class Processor:
    def __init__(self):
        self.calls = []

    def process(
        self,
        pair,
        previous_field=None,
        previous_elapsed_seconds=None,
        targeted_positions_xy_m=None,
    ):
        targeted = targeted_positions_xy_m is not None
        self.calls.append((pair.pair_id, targeted))
        available = pair.pair_id != "b__c" or targeted
        return result(pair, available=available)


def test_pair_product_round_trip_is_checked_and_immutable(tmp_path):
    cfg = config(tmp_path)
    pair = catalogue(tmp_path).adjacent_pairs("component")[0]
    products = PairProductStore(cfg)
    expected = result(pair)

    saved = products.save(pair, "primary", False, expected)
    loaded = products.load(pair, "primary", False)

    assert loaded is not None
    assert loaded.sha256 == saved.sha256
    assert loaded.match_count == 1
    assert loaded.result.field.checksum == expected.field.checksum
    np.testing.assert_array_equal(
        loaded.result.matches.source_xy_m, expected.matches.source_xy_m
    )
    assert products.count("primary") == 1

    resumed = products.save(pair, "primary", False, expected)
    assert resumed.sha256 == saved.sha256
    assert resumed.content_sha256 == saved.content_sha256

    marker = saved.path.with_suffix(".json")
    marker.unlink()
    recovered = products.save(pair, "primary", False, expected)
    assert recovered.sha256 == saved.sha256

    with pytest.raises(ValueError, match="already differs"):
        products.save(
            pair,
            "primary",
            False,
            result(pair, available=False),
        )

    metadata = json.loads(marker.read_text())
    metadata["field_sha256"] = "f" * 64
    marker.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="marker failed checksum"):
        products.load(pair, "primary", False)


def test_recovery_product_is_bound_to_measured_source_positions(tmp_path):
    cfg = config(tmp_path)
    images = catalogue(tmp_path).chronological()
    pair = ImagePair(images[0], images[2])
    products = PairProductStore(cfg)
    positions = np.array([[0.0, 0.0], [1_000.0, 0.0]])
    products.save(pair, "recovery", True, result(pair), positions)

    changed = positions.copy()
    changed[0, 0] = 1.0
    with pytest.raises(ValueError, match="targeted_positions_sha256"):
        products.load(pair, "recovery", True, changed)


def test_staged_workers_do_not_write_sqlite_and_batches_recompose_globally(tmp_path):
    cfg = config(tmp_path)
    images = catalogue(tmp_path)
    processor = Processor()
    stages = RunStages(cfg, images, processor)
    prepared = stages.prepare()

    first = stages.process_pairs("primary", batch_index=0, batch_count=2)
    second = stages.process_pairs("primary", batch_index=1, batch_count=2)
    with sqlite3.connect(cfg.database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0] == 0

    assert first["assigned_pairs"] + second["assigned_pairs"] == 2
    assert first["computed_pairs"] + second["computed_pairs"] == 2
    assert prepared["primary_pairs"] == 2

    primary = stages.compose("primary")
    assert primary["pair_product_import"]["imported_pair_products"] == 2
    with sqlite3.connect(cfg.database) as connection:
        dormant = connection.execute(
            "SELECT COUNT(*) FROM trajectory_points "
            "WHERE state='dormant' AND x_m IS NULL AND y_m IS NULL"
        ).fetchone()[0]
    assert dormant > 0
    assert primary["recovery_pairs"] == 0

    recovery = stages.process_pairs("recovery")
    assert recovery["planned_pairs"] == 1
    final = stages.compose("final", ["limosat", "compose", "config", "--phase", "final"])
    assert final["pair_product_import"]["imported_pair_products"] == 1

    with sqlite3.connect(cfg.database) as connection:
        reappeared = connection.execute(
            "SELECT COUNT(*) FROM trajectory_points "
            "WHERE state='reappeared' AND source_pair_id='a__c'"
        ).fetchone()[0]
        deformation = dict(
            connection.execute(
                """
                SELECT pairs.kind,COUNT(deformation_cells.triangle_index)
                FROM pairs LEFT JOIN deformation_cells USING(run_id,pair_id)
                GROUP BY pairs.kind
                """
            )
        )
    assert reappeared > 0
    assert deformation["primary"] > 0
    assert deformation["recovery"] == 0
    assert final["manifest_sha256"]

    with sqlite3.connect(cfg.database) as connection:
        staged_rows = connection.execute(
            """
            SELECT trajectory_id,image_id,time_utc,state,position_basis,x_m,y_m,
                   source_pair_id,selected_matches,support_radius_m,
                   maximum_residual_m
            FROM trajectory_points ORDER BY trajectory_id,time_utc,image_id
            """
        ).fetchall()
    single = replace(
        cfg,
        database=str(tmp_path / "single.sqlite"),
        output_directory=str(tmp_path / "single-products"),
        pair_product_directory=str(tmp_path / "single-work" / "pairs"),
    )
    LiMOSATRun(single, images, Processor()).execute()
    with sqlite3.connect(single.database) as connection:
        single_rows = connection.execute(
            """
            SELECT trajectory_id,image_id,time_utc,state,position_basis,x_m,y_m,
                   source_pair_id,selected_matches,support_radius_m,
                   maximum_residual_m
            FROM trajectory_points ORDER BY trajectory_id,time_utc,image_id
            """
        ).fetchall()
    assert staged_rows == single_rows


def test_field_only_worker_product_preserves_match_count_on_import(tmp_path):
    cfg = config(tmp_path, retain_matches=False)
    images = catalogue(tmp_path)
    stages = RunStages(cfg, images, Processor())
    stages.prepare()
    stages.process_pairs("primary")
    stages.compose("primary")

    with sqlite3.connect(cfg.database) as connection:
        counts = connection.execute(
            "SELECT match_count FROM pairs ORDER BY pair_id"
        ).fetchall()
        archives = connection.execute(
            "SELECT COUNT(*) FROM pair_match_archives"
        ).fetchone()[0]
    assert counts == [(1,), (1,)]
    assert archives == 0


def test_disabled_recovery_schedules_no_pair_work(tmp_path):
    cfg = replace(
        config(tmp_path),
        routing=RoutingConfig(initial="same_center", targeted_recovery=False),
    )
    images = catalogue(tmp_path)
    processor = Processor()
    stages = RunStages(cfg, images, processor)
    stages.prepare()
    stages.process_pairs("primary")
    stages.compose("primary")

    recovery = stages.process_pairs("recovery")
    final = stages.compose(
        "final", ["limosat", "compose", "config", "--phase", "final"]
    )

    assert recovery == {
        "kind": "recovery",
        "batch_index": 0,
        "batch_count": 1,
        "planned_pairs": 0,
        "assigned_pairs": 0,
        "computed_pairs": 0,
        "resumed_pair_products": 0,
        "resumed_sqlite_pairs": 0,
    }
    assert not any(targeted for _pair_id, targeted in processor.calls)
    assert final["recovery_pairs"] == 0


def test_default_pair_product_path_is_database_specific(tmp_path):
    first = replace(config(tmp_path), pair_product_directory="")
    second = replace(first, database=str(tmp_path / "other.sqlite"))
    assert first.pair_products != second.pair_products


def test_pair_stage_requires_a_registered_candidate_plan(tmp_path):
    cfg = config(tmp_path)
    RunStore(cfg)

    with pytest.raises(RuntimeError, match="run prepare first"):
        RunStages(cfg, catalogue(tmp_path), Processor()).process_pairs("primary")


def test_pair_stage_rejects_changed_registered_catalogue_metadata(tmp_path):
    cfg = config(tmp_path)
    original = catalogue(tmp_path)
    RunStages(cfg, original, Processor()).prepare()
    changed = ImageCatalogue(
        [replace(image, component_id="changed") for image in original.records]
    )

    with pytest.raises(ValueError, match="catalogue image metadata changed"):
        RunStages(cfg, changed, Processor()).process_pairs("primary")


def test_interrupted_streamed_composition_rolls_back_existing_rows(tmp_path):
    cfg = config(tmp_path)
    store = RunStore(cfg)
    image = catalogue(tmp_path).chronological()[0]
    original = (
        TrajectoryPoint(
            "parcel",
            image.image_id,
            image.time_utc,
            "created",
            "seed_grid",
            0.0,
            0.0,
            None,
        ),
    )
    store.replace_global_trajectories(original)

    def interrupted():
        yield original
        raise RuntimeError("synthetic composition interruption")

    with pytest.raises(RuntimeError, match="synthetic composition interruption"):
        store.replace_global_trajectory_batches(interrupted())

    with sqlite3.connect(cfg.database) as connection:
        rows = connection.execute(
            "SELECT trajectory_id,image_id,x_m,y_m FROM trajectory_points"
        ).fetchall()
        index = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' "
            "AND name='trajectory_points_run_image_state'"
        ).fetchone()
    assert rows == [("parcel", "a", 0.0, 0.0)]
    assert index == (1,)
