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
    MatcherConfig,
    MotionMatches,
    PairResult,
    RoutingConfig,
    RunConfig,
    TrajectoryPoint,
)
from limosat.cli import main
from limosat.store import RunStore


START = datetime(2020, 1, 1, tzinfo=timezone.utc)
GRID = np.array([[0.0, 0.0], [1_000.0, 0.0], [0.0, 1_000.0], [1_000.0, 1_000.0]])


def _config(tmp_path):
    return RunConfig(
        run_id="synthetic-run",
        catalogue=str(tmp_path / "catalogue.csv"),
        database=str(tmp_path / "run.sqlite"),
        output_directory=str(tmp_path / "products"),
        matcher=MatcherConfig(),
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


def _catalogue(tmp_path):
    records = []
    for index, name in enumerate(("a", "b", "c")):
        path = tmp_path / f"{name}.tif"
        path.write_bytes(name.encode())
        records.append(
            ImageRecord(name, path, START + timedelta(days=index), "component")
        )
    return ImageCatalogue(records)


def _result(pair, available=True):
    support = np.full(4, available)
    displacement = np.tile([100.0 * (pair.elapsed_seconds / 86_400.0), 0.0], (4, 1))
    displacement[~support] = np.nan
    field = DisplacementField(
        pair_id=pair.pair_id,
        source_image_id=pair.source.image_id,
        target_image_id=pair.target.image_id,
        source_time_utc=pair.source.time_utc,
        target_time_utc=pair.target.time_utc,
        grid_row=np.array([0, 0, 1, 1]),
        grid_column=np.array([0, 1, 0, 1]),
        source_xy_m=(
            GRID
            + np.array(
                [100.0 * ((pair.source.time_utc - START).total_seconds() / 86_400.0), 0.0]
            )
        ),
        displacement_m=displacement,
        available=support,
        selected_matches=np.full(4, 8 if available else 0),
        candidate_matches=np.full(4, 12 if available else 0),
        support_radius_m=np.full(4, 500.0 if available else np.nan),
        maximum_residual_m=np.full(4, 20.0 if available else np.nan),
    )
    return PairResult(
        MotionMatches.empty(),
        field,
        np.empty(0, dtype=int),
        {"sampling": 0.1, "matching": 0.2, "field": 0.01, "total": 0.31},
        1,
    )


class SyntheticProcessor:
    def __init__(self):
        self.calls = []

    def process(
        self,
        pair,
        previous_field=None,
        previous_elapsed_seconds=None,
        targeted_positions_xy_m=None,
    ):
        self.calls.append((pair.pair_id, targeted_positions_xy_m is not None))
        unavailable = pair.pair_id == "b__c" and targeted_positions_xy_m is None
        return _result(pair, available=not unavailable)


class FailingProcessor:
    def process(self, *args, **kwargs):
        raise AssertionError("completed pairs must resume without recomputation")


def test_interrupted_pair_retries_but_complete_product_is_immutable(tmp_path):
    config = _config(tmp_path)
    catalogue = _catalogue(tmp_path)
    pair = catalogue.adjacent_pairs("component")[0]
    store = RunStore(config)
    store.register_catalogue(catalogue)
    store.start_run()

    assert store.claim_pair(pair, "component", "primary", False)
    resumed_store = RunStore(config)
    assert resumed_store.claim_pair(pair, "component", "primary", False)
    assert resumed_store.save_pair(pair, _result(pair))
    checksum = resumed_store.load_field(pair.pair_id).checksum

    assert not resumed_store.claim_pair(pair, "component", "primary", False)
    assert not resumed_store.save_pair(pair, _result(pair, available=False))
    assert resumed_store.load_field(pair.pair_id).checksum == checksum


def test_sequence_recovers_measured_loss_and_resumes_with_versioned_manifest(tmp_path):
    config = _config(tmp_path)
    catalogue = _catalogue(tmp_path)
    processor = SyntheticProcessor()

    first = LiMOSATRun(config, catalogue, processor).execute(["limosat", "run", "config.yaml"])
    manifest = json.loads((tmp_path / "products" / "run-manifest-v2.json").read_text())

    assert first["computed_pairs"] == 3
    assert processor.calls == [("a__b", False), ("b__c", False), ("a__c", True)]
    assert manifest["manifest_schema_version"] == 2
    assert len(manifest["implementation_sha256"]) == 64
    assert manifest["product_schemas"]["lagrangian_trajectory"] == 3
    assert manifest["coordinates"]["crs"] == "EPSG:3413"
    assert manifest["product_counts"]["trajectories"] == 4
    recovery = [pair for pair in manifest["pairs"] if pair["kind"] == "recovery"]
    assert len(recovery) == 1 and recovery[0]["targeted"] == 1

    second = LiMOSATRun(config, catalogue, FailingProcessor()).execute()

    assert second["computed_pairs"] == 0
    assert second["resumed_pairs"] == 3


def test_cli_status_imports_without_loading_model(tmp_path, capsys):
    config = _config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))

    assert main(["status", str(config_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["run"]["schema_version"] == 2


def test_schema_v2_is_global_and_persists_null_dormant_coordinates(tmp_path):
    config = _config(tmp_path)
    store = RunStore(config)
    points = (
        TrajectoryPoint(
            "parcel",
            "a",
            START,
            "created",
            "seed_grid",
            1.0,
            2.0,
            None,
        ),
        TrajectoryPoint(
            "parcel",
            "b",
            START + timedelta(days=1),
            "dormant",
            "missing",
            None,
            None,
            None,
        ),
    )
    store.replace_global_trajectories(points)

    with sqlite3.connect(config.database) as connection:
        trajectory_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(trajectories)")
        }
        dormant = connection.execute(
            "SELECT x_m,y_m FROM trajectory_points WHERE state='dormant'"
        ).fetchone()
        version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert "component_id" not in trajectory_columns
    assert dormant == (None, None)
    assert version == 2


def test_legacy_database_is_rejected_without_migration(tmp_path):
    legacy = tmp_path / "legacy.sqlite"
    with sqlite3.connect(legacy) as connection:
        connection.execute("CREATE TABLE runs(run_id TEXT PRIMARY KEY)")

    with pytest.raises(ValueError, match="new schema-v2 database path and run_id"):
        RunStore(replace(_config(tmp_path), database=str(legacy)))
