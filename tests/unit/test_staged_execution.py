import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
import pytest

from limosat import (
    DisplacementField,
    FieldConfig,
    ImageCatalogue,
    ImageRecord,
    MotionMatches,
    PairProductStore,
    PairResult,
    RoutingConfig,
    RunConfig,
    RunStages,
    RunStore,
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
            missing_node_fallback=retain_matches,
        ),
        routing=RoutingConfig(coarse_matching=False, initial="same_center"),
    )


def catalogue(tmp_path):
    records = []
    for index, name in enumerate(("a", "b", "c")):
        path = tmp_path / f"{name}.tif"
        with rasterio.open(
            path, "w", driver="GTiff", width=40, height=40, count=2,
            dtype="uint8", crs="EPSG:3413",
            transform=from_origin(-1_000, 39_000, 1_000, 1_000),
        ) as dataset:
            dataset.write(np.full((40, 40), 100, dtype=np.uint8), 1)
            dataset.write(np.ones((40, 40), dtype=np.uint8), 2)
        records.append(
            ImageRecord(name, path, START + timedelta(days=index), "component")
        )
    (tmp_path / "catalogue.csv").write_text(
        "image_id,path,time_utc,component_id\n"
        + "".join(f"{r.image_id},{r.path},{r.time_utc.isoformat()},component\n" for r in records)
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

    def process(self, pair, previous_field=None, previous_elapsed_seconds=None):
        self.calls.append(pair.pair_id)
        return result(pair, available=pair.pair_id != "b__c")


def test_pair_product_round_trip_is_checked_and_immutable(tmp_path):
    cfg = config(tmp_path)
    pair = catalogue(tmp_path).adjacent_pairs("component")[0]
    products = PairProductStore(cfg)
    expected = result(pair)

    saved = products.save(pair, expected)
    loaded = products.load(pair)

    assert loaded is not None
    assert loaded.sha256 == saved.sha256
    assert loaded.match_count == 1
    assert loaded.result.field.checksum == expected.field.checksum
    np.testing.assert_array_equal(
        loaded.result.matches.source_xy_m, expected.matches.source_xy_m
    )
    assert products.count() == 1
    assert saved.path.parent == cfg.pair_products / "primary"
    metadata = json.loads(saved.path.with_suffix(".json").read_text())
    # Stored products keep the identity fields of the former product kinds.
    assert (metadata["kind"], metadata["targeted"], metadata["targeted_positions_sha256"]) == (
        "primary", False, None
    )

    resumed = products.save(pair, expected)
    assert resumed.sha256 == saved.sha256
    assert resumed.content_sha256 == saved.content_sha256

    marker = saved.path.with_suffix(".json")
    marker.unlink()
    recovered = products.save(pair, expected)
    assert recovered.sha256 == saved.sha256

    # First writer wins: a later, different result for the same pair keeps the stored one.
    adopted = products.save(pair, result(pair, available=False))
    assert adopted.content_sha256 == saved.content_sha256

    # Different data without a marker means another writer is mid-publication; a writer
    # that loses the race waits for that marker and fails if it never appears.
    marker.unlink()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("limosat.pair_products.PUBLICATION_WAIT_SECONDS", 0.0)
        with pytest.raises(ValueError, match="marker never appeared"):
            products.save(pair, result(pair, available=False))
    products.save(pair, expected)

    metadata = json.loads(marker.read_text())
    metadata["field_sha256"] = "f" * 64
    marker.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="marker failed checksum"):
        products.load(pair)


def test_pair_products_outlive_code_changes_until_the_matching_version_changes(tmp_path):
    cfg = config(tmp_path)
    pair = catalogue(tmp_path).adjacent_pairs("component")[0]
    products = PairProductStore(cfg)
    saved = products.save(pair, result(pair))
    producer_hash = saved.producer_implementation_sha256

    products.implementation_sha256 = "f" * 64
    loaded = products.load(pair)
    assert loaded is not None
    assert loaded.producer_implementation_sha256 == producer_hash
    assert loaded.result.field.checksum == saved.result.field.checksum

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("limosat.pair_products.MATCHING_VERSION", 2)
        with pytest.raises(ValueError, match="matching_version"):
            products.load(pair)


def test_run_bound_schema_2_products_are_refused(tmp_path):
    import hashlib

    cfg = config(tmp_path)
    pair = catalogue(tmp_path).adjacent_pairs("component")[0]
    products = PairProductStore(cfg)
    marker = products.save(pair, result(pair)).path.with_suffix(".json")
    metadata = json.loads(marker.read_text())
    metadata.pop("marker_content_sha256")
    metadata["pair_product_schema_version"] = 2
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    metadata["marker_content_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    marker.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="unsupported pair product schema 2"):
        products.load(pair)


def _rows(path):
    pq = pytest.importorskip("pyarrow.parquet")
    return pq.read_table(path).to_pylist()


def test_batched_pair_workers_and_composition_match_a_single_machine_run(tmp_path):
    cfg = config(tmp_path)
    images = catalogue(tmp_path)
    processor = Processor()
    stages = RunStages(cfg, images, processor)
    prepared = stages.prepare()
    assert prepared["primary_pairs"] == 2

    first = stages.process_pairs(batch_index=0, batch_count=2)
    second = stages.process_pairs(batch_index=1, batch_count=2)
    assert first["assigned_pairs"] + second["assigned_pairs"] == 2
    assert first["computed_pairs"] + second["computed_pairs"] == 2
    assert sorted(processor.calls) == ["a__b", "b__c"]
    with sqlite3.connect(cfg.database) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"runs", "images", "candidate_pairs", "planning_counts"}
    assert stages.process_pairs()["resumed_pair_products"] == 2

    composed = stages.compose()
    assert Path(composed["manifest"]).parent == Path(cfg.output_directory) / "primary-composition"
    points = _rows(composed["trajectory_points"])
    # b -> c supports nothing: tracks end at b and no point is stored at c.
    assert {row["image_id"] for row in points} == {"a", "b"}
    assert {row["state"] for row in points} == {"created", "observed"}

    single = replace(
        cfg,
        database=str(tmp_path / "single.sqlite"),
        output_directory=str(tmp_path / "single-products"),
        pair_product_directory=str(tmp_path / "single-work" / "pairs"),
    )
    report = RunStages(single, images, Processor()).run_all()
    assert report["computed_pairs"] == 2
    assert _rows(report["trajectory_points"]) == points


def test_field_only_worker_product_preserves_match_count(tmp_path):
    cfg = config(tmp_path, retain_matches=False)
    stages = RunStages(cfg, catalogue(tmp_path), Processor())
    stages.prepare()
    stages.process_pairs()
    for pair in catalogue(tmp_path).adjacent_pairs("component"):
        product = stages.pair_products.load(pair)
        assert product.match_count == 1 and not len(product.result.matches)


def test_default_pair_product_path_is_database_specific(tmp_path):
    first = replace(config(tmp_path), pair_product_directory="")
    second = replace(first, database=str(tmp_path / "other.sqlite"))
    assert first.pair_products != second.pair_products


def test_pair_stage_requires_a_registered_candidate_plan(tmp_path):
    cfg = config(tmp_path)
    RunStore(cfg)

    with pytest.raises(RuntimeError, match="run prepare first"):
        RunStages(cfg, catalogue(tmp_path), Processor()).process_pairs()


def test_pair_stage_rejects_changed_registered_catalogue_metadata(tmp_path):
    cfg = config(tmp_path)
    original = catalogue(tmp_path)
    RunStages(cfg, original, Processor()).prepare()
    changed = ImageCatalogue(
        [replace(image, component_id="changed") for image in original.records]
    )

    with pytest.raises(ValueError, match="catalogue image metadata changed"):
        RunStages(cfg, changed, Processor()).process_pairs()
