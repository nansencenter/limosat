import json
from datetime import datetime, timedelta, timezone
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from limosat import (
    DisplacementField,
    FieldConfig,
    FieldEdge,
    ImagePair,
    ImageRecord,
    MotionMatches,
    PairProductStore,
    PairResult,
    RoutingConfig,
    RunConfig,
    TrajectoryConfig,
)
from limosat.compose import compose_primary_parquet
from limosat.imagery import TargetPixelValidity
from limosat.trajectory import _supported_continuations, iter_global_trajectory_batches


START = datetime(2020, 1, 1, tzinfo=timezone.utc)
GRID = np.array(
    [[0.0, 0.0], [1_000.0, 0.0], [0.0, 1_000.0], [1_000.0, 1_000.0]]
)


def _field(displacement=(100.0, 0.0)):
    return DisplacementField(
        pair_id="a__b",
        source_image_id="a",
        target_image_id="b",
        source_time_utc=START,
        target_time_utc=START + timedelta(days=1),
        grid_row=np.array([0, 0, 1, 1]),
        grid_column=np.array([0, 1, 0, 1]),
        source_xy_m=GRID,
        displacement_m=np.tile(displacement, (4, 1)),
        available=np.ones(4, dtype=bool),
        selected_matches=np.full(4, 8),
        candidate_matches=np.full(4, 12),
        support_radius_m=np.full(4, 500.0),
        maximum_residual_m=np.full(4, 20.0),
    )


def _field_config():
    return FieldConfig(
        grid_spacing_m=1_000.0,
        neighbour_count=4,
        minimum_agreeing_matches=3,
        maximum_neighbour_distance_m=2_000.0,
        agreement_distance_m=100.0,
        maximum_triangle_edge_m=1_500.0,
    )


def test_target_pixel_validity_checks_actual_mask_and_raster_bounds(tmp_path):
    path = tmp_path / "target.tif"
    mask = np.ones((4, 4), dtype=np.uint8)
    mask[1, 1] = 2
    with rasterio.open(
        path, "w", driver="GTiff", width=4, height=4, count=2,
        dtype="uint8", crs="EPSG:3413",
        transform=from_origin(0, 4_000, 1_000, 1_000),
    ) as dataset:
        dataset.write(np.full((4, 4), 100, dtype=np.uint8), 1)
        dataset.write(mask, 2)

    validity = TargetPixelValidity(path)
    np.testing.assert_array_equal(
        validity(np.array([
            [500.0, 3_500.0], [1_250.0, 2_750.0],
            [-100.0, 2_500.0], [np.nan, 2_500.0],
        ])),
        [True, False, False, False],
    )


def _validity_raster(path, west, mask):
    with rasterio.open(
        path, "w", driver="GTiff", width=4, height=4, count=2,
        dtype="uint8", crs="EPSG:3413",
        transform=from_origin(west, 4_000, 1_000, 1_000),
    ) as dataset:
        dataset.write(np.full((4, 4), 100, dtype=np.uint8), 1)
        dataset.write(mask, 2)


def test_pass_pixel_validity_accepts_valid_same_pass_frame(tmp_path):
    from datetime import datetime, timedelta, timezone

    from shapely.geometry import box

    from limosat import ImageRecord
    from limosat.imagery import pass_pixel_validity_factory

    good = np.ones((4, 4), dtype=np.uint8)
    masked = good.copy()
    masked[1, 1] = 2
    for name, west, mask in (
        ("target", 0, good), ("sibling", 4_000, masked), ("other", 8_000, good)
    ):
        _validity_raster(tmp_path / f"{name}.tif", west, mask)
    start = datetime(2020, 3, 1, tzinfo=timezone.utc)

    def image(name, west, orbit, seconds):
        return ImageRecord(
            name, tmp_path / f"{name}.tif", start + timedelta(seconds=seconds),
            footprint=box(west, 0, west + 4_000, 4_000),
            platform="S1A", absolute_orbit=orbit,
        )

    target = image("target", 0, 100, 0)
    images = (target, image("sibling", 4_000, 100, 60), image("other", 8_000, 101, 0))
    validity = pass_pixel_validity_factory(images)(target)

    np.testing.assert_array_equal(
        validity(np.array([
            [500.0, 3_500.0],    # valid on the target frame
            [4_500.0, 3_500.0],  # outside the target, valid on the same-pass frame
            [5_250.0, 2_750.0],  # outside the target, masked on the same-pass frame
            [8_500.0, 3_500.0],  # valid only on a frame of a different pass
        ])),
        [True, True, False, False],
    )


def test_invalid_preferred_endpoint_does_not_hide_valid_pair():
    valid_field = _field((100.0, 0.0))
    preferred_field = replace(
        _field((200.0, 0.0)), pair_id="alternate__b",
        selected_matches=np.full(4, 20),
    )
    incoming = [
        (0, 1, FieldEdge(valid_field)),
        (0, 1, FieldEdge(preferred_field)),
    ]
    positions = [{"parcel": np.array([0.0, 0.0])}, {}]
    chosen = _supported_continuations(
        incoming, positions, ["parcel"], _field_config(),
        target_validity=lambda xy: xy[:, 0] < 150.0,
    )
    np.testing.assert_allclose(chosen["parcel"].displacement_m, [100.0, 0.0])
    assert _supported_continuations(
        incoming, positions, ["parcel"], _field_config(),
        target_validity=lambda xy: np.zeros(len(xy), dtype=bool),
    ) == {}


def test_trajectory_batches_expose_direct_extensions_without_a_second_scan():
    images = [
        ImageRecord("a", Path("/tmp/a.tif"), START),
        ImageRecord("b", Path("/tmp/b.tif"), START + timedelta(days=1)),
    ]

    batches = tuple(
        iter_global_trajectory_batches(
            [FieldEdge(_field())], images, _field_config(), TrajectoryConfig()
        )
    )

    assert len(batches) == 2
    assert len(batches[0].extensions) == 0
    assert len(batches[1].extensions) == 4
    assert {value.pair_id for value in batches[1].extensions} == {"a__b"}
    assert {value.target_state for value in batches[1].extensions} == {"observed"}
    for value in batches[1].extensions:
        np.testing.assert_allclose(value.displacement_m, [100.0, 0.0])
        assert value.source_image_id == "a"
        assert value.target_image_id == "b"


def test_compose_uses_pair_products_without_creating_sqlite(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "a.tif"
    target = tmp_path / "b.tif"
    for path in (source, target):
        mask = np.ones((40, 40), dtype=np.uint8)
        if path == target:
            mask[39, 1] = 2  # Predicted endpoint from the (0, 0) seed.
        with rasterio.open(
            path, "w", driver="GTiff", width=40, height=40, count=2,
            dtype="uint8", crs="EPSG:3413",
            transform=from_origin(-1_000, 39_000, 1_000, 1_000),
        ) as dataset:
            dataset.write(np.full((40, 40), 100, dtype=np.uint8), 1)
            dataset.write(mask, 2)
    catalogue_path = tmp_path / "catalogue.csv"
    catalogue_path.write_text(
        "image_id,path,time_utc,footprint_wkt\n"
        f'a,{source},2020-01-01T00:00:00+00:00,"POLYGON ((0 0, 40000 0, '
        '40000 40000, 0 40000, 0 0))"\n'
        f'b,{target},2020-01-02T00:00:00+00:00,"POLYGON ((0 0, 40000 0, '
        '40000 40000, 0 40000, 0 0))"\n',
        encoding="utf-8",
    )
    config = RunConfig(
        run_id="parquet-direct",
        catalogue=str(catalogue_path),
        database=str(tmp_path / "must-not-exist.sqlite"),
        output_directory=str(tmp_path / "native-output"),
        pair_product_directory=str(tmp_path / "pair-products"),
        field=_field_config(),
        routing=RoutingConfig(coarse_matching=False, initial="same_center"),
        retain_pair_matches=True,
    )
    images = [
        ImageRecord("a", source, START),
        ImageRecord("b", target, START + timedelta(days=1)),
    ]
    pair = ImagePair(images[0], images[1])
    result = PairResult(
        MotionMatches(
            np.array([[0.0, 0.0]]),
            np.array([[100.0, 0.0]]),
            np.array([0.9]),
            np.array([0]),
            np.array([0]),
        ),
        _field(),
        np.empty(0, dtype=np.int32),
        {"total": 1.0},
        1,
    )
    PairProductStore(config).save(pair, "primary", False, result)

    output = tmp_path / "parquet-output"
    composed = compose_primary_parquet(config, output)

    assert not Path(config.database).exists()
    assert composed["counts"]["trajectory_points"] == 8
    assert composed["counts"]["trajectory_extensions"] == 3
    assert composed["counts"]["trajectory_point_states"]["dormant"] == 1
    extensions = pq.read_table(composed["trajectory_extensions"])
    assert extensions.num_rows == 3
    assert extensions.column("qc_status").to_pylist() == ["accept"] * 3
    assert extensions.column("accepted").to_pylist() == [True] * 3
    assert json.loads(Path(composed["manifest"]).read_text())[
        "target_pixel_validity"
    ] == "same_pass_raster_bounds_and_band2_lt2"
