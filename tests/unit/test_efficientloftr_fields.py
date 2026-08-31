from datetime import datetime, timedelta, timezone

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely import box

from limosat import (
    FieldConfig,
    FieldEdge,
    ImagePair,
    ImageRecord,
    MatcherConfig,
    RoutingConfig,
    RunConfig,
    TrajectoryConfig,
    build_trajectories,
)
from limosat.efficientloftr import speed_limit_mask
from limosat.imagery import north_up_patch, projected_coordinates
from limosat.pairs import PairProcessor


class TranslationMatcher:
    def match(self, source, target):
        axis = np.arange(3.0, 13.0, 3.0)
        x, y = np.meshgrid(axis, axis)
        source_px = np.column_stack((x.ravel(), y.ravel()))
        return source_px, source_px + np.array([1.0, 0.0]), np.ones(len(source_px))


class EmptyMatcher:
    def match(self, source, target):
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)


class StationaryMatcher(TranslationMatcher):
    def match(self, source, target):
        source_px, _target_px, score = super().match(source, target)
        return source_px, source_px.copy(), score


def _write_image(path):
    values = np.arange(24 * 24, dtype=np.uint16).reshape(24, 24)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=24,
        height=24,
        count=1,
        dtype=values.dtype,
        crs="EPSG:3413",
        transform=from_origin(-1_200.0, 1_200.0, 100.0, 100.0),
    ) as dataset:
        dataset.write(values, 1)


def _config(tmp_path):
    return RunConfig(
        run_id="synthetic",
        catalogue=str(tmp_path / "catalog.csv"),
        database=str(tmp_path / "run.sqlite"),
        output_directory=str(tmp_path / "output"),
        matcher=MatcherConfig(
            pixel_size_m=100.0,
            tile_size_px=16,
            tile_margin_px=2,
            endpoint_support_radius_px=1,
            transform_grid_spacing_px=4,
        ),
        field=FieldConfig(
            grid_spacing_m=200.0,
            neighbour_count=4,
            minimum_agreeing_matches=3,
            maximum_neighbour_distance_m=500.0,
            agreement_distance_m=50.0,
            maximum_triangle_edge_m=400.0,
        ),
        routing=RoutingConfig(initial="same_center", residual_edge_recovery=False),
    )


def _pair(tmp_path):
    source_path, target_path = tmp_path / "source.tif", tmp_path / "target.tif"
    _write_image(source_path)
    _write_image(target_path)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    footprint = box(-800.0, -800.0, 800.0, 800.0)
    return ImagePair(
        ImageRecord("source", source_path, start, footprint=footprint),
        ImageRecord("target", target_path, start + timedelta(days=1), footprint=footprint),
    )


def test_rasterio_north_up_coordinates_are_float64_metres(tmp_path):
    image = tmp_path / "image.tif"
    _write_image(image)
    patch, valid = north_up_patch(image, (0.0, 0.0), 8, 100.0, transform_grid_spacing_px=2)
    coordinates = projected_coordinates(np.array([[3.5, 3.5]]), (0.0, 0.0), 8, 100.0)

    assert patch.shape == valid.shape == (8, 8)
    assert valid.all()
    assert coordinates.dtype == np.float64
    np.testing.assert_allclose(coordinates, [[0.0, 0.0]])


def test_tiled_pair_builds_fold_free_float64_field(tmp_path):
    result = PairProcessor(_config(tmp_path), TranslationMatcher()).process(_pair(tmp_path))

    assert result.matcher_calls > 0
    assert result.field.available.any()
    assert result.field.displacement_m.dtype == np.float64
    np.testing.assert_allclose(
        result.field.displacement_m[result.field.available]
        - np.array([100.0, 0.0]),
        0.0,
        atol=1.0e-8,
    )
    assert len(result.fold_rejected_indices) == 0


def test_empty_matches_remain_missing_not_zero(tmp_path):
    result = PairProcessor(_config(tmp_path), EmptyMatcher()).process(_pair(tmp_path))

    assert not result.field.available.any()
    assert np.isnan(result.field.displacement_m).all()


def test_speed_limit_uses_elapsed_seconds_and_metres_per_day():
    keep = speed_limit_mask(
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        np.array([[15_000.0, 0.0], [15_001.0, 0.0]]),
        43_200.0,
        30_000.0,
    )
    assert keep.tolist() == [True, False]


def test_small_synthetic_sequence_flows_from_tiles_to_trajectories(tmp_path):
    paths = [tmp_path / f"image-{index}.tif" for index in range(3)]
    for path in paths:
        _write_image(path)
    footprint = box(-800.0, -800.0, 800.0, 800.0)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    images = [
        ImageRecord(
            f"image-{index}",
            path,
            start + timedelta(days=index),
            "component",
            footprint,
        )
        for index, path in enumerate(paths)
    ]
    processor = PairProcessor(_config(tmp_path), StationaryMatcher())
    first_pair, second_pair = ImagePair(images[0], images[1]), ImagePair(images[1], images[2])
    first = processor.process(first_pair)
    second = processor.process(
        second_pair, first.field, first_pair.elapsed_seconds
    )

    trajectories = build_trajectories(
        [FieldEdge(first.field), FieldEdge(second.field)],
        images,
        _config(tmp_path).field,
        TrajectoryConfig(),
    )

    assert {point.state for point in trajectories if point.image_id == "image-2"} == {
        "observed"
    }
    assert all(point.x_m is not None for point in trajectories)
