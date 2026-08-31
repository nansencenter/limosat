import numpy as np
import pandas as pd
import pytest
import shapely
import torch

from experiments.refine_aliked_dense_topology import (
    reject_flipped_nodes_until_stable,
)
from experiments.run_aliked_dense_pair import (
    match_tiles,
    nearest_consensus_at_queries,
    regular_queries,
    tile_layout as existing_tile_layout,
)
from limosat.learned_drift import (
    ALIKEDConfig,
    ALIKEDDrift,
    DriftField,
    ImagePair,
    LearnedDriftStore,
    MotionMatches,
    PairResult,
    estimate_field,
    match_features,
    reject_folds,
    tile_layout,
)
from limosat.learned_drift.types import FeatureTile, ImageFeatures


def coherent_matches() -> pd.DataFrame:
    offsets = np.array(
        [
            [-800.0, -800.0],
            [0.0, -800.0],
            [800.0, -800.0],
            [-800.0, 0.0],
            [0.0, 0.0],
            [800.0, 0.0],
            [-800.0, 800.0],
            [0.0, 800.0],
            [800.0, 800.0],
            [1200.0, 0.0],
            [0.0, 1200.0],
            [-1200.0, 0.0],
        ]
    )
    displacement = np.column_stack(
        (
            np.linspace(920.0, 1080.0, len(offsets)),
            np.linspace(-460.0, -540.0, len(offsets)),
        )
    )
    return pd.DataFrame(
        {
            "source_feature_id": np.arange(len(offsets)),
            "source_tile_id": 0,
            "target_tile_id": 1,
            "source_x": offsets[:, 0],
            "source_y": offsets[:, 1],
            "target_x": offsets[:, 0] + displacement[:, 0],
            "target_y": offsets[:, 1] + displacement[:, 1],
            "dx_m": displacement[:, 0],
            "dy_m": displacement[:, 1],
            "matcher_score": np.linspace(0.95, 0.50, len(offsets)),
            "physics_valid": True,
        }
    )


def test_selected_field_matches_existing_dataframe_implementation():
    rows = coherent_matches()
    domain = shapely.box(-1.0, -1.0, 1.0, 1.0)
    config = ALIKEDConfig()

    expected_queries = regular_queries(domain, config.grid_spacing_m)
    expected = nearest_consensus_at_queries(
        rows,
        expected_queries,
        config.maximum_neighbour_distance_m,
        config.neighbour_count,
        config.minimum_agreeing_matches,
        config.agreement_distance_m,
    )
    actual = estimate_field(MotionMatches.from_frame(rows), domain, config).to_frame()

    exact_columns = [
        "grid_row",
        "grid_column",
        "available",
        "selected_vectors",
        "candidate_count",
    ]
    pd.testing.assert_frame_equal(
        actual[exact_columns], expected[exact_columns], check_dtype=False
    )
    for column in [
        "source_x",
        "source_y",
        "support_radius_m",
        "proposal_dx_m",
        "proposal_dy_m",
        "maximum_vector_residual_m",
    ]:
        assert np.allclose(actual[column], expected[column], equal_nan=True)


def test_tile_layout_matches_existing_anchored_implementation():
    domain = shapely.box(-12_000.0, -8_000.0, 64_000.0, 56_000.0)
    config = ALIKEDConfig()

    expected = existing_tile_layout(
        domain,
        config.tile_size_px,
        config.tile_margin_px,
        config.pixel_size_m,
        config.tile_grid_origin_m,
    )
    actual = tile_layout(domain, config)

    assert len(actual) == len(expected)
    for region, row in zip(actual, expected, strict=True):
        assert (region.tile_id, region.row, region.column) == (
            row["tile_id"],
            row["row"],
            row["column"],
        )
        assert np.allclose(region.center_xy_m, [row["center_x"], row["center_y"]])
        assert region.core.equals_exact(row["core"], tolerance=0.0)


def test_selected_matching_matches_existing_dataframe_implementation():
    class ExistingIdentityMatcher:
        matcher_name = "lightglue_direct"
        uses_laf = False
        uses_direct_keypoints = True

        def __call__(self, source, target, _source_laf, _target_laf, **_kwargs):
            count = min(len(source), len(target))
            indices = torch.arange(count, device=source.device)
            return (
                torch.linspace(0.9, 0.6, count, device=source.device),
                torch.column_stack((indices, indices)),
            )

    class NewIdentityMatcher(torch.nn.Module):
        def forward(self, source, target, _source_keypoints, _target_keypoints, _size):
            count = min(len(source), len(target))
            indices = torch.arange(count, device=source.device)
            return (
                torch.linspace(0.9, 0.6, count, device=source.device),
                torch.column_stack((indices, indices)),
            )

    config = ALIKEDConfig()
    keypoints = torch.tensor(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]]
    )
    descriptors = torch.eye(4)
    scores = torch.ones(4)
    source_xy = np.array(
        [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]]
    )
    core = shapely.box(-1000.0, -1000.0, 1000.0, 1000.0)

    def feature_tile(tile_id, xy):
        return FeatureTile(
            tile_id=tile_id,
            row=0,
            column=tile_id,
            center_xy_m=(0.0, 0.0),
            core=core,
            keypoints_px=keypoints,
            descriptors=descriptors,
            scores=scores,
            xy_m=xy,
        )

    source = feature_tile(0, source_xy)
    targets = (
        feature_tile(1, source_xy + [1000.0, 500.0]),
        feature_tile(2, source_xy + [1500.0, 500.0]),
    )
    old_source = [
        {
            "tile_id": source.tile_id,
            "core": source.core,
            "keypoints": source.keypoints_px,
            "descriptors": source.descriptors,
            "scores": source.scores,
            "xy": source.xy_m,
        }
    ]
    old_targets = [
        {
            "tile_id": tile.tile_id,
            "core": tile.core,
            "keypoints": tile.keypoints_px,
            "descriptors": tile.descriptors,
            "scores": tile.scores,
            "xy": tile.xy_m,
        }
        for tile in targets
    ]
    expected, _ = match_tiles(
        old_source,
        old_targets,
        ExistingIdentityMatcher(),
        torch.device("cpu"),
        config.tile_size_px,
        config.maximum_displacement_m(24.0),
        1.0,
        config.maximum_speed_m_per_day,
        physics_subset_matching=True,
    )
    domain = shapely.box(-2000.0, -2000.0, 2000.0, 2000.0)
    actual = match_features(
        ImageFeatures("source", domain, (source,), config.analysis_epsg),
        ImageFeatures("target", domain, targets, config.analysis_epsg),
        24.0,
        NewIdentityMatcher(),
        torch.device("cpu"),
        config,
    ).to_frame()

    columns = [
        "source_feature_id",
        "source_tile_id",
        "target_tile_id",
        "source_x",
        "source_y",
        "target_x",
        "target_y",
        "matcher_score",
    ]
    pd.testing.assert_frame_equal(
        actual[columns], expected[columns], check_dtype=False
    )


def test_pair_facade_returns_matches_and_fold_free_field():
    class IdentityMatcher(torch.nn.Module):
        def forward(self, source, target, _source_keypoints, _target_keypoints, _size):
            count = min(len(source), len(target))
            indices = torch.arange(count, device=source.device)
            return torch.ones(count, device=source.device), torch.column_stack(
                (indices, indices)
            )

    offsets = np.array(
        [
            [-800.0, -800.0],
            [0.0, -800.0],
            [800.0, -800.0],
            [-800.0, 0.0],
            [0.0, 0.0],
            [800.0, 0.0],
            [-800.0, 800.0],
            [0.0, 800.0],
            [800.0, 800.0],
            [-1200.0, 0.0],
            [1200.0, 0.0],
            [0.0, 1200.0],
        ]
    )
    keypoints = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    descriptors = torch.eye(12)
    core = shapely.box(-2000.0, -2000.0, 2000.0, 2000.0)

    def image_features(path, tile_id, xy):
        tile = FeatureTile(
            tile_id=tile_id,
            row=0,
            column=0,
            center_xy_m=(0.0, 0.0),
            core=core,
            keypoints_px=keypoints,
            descriptors=descriptors,
            scores=torch.ones(12),
            xy_m=xy,
        )
        return ImageFeatures(path, core, (tile,), 3413)

    tracker = ALIKEDDrift(
        model=torch.nn.Identity(), matcher=IdentityMatcher()
    )
    result = tracker.track_pair(
        image_features("source", 0, offsets),
        image_features("target", 1, offsets + [1000.0, -500.0]),
        elapsed_hours=24.0,
        domain=shapely.box(-1.0, -1.0, 1.0, 1.0),
    )

    assert len(result.matches) == 12
    assert result.field.available.tolist() == [True]
    assert np.allclose(result.field.displacement_m[0], [1000.0, -500.0])
    assert len(result.fold_rejected_indices) == 0


def test_sequence_facade_extracts_once_and_uses_only_preceding_field():
    class StubTracker(ALIKEDDrift):
        def __init__(self):
            self.config = ALIKEDConfig()
            self.extracted = []
            self.pairs = []

        def _pair_domains(self, source_path, target_path, elapsed_hours):
            self.pairs.append((source_path, target_path, elapsed_hours))
            domain = shapely.box(-2000.0, -2000.0, 2000.0, 2000.0)
            return domain, domain

        def extract(self, image_path, domain=None):
            self.extracted.append(image_path)
            return ImageFeatures(image_path, domain, (), 3413)

        def track_pair(
            self,
            source,
            target,
            elapsed_hours,
            domain=None,
            prior_displacement_m=None,
            prior_uncertainty_m=None,
        ):
            count = 8
            field = DriftField(
                grid_row=np.arange(count),
                grid_column=np.zeros(count, dtype=int),
                source_xy_m=np.column_stack((np.arange(count), np.zeros(count))),
                displacement_m=np.tile([1000.0, -500.0], (count, 1)),
                available=np.ones(count, dtype=bool),
                selected_matches=np.full(count, 8),
                candidate_matches=np.full(count, 12),
                support_radius_m=np.full(count, 1000.0),
                maximum_residual_m=np.zeros(count),
            )
            return PairResult(
                MotionMatches.empty(),
                field,
                np.empty(0, dtype=int),
                0.0,
                0.0,
                prior_displacement_m,
            )

    tracker = StubTracker()
    results = tracker.track_sequence(
        ["image-a", "image-b", "image-c"],
        [24.0, 12.0],
        sequential_prior_uncertainty_m=15_000.0,
    )

    assert tracker.pairs == [
        ("image-a", "image-b", 24.0),
        ("image-b", "image-c", 12.0),
    ]
    assert tracker.extracted == ["image-a", "image-b", "image-c"]
    assert results[0].prior_displacement_m is None
    assert np.allclose(results[1].prior_displacement_m, [500.0, -250.0])


def test_fold_rejection_matches_existing_dataframe_implementation():
    source = np.array(
        [(x * 4000.0, y * 4000.0) for y in range(5) for x in range(5)]
    )
    rng = np.random.default_rng(20260818)
    displacement = np.zeros_like(source)
    for trial in range(7):
        displacement = rng.normal(0.0, 3000.0, size=source.shape)
        if trial % 2 == 0:
            selected = rng.choice(len(source), 3, replace=False)
            displacement[selected] += rng.normal(0.0, 9000.0, size=(3, 2))
    frame = pd.DataFrame(
        {
            "source_x": source[:, 0],
            "source_y": source[:, 1],
            "proposal_dx_m": displacement[:, 0],
            "proposal_dy_m": displacement[:, 1],
            "available": True,
        }
    )
    expected, expected_indices, _ = reject_flipped_nodes_until_stable(
        frame, 4000.0
    )
    field = DriftField(
        grid_row=np.repeat(np.arange(5), 5),
        grid_column=np.tile(np.arange(5), 5),
        source_xy_m=source,
        displacement_m=displacement,
        available=np.ones(len(source), dtype=bool),
        selected_matches=np.full(len(source), 8),
        candidate_matches=np.full(len(source), 12),
        support_radius_m=np.full(len(source), 4000.0),
        maximum_residual_m=np.zeros(len(source)),
    )

    actual, actual_indices = reject_folds(field, 4000.0)

    assert np.array_equal(actual.available, expected["available"].to_numpy())
    assert np.array_equal(actual_indices, expected_indices)


def stored_pair_result() -> PairResult:
    matches = MotionMatches(
        source_feature_id=np.array([10, 11, 12], dtype=np.int64),
        source_tile_id=np.array([2, 2, 3], dtype=np.int32),
        target_tile_id=np.array([5, 5, 6], dtype=np.int32),
        source_xy_m=np.array([[0.0, 0.0], [4000.0, 0.0], [0.0, 4000.0]]),
        target_xy_m=np.array(
            [[120.0, -40.0], [4121.0, -39.0], [119.0, 3960.0]]
        ),
        score=np.array([0.9, 0.8, 0.7], dtype=np.float32),
    )
    field = DriftField(
        grid_row=np.array([0, 0, 1, 1]),
        grid_column=np.array([0, 1, 0, 1]),
        source_xy_m=np.array(
            [[0.0, 0.0], [4000.0, 0.0], [0.0, 4000.0], [4000.0, 4000.0]]
        ),
        displacement_m=np.array(
            [[120.0, -40.0], [121.0, -39.0], [119.0, -40.0], [np.nan, np.nan]]
        ),
        available=np.array([True, True, False, False]),
        selected_matches=np.array([9, 8, 8, 5], dtype=np.int32),
        candidate_matches=np.array([12, 12, 10, 5], dtype=np.int32),
        support_radius_m=np.array([5200.0, 5300.0, 5500.0, 5900.0]),
        maximum_residual_m=np.array([90.0, 110.0, 120.0, np.nan]),
    )
    return PairResult(
        matches=matches,
        field=field,
        fold_rejected_indices=np.array([2], dtype=np.int64),
        matching_seconds=1.25,
        field_seconds=0.5,
        prior_displacement_m=(120.0, -40.0),
    )


def test_learned_store_round_trips_pair_and_resumes(tmp_path):
    config = ALIKEDConfig()
    store = LearnedDriftStore(
        tmp_path / "run.sqlite", tmp_path / "run.zarr", "test-run", config
    )
    pair = ImagePair(
        source_image_id=740,
        target_image_id=849,
        source_path="source.tiff",
        target_path="target.tiff",
        elapsed_hours=21.4,
        source_time_utc="2020-01-07T06:02:43Z",
        target_time_utc="2020-01-08T03:27:30Z",
        prior_displacement_m=(120.0, -40.0),
        prior_uncertainty_m=15_000.0,
    )
    expected = stored_pair_result()

    assert store.load_pair(pair) is None
    pair_key = store.save_pair(pair, expected)
    actual = store.load_pair(pair)

    assert len(pair_key) == 64
    assert store.status(pair) == "complete"
    assert actual is not None
    assert store.incomplete_pair_keys() == ()
    for name in (
        "source_feature_id",
        "source_tile_id",
        "target_tile_id",
        "source_xy_m",
        "target_xy_m",
        "score",
    ):
        np.testing.assert_array_equal(
            getattr(actual.matches, name), getattr(expected.matches, name)
        )
    for name in (
        "grid_row",
        "grid_column",
        "source_xy_m",
        "displacement_m",
        "available",
        "selected_matches",
        "candidate_matches",
        "support_radius_m",
        "maximum_residual_m",
    ):
        np.testing.assert_equal(
            getattr(actual.field, name), getattr(expected.field, name)
        )
    np.testing.assert_array_equal(
        actual.fold_rejected_indices, expected.fold_rejected_indices
    )
    assert actual.matching_seconds == expected.matching_seconds
    assert actual.field_seconds == expected.field_seconds
    assert actual.prior_displacement_m == expected.prior_displacement_m
    with pytest.raises(ValueError, match="different config"):
        LearnedDriftStore(
            tmp_path / "run.sqlite",
            tmp_path / "run.zarr",
            "test-run",
            ALIKEDConfig(grid_spacing_m=8000.0),
        )


def test_learned_store_retries_failed_pair(tmp_path, monkeypatch):
    store = LearnedDriftStore(
        tmp_path / "run.sqlite",
        tmp_path / "run.zarr",
        "test-run",
        ALIKEDConfig(),
    )
    pair = ImagePair(
        740,
        849,
        "source.tiff",
        "target.tiff",
        21.4,
        prior_displacement_m=(120.0, -40.0),
    )
    result = stored_pair_result()
    write_result = store._write_result

    def fail_write(_pair_key, _result):
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(store, "_write_result", fail_write)
    with pytest.raises(OSError, match="simulated interrupted write"):
        store.save_pair(pair, result)
    assert store.status(pair) == "failed"
    assert len(store.incomplete_pair_keys()) == 1
    assert store.load_pair(pair) is None

    monkeypatch.setattr(store, "_write_result", write_result)
    store.save_pair(pair, result)
    assert store.status(pair) == "complete"
    assert store.load_pair(pair) is not None
