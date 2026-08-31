import numpy as np
import pandas as pd
import shapely
import torch
from kornia.feature import match_mnn, match_smnn
from kornia.feature.lightglue import LightGlue

from experiments.aliked_matchers import (
    DirectALIKEDLightGlueMatcher,
    MutualNearestDescriptorMatcher,
    build_aliked_matcher,
)

from experiments.run_aliked_dense_pair import (
    adaptive_consensus_at_queries,
    anchored_axis_centres,
    attach_buoy_source_positions,
    axis_centres,
    consensus_at_queries,
    match_tiles,
    nearest_consensus_at_queries,
    select_pair_cases,
    spatially_thin_tiles_for_matching,
    topology_summary,
)
from experiments.run_aliked_selected_sequence import (
    restrict_tiles_to_domain,
    summarize_buoys,
)
from experiments.refine_aliked_dense_topology import (
    flipped_node_indices,
    reject_flipped_nodes_until_stable,
)
from experiments.validate_aliked_controlled_warp import (
    fit_affine,
    piecewise_truth_at_queries,
    truth_at_queries,
)


def test_axis_centres_cover_requested_interval():
    centres = axis_centres(0.0, 100.0, 40.0)

    assert np.allclose(centres, [20.0, 60.0, 100.0])
    assert centres[0] - 20.0 <= 0.0
    assert centres[-1] + 20.0 >= 100.0

    anchored = anchored_axis_centres(10.0, 100.0, 40.0, 0.0)
    assert np.allclose(anchored, [20.0, 60.0, 100.0])


def test_sequence_domain_restriction_preserves_only_inside_features():
    tile = {
        "tile_id": 3,
        "core": shapely.box(0.0, 0.0, 100.0, 100.0),
        "keypoints": torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        "descriptors": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "scores": torch.tensor([0.8, 0.7]),
        "xy": np.array([[25.0, 25.0], [75.0, 75.0]]),
    }

    restricted = restrict_tiles_to_domain(
        [tile], shapely.box(0.0, 0.0, 50.0, 50.0)
    )

    assert len(restricted) == 1
    assert restricted[0]["tile_id"] == 3
    assert restricted[0]["xy"].tolist() == [[25.0, 25.0]]
    assert restricted[0]["descriptors"].tolist() == [[0.0, 1.0, 2.0, 3.0]]


def test_empty_unlabelled_pair_has_zero_buoy_counts():
    summary = summarize_buoys(pd.DataFrame())

    assert summary["expected"] == 0
    assert summary["available"] == 0
    assert summary["median_error_m"] is None


def test_match_tiles_device_reuse_preserves_matches():
    class IdentityMatcher:
        def __call__(self, source, target, source_laf, target_laf, **kwargs):
            count = min(len(source), len(target))
            indices = torch.arange(count, dtype=torch.long)
            return (
                torch.linspace(1.0, 0.5, count),
                torch.column_stack([indices, indices]),
            )

    def tiles(offset):
        keypoints = torch.tensor(
            [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]]
        )
        return [
            {
                "tile_id": 0,
                "core": shapely.box(-100.0, -100.0, 100.0, 100.0),
                "keypoints": keypoints,
                "descriptors": torch.eye(4),
                "scores": torch.ones(4),
                "xy": keypoints.numpy() + offset,
            }
        ]

    baseline, _ = match_tiles(
        tiles(0.0),
        tiles(100.0),
        IdentityMatcher(),
        torch.device("cpu"),
        512,
        1000.0,
        1.0,
        1000.0,
        reuse_device_features=False,
    )
    reused, _ = match_tiles(
        tiles(0.0),
        tiles(100.0),
        IdentityMatcher(),
        torch.device("cpu"),
        512,
        1000.0,
        1.0,
        1000.0,
        reuse_device_features=True,
    )

    pd.testing.assert_frame_equal(baseline, reused)


def test_match_tiles_batches_target_hypotheses_without_changing_matches():
    class BatchIdentityMatcher:
        matcher_name = "lightglue_direct"
        uses_laf = False
        uses_direct_keypoints = True

        def __call__(
            self,
            source,
            target,
            source_laf,
            target_laf,
            **kwargs,
        ):
            count = min(len(source), len(target))
            indices = torch.arange(count, dtype=torch.long)
            return torch.ones(count, 1), torch.column_stack([indices, indices])

        def forward_batch(
            self,
            source,
            target,
            *,
            source_keypoints,
            target_keypoints,
            hw1,
            hw2,
        ):
            del source_keypoints, target_keypoints, hw1, hw2
            results = []
            for desc0, desc1 in zip(source, target, strict=True):
                scores, indexes = self(desc0, desc1, None, None)
                results.append((scores, indexes, {}))
            return results

    keypoints = torch.tensor(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]]
    )

    def tile(tile_id, offset):
        return {
            "tile_id": tile_id,
            "core": shapely.box(-100.0, -100.0, 100.0, 100.0),
            "keypoints": keypoints,
            "descriptors": torch.eye(4),
            "scores": torch.ones(4),
            "xy": keypoints.numpy() + offset,
        }

    source = [tile(0, 0.0)]
    targets = [tile(tile_id, float(tile_id)) for tile_id in (1, 2, 3)]
    individual, individual_audit = match_tiles(
        source,
        targets,
        BatchIdentityMatcher(),
        torch.device("cpu"),
        512,
        1000.0,
        1.0,
        1000.0,
    )
    batched, batched_audit = match_tiles(
        source,
        targets,
        BatchIdentityMatcher(),
        torch.device("cpu"),
        512,
        1000.0,
        1.0,
        1000.0,
        lightglue_target_batch_size=3,
    )

    pd.testing.assert_frame_equal(individual, batched)
    assert individual_audit[0]["matcher_invocations"] == 3
    assert batched_audit[0]["matcher_invocations"] == 1


def test_exact_cosine_mnn_matches_kornia_l2_mnn_for_unit_descriptors():
    generator = torch.Generator().manual_seed(20260820)
    source = torch.nn.functional.normalize(torch.randn(12, 8, generator=generator), dim=1)
    target = torch.nn.functional.normalize(torch.randn(15, 8, generator=generator), dim=1)

    _, expected = match_mnn(source, target)
    scores, actual = MutualNearestDescriptorMatcher()(source, target)

    assert actual.tolist() == expected.tolist()
    assert torch.all(scores <= 1.0 + 1.0e-6)
    assert torch.all(scores >= -1.0 - 1.0e-6)


def test_cosine_smnn_matches_kornia_symmetric_l2_ratio_for_unit_descriptors():
    generator = torch.Generator().manual_seed(20260821)
    source = torch.nn.functional.normalize(torch.randn(12, 8, generator=generator), dim=1)
    target = torch.nn.functional.normalize(torch.randn(15, 8, generator=generator), dim=1)

    _, expected = match_smnn(source, target, th=0.95)
    _, actual = MutualNearestDescriptorMatcher(ratio=0.95)(source, target)

    assert actual.tolist() == expected.tolist()


def test_descriptor_matcher_rejects_invalid_ratio():
    with np.testing.assert_raises_regex(ValueError, "between zero and one"):
        MutualNearestDescriptorMatcher(ratio=1.0)


def test_lightglue_layer_count_is_bounded_before_model_loading():
    with np.testing.assert_raises_regex(ValueError, "between one and nine"):
        build_aliked_matcher("lightglue", torch.device("cpu"), lightglue_layers=0)


def test_direct_lightglue_adapter_uses_keypoints_without_lafs():
    class FakeRawMatcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.data = None

        def forward(self, data):
            self.data = data
            return {
                "matches0": torch.tensor([[1, -1, 0]]),
                "matching_scores0": torch.tensor([[0.8, 0.0, 0.7]]),
                "stop": 2,
                "prune0": torch.tensor([[2, 1, 2]]),
                "prune1": torch.tensor([[2, 2, 1, 2]]),
            }

    raw = FakeRawMatcher()
    adapter = DirectALIKEDLightGlueMatcher({}, raw_matcher=raw)
    source = torch.randn(3, 128)
    target = torch.randn(4, 128)
    source_keypoints = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    target_keypoints = torch.tensor(
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0]]
    )

    scores, indexes = adapter(
        source,
        target,
        source_keypoints=source_keypoints,
        target_keypoints=target_keypoints,
        hw1=(512, 256),
        hw2=(128, 64),
    )

    assert indexes.tolist() == [[0, 1], [2, 0]]
    assert torch.allclose(scores[:, 0], torch.tensor([0.8, 0.7]))
    assert torch.equal(raw.data["image0"]["keypoints"], source_keypoints[None])
    assert raw.data["image0"]["image_size"].tolist() == [[256.0, 512.0]]
    assert raw.data["image1"]["image_size"].tolist() == [[64.0, 128.0]]
    assert adapter.last_diagnostics == {
        "stop_layer": 2,
        "source_input_features": 3,
        "target_input_features": 4,
        "source_pruned_features": 1,
        "target_pruned_features": 1,
    }


def test_direct_lightglue_variable_length_batch_matches_individual_calls():
    torch.manual_seed(20260823)
    raw = LightGlue(
        None,
        input_dim=8,
        descriptor_dim=8,
        n_layers=3,
        num_heads=2,
        flash=False,
        depth_confidence=0.95,
        width_confidence=0.99,
        filter_threshold=0.0,
    ).eval()
    adapter = DirectALIKEDLightGlueMatcher({}, raw_matcher=raw)
    source = [torch.randn(7, 8), torch.randn(11, 8)]
    target = [torch.randn(9, 8), torch.randn(6, 8)]
    source_keypoints = [
        torch.rand(len(values), 2) * 511.0 for values in source
    ]
    target_keypoints = [
        torch.rand(len(values), 2) * 511.0 for values in target
    ]
    expected = []
    for desc0, desc1, keypoints0, keypoints1 in zip(
        source, target, source_keypoints, target_keypoints, strict=True
    ):
        scores, indexes = adapter(
            desc0,
            desc1,
            source_keypoints=keypoints0,
            target_keypoints=keypoints1,
            hw1=(512, 512),
            hw2=(512, 512),
        )
        expected.append((scores, indexes, dict(adapter.last_diagnostics)))

    actual = adapter.forward_batch(
        source,
        target,
        source_keypoints=source_keypoints,
        target_keypoints=target_keypoints,
        hw1=(512, 512),
        hw2=(512, 512),
    )

    for expected_values, actual_values in zip(expected, actual, strict=True):
        expected_scores, expected_indexes, expected_diagnostics = expected_values
        actual_scores, actual_indexes, actual_diagnostics = actual_values
        assert actual_indexes.tolist() == expected_indexes.tolist()
        torch.testing.assert_close(actual_scores, expected_scores, atol=1e-6, rtol=1e-5)
        assert actual_diagnostics == expected_diagnostics


def test_consensus_accepts_generic_matcher_scores():
    matches = pd.DataFrame(
        {
            "source_x": [100.0, 200.0, 300.0],
            "source_y": [0.0, 0.0, 0.0],
            "dx_m": [1000.0, 1005.0, 995.0],
            "dy_m": [200.0, 205.0, 195.0],
            "matcher_score": [0.9, 0.8, 0.7],
            "physics_valid": [True, True, True],
        }
    )

    proposal = consensus_at_queries(
        matches,
        pd.DataFrame([{"source_x": 0.0, "source_y": 0.0}]),
        tight_radius_m=2000.0,
        consensus_radius_m=1000.0,
    ).iloc[0]

    assert proposal.available
    assert proposal.selected_vectors == 3


def test_match_tiles_prior_restricts_target_window_without_thinning_features():
    class IdentityMatcher:
        def __call__(self, source, target, source_laf, target_laf, **kwargs):
            count = min(len(source), len(target))
            indices = torch.arange(count, dtype=torch.long)
            return torch.ones(count), torch.column_stack([indices, indices])

    keypoints = torch.tensor(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]]
    )

    def tile(tile_id, core, offset):
        return {
            "tile_id": tile_id,
            "core": core,
            "keypoints": keypoints,
            "descriptors": torch.eye(4),
            "scores": torch.ones(4),
            "xy": keypoints.numpy() + offset,
        }

    matches, audit = match_tiles(
        [tile(0, shapely.box(0.0, 0.0, 100.0, 100.0), 0.0)],
        [
            tile(1, shapely.box(0.0, 0.0, 100.0, 100.0), 0.0),
            tile(
                2,
                shapely.box(200.0, 0.0, 300.0, 100.0),
                np.array([200.0, 0.0]),
            ),
        ],
        IdentityMatcher(),
        torch.device("cpu"),
        512,
        300.0,
        1.0,
        1000.0,
        physics_subset_matching=True,
        matching_prior_displacement_m=(200.0, 0.0),
        matching_prior_uncertainty_m=20.0,
    )

    assert audit[0]["candidate_target_tiles"] == 1
    assert len(matches) == 4
    np.testing.assert_allclose(
        matches[["dx_m", "dy_m"]], np.tile([200.0, 0.0], (4, 1))
    )


def test_mnn_candidate_limit_executes_only_highest_support_target_tile():
    class CountingMatcher:
        matcher_name = "lightglue"
        uses_laf = True

        def __init__(self):
            self.calls = 0

        def __call__(self, source, target, source_laf, target_laf, **kwargs):
            self.calls += 1
            count = min(len(source), len(target))
            indices = torch.arange(count, dtype=torch.long)
            return torch.ones(count), torch.column_stack([indices, indices])

    keypoints = torch.tensor(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]]
    )

    def tile(tile_id, descriptors, x_offset):
        return {
            "tile_id": tile_id,
            "core": shapely.box(-100.0, -100.0, 100.0, 100.0),
            "keypoints": keypoints,
            "descriptors": descriptors,
            "scores": torch.ones(4),
            "xy": keypoints.numpy() + np.array([x_offset, 0.0]),
        }

    matcher = CountingMatcher()
    calls = []
    matches, audit = match_tiles(
        [tile(0, torch.eye(4), 0.0)],
        [
            tile(1, torch.eye(4), 100.0),
            tile(2, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1), 200.0),
        ],
        matcher,
        torch.device("cpu"),
        512,
        300.0,
        1.0,
        1000.0,
        matcher_call_audit=calls,
        mnn_candidate_limit=1,
    )

    assert matcher.calls == 1
    assert matches["target_tile_id"].unique().tolist() == [1]
    assert audit[0]["candidate_target_tiles"] == 2
    assert audit[0]["executed_target_tiles"] == 1
    assert sum(bool(row["matcher_executed"]) for row in calls) == 1


def test_matching_thinning_distributes_features_across_spatial_cells():
    keypoints = torch.tensor(
        [
            [40.0, 40.0],
            [50.0, 50.0],
            [300.0, 40.0],
            [310.0, 50.0],
            [40.0, 300.0],
            [50.0, 310.0],
            [300.0, 300.0],
            [310.0, 310.0],
        ]
    )
    tile = {
        "tile_id": 0,
        "keypoints": keypoints,
        "descriptors": torch.arange(32, dtype=torch.float32).reshape(8, 4),
        "scores": torch.tensor([0.99, 0.98, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
        "xy": keypoints.numpy() * 80.0,
    }

    result = spatially_thin_tiles_for_matching(
        [tile], 4, cells_per_axis=2, tile_pixels=512, margin_pixels=32
    )[0]

    assert result["features_before_matching_thinning"] == 8
    assert result["keypoints"].tolist() == keypoints[[0, 2, 4, 6]].tolist()
    assert result["descriptors"].tolist() == tile["descriptors"][[0, 2, 4, 6]].tolist()
    np.testing.assert_allclose(result["xy"], tile["xy"][[0, 2, 4, 6]])


def test_attach_buoy_source_positions_from_observations(tmp_path):
    pair = pd.DataFrame(
        [{"buoy_id": "001", "source_image_id": 7, "target_image_id": 8}]
    )
    observations = tmp_path / "observations.csv"
    pd.DataFrame(
        [
            {
                "buoy_id": "001",
                "image_id": 7,
                "x": 1200.0,
                "y": -3400.0,
                "analysis_crs": "EPSG:3413",
            }
        ]
    ).to_csv(observations, index=False)

    result = attach_buoy_source_positions(pair, observations).iloc[0]

    assert result.source_x == 1200.0
    assert result.source_y == -3400.0
    assert result.source_analysis_crs == "EPSG:3413"


def test_select_pair_cases_requires_explicit_split_for_mixed_input():
    cases = pd.DataFrame(
        [
            {
                "source_image_id": 7,
                "target_image_id": 8,
                "within_dataset_split": split,
            }
            for split in ("development", "final_holdout")
        ]
    )

    with np.testing.assert_raises_regex(ValueError, "multiple data splits"):
        select_pair_cases(cases, 7, 8, None)

    selected = select_pair_cases(cases, 7, 8, "development")
    assert selected["within_dataset_split"].tolist() == ["development"]


def test_consensus_and_topology_preserve_uniform_translation():
    matches = pd.DataFrame(
        [
            {
                "source_x": 0.0,
                "source_y": 0.0,
                "dx_m": 1200.0,
                "dy_m": -300.0,
                "lightglue_score": 0.9,
                "physics_valid": True,
            }
        ]
    )
    queries = pd.DataFrame(
        [{"grid_row": 0, "grid_column": 0, "source_x": 0.0, "source_y": 0.0}]
    )
    proposal = consensus_at_queries(matches, queries, 2000.0, 1000.0).iloc[0]

    assert proposal.available
    assert proposal.proposal_dx_m == 1200.0
    assert proposal.proposal_dy_m == -300.0

    missing = consensus_at_queries(
        matches,
        pd.DataFrame(
            [
                {
                    "grid_row": 1,
                    "grid_column": 1,
                    "source_x": 10000.0,
                    "source_y": 10000.0,
                }
            ]
        ),
        2000.0,
        1000.0,
    ).iloc[0]
    assert not missing.available

    field = pd.DataFrame(
        [
            {
                "available": True,
                "source_x": x,
                "source_y": y,
                "proposal_dx_m": 1200.0,
                "proposal_dy_m": -300.0,
            }
            for y in (0.0, 4000.0, 8000.0)
            for x in (0.0, 4000.0, 8000.0)
        ]
    )
    topology = topology_summary(field, 4000.0)

    assert topology["triangles"] == 8
    assert topology["flipped_triangles"] == 0
    assert np.isclose(topology["area_ratio_median"], 1.0)


def test_flipped_node_selection_returns_only_triangle_vertices():
    field = pd.DataFrame(
        [
            {
                "available": True,
                "source_x": 0.0,
                "source_y": 0.0,
                "proposal_dx_m": 0.0,
                "proposal_dy_m": 0.0,
            },
            {
                "available": True,
                "source_x": 4000.0,
                "source_y": 0.0,
                "proposal_dx_m": 0.0,
                "proposal_dy_m": 0.0,
            },
            {
                "available": True,
                "source_x": 0.0,
                "source_y": 4000.0,
                "proposal_dx_m": 0.0,
                "proposal_dy_m": -8000.0,
            },
        ]
    )

    assert np.array_equal(flipped_node_indices(field, 4000.0), [0, 1, 2])


def test_flipped_node_rejection_rechecks_after_retriangulation():
    source = np.array(
        [(x * 4000.0, y * 4000.0) for y in range(5) for x in range(5)]
    )
    rng = np.random.default_rng(20260818)
    displacement = None
    for trial in range(7):
        displacement = rng.normal(0.0, 3000.0, size=source.shape)
        if trial % 2 == 0:
            selected = rng.choice(len(source), 3, replace=False)
            displacement[selected] += rng.normal(0.0, 9000.0, size=(3, 2))
    field = pd.DataFrame(
        {
            "source_x": source[:, 0],
            "source_y": source[:, 1],
            "proposal_dx_m": displacement[:, 0],
            "proposal_dy_m": displacement[:, 1],
            "available": True,
        }
    )
    first = flipped_node_indices(field, 4000.0)
    one_pass = field.copy()
    one_pass.loc[first, "available"] = False

    rejected, rejected_indices, iteration_counts = (
        reject_flipped_nodes_until_stable(field, 4000.0)
    )

    assert len(flipped_node_indices(one_pass, 4000.0)) > 0
    assert len(flipped_node_indices(rejected, 4000.0)) == 0
    assert iteration_counts == [14, 3]
    assert len(rejected_indices) == 17


def test_adaptive_consensus_uses_smallest_radius_with_enough_support():
    matches = pd.DataFrame(
        [
            {
                "source_x": x,
                "source_y": 0.0,
                "dx_m": 1000.0,
                "dy_m": 200.0,
                "lightglue_score": 0.8,
                "physics_valid": True,
            }
            for x in (500.0, 2500.0, -2500.0)
        ]
    )
    queries = pd.DataFrame(
        [{"grid_row": 0, "grid_column": 0, "source_x": 0.0, "source_y": 0.0}]
    )

    result = adaptive_consensus_at_queries(
        matches, queries, [2000.0, 3000.0], 3, 1000.0
    ).iloc[0]

    assert result.available
    assert result.selected_vectors == 3
    assert result.support_radius_m == 3000.0

    insufficient = adaptive_consensus_at_queries(
        matches.iloc[:2], queries, [2000.0, 3000.0], 3, 1000.0
    ).iloc[0]
    assert not insufficient.available
    assert np.isnan(insufficient.support_radius_m)

    local_matches = pd.DataFrame(
        [
            {
                "source_x": distance,
                "source_y": 0.0,
                "dx_m": 800.0 if index < 5 else 1400.0,
                "dy_m": -400.0,
                "lightglue_score": 1.0,
                "physics_valid": True,
            }
            for index, distance in enumerate(
                [100.0, -200.0, 300.0, -400.0, 500.0, -600.0, 700.0, -800.0]
            )
        ]
    )
    nearest = nearest_consensus_at_queries(
        local_matches, queries, 2000.0, 8, 8, 1000.0
    ).iloc[0]
    assert nearest.available
    assert nearest.selected_vectors == 8
    assert np.isclose(nearest.proposal_dx_m, 800.0)


def test_controlled_affine_truth_is_recovered_exactly():
    queries = pd.DataFrame(
        {
            "source_x": [0.0, 0.0, 20.0, 20.0],
            "source_y": [0.0, 20.0, 0.0, 20.0],
        }
    )
    gradient = np.array([[1.01, 0.02], [-0.01, 0.99]])
    translation = np.array([3.0, -2.0])
    truth = truth_at_queries(
        queries, (21, 21), 1.0, gradient, translation
    )
    field = queries.assign(
        available=True,
        proposal_dx_m=truth[:, 0],
        proposal_dy_m=truth[:, 1],
    )

    fitted_gradient, fitted_intercept = fit_affine(field)
    center = np.array([10.0, 10.0])
    recovered_translation = fitted_intercept - (center - gradient @ center)

    assert np.allclose(fitted_gradient, gradient)
    assert np.allclose(recovered_translation, translation)

    piecewise = piecewise_truth_at_queries(
        pd.DataFrame({"source_x": [5.0, 15.0], "source_y": [10.0, 10.0]}),
        (21, 21),
        1.0,
        np.array([1.0, 2.0]),
        np.array([4.0, 2.0]),
    )
    assert np.allclose(piecewise, [[1.0, 2.0], [4.0, 2.0]])
