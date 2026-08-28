"""Physics-routed direct LightGlue matching for ALIKED tiles."""

from __future__ import annotations

import numpy as np
import shapely
import torch
from kornia.feature.lightglue import LightGlue
from shapely.affinity import translate

from .config import ALIKEDConfig
from .types import FeatureTile, ImageFeatures, MotionMatches


class DirectALIKEDLightGlue(torch.nn.Module):
    """Use ALIKED keypoints directly, without unused affine-frame wrappers."""

    def __init__(
        self,
        config: ALIKEDConfig,
        model: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.model = model or LightGlue(
            "aliked",
            n_layers=config.lightglue_layers,
            depth_confidence=config.lightglue_depth_confidence,
            width_confidence=config.lightglue_width_confidence,
            filter_threshold=config.lightglue_match_threshold,
        )
        self.last_call_degenerate = False

    def forward(
        self,
        source_descriptors: torch.Tensor,
        target_descriptors: torch.Tensor,
        source_keypoints_px: torch.Tensor,
        target_keypoints_px: torch.Tensor,
        tile_size_px: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_call_degenerate = False
        if len(source_descriptors) < 2 or len(target_descriptors) < 2:
            return _empty_match(source_descriptors)
        image_size = source_descriptors.new_tensor(
            [[tile_size_px, tile_size_px]]
        )
        try:
            prediction = self.model(
                {
                    "image0": {
                        "keypoints": source_keypoints_px[None],
                        "descriptors": source_descriptors[None],
                        "image_size": image_size,
                    },
                    "image1": {
                        "keypoints": target_keypoints_px[None],
                        "descriptors": target_descriptors[None],
                        "image_size": image_size,
                    },
                }
            )
        except IndexError as error:
            if "non-zero size" not in str(error):
                raise
            self.last_call_degenerate = True
            return _empty_match(source_descriptors)
        matches0 = prediction["matches0"]
        scores0 = prediction["matching_scores0"]
        valid = matches0 > -1
        indices = torch.stack([torch.where(valid)[1], matches0[valid]], dim=-1)
        return scores0[valid].reshape(-1), indices


def _empty_match(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return reference.new_empty(0), torch.empty(
        (0, 2), dtype=torch.long, device=reference.device
    )


def _source_indices_for_target(
    source: FeatureTile,
    target: FeatureTile,
    maximum_displacement_m: float,
    prior_displacement_m: np.ndarray | None,
    prior_uncertainty_m: float | None,
) -> np.ndarray:
    source_xy_m = source.xy_m
    radius_m = maximum_displacement_m
    if prior_displacement_m is not None:
        source_xy_m = source_xy_m + prior_displacement_m
        radius_m = float(prior_uncertainty_m)
    return np.flatnonzero(
        shapely.intersects_xy(
            target.core.buffer(radius_m),
            source_xy_m[:, 0],
            source_xy_m[:, 1],
        )
    )


def physical_target_tiles(
    source: FeatureTile,
    targets: tuple[FeatureTile, ...],
    maximum_displacement_m: float,
    minimum_features: int,
    prior_displacement_m: np.ndarray | None = None,
    prior_uncertainty_m: float | None = None,
) -> tuple[FeatureTile, ...]:
    """Return target cores reachable under the speed gate or supplied prior."""
    reachable = source.core.buffer(maximum_displacement_m)
    if prior_displacement_m is not None:
        reachable = translate(
            source.core,
            xoff=float(prior_displacement_m[0]),
            yoff=float(prior_displacement_m[1]),
        ).buffer(float(prior_uncertainty_m))
    return tuple(
        target
        for target in targets
        if len(target) >= minimum_features and reachable.intersects(target.core)
    )


def _mutual_nearest(
    source_descriptors: torch.Tensor,
    target_descriptors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not len(source_descriptors) or not len(target_descriptors):
        return _empty_match(source_descriptors)
    similarity = source_descriptors @ target_descriptors.transpose(0, 1)
    source_best = similarity.argmax(dim=1)
    target_best = similarity.argmax(dim=0)
    source_indices = torch.arange(len(source_descriptors), device=similarity.device)
    mutual = target_best[source_best] == source_indices
    selected_source = source_indices[mutual]
    selected_target = source_best[mutual]
    return (
        similarity[selected_source, selected_target],
        torch.column_stack((selected_source, selected_target)),
    )


def _rank_target_tiles(
    source: FeatureTile,
    targets: tuple[FeatureTile, ...],
    elapsed_days: float,
    maximum_displacement_m: float,
    config: ALIKEDConfig,
    prior_displacement_m: np.ndarray | None,
    prior_uncertainty_m: float | None,
) -> tuple[FeatureTile, ...]:
    limit = config.target_tile_limit
    if limit is None or len(targets) <= limit:
        return targets
    ranked = []
    for target in targets:
        source_indices = _source_indices_for_target(
            source,
            target,
            maximum_displacement_m,
            prior_displacement_m,
            prior_uncertainty_m,
        )
        if len(source_indices) < config.minimum_features_per_match:
            continue
        tensor_indices = torch.as_tensor(
            source_indices, device=source.descriptors.device
        )
        scores, indices = _mutual_nearest(
            source.descriptors.index_select(0, tensor_indices),
            target.descriptors,
        )
        indices_np = indices.detach().cpu().numpy()
        valid = np.zeros(len(indices_np), dtype=bool)
        if len(indices_np):
            matched_source = source_indices[indices_np[:, 0]]
            displacement_m = (
                target.xy_m[indices_np[:, 1]] - source.xy_m[matched_source]
            )
            valid = (
                np.linalg.norm(displacement_m, axis=1) / elapsed_days
                <= config.maximum_speed_m_per_day
            )
            if prior_displacement_m is not None:
                valid &= (
                    np.linalg.norm(displacement_m - prior_displacement_m, axis=1)
                    <= float(prior_uncertainty_m)
                )
        score_values = scores.detach().cpu().numpy()
        ranked.append(
            (
                -int(valid.sum()),
                -float(np.median(score_values[valid])) if valid.any() else np.inf,
                target.tile_id,
                target,
            )
        )
    return tuple(row[-1] for row in sorted(ranked, key=lambda row: row[:3])[:limit])


def _best_per_source(source_indices: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if not len(source_indices):
        return np.empty(0, dtype=int)
    order = np.argsort(-scores, kind="stable")
    _, first = np.unique(source_indices[order], return_index=True)
    return np.sort(order[first])


def match_features(
    source_features: ImageFeatures,
    target_features: ImageFeatures,
    elapsed_hours: float,
    matcher,
    device: torch.device,
    config: ALIKEDConfig,
    prior_displacement_m: tuple[float, float] | None = None,
    prior_uncertainty_m: float | None = None,
) -> MotionMatches:
    """Match a pair and retain one speed-valid target per source feature."""
    if elapsed_hours <= 0:
        raise ValueError("elapsed hours must be positive")
    if source_features.analysis_epsg != target_features.analysis_epsg:
        raise ValueError("source and target feature CRS must agree")
    if source_features.analysis_epsg != config.analysis_epsg:
        raise ValueError("feature CRS must agree with the ALIKED config")
    if (prior_displacement_m is None) != (prior_uncertainty_m is None):
        raise ValueError("prior displacement and uncertainty must be paired")
    if prior_uncertainty_m is not None and prior_uncertainty_m <= 0:
        raise ValueError("prior uncertainty must be positive")

    source_features = source_features.to(device)
    target_features = target_features.to(device)
    prior = (
        np.asarray(prior_displacement_m, dtype=float)
        if prior_displacement_m is not None
        else None
    )
    elapsed_days = elapsed_hours / 24.0
    maximum_displacement_m = config.maximum_displacement_m(elapsed_hours)
    source_ids = []
    source_tile_ids = []
    target_tile_ids = []
    source_xy = []
    target_xy = []
    match_scores = []
    global_source_offset = 0

    for source in source_features.tiles:
        candidate_targets = physical_target_tiles(
            source,
            target_features.tiles,
            maximum_displacement_m,
            config.minimum_features_per_match,
            prior,
            prior_uncertainty_m,
        )
        candidate_targets = _rank_target_tiles(
            source,
            candidate_targets,
            elapsed_days,
            maximum_displacement_m,
            config,
            prior,
            prior_uncertainty_m,
        )
        local_source_indices = []
        local_target_xy = []
        local_target_tile_ids = []
        local_scores = []
        if len(source) >= config.minimum_features_per_match:
            for target in candidate_targets:
                subset = _source_indices_for_target(
                    source,
                    target,
                    maximum_displacement_m,
                    prior,
                    prior_uncertainty_m,
                )
                if len(subset) < config.minimum_features_per_match:
                    continue
                tensor_subset = torch.as_tensor(
                    subset, device=device, dtype=torch.long
                )
                with torch.inference_mode():
                    scores, indices = matcher(
                        source.descriptors.index_select(0, tensor_subset),
                        target.descriptors,
                        source.keypoints_px.index_select(0, tensor_subset),
                        target.keypoints_px,
                        config.tile_size_px,
                    )
                indices = indices.detach().cpu().numpy()
                if not len(indices):
                    continue
                local_source_indices.append(subset[indices[:, 0]])
                local_target_xy.append(target.xy_m[indices[:, 1]])
                local_target_tile_ids.append(
                    np.full(len(indices), target.tile_id, dtype=np.int32)
                )
                local_scores.append(scores.detach().cpu().numpy().reshape(-1))

        if local_source_indices:
            selected_source = np.concatenate(local_source_indices)
            selected_target_xy = np.concatenate(local_target_xy)
            selected_target_tiles = np.concatenate(local_target_tile_ids)
            selected_scores = np.concatenate(local_scores)
            keep = _best_per_source(selected_source, selected_scores)
            selected_source = selected_source[keep]
            selected_target_xy = selected_target_xy[keep]
            selected_target_tiles = selected_target_tiles[keep]
            selected_scores = selected_scores[keep]
            selected_source_xy = source.xy_m[selected_source]
            displacement_m = selected_target_xy - selected_source_xy
            valid = (
                np.linalg.norm(displacement_m, axis=1) / elapsed_days
                <= config.maximum_speed_m_per_day
            )
            if prior is not None:
                valid &= (
                    np.linalg.norm(displacement_m - prior, axis=1)
                    <= float(prior_uncertainty_m)
                )
            valid_indices = np.flatnonzero(valid)
            source_ids.append(global_source_offset + selected_source[valid_indices])
            source_tile_ids.append(
                np.full(len(valid_indices), source.tile_id, dtype=np.int32)
            )
            target_tile_ids.append(selected_target_tiles[valid_indices])
            source_xy.append(selected_source_xy[valid_indices])
            target_xy.append(selected_target_xy[valid_indices])
            match_scores.append(selected_scores[valid_indices])
        global_source_offset += len(source)

    if not source_ids:
        return MotionMatches.empty()
    return MotionMatches(
        source_feature_id=np.concatenate(source_ids).astype(np.int64, copy=False),
        source_tile_id=np.concatenate(source_tile_ids),
        target_tile_id=np.concatenate(target_tile_ids),
        source_xy_m=np.concatenate(source_xy).astype(np.float64, copy=False),
        target_xy_m=np.concatenate(target_xy).astype(np.float64, copy=False),
        score=np.concatenate(match_scores).astype(np.float32, copy=False),
    )
