"""Experimental matcher alternatives and audits for frozen ALIKED descriptors.

The selected direct matcher is maintained in ``limosat.learned_drift``. This
module retains rejected and diagnostic variants needed to reproduce experiments.
"""

from __future__ import annotations

import torch
from kornia.feature import LightGlueMatcher
from kornia.feature.lightglue import LightGlue, filter_matches, normalize_keypoints


def _lightglue_diagnostics(prediction: dict) -> dict[str, int]:
    """Reduce one LightGlue result to inexpensive call-level diagnostics."""
    stop = int(prediction["stop"])
    prune0 = prediction["prune0"]
    prune1 = prediction["prune1"]
    return {
        "stop_layer": stop,
        "source_input_features": int(prune0.shape[-1]),
        "target_input_features": int(prune1.shape[-1]),
        "source_pruned_features": int((prune0 < stop).sum().item()),
        "target_pruned_features": int((prune1 < stop).sum().item()),
    }


class AuditedLightGlueMatcher(LightGlueMatcher):
    """Kornia's adapter with the underlying LightGlue audit retained."""

    uses_laf = True
    uses_direct_keypoints = False

    def __init__(self, feature_name: str, params: dict):
        super().__init__(feature_name, params)
        self.last_diagnostics: dict[str, int] = {}
        self.matcher.register_forward_hook(self._capture_diagnostics)

    def _capture_diagnostics(self, _module, _inputs, output) -> None:
        self.last_diagnostics = _lightglue_diagnostics(output)


class DirectALIKEDLightGlueMatcher(torch.nn.Module):
    """Call ALIKED-LightGlue without constructing unused local affine frames."""

    uses_laf = False
    uses_direct_keypoints = True
    matcher_name = "lightglue_direct"

    def __init__(self, params: dict, raw_matcher: torch.nn.Module | None = None):
        super().__init__()
        self.matcher = raw_matcher or LightGlue("aliked", **params)
        self.last_diagnostics: dict[str, int] = {}

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_laf=None,
        target_laf=None,
        *,
        source_keypoints: torch.Tensor,
        target_keypoints: torch.Tensor,
        hw1: tuple[int, int],
        hw2: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del source_laf, target_laf
        if len(source) < 2 or len(target) < 2:
            self.last_diagnostics = {
                "stop_layer": 0,
                "source_input_features": int(len(source)),
                "target_input_features": int(len(target)),
                "source_pruned_features": 0,
                "target_pruned_features": 0,
            }
            return source.new_empty((0, 1)), torch.empty(
                (0, 2), dtype=torch.long, device=source.device
            )
        prediction = self.matcher(
            {
                "image0": {
                    "keypoints": source_keypoints[None],
                    "descriptors": source[None],
                    "image_size": source.new_tensor(
                        [[hw1[1], hw1[0]]]
                    ),
                },
                "image1": {
                    "keypoints": target_keypoints[None],
                    "descriptors": target[None],
                    "image_size": target.new_tensor(
                        [[hw2[1], hw2[0]]]
                    ),
                },
            }
        )
        self.last_diagnostics = _lightglue_diagnostics(prediction)
        matches0 = prediction["matches0"]
        scores0 = prediction["matching_scores0"]
        valid = matches0 > -1
        indexes = torch.stack([torch.where(valid)[1], matches0[valid]], -1)
        return scores0[valid].reshape(-1, 1), indexes

    def forward_batch(
        self,
        source: list[torch.Tensor],
        target: list[torch.Tensor],
        *,
        source_keypoints: list[torch.Tensor],
        target_keypoints: list[torch.Tensor],
        hw1: tuple[int, int],
        hw2: tuple[int, int],
    ) -> list[tuple[torch.Tensor, torch.Tensor, dict[str, int]]]:
        """Match variable-length tile pairs in one masked LightGlue batch."""
        batch_size = len(source)
        if not (
            batch_size
            == len(target)
            == len(source_keypoints)
            == len(target_keypoints)
        ):
            raise ValueError("batched LightGlue inputs must have equal list lengths")
        if batch_size < 1:
            return []

        source_lengths = torch.tensor(
            [len(values) for values in source], device=source[0].device
        )
        target_lengths = torch.tensor(
            [len(values) for values in target], device=target[0].device
        )
        source_desc = torch.nn.utils.rnn.pad_sequence(source, batch_first=True)
        target_desc = torch.nn.utils.rnn.pad_sequence(target, batch_first=True)
        source_kpts = torch.nn.utils.rnn.pad_sequence(
            source_keypoints, batch_first=True
        )
        target_kpts = torch.nn.utils.rnn.pad_sequence(
            target_keypoints, batch_first=True
        )
        source_valid = (
            torch.arange(source_desc.shape[1], device=source_desc.device)[None]
            < source_lengths[:, None]
        )
        target_valid = (
            torch.arange(target_desc.shape[1], device=target_desc.device)[None]
            < target_lengths[:, None]
        )

        raw = self.matcher
        source_size = source_desc.new_tensor([[hw1[1], hw1[0]]]).repeat(
            batch_size, 1
        )
        target_size = target_desc.new_tensor([[hw2[1], hw2[0]]]).repeat(
            batch_size, 1
        )
        source_kpts = normalize_keypoints(source_kpts, source_size)
        target_kpts = normalize_keypoints(target_kpts, target_size)
        source_desc = raw.input_proj(source_desc.detach().contiguous())
        target_desc = raw.input_proj(target_desc.detach().contiguous())
        source_encoding = raw.posenc(source_kpts)
        target_encoding = raw.posenc(target_kpts)

        results: list[tuple[torch.Tensor, torch.Tensor, dict[str, int]] | None] = [
            None
        ] * batch_size
        active = torch.ones(batch_size, dtype=torch.bool, device=source_desc.device)
        pruning_threshold = raw.pruning_min_kpts(source_desc.device)

        def finish(batch_index: int, layer_index: int) -> None:
            source_indices = torch.where(source_valid[batch_index])[0]
            target_indices = torch.where(target_valid[batch_index])[0]
            assignment, _ = raw.log_assignment[layer_index](
                source_desc[batch_index : batch_index + 1].index_select(
                    1, source_indices
                ),
                target_desc[batch_index : batch_index + 1].index_select(
                    1, target_indices
                ),
            )
            matches0, _, scores0, _ = filter_matches(
                assignment, raw.conf.filter_threshold
            )
            valid_matches = matches0[0] > -1
            indexes = torch.column_stack(
                (
                    source_indices[valid_matches],
                    target_indices[matches0[0, valid_matches]],
                )
            )
            diagnostics = {
                "stop_layer": layer_index + 1,
                "source_input_features": int(source_lengths[batch_index]),
                "target_input_features": int(target_lengths[batch_index]),
                "source_pruned_features": int(
                    source_lengths[batch_index]
                    - source_valid[batch_index].sum()
                ),
                "target_pruned_features": int(
                    target_lengths[batch_index]
                    - target_valid[batch_index].sum()
                ),
            }
            results[batch_index] = (
                scores0[0, valid_matches].reshape(-1, 1),
                indexes,
                diagnostics,
            )

        for layer_index, layer in enumerate(raw.transformers):
            active_indices = torch.where(active)[0]
            if not len(active_indices):
                break
            desc0 = source_desc.index_select(0, active_indices)
            desc1 = target_desc.index_select(0, active_indices)
            encoding0 = source_encoding.index_select(1, active_indices)
            encoding1 = target_encoding.index_select(1, active_indices)
            valid0 = source_valid.index_select(0, active_indices)
            valid1 = target_valid.index_select(0, active_indices)
            self_mask0 = valid0[:, None, :, None] & valid0[:, None, None, :]
            self_mask1 = valid1[:, None, :, None] & valid1[:, None, None, :]
            cross_mask = valid0[:, None, :, None] & valid1[:, None, None, :]
            desc0 = layer.self_attn(desc0, encoding0, self_mask0)
            desc1 = layer.self_attn(desc1, encoding1, self_mask1)
            desc0, desc1 = layer.cross_attn(desc0, desc1, cross_mask)
            source_desc.index_copy_(0, active_indices, desc0)
            target_desc.index_copy_(0, active_indices, desc1)

            if layer_index == raw.conf.n_layers - 1:
                stopped_local = torch.ones_like(active_indices, dtype=torch.bool)
                token0 = token1 = None
            else:
                token0, token1 = raw.token_confidence[layer_index](desc0, desc1)
                if raw.conf.depth_confidence > 0:
                    threshold = raw.confidence_thresholds[layer_index]
                    low_confidence = (
                        ((token0 < threshold) & valid0).sum(1)
                        + ((token1 < threshold) & valid1).sum(1)
                    )
                    point_count = valid0.sum(1) + valid1.sum(1)
                    confident_fraction = 1.0 - low_confidence / point_count
                    stopped_local = (
                        confident_fraction > raw.conf.depth_confidence
                    )
                else:
                    stopped_local = torch.zeros_like(
                        active_indices, dtype=torch.bool
                    )

            for batch_index in active_indices[stopped_local].tolist():
                finish(batch_index, layer_index)
            active[active_indices[stopped_local]] = False
            continuing = ~stopped_local
            if not continuing.any() or layer_index == raw.conf.n_layers - 1:
                continue
            if raw.conf.width_confidence <= 0:
                continue

            continuing_indices = active_indices[continuing]
            for values, tokens, valid, descriptors in (
                (
                    source_valid,
                    token0[continuing],
                    valid0[continuing],
                    desc0[continuing],
                ),
                (
                    target_valid,
                    token1[continuing],
                    valid1[continuing],
                    desc1[continuing],
                ),
            ):
                matchability = raw.log_assignment[
                    layer_index
                ].get_matchability(descriptors)
                keep = raw.get_pruning_mask(tokens, matchability, layer_index)
                for local_index, batch_index in enumerate(continuing_indices):
                    if int(valid[local_index].sum()) > pruning_threshold:
                        values[batch_index] &= keep[local_index]

        if any(result is None for result in results):
            raise RuntimeError("batched LightGlue left an unfinished tile pair")
        return results  # type: ignore[return-value]


class MutualNearestDescriptorMatcher(torch.nn.Module):
    """Exact cosine MNN, optionally with a symmetric L2 ratio test.

    ALIKED emits unit-normalized descriptors, so cosine and L2 have identical
    nearest-neighbour rankings. Scores are cosine similarities so matches from
    overlapping target tiles remain directly comparable.
    """

    uses_laf = False

    def __init__(self, ratio: float | None = None):
        super().__init__()
        if ratio is not None and not 0.0 < ratio < 1.0:
            raise ValueError("symmetric ratio must be between zero and one")
        self.ratio = ratio
        self.matcher_name = "smnn" if ratio is not None else "mnn"

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_laf=None,
        target_laf=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del source_laf, target_laf, kwargs
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("descriptor tensors must have shape (features, dimension)")
        if source.shape[1] != target.shape[1]:
            raise ValueError("source and target descriptor dimensions must agree")
        if not len(source) or not len(target):
            return source.new_empty((0,)), torch.empty(
                (0, 2), dtype=torch.long, device=source.device
            )

        similarity = source @ target.transpose(0, 1)
        source_best = similarity.argmax(dim=1)
        target_best = similarity.argmax(dim=0)
        source_indices = torch.arange(len(source), device=source.device)
        mutual = target_best[source_best].eq(source_indices)

        if self.ratio is not None:
            if len(source) < 2 or len(target) < 2:
                mutual = torch.zeros_like(mutual)
            else:
                source_top2 = similarity.topk(2, dim=1).values
                target_top2 = similarity.topk(2, dim=0).values
                source_distances = torch.sqrt(
                    torch.clamp(2.0 - 2.0 * source_top2, min=0.0)
                )
                target_distances = torch.sqrt(
                    torch.clamp(2.0 - 2.0 * target_top2, min=0.0)
                )
                epsilon = torch.finfo(similarity.dtype).eps
                source_ratio_ok = source_distances[:, 0] <= self.ratio * torch.clamp(
                    source_distances[:, 1], min=epsilon
                )
                target_ratio_ok = target_distances[0] <= self.ratio * torch.clamp(
                    target_distances[1], min=epsilon
                )
                mutual &= source_ratio_ok & target_ratio_ok[source_best]

        selected_source = source_indices[mutual]
        selected_target = source_best[mutual]
        indexes = torch.column_stack((selected_source, selected_target))
        scores = similarity[selected_source, selected_target]
        return scores, indexes


def build_aliked_matcher(
    matcher_name: str,
    device: torch.device,
    smnn_ratio: float = 0.95,
    lightglue_layers: int = 9,
    lightglue_depth_confidence: float = 0.95,
    lightglue_width_confidence: float = 0.99,
    lightglue_filter_threshold: float = 0.1,
    lightglue_adapter: str = "kornia",
    lightglue_compile: bool = False,
):
    """Construct one declared matcher without changing ALIKED extraction."""
    if not 1 <= lightglue_layers <= 9:
        raise ValueError("LightGlue layers must be between one and nine")
    for name, value in (
        ("depth confidence", lightglue_depth_confidence),
        ("width confidence", lightglue_width_confidence),
    ):
        if value != -1.0 and not 0.0 < value < 1.0:
            raise ValueError(f"LightGlue {name} must be -1 or between zero and one")
    if not 0.0 <= lightglue_filter_threshold < 1.0:
        raise ValueError("LightGlue filter threshold must be in [0, 1)")
    if matcher_name == "lightglue":
        params = {
            "n_layers": lightglue_layers,
            "depth_confidence": lightglue_depth_confidence,
            "width_confidence": lightglue_width_confidence,
            "filter_threshold": lightglue_filter_threshold,
        }
        if lightglue_adapter == "kornia":
            matcher = AuditedLightGlueMatcher("aliked", params)
            matcher.matcher_name = "lightglue"
        elif lightglue_adapter == "direct":
            matcher = DirectALIKEDLightGlueMatcher(params)
        else:
            raise ValueError("LightGlue adapter must be 'kornia' or 'direct'")
        matcher.lightglue_layers = lightglue_layers
        if lightglue_compile:
            matcher.matcher.compile(mode="reduce-overhead")
    elif matcher_name == "mnn":
        matcher = MutualNearestDescriptorMatcher()
    elif matcher_name == "smnn":
        matcher = MutualNearestDescriptorMatcher(ratio=smnn_ratio)
    else:
        raise ValueError(f"unsupported ALIKED matcher: {matcher_name}")
    return matcher.to(device).eval()
