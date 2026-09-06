"""Official EfficientLoFTR loading and inference contract."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import cv2
import numpy as np

from .config import MatcherConfig


class EfficientLoFTR:
    """One EfficientLoFTR model; no matcher registry or alternate implementation."""

    def __init__(self, config: MatcherConfig) -> None:
        if not config.repository or not config.checkpoint:
            raise ValueError("EfficientLoFTR repository and checkpoint are required")
        import torch

        self.config = config
        self.device = torch.device(config.device)
        self.model = _load_optimized_model(
            Path(config.repository),
            Path(config.checkpoint),
            self.device,
            graph_mode=config.prefix_cuda_graph and self.device.type == "cuda",
        )
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = True
        use_graph = config.prefix_cuda_graph and self.device.type == "cuda"
        self.execution_mode = "prefix_cuda_graph" if use_graph else "eager"
        self._runner = (
            _PrefixGraphRunner(
                self.model,
                config.tile_batch_size,
                config.tile_size_px,
                config.cuda_graph_warmup_batches,
                self.device,
            )
            if use_graph
            else _EagerRunner(self.model)
        )

    def match(
        self, source: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return source pixels, target pixels, and normalized confidence."""
        return self.match_batch((source,), (target,))[0]

    def match_batch(
        self,
        sources: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Match up to the configured number of independent tile pairs."""
        import torch

        if len(sources) != len(targets):
            raise ValueError("source and target tile counts differ")
        if not 1 <= len(sources) <= self.config.tile_batch_size:
            raise ValueError("tile batch is empty or exceeds tile_batch_size")
        valid_items = len(sources)
        padded_sources = list(sources)
        padded_targets = list(targets)
        while len(padded_sources) < self.config.tile_batch_size:
            padded_sources.append(padded_sources[-1])
            padded_targets.append(padded_targets[-1])

        def tensor(images: Sequence[np.ndarray]):
            values = np.stack(
                [_tile_image(image, self.config.tile_size_px) for image in images]
            )
            return (
                torch.from_numpy(values)
                .to(device=self.device, dtype=torch.float32)[:, None]
                / 255.0
            )

        inputs = self._runner(tensor(padded_sources), tensor(padded_targets))
        _synchronize(self.device)
        return _split_matches(inputs, valid_items)


class _EagerRunner:
    def __init__(self, model: Any) -> None:
        self.model = model

    def __call__(self, image0, image1) -> dict[str, Any]:
        import torch

        inputs = {"image0": image0, "image1": image1}
        with torch.inference_mode():
            self.model(inputs)
        return inputs


class _PrefixGraphRunner:
    """Replay the fixed CNN/attention prefix; run variable matching eagerly."""

    def __init__(
        self,
        model: Any,
        batch_size: int,
        tile_size_px: int,
        warmup_batches: int,
        device: Any,
    ) -> None:
        import torch

        class StaticPrefix(torch.nn.Module):
            def __init__(self, matcher: Any) -> None:
                super().__init__()
                self.backbone = matcher.backbone
                self.coarse_attention = matcher.loftr_coarse

            def forward(self, image0, image1):
                batch = image0.shape[0]
                features = self.backbone(torch.cat((image0, image1), dim=0))
                coarse0, coarse1 = features["feats_c"].split(batch)
                coarse0, coarse1 = self.coarse_attention(coarse0, coarse1)
                return coarse0, coarse1, features["feats_x2"], features["feats_x1"]

        self.model = model
        self.device = device
        self.image_shape = (tile_size_px, tile_size_px)
        self.prefix = StaticPrefix(model).eval()
        shape = (batch_size, 1, tile_size_px, tile_size_px)
        with torch.cuda.device(device):
            self.static0 = torch.zeros(shape, device=device, dtype=torch.float32)
            self.static1 = torch.zeros_like(self.static0)
            warmup_stream = torch.cuda.Stream(device=device)
            warmup_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(warmup_stream):
                for _ in range(warmup_batches):
                    with torch.inference_mode():
                        self.prefix(self.static0, self.static1)
            torch.cuda.current_stream(device).wait_stream(warmup_stream)
            torch.cuda.synchronize(device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(self.graph):
                    self.encoded = self.prefix(self.static0, self.static1)

    def __call__(self, image0, image1) -> dict[str, Any]:
        import torch

        with torch.cuda.device(self.device), torch.inference_mode():
            self.static0.copy_(image0)
            self.static1.copy_(image1)
            self.graph.replay()
            return _run_dynamic_suffix(self.model, self.encoded, self.image_shape)


def _run_dynamic_suffix(model: Any, encoded, image_shape) -> dict[str, Any]:
    """Run data-dependent MNN selection and fine matching outside the graph."""
    coarse0, coarse1, features_x2, features_x1 = encoded
    data: dict[str, Any] = {
        "bs": coarse0.shape[0],
        "hw0_i": image_shape,
        "hw1_i": image_shape,
        "hw0_c": coarse0.shape[2:],
        "hw1_c": coarse1.shape[2:],
        "feats_x2": features_x2,
        "feats_x1": features_x1,
    }
    multiplier = model.config["resolution"][0] // model.config["resolution"][1]
    data["hw0_f"] = [coarse0.shape[2] * multiplier, coarse0.shape[3] * multiplier]
    data["hw1_f"] = [coarse1.shape[2] * multiplier, coarse1.shape[3] * multiplier]
    coarse0 = coarse0.flatten(2).transpose(1, 2)
    coarse1 = coarse1.flatten(2).transpose(1, 2)
    model.coarse_matching(coarse0, coarse1, data)
    coarse0 = coarse0 / coarse0.shape[-1] ** 0.5
    coarse1 = coarse1 / coarse1.shape[-1] ** 0.5
    fine0, fine1 = model.fine_preprocess(coarse0, coarse1, data)
    model.fine_matching(fine0, fine1, data)
    return data


def _tile_image(image: np.ndarray, tile_size_px: int) -> np.ndarray:
    values = np.asarray(image)
    if values.shape != (tile_size_px, tile_size_px):
        raise ValueError(
            f"matcher tile must have shape {(tile_size_px, tile_size_px)}"
        )
    return np.ascontiguousarray(values)


def _split_matches(
    inputs: dict[str, Any], valid_items: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    batch_ids = inputs["m_bids"].detach().cpu().numpy()
    source_px = inputs["mkpts0_f"].detach().cpu().numpy()
    target_px = inputs["mkpts1_f"].detach().cpu().numpy()
    score = inputs["mconf"].detach().cpu().numpy().astype(np.float64)
    if not (len(batch_ids) == len(source_px) == len(target_px) == len(score)):
        raise ValueError("EfficientLoFTR returned inconsistent match arrays")
    output = []
    for batch_id in range(valid_items):
        selected = batch_ids == batch_id
        output.append(
            (
                source_px[selected],
                target_px[selected],
                _normalize_score(score[selected]),
            )
        )
    return output


def _normalize_score(score: np.ndarray) -> np.ndarray:
    if not len(score):
        return score
    lower = min(20.0, float(score.min()))
    upper = max(30.0, float(score.max()))
    return (score - lower) / (upper - lower)


def source_core_mask(
    points_px: np.ndarray, tile_size_px: int, margin_px: int
) -> np.ndarray:
    points = _points(points_px)
    if margin_px < 0 or margin_px * 2 >= tile_size_px:
        raise ValueError("tile margin leaves no core")
    return (
        (points[:, 0] >= margin_px)
        & (points[:, 0] < tile_size_px - margin_px)
        & (points[:, 1] >= margin_px)
        & (points[:, 1] < tile_size_px - margin_px)
    )


def valid_support(valid: np.ndarray, radius_px: int) -> np.ndarray:
    mask = np.asarray(valid, dtype=bool)
    if mask.ndim != 2 or radius_px < 0:
        raise ValueError("valid support requires a 2-D mask and non-negative radius")
    if radius_px == 0:
        return mask.copy()
    kernel = np.ones((2 * radius_px + 1, 2 * radius_px + 1), dtype=np.uint8)
    return cv2.erode(
        mask.astype(np.uint8),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def valid_endpoints(points_px: np.ndarray, valid: np.ndarray) -> np.ndarray:
    points = _points(points_px)
    mask = np.asarray(valid, dtype=bool)
    rounded = np.rint(points).astype(int)
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < mask.shape[0])
    )
    accepted = np.zeros(len(points), dtype=bool)
    accepted[inside] = mask[rounded[inside, 1], rounded[inside, 0]]
    return accepted


def speed_limit_mask(
    source_xy_m: np.ndarray,
    target_xy_m: np.ndarray,
    elapsed_seconds: float,
    maximum_speed_m_per_day: float,
) -> np.ndarray:
    source = _points(source_xy_m)
    target = _points(target_xy_m)
    if source.shape != target.shape or elapsed_seconds <= 0:
        raise ValueError("speed filter inputs are inconsistent")
    limit = maximum_speed_m_per_day * elapsed_seconds / 86_400.0
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    return finite & (np.linalg.norm(target - source, axis=1) <= limit)


def _points(values: np.ndarray) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("coordinates must have shape (n, 2)")
    return points


def _synchronize(device) -> None:
    import torch

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_optimized_model(
    repo: Path, checkpoint: Path, device, *, graph_mode: bool = False
):
    import torch
    from kornia.geometry import create_meshgrid

    if not repo.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("EfficientLoFTR repository or checkpoint is missing")
    grid_module = ModuleType("kornia.utils.grid")
    grid_module.create_meshgrid = create_meshgrid
    sys.modules.setdefault("kornia.utils.grid", grid_module)
    if "pytorch_lightning.utilities" not in sys.modules:
        lightning = ModuleType("pytorch_lightning")
        utilities = ModuleType("pytorch_lightning.utilities")

        class RankZeroOnly:
            rank = 0

            def __call__(self, function):
                return function

        utilities.rank_zero_only = RankZeroOnly()
        lightning.utilities = utilities
        sys.modules.setdefault("pytorch_lightning", lightning)
        sys.modules.setdefault("pytorch_lightning.utilities", utilities)
    sys.path.insert(0, str(repo))
    from src.loftr import LoFTR, opt_default_cfg, reparameter

    class ModelCheckpoint:
        pass

    safe_global = (
        ModelCheckpoint,
        "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
    )
    with torch.serialization.safe_globals([safe_global]):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)[
            "state_dict"
        ]
    model_config = deepcopy(opt_default_cfg)
    if graph_mode:
        # Avoid a tensor-to-Python finite check inside CUDA Graph capture.
        model_config["replace_nan"] = False
    model = LoFTR(config=model_config)
    model.load_state_dict(state)
    return reparameter(model).eval().to(device)
