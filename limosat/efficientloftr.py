"""Official EfficientLoFTR loading and inference contract."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

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
            Path(config.repository), Path(config.checkpoint), self.device
        )

    def match(
        self, source: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return source pixels, target pixels, and normalized confidence."""
        import torch

        def tensor(image: np.ndarray):
            return (
                torch.from_numpy(image.copy())
                .to(device=self.device, dtype=torch.float32)[None, None]
                / 255.0
            )

        inputs = {"image0": tensor(source), "image1": tensor(target)}
        with torch.inference_mode():
            self.model(inputs)
            _synchronize(self.device)
        source_px = inputs["mkpts0_f"].detach().cpu().numpy()
        target_px = inputs["mkpts1_f"].detach().cpu().numpy()
        score = inputs["mconf"].detach().cpu().numpy().astype(np.float64)
        if len(score):
            lower = min(20.0, float(score.min()))
            upper = max(30.0, float(score.max()))
            score = (score - lower) / (upper - lower)
        return source_px, target_px, score


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


def _load_optimized_model(repo: Path, checkpoint: Path, device):
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
    model = LoFTR(config=deepcopy(opt_default_cfg))
    model.load_state_dict(state)
    return reparameter(model).eval().to(device)
