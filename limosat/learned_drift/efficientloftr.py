"""EfficientLoFTR loading, inference, and projected-coordinate filters."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import torch


def matcher_inputs(
    source: np.ndarray,
    target: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert two uint8 SAR tiles to the official model input contract."""

    def tensor(image: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(image.copy()).to(device=device, dtype=torch.float32)[
                None, None
            ]
            / 255.0
        )

    return {"image0": tensor(source), "image1": tensor(target)}


def run_optimized_matcher(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the official optimized model and return pixel matches and scores."""
    model(inputs)
    source_px = inputs["mkpts0_f"].detach().cpu().numpy()
    target_px = inputs["mkpts1_f"].detach().cpu().numpy()
    score = inputs["mconf"].detach().cpu().numpy()
    if len(score):
        lower = min(20.0, float(score.min()))
        upper = max(30.0, float(score.max()))
        score = (score - lower) / (upper - lower)
    return source_px, target_px, score


def synchronize(device: torch.device) -> None:
    """Wait for asynchronous accelerator work before recording elapsed time."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def load_optimized_model(
    repo: Path,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load the authors' optimized outdoor model with narrow compatibility shims."""
    from kornia.geometry import create_meshgrid

    grid_module = ModuleType("kornia.utils.grid")
    grid_module.create_meshgrid = create_meshgrid
    sys.modules.setdefault("kornia.utils.grid", grid_module)
    if "pytorch_lightning.utilities" not in sys.modules:
        lightning_module = ModuleType("pytorch_lightning")
        utilities_module = ModuleType("pytorch_lightning.utilities")

        class RankZeroOnly:
            rank = 0

            def __call__(self, function):
                return function

        utilities_module.rank_zero_only = RankZeroOnly()
        lightning_module.utilities = utilities_module
        sys.modules.setdefault("pytorch_lightning", lightning_module)
        sys.modules.setdefault("pytorch_lightning.utilities", utilities_module)

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


def source_core_mask(
    points_px: np.ndarray, tile_size_px: int, margin_px: int
) -> np.ndarray:
    """Assign each source match to the non-overlapping core of one tile."""
    points_px = np.asarray(points_px, dtype=float)
    if points_px.ndim != 2 or points_px.shape[1] != 2:
        raise ValueError("pixel endpoints must have shape (n, 2)")
    if tile_size_px <= 0:
        raise ValueError("tile size must be positive")
    if margin_px < 0 or margin_px * 2 >= tile_size_px:
        raise ValueError("tile margin leaves no tile core")
    return (
        (points_px[:, 0] >= margin_px)
        & (points_px[:, 0] < tile_size_px - margin_px)
        & (points_px[:, 1] >= margin_px)
        & (points_px[:, 1] < tile_size_px - margin_px)
    )


def valid_support(valid: np.ndarray, radius_px: int) -> np.ndarray:
    """Erode an image-validity mask by a documented endpoint radius."""
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("validity mask must be two-dimensional")
    if radius_px < 0:
        raise ValueError("support radius cannot be negative")
    if radius_px == 0:
        return valid.copy()
    kernel = np.ones((2 * radius_px + 1, 2 * radius_px + 1), dtype=np.uint8)
    return cv2.erode(
        valid.astype(np.uint8),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def valid_endpoints(points_px: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return whether rounded pixel endpoints lie in a validity mask."""
    points_px = np.asarray(points_px, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if points_px.ndim != 2 or points_px.shape[1] != 2:
        raise ValueError("pixel endpoints must have shape (n, 2)")
    if valid.ndim != 2:
        raise ValueError("validity mask must be two-dimensional")
    rounded = np.rint(points_px).astype(int)
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < valid.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < valid.shape[0])
    )
    accepted = np.zeros(len(points_px), dtype=bool)
    accepted[inside] = valid[rounded[inside, 1], rounded[inside, 0]]
    return accepted


def speed_limit_mask(
    source_xy_m: np.ndarray,
    target_xy_m: np.ndarray,
    elapsed_hours: float,
    maximum_speed_m_per_day: float,
) -> np.ndarray:
    """Apply a radial sea-ice speed limit in projected metre coordinates."""
    source_xy_m = np.asarray(source_xy_m, dtype=float)
    target_xy_m = np.asarray(target_xy_m, dtype=float)
    if source_xy_m.shape != target_xy_m.shape:
        raise ValueError("source and target coordinates must have the same shape")
    if source_xy_m.ndim != 2 or source_xy_m.shape[1] != 2:
        raise ValueError("coordinates must have shape (n, 2)")
    if not np.isfinite(elapsed_hours) or elapsed_hours <= 0:
        raise ValueError("elapsed hours must be finite and positive")
    if not np.isfinite(maximum_speed_m_per_day) or maximum_speed_m_per_day <= 0:
        raise ValueError("maximum speed must be finite and positive")
    maximum_displacement_m = maximum_speed_m_per_day * elapsed_hours / 24.0
    finite = np.isfinite(source_xy_m).all(axis=1) & np.isfinite(target_xy_m).all(axis=1)
    displacement_m = target_xy_m - source_xy_m
    return finite & (np.linalg.norm(displacement_m, axis=1) <= maximum_displacement_m)
