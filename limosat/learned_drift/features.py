"""Spatial tile layout and ALIKED feature extraction."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import shapely
import torch
from shapely.geometry.base import BaseGeometry

from .config import ALIKEDConfig, EfficientLoFTRConfig
from .imagery import north_up_patch, projected_coordinates
from .types import FeatureTile, ImageFeatures


@dataclass(frozen=True)
class TileRegion:
    tile_id: int
    row: int
    column: int
    center_xy_m: tuple[float, float]
    core: BaseGeometry


def _anchored_centres(
    lower: float,
    upper: float,
    extent_m: float,
    origin_m: float,
) -> np.ndarray:
    first_edge = math.floor((lower - origin_m) / extent_m) * extent_m + origin_m
    count = max(1, math.ceil((upper - first_edge) / extent_m))
    return first_edge + (np.arange(count, dtype=float) + 0.5) * extent_m


def tile_layout(
    domain: BaseGeometry, config: ALIKEDConfig | EfficientLoFTRConfig
) -> tuple[TileRegion, ...]:
    """Lay out stable, non-overlapping tile cores across a projected domain."""
    minx, miny, maxx, maxy = domain.bounds
    extent_m = config.tile_core_size_m
    centers_x = _anchored_centres(
        minx, maxx, extent_m, config.tile_grid_origin_m
    )
    centers_y = _anchored_centres(
        miny, maxy, extent_m, config.tile_grid_origin_m
    )
    regions = []
    for row, center_y in enumerate(centers_y):
        for column, center_x in enumerate(centers_x):
            core = shapely.box(
                center_x - extent_m / 2.0,
                center_y - extent_m / 2.0,
                center_x + extent_m / 2.0,
                center_y + extent_m / 2.0,
            )
            if domain.intersects(core):
                regions.append(
                    TileRegion(
                        tile_id=len(regions),
                        row=row,
                        column=column,
                        center_xy_m=(center_x, center_y),
                        core=core,
                    )
                )
    return tuple(regions)


def _valid_features(raw, valid: np.ndarray, config: ALIKEDConfig):
    kernel_size = 2 * config.feature_support_radius_px + 1
    support = cv2.erode(
        valid.astype(np.uint8),
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    rounded = torch.round(raw.keypoints).to(torch.long)
    columns = rounded[:, 0].clamp(0, support.shape[1] - 1).cpu().numpy()
    rows = rounded[:, 1].clamp(0, support.shape[0] - 1).cpu().numpy()
    keep = np.flatnonzero(support[rows, columns])
    keep_tensor = torch.as_tensor(keep, device=raw.keypoint_scores.device)
    if len(keep) > config.features_per_tile:
        strongest = torch.topk(
            raw.keypoint_scores[keep_tensor],
            config.features_per_tile,
            sorted=True,
        ).indices
        keep_tensor = keep_tensor[strongest]
    return (
        raw.keypoints[keep_tensor],
        raw.descriptors[keep_tensor],
        raw.keypoint_scores[keep_tensor],
    )


def _cache_path(
    cache_dir: Path,
    image_path: str,
    region: TileRegion,
    config: ALIKEDConfig,
) -> Path:
    identity = "|".join(
        [
            str(Path(image_path).resolve()),
            f"{region.center_xy_m[0]:.3f}",
            f"{region.center_xy_m[1]:.3f}",
            str(config.tile_size_px),
            str(config.tile_margin_px),
            f"{config.pixel_size_m:.6f}",
            str(config.features_per_tile),
            str(config.feature_support_radius_px),
            f"{config.model_name}-threshold{config.detection_threshold:g}",
        ]
    )
    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return cache_dir / f"{Path(image_path).stem}_{key}.pt"


def _extract_region(
    image_path: str,
    region: TileRegion,
    model,
    device: torch.device,
    config: ALIKEDConfig,
    cache_dir: Path | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_path = (
        _cache_path(cache_dir, image_path, region, config)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cached["keypoints"], cached["descriptors"], cached["scores"]

    patch, valid = north_up_patch(
        image_path,
        region.center_xy_m,
        config.tile_size_px,
        config.pixel_size_m,
        config.analysis_epsg,
        config.transform_grid_spacing_px,
    )
    tensor = (
        torch.from_numpy(patch.copy()).to(device=device, dtype=torch.float32)[
            None, None
        ]
        / 255.0
    )
    with torch.inference_mode():
        raw = model(tensor)[0]
    keypoints, descriptors, scores = _valid_features(raw, valid, config)
    margin = config.tile_margin_px
    inside_core = (
        (keypoints[:, 0] >= margin)
        & (keypoints[:, 0] < config.tile_size_px - margin)
        & (keypoints[:, 1] >= margin)
        & (keypoints[:, 1] < config.tile_size_px - margin)
    )
    keypoints = keypoints[inside_core].detach().cpu()
    descriptors = descriptors[inside_core].detach().cpu()
    scores = scores[inside_core].detach().cpu()
    if cache_path is not None:
        torch.save(
            {
                "keypoints": keypoints,
                "descriptors": descriptors,
                "scores": scores,
            },
            cache_path,
        )
    return keypoints, descriptors, scores


def extract_image_features(
    image_path: str,
    domain: BaseGeometry,
    model,
    device: torch.device,
    config: ALIKEDConfig,
    cache_dir: Path | None = None,
) -> ImageFeatures:
    """Detect ALIKED features once in each stable tile core."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for region in tile_layout(domain, config):
        keypoints, descriptors, scores = _extract_region(
            image_path, region, model, device, config, cache_dir
        )
        xy_m = projected_coordinates(
            keypoints,
            region.center_xy_m,
            config.tile_size_px,
            config.pixel_size_m,
        )
        if len(xy_m):
            keep = np.flatnonzero(
                shapely.intersects_xy(domain, xy_m[:, 0], xy_m[:, 1])
            )
            tensor_keep = torch.as_tensor(keep)
            keypoints = keypoints[tensor_keep]
            descriptors = descriptors[tensor_keep]
            scores = scores[tensor_keep]
            xy_m = xy_m[keep]
        tiles.append(
            FeatureTile(
                tile_id=region.tile_id,
                row=region.row,
                column=region.column,
                center_xy_m=region.center_xy_m,
                core=region.core,
                keypoints_px=keypoints,
                descriptors=descriptors,
                scores=scores,
                xy_m=xy_m,
            )
        )
    return ImageFeatures(
        image_path=image_path,
        domain=domain,
        tiles=tuple(tiles),
        analysis_epsg=config.analysis_epsg,
    )


def restrict_features(
    features: ImageFeatures, domain: BaseGeometry
) -> ImageFeatures:
    """Return a pair-specific view of features extracted over a union domain."""
    tiles = []
    for tile in features.tiles:
        if not domain.intersects(tile.core):
            continue
        keep = (
            np.flatnonzero(
                shapely.intersects_xy(domain, tile.xy_m[:, 0], tile.xy_m[:, 1])
            )
            if len(tile)
            else np.empty(0, dtype=int)
        )
        tiles.append(tile.select(keep))
    return ImageFeatures(
        image_path=features.image_path,
        domain=domain,
        tiles=tuple(tiles),
        analysis_epsg=features.analysis_epsg,
    )
