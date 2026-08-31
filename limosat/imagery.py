"""Rasterio-based sampling of north-up EPSG:3413 image tiles."""

from __future__ import annotations

import atexit
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import AffineTransformer, GCPTransformer
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform


@lru_cache(maxsize=4)
def read_scene(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read one grayscale image and its optional LiMOSAT invalid-pixel band."""
    with rasterio.open(path) as dataset:
        image = dataset.read(1)
        mask = dataset.read(2).astype(np.uint8) if dataset.count >= 2 else None
    if image.dtype != np.uint8:
        finite = np.isfinite(image)
        if not finite.any():
            raise ValueError(f"image has no finite pixels: {path}")
        low, high = np.nanpercentile(image[finite], [1, 99])
        image = np.clip(
            (image - low) * 255.0 / max(float(high - low), 1.0e-9), 0, 255
        ).astype(np.uint8)
    else:
        image = image.copy()
    if mask is not None:
        image[mask >= 2] = 0
    return image, mask


@dataclass
class _SceneTransform:
    width: int
    height: int
    transformer: AffineTransformer | GCPTransformer
    analysis_to_native: Transformer
    native_to_analysis: Transformer

    def analysis_to_pixels(
        self, x_m: np.ndarray, y_m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        native_x, native_y = self.analysis_to_native.transform(x_m, y_m)
        rows, columns = self.transformer.rowcol(
            native_x, native_y, op=lambda value: value
        )
        return np.asarray(columns, dtype=np.float64), np.asarray(rows, dtype=np.float64)

    def pixels_to_analysis(
        self, columns: np.ndarray, rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        native_x, native_y = self.transformer.xy(rows, columns, offset="ul")
        x_m, y_m = self.native_to_analysis.transform(native_x, native_y)
        return np.asarray(x_m, dtype=np.float64), np.asarray(y_m, dtype=np.float64)


@lru_cache(maxsize=8)
def scene_transform(path: str | Path, analysis_epsg: int = 3413) -> _SceneTransform:
    """Return cached affine or thin-plate-spline pixel transforms."""
    with rasterio.open(path) as dataset:
        gcps, gcp_crs = dataset.gcps
        if gcps:
            if gcp_crs is None:
                raise ValueError(f"ground-control points have no CRS: {path}")
            native_crs = gcp_crs
            pixel_transformer = GCPTransformer(gcps, tps=True)
        else:
            if dataset.crs is None:
                raise ValueError(f"image has neither CRS nor GCP CRS: {path}")
            native_crs = dataset.crs
            pixel_transformer = AffineTransformer(dataset.transform)
        return _SceneTransform(
            dataset.width,
            dataset.height,
            pixel_transformer,
            Transformer.from_crs(analysis_epsg, native_crs, always_xy=True),
            Transformer.from_crs(native_crs, analysis_epsg, always_xy=True),
        )


atexit.register(scene_transform.cache_clear)


def projected_footprint(path: str | Path, analysis_epsg: int = 3413) -> BaseGeometry:
    """Trace an image border in the analysis CRS."""
    scene = scene_transform(path, analysis_epsg)
    x = _border_pixels(scene.width)
    y = _border_pixels(scene.height)
    columns = np.asarray(
        x + [scene.width] * len(y) + x[::-1] + [0] * len(y), dtype=np.float64
    )
    rows = np.asarray(
        [0] * len(x) + y + [scene.height] * len(x) + y[::-1], dtype=np.float64
    )
    projected_x, projected_y = scene.pixels_to_analysis(columns, rows)
    return Polygon(np.column_stack((projected_x, projected_y))).buffer(0)


def north_up_patch(
    path: str | Path,
    center_xy_m: tuple[float, float],
    pixels: int,
    pixel_size_m: float,
    analysis_epsg: int = 3413,
    transform_grid_spacing_px: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample one projected north-up grayscale tile and validity mask."""
    image, mask = read_scene(path)
    coarse_pixels = math.ceil((pixels - 1) / transform_grid_spacing_px) + 1
    offsets = np.linspace(
        -(pixels - 1) / 2.0 * pixel_size_m,
        (pixels - 1) / 2.0 * pixel_size_m,
        coarse_pixels,
        dtype=np.float64,
    )
    projected_x, y_offset = np.meshgrid(center_xy_m[0] + offsets, offsets)
    projected_y = center_xy_m[1] - y_offset
    columns, rows = scene_transform(path, analysis_epsg).analysis_to_pixels(
        projected_x.ravel(), projected_y.ravel()
    )
    sample_x = _interpolate_grid(columns.reshape(coarse_pixels, coarse_pixels), pixels)
    sample_y = _interpolate_grid(rows.reshape(coarse_pixels, coarse_pixels), pixels)
    finite = np.isfinite(sample_x) & np.isfinite(sample_y)
    safe_x = np.where(finite, sample_x, -1).astype(np.float32)
    safe_y = np.where(finite, sample_y, -1).astype(np.float32)
    patch = cv2.remap(
        image,
        safe_x,
        safe_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = (
        finite
        & (sample_x >= 0)
        & (sample_x <= image.shape[1] - 1)
        & (sample_y >= 0)
        & (sample_y <= image.shape[0] - 1)
    )
    if mask is not None:
        sampled_mask = cv2.remap(
            mask,
            safe_x,
            safe_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=2,
        )
        valid &= sampled_mask < 2
    patch = patch.astype(np.uint8, copy=False)
    patch[~valid] = 0
    return patch, valid


def projected_coordinates(
    points_px: np.ndarray,
    center_xy_m: tuple[float, float],
    pixels: int,
    pixel_size_m: float,
) -> np.ndarray:
    """Convert north-up tile pixels to float64 projected metre coordinates."""
    values = np.asarray(points_px, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points_px must have shape (n, 2)")
    center_px = (pixels - 1) / 2.0
    return np.column_stack(
        (
            center_xy_m[0] + (values[:, 0] - center_px) * pixel_size_m,
            center_xy_m[1] - (values[:, 1] - center_px) * pixel_size_m,
        )
    ).astype(np.float64)


def _border_pixels(size: int, points: int = 10) -> list[int]:
    step = max(1, int(size / points))
    return list(range(0, size, step))[:points] + [size]


def _interpolate_grid(coarse: np.ndarray, pixels: int) -> np.ndarray:
    positions = np.linspace(0.0, pixels - 1.0, coarse.shape[0])
    full_positions = np.arange(pixels, dtype=float)
    horizontal = np.vstack(
        [np.interp(full_positions, positions, row) for row in coarse]
    )
    return np.column_stack(
        [
            np.interp(full_positions, positions, horizontal[:, column])
            for column in range(pixels)
        ]
    ).astype(np.float32)
