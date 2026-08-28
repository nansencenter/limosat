"""North-up sampling of the standard LiMOSAT VAE imagery."""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np
import shapely
from nansat import NSR
from osgeo import gdal
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from limosat.image import Image

gdal.UseExceptions()


@lru_cache(maxsize=4)
def read_scene(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Read one VAE image and its optional invalid-pixel mask."""
    dataset = gdal.Open(path)
    if dataset is None:
        raise FileNotFoundError(path)
    image = np.asarray(dataset.GetRasterBand(1).ReadAsArray())
    if image.dtype != np.uint8:
        finite = np.isfinite(image)
        if not finite.any():
            raise ValueError(f"image has no finite pixels: {path}")
        low, high = np.nanpercentile(image[finite], [1, 99])
        scale = max(float(high - low), 1.0e-9)
        image = np.clip((image - low) * 255.0 / scale, 0, 255).astype(np.uint8)
    else:
        image = image.copy()
    mask = (
        np.asarray(dataset.GetRasterBand(2).ReadAsArray(), dtype=np.uint8)
        if dataset.RasterCount >= 2
        else None
    )
    if mask is not None:
        image[mask >= 2] = 0
    return image, mask


@lru_cache(maxsize=4)
def image_object(path: str, analysis_epsg: int) -> Image:
    return Image(path, srs=NSR(analysis_epsg))


def projected_footprint(path: str, analysis_epsg: int) -> BaseGeometry:
    """Return the image footprint in the metre-based analysis CRS."""
    geometry = shapely.from_geojson(
        image_object(path, analysis_epsg).get_border_geojson()
    )
    projector = Transformer.from_crs(
        4326, analysis_epsg, always_xy=True
    ).transform
    return transform(projector, geometry).buffer(0)


def _interpolate_transform_grid(coarse: np.ndarray, pixels: int) -> np.ndarray:
    positions = np.linspace(0.0, pixels - 1.0, coarse.shape[0])
    full_positions = np.arange(pixels, dtype=float)
    horizontal = np.empty((coarse.shape[0], pixels), dtype=np.float64)
    for row in range(coarse.shape[0]):
        horizontal[row] = np.interp(full_positions, positions, coarse[row])
    full = np.empty((pixels, pixels), dtype=np.float32)
    for column in range(pixels):
        full[:, column] = np.interp(
            full_positions, positions, horizontal[:, column]
        )
    return full


def north_up_patch(
    path: str,
    center_xy_m: tuple[float, float],
    pixels: int,
    pixel_size_m: float,
    analysis_epsg: int,
    transform_grid_spacing_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a projected north-up grayscale tile and validity mask."""
    image, mask = read_scene(path)
    coarse_pixels = math.ceil((pixels - 1) / transform_grid_spacing_px) + 1
    offsets = np.linspace(
        -(pixels - 1) / 2.0 * pixel_size_m,
        (pixels - 1) / 2.0 * pixel_size_m,
        coarse_pixels,
        dtype=np.float64,
    )
    projected_x, projected_y_offset = np.meshgrid(center_xy_m[0] + offsets, offsets)
    projected_y = center_xy_m[1] - projected_y_offset
    columns, rows = image_object(path, analysis_epsg).transform_points(
        projected_x.ravel(),
        projected_y.ravel(),
        DstToSrc=1,
        dst_srs=NSR(analysis_epsg),
    )
    coarse_x = np.asarray(columns, dtype=np.float32).reshape(
        coarse_pixels, coarse_pixels
    )
    coarse_y = np.asarray(rows, dtype=np.float32).reshape(
        coarse_pixels, coarse_pixels
    )
    sample_x = _interpolate_transform_grid(coarse_x, pixels)
    sample_y = _interpolate_transform_grid(coarse_y, pixels)
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
    in_bounds = (
        finite
        & (sample_x >= 0)
        & (sample_x <= image.shape[1] - 1)
        & (sample_y >= 0)
        & (sample_y <= image.shape[0] - 1)
    )
    if mask is None:
        valid = in_bounds
    else:
        sampled_mask = cv2.remap(
            mask,
            safe_x,
            safe_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=2,
        )
        valid = in_bounds & (sampled_mask < 2)
    patch = patch.astype(np.uint8, copy=False)
    patch[~valid] = 0
    return patch, valid


def projected_coordinates(
    keypoints_px,
    center_xy_m: tuple[float, float],
    pixels: int,
    pixel_size_m: float,
) -> np.ndarray:
    """Convert north-up tile pixels to projected x/y metres."""
    values = keypoints_px.detach().cpu().numpy()
    center_px = (pixels - 1) / 2.0
    return np.column_stack(
        (
            center_xy_m[0] + (values[:, 0] - center_px) * pixel_size_m,
            center_xy_m[1] - (values[:, 1] - center_px) * pixel_size_m,
        )
    )
