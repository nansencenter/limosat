"""North-up sampling of the standard LiMOSAT VAE imagery with Rasterio."""

from __future__ import annotations

import atexit
import math
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import rasterio
from rasterio.transform import AffineTransformer, GCPTransformer
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform


@lru_cache(maxsize=4)
def read_scene(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Read one VAE image and its optional invalid-pixel mask."""
    with rasterio.open(path) as dataset:
        image = dataset.read(1)
        mask = dataset.read(2).astype(np.uint8) if dataset.count >= 2 else None
    if image.dtype != np.uint8:
        finite = np.isfinite(image)
        if not finite.any():
            raise ValueError(f"image has no finite pixels: {path}")
        low, high = np.nanpercentile(image[finite], [1, 99])
        scale = max(float(high - low), 1.0e-9)
        image = np.clip((image - low) * 255.0 / scale, 0, 255).astype(np.uint8)
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
    native_to_lonlat: Transformer

    def analysis_to_pixels(
        self, x_m: np.ndarray, y_m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        native_x, native_y = self.analysis_to_native.transform(x_m, y_m)
        rows, columns = self.transformer.rowcol(
            native_x, native_y, op=lambda value: value
        )
        return (
            np.asarray(columns, dtype=np.float64),
            np.asarray(rows, dtype=np.float64),
        )

    def pixels_to_lonlat(
        self, columns: np.ndarray, rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        native_x, native_y = self.transformer.xy(rows, columns, offset="ul")
        longitude, latitude = self.native_to_lonlat.transform(native_x, native_y)
        return (
            np.asarray(longitude, dtype=np.float64),
            np.asarray(latitude, dtype=np.float64),
        )


@lru_cache(maxsize=4)
def image_object(path: str, analysis_epsg: int) -> _SceneTransform:
    """Return the cached native/pixel transform for one scene.

    The standard VAE GeoTIFFs use ground-control points rather than a single
    affine geotransform. GDAL's thin-plate-spline transformer is selected for
    those files, matching the previous Nansat sampling path.
    """
    with rasterio.open(path) as dataset:
        gcps, gcp_crs = dataset.gcps
        if gcps:
            if gcp_crs is None:
                raise ValueError(f"ground-control points have no CRS: {path}")
            native_crs = gcp_crs
            pixel_transformer = GCPTransformer(gcps, tps=True)
        else:
            if dataset.crs is None:
                raise ValueError(
                    f"image has neither ground-control points nor a CRS: {path}"
                )
            native_crs = dataset.crs
            pixel_transformer = AffineTransformer(dataset.transform)
        return _SceneTransform(
            width=dataset.width,
            height=dataset.height,
            transformer=pixel_transformer,
            analysis_to_native=Transformer.from_crs(
                analysis_epsg, native_crs, always_xy=True
            ),
            native_to_lonlat=Transformer.from_crs(
                native_crs, 4326, always_xy=True
            ),
        )


atexit.register(image_object.cache_clear)


def _border_pixels(size: int, points: int = 10) -> list[int]:
    step = max(1, int(size / points))
    return list(range(0, size, step))[:points] + [size]


def projected_footprint(path: str, analysis_epsg: int) -> BaseGeometry:
    """Return the image footprint in the metre-based analysis CRS."""
    scene = image_object(path, analysis_epsg)
    x_vector = _border_pixels(scene.width)
    y_vector = _border_pixels(scene.height)
    columns = np.asarray(
        x_vector
        + [scene.width] * len(y_vector)
        + x_vector[::-1]
        + [0] * len(y_vector),
        dtype=np.float64,
    )
    rows = np.asarray(
        [0] * len(x_vector)
        + y_vector
        + [scene.height] * len(x_vector)
        + y_vector[::-1],
        dtype=np.float64,
    )
    longitude, latitude = scene.pixels_to_lonlat(columns, rows)
    # Preserve Nansat get_border()'s established four-decimal-degree footprint.
    geometry = Polygon(np.column_stack((longitude.round(4), latitude.round(4))))
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
    columns, rows = image_object(path, analysis_epsg).analysis_to_pixels(
        projected_x.ravel(), projected_y.ravel()
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
