"""Land and point open-water exclusion for matching, composition and seeding.

Land comes from an inland-distance raster on a regular EPSG:3413 grid (``.npz``
with ``inland_km``, ``x0`` west edge, ``y1`` north edge, ``res`` metres), for
example the 1 km OpenStreetMap raster used by the ORB trajectory QC. Cells at
least ``land_minimum_inland_km`` inland are land; the coastal band stays
matchable so coastal and fast ice can be tracked.

Open water uses daily University of Bremen AMSR2 ASI sea-ice concentration in
percent, either GeoTIFF (``<root>/<YYYY>/<mon>/asi-AMSR2-<grid>-<YYYYMMDD>-v5.4.tif``,
values above 100 are land or missing) or single-variable NetCDF
(``<root>/<YYYY>/...v5.4.nc``, NaN where missing), nearest cell in the file's own CRS, with an
optional coarser fallback product for missing days. A position is open water
only where a value exists and is below the threshold; missing values keep it.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np


class LandMask:
    def __init__(self, path: str | Path, minimum_inland_km: float) -> None:
        data = np.load(path)
        self.inland_km = np.asarray(data["inland_km"], dtype=np.float32)
        self.x0 = float(data["x0"])
        self.y1 = float(data["y1"])
        self.res = float(data["res"])
        self.minimum_inland_km = float(minimum_inland_km)

    def is_land(self, x_m, y_m) -> np.ndarray:
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        col = np.floor((x - self.x0) / self.res).astype(np.int64)
        row = np.floor((self.y1 - y) / self.res).astype(np.int64)
        inside = (
            (col >= 0) & (col < self.inland_km.shape[1])
            & (row >= 0) & (row < self.inland_km.shape[0])
        )
        land = np.zeros(x.shape, dtype=bool)
        land[inside] = self.inland_km[row[inside], col[inside]] >= self.minimum_inland_km
        return land

    def tile(self, center_xy_m, pixels: int, pixel_size_m: float) -> np.ndarray:
        """Land mask for a north-up tile as sampled by ``imagery.north_up_patch``."""
        offsets = (np.arange(pixels) - (pixels - 1) / 2.0) * pixel_size_m
        # Columns depend only on x and rows only on y, so index the raster once per axis.
        col = np.floor((center_xy_m[0] + offsets - self.x0) / self.res).astype(np.int64)
        row = np.floor((self.y1 - (center_xy_m[1] - offsets)) / self.res).astype(np.int64)
        col_inside = (col >= 0) & (col < self.inland_km.shape[1])
        row_inside = (row >= 0) & (row < self.inland_km.shape[0])
        land = np.zeros((pixels, pixels), dtype=bool)
        land[np.ix_(row_inside, col_inside)] = (
            self.inland_km[np.ix_(row[row_inside], col[col_inside])] >= self.minimum_inland_km
        )
        return land


class AsiConcentration:
    """Nearest-cell daily AMSR2 ASI concentration in percent (NaN where missing)."""

    def __init__(self, roots: tuple[tuple[str, str], ...]) -> None:
        self.roots = tuple((Path(root), grid) for root, grid in roots)
        self._cache: dict[str, tuple | None] = {}

    def _path(self, day: datetime) -> str | None:
        for root, grid in self.roots:
            name = f"asi-AMSR2-{grid}-{day:%Y%m%d}-v5.4"
            tif = root / f"{day:%Y}" / day.strftime("%b").lower() / f"{name}.tif"
            if tif.is_file():
                return str(tif)
            netcdf = root / f"{day:%Y}" / f"{name}.nc"
            if netcdf.is_file():
                return str(netcdf)
        return None

    def _day(self, day: datetime):
        key = f"{day:%Y%m%d}"
        if key not in self._cache:
            import rasterio
            from pyproj import Transformer

            path = self._path(day)
            self._cache[key] = None
            if path is not None:
                with rasterio.open(path) as source:
                    to_grid = Transformer.from_crs(3413, source.crs, always_xy=True).transform
                    self._cache[key] = (to_grid, source.transform, source.read(1).astype(np.float32))
            if len(self._cache) > 16:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def sample(self, x_m, y_m, day: datetime) -> np.ndarray:
        x = np.asarray(x_m, dtype=np.float64)
        out = np.full(x.shape, np.nan, dtype=np.float32)
        grid = self._day(day)
        if grid is None:
            return out
        to_grid, transform, values = grid
        gx, gy = to_grid(x, np.asarray(y_m, dtype=np.float64))
        col = np.floor((np.asarray(gx) - transform.c) / transform.a).astype(np.int64)
        row = np.floor((np.asarray(gy) - transform.f) / transform.e).astype(np.int64)
        inside = (col >= 0) & (col < values.shape[1]) & (row >= 0) & (row < values.shape[0])
        sampled = values[row[inside], col[inside]]
        sampled[sampled > 100] = np.nan
        out[inside] = sampled
        return out


class SurfaceGate:
    """Exclude land and point open water; ``None`` parts are disabled."""

    def __init__(self, land: LandMask | None, concentration: AsiConcentration | None,
                 open_water_threshold_percent: float) -> None:
        self.land = land
        self.concentration = concentration
        self.threshold = float(open_water_threshold_percent)

    def excluded(self, xy_m, time_utc: datetime) -> np.ndarray:
        xy = np.asarray(xy_m, dtype=np.float64).reshape(-1, 2)
        out = np.zeros(len(xy), dtype=bool)
        if self.land is not None:
            out |= self.land.is_land(xy[:, 0], xy[:, 1])
        if self.concentration is not None:
            sic = self.concentration.sample(xy[:, 0], xy[:, 1], time_utc)
            out |= np.isfinite(sic) & (sic < self.threshold)
        return out


@lru_cache(maxsize=4)
def _gate(land_raster: str, minimum_inland_km: float, sic_roots: tuple, threshold: float) -> SurfaceGate | None:
    land = LandMask(land_raster, minimum_inland_km) if land_raster else None
    concentration = AsiConcentration(sic_roots) if sic_roots else None
    if land is None and concentration is None:
        return None
    return SurfaceGate(land, concentration, threshold)


def surface_gate(config) -> SurfaceGate | None:
    """The configured gate, or ``None`` when neither land nor point SIC is set."""
    surface = config.surface
    roots = tuple(
        (root, grid) for root, grid in (
            (surface.point_sic_root, surface.point_sic_grid),
            (surface.point_sic_fallback_root, surface.point_sic_fallback_grid),
        ) if root
    )
    return _gate(surface.land_raster, surface.land_minimum_inland_km, roots,
                 surface.point_open_water_threshold_percent)


def surface_validity_factory(base_factory, gate: SurfaceGate | None):
    """Wrap an ``image -> validity`` factory so land and open-water endpoints fail."""
    if gate is None:
        return base_factory

    def factory(image):
        base = base_factory(image)
        return lambda xy_m: np.asarray(base(xy_m), dtype=bool) & ~gate.excluded(xy_m, image.time_utc)

    return factory


def seed_filter_factory(gate: SurfaceGate | None):
    """``image -> (xy -> keep mask)`` for new trajectory seeds, or ``None``."""
    if gate is None:
        return None
    return lambda image: (lambda xy_m: ~gate.excluded(xy_m, image.time_utc))
