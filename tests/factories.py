import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box


class ImageStub:
    def __init__(self, angle: float = 0.0, date: pd.Timestamp | None = None, orbit_num: int = 1,
                 width: int = 64, height: int = 64, band_name: str = 's0_HV'):
        self._w = width
        self._h = height
        self._band_name = band_name
        self._band = (np.indices((height, width))[0] * 0 + 127).astype(np.uint8)
        self.srs = 'EPSG:3413'
        self.angle = float(angle)
        self.date = pd.Timestamp('2025-01-01 00:00:00') if date is None else pd.Timestamp(date)
        self.orbit_num = int(orbit_num)
        # Projected coords: treat pixel col/row as x/y directly
        self._poly = box(0, 0, width - 1, height - 1)

    @property
    def poly(self):
        return self._poly

    def __getitem__(self, band):
        # Only one band; ignore band name and return array
        return self._band

    def transform_points(self, x, y, DstToSrc=0, dst_srs=None):
        x = np.asarray(x)
        y = np.asarray(y)
        if DstToSrc == 1:
            # Treat inputs as projected XY; return cols, rows clipped to bounds
            cols = np.clip(x, 0, self._w - 1).astype(float)
            rows = np.clip(y, 0, self._h - 1).astype(float)
            return cols, rows
        else:
            # Identity mapping back to projected
            return x.astype(float), y.astype(float)


class MatcherStub:
    def __init__(self):
        self.spatial_distance_max = 1000

    def match_with_grid(self, points_poly, points_grid):
        n = min(len(points_poly), len(points_grid))
        if n == 0:
            # Return empty consistent outputs
            return (points_poly.iloc[:0].copy(), points_grid.iloc[:0].copy(), np.zeros((0, 2)))
        idx = np.arange(n)
        return (points_poly.iloc[idx].copy(), points_grid.iloc[idx].copy(), np.zeros((n, 2)))

    def match_with_lowe_ratio(self, *args, **kwargs):
        # Passthrough
        return []


def make_templates(tids, hs):
    size = 2 * hs + 1
    data = np.zeros((len(tids), size, size), dtype=np.uint8)
    c = hs
    for i in range(len(tids)):
        data[i, c - 1:c + 2, c - 1:c + 2] = 255
    da = xr.DataArray(
        data,
        dims=("trajectory_id", "height", "width"),
        coords={"trajectory_id": np.asarray(tids, dtype=np.int64),
                "height": np.arange(size),
                "width": np.arange(size)},
        name="template_data",
    )
    return da


def make_keypoints(n, image_id, t0, step_s=0, base_xy=(10, 10), du=(1, 0)):
    t0 = pd.Timestamp(t0)
    xs = base_xy[0] + np.arange(n) * du[0]
    ys = base_xy[1] + np.arange(n) * du[1]
    times = [t0 + pd.Timedelta(seconds=i * step_s) for i in range(n)]
    gdf = gpd.GeoDataFrame(
        {
            'image_id': np.full(n, image_id, dtype=np.int32),
            'is_last': np.ones(n, dtype=np.int32),
            'trajectory_id': np.arange(n, dtype=np.int64),
            'geometry': gpd.points_from_xy(xs, ys),
            'descriptors': [np.zeros((32,), dtype=np.uint8) for _ in range(n)],
            'angle': np.zeros(n),
            'corr': np.zeros(n),
            'time': times,
            'interpolated': np.zeros(n, dtype=np.int32),
            'orbit_num': np.full(n, 1, dtype=np.int32),
            'stopped': np.zeros(n, dtype=bool),
            'converged_to': [None] * n,
        },
        geometry='geometry',
        crs='EPSG:3413',
    )
    return gdf
