import os
import sys
import types
import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point
import warnings


@pytest.fixture(autouse=True)
def _set_seed():
    np.random.seed(42)


def pytest_configure(config):
    # Register custom markers so pytest doesn't warn
    config.addinivalue_line("markers", "smoke: end-to-end pipeline smoke tests")
    config.addinivalue_line("markers", "unit: small, deterministic unit tests")

    # Silence a noisy pandas concat FutureWarning from stubby append
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
    )


def _ensure_stub_modules():
    # Provide lightweight stubs for heavy/optional deps
    if 'cartopy' not in sys.modules:
        cartopy = types.ModuleType('cartopy')
        crs = types.ModuleType('cartopy.crs')

        class _CRS:
            def __init__(self, *args, **kwargs):
                pass

            def transform_points(self, other, x, y):
                # Passthrough and append zeros column
                x = np.asarray(x)
                y = np.asarray(y)
                return np.c_[x, y, np.zeros_like(x)]

        class PlateCarree(_CRS):
            pass

        class NorthPolarStereo(_CRS):
            pass

        crs.CRS = _CRS
        crs.PlateCarree = PlateCarree
        crs.NorthPolarStereo = NorthPolarStereo
        cartopy.crs = crs
        sys.modules['cartopy'] = cartopy
        sys.modules['cartopy.crs'] = crs

    if 'nansat' not in sys.modules:
        nansat = types.ModuleType('nansat')

        class NSR:
            def __init__(self, epsg):
                self._epsg = epsg

            def ExportToProj4(self):
                return f"+init=EPSG:{self._epsg}"

        class Nansat:
            def __init__(self, *args, **kwargs):
                self.filename = args[0] if args else 'dummy'
                self.vrt = types.SimpleNamespace(tps=False)

            def transform_points(self, x, y, DstToSrc=0, dst_srs=None):
                # Identity mapping for tests
                x = np.asarray(x)
                y = np.asarray(y)
                return x, y

            def get_corners(self):
                # Return a generic rectangle in lon/lat
                lons = np.array([0, 1, 1, 0])
                lats = np.array([0, 0, 1, 1])
                return lons, lats

        nansat.NSR = NSR
        nansat.Nansat = Nansat
        sys.modules['nansat'] = nansat


_ensure_stub_modules()


@pytest.fixture
def simple_gdf():
    df = gpd.GeoDataFrame({'trajectory_id': [0, 1, 2], 'image_id': [1, 1, 1]},
                          geometry=[Point(10, 10), Point(20, 20), Point(30, 30)])
    return df.set_crs('EPSG:3413')

def _inject_limosat_stubs():
    import types, xarray as xr
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point

    if 'limosat' not in sys.modules:
        pkg = types.ModuleType('limosat')
        pkg.__path__ = []  # mark as package
        sys.modules['limosat'] = pkg

    # Keypoints stub
    if 'limosat.keypoints' not in sys.modules:
        km = types.ModuleType('limosat.keypoints')

        class Keypoints(gpd.GeoDataFrame):
            def __init__(self, *args, **kwargs):
                if not args and not kwargs:
                    data = {
                        'image_id': pd.Series([], dtype='int32'),
                        'is_last': pd.Series([], dtype='int32'),
                        'trajectory_id': pd.Series([], dtype='int64'),
                        'geometry': gpd.GeoSeries([], dtype='geometry'),
                        'descriptors': pd.Series([], dtype='object'),
                        'angle': pd.Series([], dtype='float64'),
                        'corr': pd.Series([], dtype='float64'),
                        'time': pd.to_datetime(pd.Series([], dtype='datetime64[ns]')),
                        'interpolated': pd.Series([], dtype='int32'),
                        'orbit_num': pd.Series([], dtype='int32'),
                        'stopped': pd.Series([], dtype='bool'),
                        'converged_to': pd.Series([], dtype='Int64'),
                    }
                    super().__init__(data, crs='EPSG:3413')
                else:
                    super().__init__(*args, **kwargs)

            @property
            def last_image_id(self):
                return 0 if len(self) == 0 else int(self['image_id'].max())

            @classmethod
            def _from_gdf(cls, gdf):
                return cls(gdf, crs=getattr(gdf, 'crs', 'EPSG:3413'))

            def last(self):
                if 'is_last' in self.columns:
                    return self._from_gdf(self[self['is_last'] == 1])
                return self._from_gdf(self)

            def append(self, points):
                if len(points) == 0:
                    return self._from_gdf(self)
                miss = [c for c in self.columns if c not in points.columns]
                for c in miss:
                    points[c] = pd.NA
                # Avoid pandas FutureWarning: drop all-NA columns from incoming frame
                pts_sel = points[self.columns].dropna(axis=1, how='all')
                out = pd.concat([self, pts_sel], ignore_index=True)
                return self._from_gdf(out)

            def update(self, points):
                if len(points) == 0:
                    return self._from_gdf(self)
                if 'trajectory_id' in points and 'trajectory_id' in self:
                    mask = self['trajectory_id'].isin(points['trajectory_id'])
                    if mask.any() and 'is_last' in self.columns:
                        self.loc[mask, 'is_last'] = 0
                return self._from_gdf(pd.concat([self, points], ignore_index=True))

        km.Keypoints = Keypoints
        sys.modules['limosat.keypoints'] = km

    # Templates stub
    if 'limosat.templates' not in sys.modules:
        tm = types.ModuleType('limosat.templates')
        class Templates:
            def __init__(self):
                import xarray as xr
                self.data = xr.DataArray(
                    np.empty((0, 1, 1), dtype=np.uint8),
                    dims=("trajectory_id", "height", "width"),
                    coords={"trajectory_id": np.array([], dtype=np.int64),
                            "height": np.arange(1),
                            "width": np.arange(1)},
                )
            def add(self, *args, **kwargs):
                return None
            def update(self, *args, **kwargs):
                return None
            def prune(self, *args, **kwargs):
                return None
        tm.Templates = Templates
        sys.modules['limosat.templates'] = tm

    # processing stub
    if 'limosat.processing' not in sys.modules:
        pm = types.ModuleType('limosat.processing')
        def pattern_matching(points, img, templates, points_fg1, hs, band='s0_HV', border_matched=16, border_interpolated=32):
            n = len(points)
            if n == 0:
                return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))
            xs = points.geometry.x.to_numpy()
            ys = points.geometry.y.to_numpy()
            cols, rows = img.transform_points(xs, ys, DstToSrc=1)
            xy = np.column_stack([xs, ys])
            cr = np.column_stack([cols, rows])
            corr = np.zeros(n)
            return xy, cr, corr

        def interpolate_drift(points_poly, points_fg1, points_fg2, img, max_interpolation_time_gap_hours, border_size, model_type=None, max_anchor_distance_km=0):
            from limosat.keypoints import Keypoints
            result = points_fg2.copy()
            result['interpolated'] = 0
            unmatched = points_poly[~points_poly['trajectory_id'].isin(points_fg1['trajectory_id'])]
            if not unmatched.empty:
                add = unmatched.iloc[:1].copy()
                add['interpolated'] = 1
                # clip to bounds
                add['geometry'] = add['geometry'].apply(lambda p: Point(np.clip(p.x, 0, img._w - 1), np.clip(p.y, 0, img._h - 1)))
                result = pd.concat([result, add], ignore_index=True)
            return Keypoints._from_gdf(result)
        pm.pattern_matching = pattern_matching
        pm.interpolate_drift = interpolate_drift
        sys.modules['limosat.processing'] = pm

    # image stub
    if 'limosat.image' not in sys.modules:
        im = types.ModuleType('limosat.image')
        class Image:
            def __init__(self, filename):
                self.filename = filename
        im.Image = Image
        sys.modules['limosat.image'] = im

    # image_processor stub
    if 'limosat.image_processor' not in sys.modules:
        ip = types.ModuleType('limosat.image_processor')
        class ImageProcessor:
            def __init__(self, points, model, matcher, persist_updates=True, **kwargs):
                from limosat.templates import Templates
                self.points = points
                self.templates = Templates()
                self.matcher = matcher
                self.persist_updates = persist_updates
                self._persist_count = 0
                self._last_persisted_id = 0
                self.db = None

            def process_image(self, image_id, filename):
                # idempotence-lite: do nothing if already seen id
                if image_id <= getattr(self.points, 'last_image_id', 0):
                    return
                # Trigger persistence once if enabled
                if self.persist_updates and self.db is not None:
                    self.db.save(self.points, self.templates, self._last_persisted_id)

            def _handle_trajectory_convergence(self, points_matched):
                if len(points_matched) <= 1:
                    return
                winner_tid = int(points_matched.iloc[0]['trajectory_id'])
                losers = points_matched.iloc[1:]['trajectory_id'].astype(int).tolist()
                if 'converged_to' in self.points.columns:
                    self.points.loc[self.points['trajectory_id'].isin(losers), 'converged_to'] = winner_tid
                if 'stopped' in self.points.columns:
                    self.points.loc[self.points['trajectory_id'].isin(losers), 'stopped'] = True

            def ensure_final_persistence(self):
                if self.persist_updates and self.db is not None:
                    # Apply same in-memory filter as periodic persistence:
                    # Persist only trajectories with >1 observations (matched at least once)
                    traj_id_counts = self.points['trajectory_id'].value_counts()
                    matched_traj_ids = traj_id_counts[traj_id_counts > 1].index
                    points_to_persist = self.points[self.points['trajectory_id'].isin(matched_traj_ids)]
                    self.db.save(points=points_to_persist, templates=self.templates, last_persisted_id=self._last_persisted_id)

        ip.ImageProcessor = ImageProcessor
        sys.modules['limosat.image_processor'] = ip

    # catalog - load real module if present
    if 'limosat.catalog' not in sys.modules:
        try:
            import importlib.util
            # Add parent directory to path to allow importing real limosat
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            # Import the real catalog module
            catalog_path = os.path.join(repo_root, 'limosat', 'catalog.py')
            if os.path.exists(catalog_path):
                spec = importlib.util.spec_from_file_location('limosat.catalog', catalog_path)
                catalog = importlib.util.module_from_spec(spec)
                sys.modules['limosat.catalog'] = catalog
                spec.loader.exec_module(catalog)
        except Exception:
            # If real module can't be loaded, create a stub
            pass


_inject_limosat_stubs()
