import sys
import types
import warnings

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _set_seed():
    np.random.seed(42)


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: end-to-end pipeline smoke tests")
    config.addinivalue_line("markers", "unit: small, deterministic unit tests")
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
    )


def _install_optional_dependency_stubs():
    """Allow pure unit tests to run when geolocation libraries are unavailable."""
    try:
        import cartopy.crs  # noqa: F401
    except ImportError:
        cartopy = types.ModuleType("cartopy")
        crs = types.ModuleType("cartopy.crs")

        class _CRS:
            def __init__(self, *args, **kwargs):
                pass

            def transform_points(self, other, x, y):
                x = np.asarray(x)
                y = np.asarray(y)
                return np.c_[x, y, np.zeros_like(x)]

        crs.CRS = _CRS
        crs.PlateCarree = _CRS
        crs.NorthPolarStereo = _CRS
        cartopy.crs = crs
        sys.modules["cartopy"] = cartopy
        sys.modules["cartopy.crs"] = crs

    try:
        import nansat  # noqa: F401
    except ImportError:
        nansat = types.ModuleType("nansat")

        class NSR:
            def __init__(self, epsg):
                self._epsg = epsg

            def ExportToProj4(self):
                return f"+init=EPSG:{self._epsg}"

        class Nansat:
            def __init__(self, *args, **kwargs):
                self.filename = args[0] if args else "dummy"
                self.vrt = types.SimpleNamespace(tps=False)

            def transform_points(self, x, y, DstToSrc=0, dst_srs=None):
                return np.asarray(x), np.asarray(y)

            def get_corners(self):
                return np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1])

        nansat.NSR = NSR
        nansat.Nansat = Nansat
        sys.modules["nansat"] = nansat


_install_optional_dependency_stubs()
