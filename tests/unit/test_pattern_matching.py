import numpy as np
import cv2
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point

from limosat.processing import pattern_matching, refine_correlation_peak_quadratic
from tests.factories import ImageStub, make_templates


@pytest.mark.unit
def test_quadratic_peak_refinement_recovers_analytic_maximum():
    cols, rows = np.meshgrid(np.arange(7), np.arange(7))
    expected = np.array([3.35, 2.60])
    response = -(
        1.4 * (cols - expected[0]) ** 2
        + 0.3 * (cols - expected[0]) * (rows - expected[1])
        + 0.9 * (rows - expected[1]) ** 2
    )
    peak = cv2.minMaxLoc(response.astype(np.float32))[3]

    dc, dr, status = refine_correlation_peak_quadratic(response, peak)

    assert status == "quadratic"
    np.testing.assert_allclose(np.array(peak) + [dc, dr], expected, atol=1e-12)


@pytest.mark.unit
def test_quadratic_peak_refinement_falls_back_at_boundary():
    dc, dr, status = refine_correlation_peak_quadratic(np.ones((5, 5)), (0, 2))

    assert (dc, dr, status) == (0.0, 0.0, "response_boundary")


@pytest.mark.unit
@pytest.mark.parametrize(
    "translation",
    [(0.30, -0.35), (-0.40, 0.25), (0.45, 0.40)],
)
def test_quadratic_peak_refinement_recovers_fractional_image_translation(
    translation,
):
    rng = np.random.default_rng(20260817)
    texture = rng.normal(size=(129, 129)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), 1.2)
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    center = np.array([64, 64])
    half_size = 16
    border = 8
    template = texture[
        center[1] - half_size : center[1] + half_size + 1,
        center[0] - half_size : center[0] + half_size + 1,
    ]
    shifted = cv2.warpAffine(
        texture,
        np.array([[1.0, 0.0, translation[0]], [0.0, 1.0, translation[1]]]),
        (texture.shape[1], texture.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )
    shifted = np.clip(
        0.72 * shifted.astype(np.float32)
        + 31.0
        + rng.normal(0.0, 1.5, shifted.shape),
        0,
        255,
    ).astype(np.uint8)
    search = shifted[
        center[1] - half_size - border : center[1] + half_size + border + 1,
        center[0] - half_size - border : center[0] + half_size + border + 1,
    ]
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    peak = cv2.minMaxLoc(response)[3]

    dc, dr, status = refine_correlation_peak_quadratic(response, peak)
    estimated = np.array(peak, dtype=float) + [dc, dr] - border

    assert status == "quadratic"
    np.testing.assert_allclose(estimated, translation, atol=0.10)


@pytest.mark.unit
def test_quadratic_peak_refinement_rejects_non_concave_neighbourhood():
    cols, rows = np.meshgrid(np.arange(5), np.arange(5))
    response = (cols - 2.0) ** 2 + (rows - 2.0) ** 2

    dc, dr, status = refine_correlation_peak_quadratic(response, (2, 2))

    assert (dc, dr, status) == (0.0, 0.0, "non_concave")


@pytest.mark.unit
def test_pattern_matching_shapes(monkeypatch):
    # monkeypatch cartopy transforms to pass-through handled by conftest
    img = ImageStub()
    n = 5
    points = gpd.GeoDataFrame(
        {
            'trajectory_id': np.arange(n, dtype=np.int64),
            'interpolated': np.zeros(n, dtype=int),
        },
        geometry=[Point(5 + i, 5 + i) for i in range(n)],
        crs='EPSG:3413',
    )
    points_fg1 = points.copy()
    points_fg1['angle'] = 0

    templates = make_templates(points['trajectory_id'].values, hs=2)

    xy, cr, corr = pattern_matching(points, img, templates, points_fg1, hs=2)

    assert len(xy) == len(points) == len(corr)
    assert cr.shape == (n, 2)

    # boundary case: point outside still clipped by ImageStub transform
    points.loc[0, 'geometry'] = Point(1000, 1000)
    xy, cr, corr = pattern_matching(points, img, templates, points_fg1, hs=2)
    assert cr.shape == (n, 2)
