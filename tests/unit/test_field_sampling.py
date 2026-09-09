from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from scipy.spatial import Delaunay

from limosat import DisplacementField
from limosat.field import _recover_boundary_simplices, sample_field


POINTS = np.array([[0., 0.], [4000., 0.], [8000., 0.], [0., 4000.]])


def boundary_field():
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return DisplacementField(
        pair_id="a__b", source_image_id="a", target_image_id="b",
        source_time_utc=start, target_time_utc=start + timedelta(days=1),
        grid_row=np.array([0, 0, 0, 1]), grid_column=np.array([0, 1, 2, 0]),
        source_xy_m=POINTS,
        displacement_m=np.column_stack((100. + POINTS[:, 0] / 100.,
                                         200. + POINTS[:, 1] / 100.)),
        available=np.ones(4, bool), selected_matches=np.array([8, 10, 12, 14]),
        candidate_matches=np.full(4, 12),
        support_radius_m=np.array([100., 200., 300., 400.]),
        maximum_residual_m=np.array([10., 20., 30., 40.]),
    )


def test_available_boundary_vertex_and_edge_use_only_local_support():
    field = boundary_field()
    query = np.vstack((POINTS, [6000., 500.], [2000., 2000.], [-1., 0.]))
    samples = sample_field(field, query, 6400.)

    assert samples.available.tolist() == [True, True, False, True, False, True, False]
    np.testing.assert_allclose(samples.displacement_m[[0, 1, 3]],
                               field.displacement_m[[0, 1, 3]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(samples.displacement_m[5], [120., 220.], rtol=0, atol=1e-12)
    np.testing.assert_allclose(samples.selected_matches[5], 12.)
    np.testing.assert_allclose(samples.support_radius_m[5], 300.)
    np.testing.assert_allclose(samples.maximum_residual_m[5], 30.)
    assert np.isnan(samples.displacement_m[~samples.available]).all()


def test_boundary_selection_preserves_success_and_recovers_every_incident_choice():
    tri = Delaunay(POINTS)
    edge = np.linalg.norm(POINTS[tri.simplices] -
                          np.roll(POINTS[tri.simplices], 1, axis=1), axis=2).max(axis=1)
    usable = edge <= 6400.
    good = int(np.flatnonzero(usable)[0])
    bad = int(np.flatnonzero(~usable)[0])
    # Explicit lookup results exercise both sides of a boundary independently
    # of which triangle a particular SciPy version's directed walk selects.
    query = np.array([[0., 4000.], [0., 4000.], [0., 4000.], [2000., 2000.],
                      [6000., 500.], [8000., 0.], [-1., 0.]])
    chosen = np.array([good, bad, -1, bad, bad, bad, -1])
    _recover_boundary_simplices(tri, usable, query, chosen)
    np.testing.assert_array_equal(chosen, [good, good, good, good, bad, bad, -1])


def test_vertex_fallback_can_cross_more_than_one_incident_triangle():
    points = np.array([[0., 0.], [4000., 0.], [4000., 4000.], [0., 4000.],
                       [-12000., 0.], [0., -12000.]])
    tri = Delaunay(points)
    edges = np.linalg.norm(points[tri.simplices] -
                           np.roll(points[tri.simplices], 1, axis=1), axis=2).max(axis=1)
    usable = edges <= 6400.
    incident = np.flatnonzero((tri.simplices == 0).any(axis=1))
    assert usable[incident].any() and (~usable[incident]).sum() >= 2
    rejected = incident[~usable[incident]]
    chosen = rejected.copy()
    _recover_boundary_simplices(tri, usable, np.zeros((len(chosen), 2)), chosen)
    assert usable[chosen].all()
    assert len(np.unique(chosen)) == 1


def test_sampling_availability_and_values_do_not_depend_on_query_batches():
    field = boundary_field()
    query = np.vstack((POINTS, [6000., 500.], [2000., 2000.], [-1., 0.]))
    expected = sample_field(field, query, 6400.)
    for order in [np.arange(len(query))[::-1], np.array([4, 3, 2, 5, 1, 6, 0])]:
        reordered = sample_field(field, query[order], 6400.)
        for name in expected.__dataclass_fields__:
            np.testing.assert_allclose(getattr(reordered, name),
                                       getattr(expected, name)[order],
                                       rtol=0, atol=1e-12, equal_nan=True)
    single = [sample_field(field, point[None, :], 6400.) for point in query]
    for name in expected.__dataclass_fields__:
        np.testing.assert_allclose(np.concatenate([getattr(s, name) for s in single]),
                                   getattr(expected, name), rtol=0, atol=1e-12,
                                   equal_nan=True)


@pytest.mark.parametrize("target", [np.zeros((4, 2)), POINTS * [-1., 1.]])
def test_collapsed_or_reversed_triangles_remain_unavailable(target):
    field = replace(boundary_field(), displacement_m=target - POINTS)
    assert not sample_field(field, POINTS, 6400.).available.any()


def test_missing_vertex_and_nonlocal_support_are_not_filled():
    field = boundary_field().with_available(np.array([True, True, True, False]))
    assert not sample_field(field, POINTS, 6400.).available.any()
    assert not sample_field(boundary_field(), POINTS, 3999.).available.any()
    assert len(sample_field(boundary_field(), np.empty((0, 2)), 6400.).available) == 0
