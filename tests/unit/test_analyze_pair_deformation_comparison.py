import numpy as np
import pandas as pd

from experiments.analyze_pair_deformation_comparison import (
    direct_local_affine_field,
    triangle_field,
)


def affine_vectors(points, gradient, translation):
    displacement = points @ gradient.T + translation
    return pd.DataFrame(
        {
            "source_x": points[:, 0],
            "source_y": points[:, 1],
            "dx_m": displacement[:, 0],
            "dy_m": displacement[:, 1],
        }
    )


def test_triangle_field_recovers_known_affine_deformation():
    axis = np.arange(0.0, 16001.0, 4000.0)
    points = np.array([(x, y) for x in axis for y in axis])
    gradient = np.array([[0.01, 0.004], [-0.002, -0.006]])
    vectors = affine_vectors(points, gradient, np.array([400.0, -250.0]))
    queries = vectors[["source_x", "source_y"]].copy()

    field, _ = triangle_field(vectors, queries, 1.0, maximum_edge_m=6000.0)

    available = field.available
    assert np.allclose(field.loc[available, "divergence_per_day"], 0.004)
    assert np.allclose(
        field.loc[available, "shear_per_day"], np.hypot(0.016, 0.002)
    )


def test_direct_local_affine_recovers_known_gradient():
    axis = np.arange(-12000.0, 12001.0, 3000.0)
    points = np.array([(x, y) for x in axis for y in axis])
    gradient = np.array([[0.008, 0.003], [0.001, -0.004]])
    vectors = affine_vectors(points, gradient, np.array([200.0, -300.0]))
    queries = pd.DataFrame({"source_x": [0.0], "source_y": [0.0]})

    field, _ = direct_local_affine_field(vectors, queries, 1.0)

    assert bool(field.available.iloc[0])
    assert np.isclose(field.divergence_per_day.iloc[0], 0.004, atol=1e-8)
    assert np.isclose(
        field.shear_per_day.iloc[0], np.hypot(0.012, 0.004), atol=1e-8
    )
