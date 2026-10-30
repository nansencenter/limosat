import numpy as np
from scipy.spatial import cKDTree

from limosat.qc import ExtensionQCConfig, _local_diagnostics


def _reference(xy, uv, config):
    """Per-row implementation the vectorised diagnostics must reproduce exactly."""
    count = len(xy)
    result = {
        "count": np.zeros(count, dtype=np.int32),
        "support_radius_m": np.full(count, np.nan),
        "median_uv": np.full((count, 2), np.nan),
        "residual_m": np.full(count, np.nan),
        "robust_z": np.full(count, np.nan),
        "review": np.zeros(count, dtype=bool),
    }
    if count <= config.minimum_neighbours:
        return result
    k = min(config.maximum_neighbours + 1, count)
    distances, neighbours = cKDTree(xy).query(xy, k=k, distance_upper_bound=config.neighbour_search_radius_m)
    for index in range(count):
        valid = np.isfinite(distances[index]) & (neighbours[index] < count) & (neighbours[index] != index)
        selected = neighbours[index, valid].astype(int)
        result["count"][index] = len(selected)
        if len(selected) < config.minimum_neighbours:
            continue
        neighbour_uv = uv[selected]
        median_uv = np.median(neighbour_uv, axis=0)
        neighbour_residuals = np.linalg.norm(neighbour_uv - median_uv, axis=1)
        residual = float(np.linalg.norm(uv[index] - median_uv))
        median_residual = float(np.median(neighbour_residuals))
        mad = float(np.median(np.abs(neighbour_residuals - median_residual)))
        scale = max(1.4826 * mad, config.residual_noise_floor_m)
        result["support_radius_m"][index] = float(np.max(distances[index, valid]))
        result["median_uv"][index] = median_uv
        result["residual_m"][index] = residual
        result["robust_z"][index] = max(0.0, (residual - median_residual) / scale)
        result["review"][index] = residual > max(
            config.local_review_floor_m, median_residual + config.local_review_z * scale
        )
    return result


def test_vectorised_local_diagnostics_match_reference():
    config = ExtensionQCConfig(configured_speed_m_per_day=43200.0, hard_speed_m_per_day=60000.0)
    rng = np.random.default_rng(3)
    for count in (3, 9, 40, 400):
        # Clustered and sparse points so neighbour counts vary, including below the minimum.
        xy = np.concatenate([rng.normal(0, 4000, (count // 2, 2)), rng.uniform(-1e5, 1e5, (count - count // 2, 2))])
        uv = rng.normal(0, 800, (count, 2))
        uv[rng.random(count) < 0.05] *= 30
        expected, actual = _reference(xy, uv, config), _local_diagnostics(xy, uv, config)
        for key in expected:
            if key in ("residual_m", "robust_z"):
                # A vector norm per row may round differently from the 1-D norm in the last bit.
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=0, err_msg=key)
            else:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"{key} n={count}")
