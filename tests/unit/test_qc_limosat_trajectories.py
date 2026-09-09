import numpy as np
import pandas as pd
import pytest

from limosat.qc import core as qc


pytestmark = pytest.mark.unit


def affine_vector_fixture(size=15, seed=4):
    rng = np.random.default_rng(seed)
    axis = np.arange(size, dtype=float) * 5_000.0
    x, y = np.meshgrid(axis, axis)
    x = x.ravel()
    y = y.ravel()
    u = 7_000.0 + 0.005 * x - 0.002 * y + rng.normal(0, 35, len(x))
    v = 3_000.0 + 0.001 * x + 0.003 * y + rng.normal(0, 35, len(x))
    count = len(x)
    return pd.DataFrame(
        {
            "trajectory_id": np.arange(count),
            "source_rowid": 2 * np.arange(count) + 1,
            "target_rowid": 2 * np.arange(count) + 2,
            "source_image_id": 1,
            "target_image_id": 2,
            "source_time": pd.Timestamp("2024-01-01"),
            "target_time": pd.Timestamp("2024-01-02"),
            "elapsed_days": 1.0,
            "x0_m": x,
            "y0_m": y,
            "x1_m": x + u,
            "y1_m": y + v,
            "u_m": u,
            "v_m": v,
            "magnitude_m": np.hypot(u, v),
            "speed_m_per_day": np.hypot(u, v),
            "corr": 0.7,
            "interpolated": 0,
            "image_gap": 1,
        }
    )


def test_local_affine_qc_recovers_injected_errors_without_clean_loss():
    vectors = affine_vector_fixture()
    injected = np.array([32, 66, 112, 158, 190])
    vectors.loc[injected, "u_m"] += 15_000.0
    vectors.loc[injected, "v_m"] -= 12_000.0
    vectors.loc[injected, "x1_m"] = vectors.loc[injected, "x0_m"] + vectors.loc[injected, "u_m"]
    vectors.loc[injected, "y1_m"] = vectors.loc[injected, "y0_m"] + vectors.loc[injected, "v_m"]
    vectors.loc[injected, "magnitude_m"] = np.hypot(
        vectors.loc[injected, "u_m"], vectors.loc[injected, "v_m"]
    )
    vectors.loc[injected, "speed_m_per_day"] = vectors.loc[injected, "magnitude_m"]

    result = qc.score_vectors(vectors, qc.QCConfig())

    assert set(result.index[result.reject]) == set(injected)
    assert result.loc[injected, "local_supported"].all()


def test_piecewise_motion_boundary_is_retained():
    vectors = affine_vector_fixture()
    right = vectors["x0_m"] >= vectors["x0_m"].median()
    vectors.loc[right, "u_m"] += 2_000.0
    vectors["x1_m"] = vectors["x0_m"] + vectors["u_m"]
    vectors["magnitude_m"] = np.hypot(vectors["u_m"], vectors["v_m"])
    vectors["speed_m_per_day"] = vectors["magnitude_m"]

    result = qc.score_vectors(vectors, qc.QCConfig())

    assert not result.reject.any()


def test_bundled_protocol_matches_code_defaults():
    protocol = qc.load_protocol()

    assert protocol["protocol_id"] == qc.PROTOCOL_ID
    qc.validate_frozen_protocol(qc.QCConfig(configured_speed_m_per_day=50_000.0))

    with pytest.raises(ValueError, match="does not match frozen protocol"):
        qc.validate_frozen_protocol(qc.QCConfig(search_radius_m=20_000.0))


@pytest.mark.parametrize("prescreen", [1_000.0, 1_000_000.0])
def test_archive_prescreen_preserves_full_qc_decisions(prescreen):
    vectors = affine_vector_fixture()
    injected = np.array([32, 66, 112, 158, 190])
    vectors.loc[injected, "u_m"] += 15_000.0
    vectors.loc[injected, "v_m"] -= 12_000.0
    vectors["x1_m"] = vectors["x0_m"] + vectors["u_m"]
    vectors["y1_m"] = vectors["y0_m"] + vectors["v_m"]
    vectors["magnitude_m"] = np.hypot(vectors["u_m"], vectors["v_m"])
    vectors["speed_m_per_day"] = vectors["magnitude_m"]

    full = qc.score_vectors(vectors, qc.QCConfig())
    accelerated = qc.score_vectors(
        vectors, qc.QCConfig(), prescreen_residual_m=prescreen
    )

    assert accelerated.reject.tolist() == full.reject.tolist()
    assert accelerated.review.tolist() == full.review.tolist()
    assert accelerated.local_evaluated.sum() < full.local_evaluated.sum()


def geometry_fixture(x, y, u, v=None):
    """Float64 positions/displacements in metres; all vectors span one day."""
    x, y, u = (np.asarray(values, dtype=float) for values in (x, y, u))
    v = np.zeros_like(u) if v is None else np.asarray(v, dtype=float)
    return pd.DataFrame(dict(
        source_image_id=1, target_image_id=2,
        x0_m=x, y0_m=y, x1_m=x + u, y1_m=y + v,
        u_m=u, v_m=v, speed_m_per_day=np.hypot(u, v),
    ))


@pytest.mark.parametrize("prescreen", [None, 1_000.0])
def test_narrow_neighbour_geometry_does_not_amplify_small_motion_noise(prescreen):
    rng = np.random.default_rng(4)
    vectors = geometry_fixture(
        np.r_[0., np.linspace(5_000., 15_000., 12)],
        np.r_[0., 1_000. + rng.normal(0, .1, 12)],
        np.r_[3_000., 1_000. + rng.normal(0, 35., 12)],
    )
    result = qc.score_vectors(vectors, qc.QCConfig(), prescreen_residual_m=prescreen)
    assert result.loc[0, "local_evaluated"]
    assert result.loc[0, "local_residual_m"] < 2_100
    assert not result.loc[0, "reject"]
    assert not result.loc[0, "review"]


def test_prescreen_handles_zero_median_residual_with_one_sided_neighbours():
    x, y = np.meshgrid([10_000., 10_500., 11_000., 11_500.], [-1_000., 0., 1_000.])
    vectors = geometry_fixture(
        np.r_[0., x.ravel()], np.r_[0., y.ravel()],
        np.r_[1_500., 2 * x.ravel() - 20_000.],
    )
    full = qc.score_vectors(vectors, qc.QCConfig())
    fast = qc.score_vectors(vectors, qc.QCConfig(), prescreen_residual_m=1_000.)
    assert full.loc[0, "coarse_local_residual_m"] == 0
    assert fast.loc[0, "local_evaluated"]
    assert not full.loc[0, "reject"]  # Unsafe extrapolation falls back to translation.
    pd.testing.assert_frame_equal(full[["reject", "review"]], fast[["reject", "review"]])


@pytest.mark.parametrize("seed", range(5))
def test_prescreen_decisions_match_for_irregular_contaminated_fields(seed):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-20_000., 20_000., (80, 2))
    uv = xy @ np.array([[.4, .2], [-.1, .3]]) + rng.normal(0, 100., (80, 2))
    uv[::7] += rng.normal(0, 25_000., uv[::7].shape)
    vectors = geometry_fixture(*xy.T, *uv.T)
    full = qc.score_vectors(vectors, qc.QCConfig())
    fast = qc.score_vectors(vectors, qc.QCConfig(), prescreen_residual_m=1_000.)
    pd.testing.assert_frame_equal(full[["reject", "review"]], fast[["reject", "review"]])


@pytest.mark.parametrize("matrix", [
    [[.2, .1], [0., 0.]],  # Coherent shear.
    [[np.cos(.3) - 1, -np.sin(.3)], [np.sin(.3), np.cos(.3) - 1]],
])
def test_coherent_shear_and_rotation_are_retained(matrix):
    x, y = np.meshgrid(np.linspace(-20_000., 20_000., 9), np.linspace(-20_000., 20_000., 9))
    xy = np.column_stack([x.ravel(), y.ravel()])
    uv = xy @ np.asarray(matrix).T
    result = qc.score_vectors(geometry_fixture(*xy.T, *uv.T), qc.QCConfig())
    assert not result.reject.any()
    assert not result.review.any()


def test_sparse_pair_is_only_subject_to_hard_speed_gate():
    vectors = affine_vector_fixture(size=2)
    vectors.loc[0, ["u_m", "x1_m", "magnitude_m", "speed_m_per_day"]] = [
        70_000.0,
        vectors.loc[0, "x0_m"] + 70_000.0,
        70_000.0,
        70_000.0,
    ]

    result = qc.score_vectors(vectors, qc.QCConfig())

    assert not result.local_supported.any()
    assert result.reject.tolist() == [True, False, False, False]
    assert result.loc[0, "decision_reason"] == "hard_speed"


@pytest.mark.parametrize("prescreen", [None, 1_000.0])
def test_legacy_scoring_entry_point_preserves_values_and_keyword_arguments(prescreen):
    vectors = affine_vector_fixture()
    config = qc.QCConfig()
    current = qc.score_vectors(vectors, config, prescreen_residual_m=prescreen)
    legacy = qc.score_edges(edges=vectors, config=config, prescreen_residual_m=prescreen)
    pd.testing.assert_frame_equal(current, legacy, check_exact=True)


def test_orientation_diagnostic_detects_and_removes_fold():
    vectors = affine_vector_fixture(size=4)
    centre = np.argmin(
        (vectors.x0_m - 5_000.0) ** 2 + (vectors.y0_m - 5_000.0) ** 2
    )
    vectors.loc[centre, "x1_m"] += 20_000.0
    vectors.loc[centre, "y1_m"] += 20_000.0
    before = qc.annotate_topology_incidence(vectors, maximum_edge_m=8_000.0)
    after = qc.annotate_topology_incidence(
        vectors.drop(index=centre).reset_index(drop=True), maximum_edge_m=8_000.0
    )

    assert before["topology_flip_count"].sum() > 0
    assert after["topology_flip_count"].sum() == 0


def test_v3_keeps_v2_vector_scoring_rules():
    import json

    previous = json.loads((qc.PROTOCOL_PATH.parent / "trajectory_link_qc_v2.json").read_text())
    current = qc.load_protocol()
    for field in ("parameters", "neighbour_model", "automatic_rejection", "retained_for_review",
                  "not_used_for_rejection", "archive_prescreen"):
        assert current[field] == previous[field]
    assert current["protocol_id"] != previous["protocol_id"]
    assert current["split_semantics"]["minimum_segment_points"] == 2
