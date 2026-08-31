import numpy as np
import pandas as pd
import pytest

from experiments.analyze_buoy_array_deformation import analyze, validate_input


def fixture(
    truth_gradient=None,
    estimate_gradient=None,
    truth_translation=(100.0, -50.0),
    estimate_translation=(105.0, -53.0),
):
    source = np.array(
        [
            [400_000.0, -300_000.0],
            [500_000.0, -300_000.0],
            [400_000.0, -200_000.0],
            [500_000.0, -200_000.0],
            [450_000.0, -250_000.0],
        ]
    )
    centred = source - source.mean(axis=0)
    truth_gradient = np.zeros((2, 2)) if truth_gradient is None else truth_gradient
    estimate_gradient = (
        truth_gradient if estimate_gradient is None else estimate_gradient
    )
    truth = centred @ truth_gradient.T + np.asarray(truth_translation)
    estimate = centred @ estimate_gradient.T + np.asarray(estimate_translation)
    return pd.DataFrame(
        {
            "source_image_id": 7,
            "target_image_id": 8,
            "method": "candidate",
            "buoy_id": [f"b{index}" for index in range(len(source))],
            "source_x": source[:, 0],
            "source_y": source[:, 1],
            "truth_dx_m": truth[:, 0],
            "truth_dy_m": truth[:, 1],
            "estimated_dx_m": estimate[:, 0],
            "estimated_dy_m": estimate[:, 1],
            "available": True,
            "elapsed_hours": 24.0,
            "analysis_crs": "EPSG:3413",
        }
    )


def test_common_translation_bias_does_not_look_like_deformation_error():
    outputs = analyze(fixture(), bootstrap_replicates=50, random_seed=4)
    summary = outputs["pair_summary"].iloc[0]

    assert summary.expected_buoys == 5
    assert summary.available_buoys == 5
    assert summary.median_endpoint_error_m == pytest.approx(np.hypot(5.0, -3.0))
    assert summary.vector_bias_dx_m == pytest.approx(5.0)
    assert summary.vector_bias_dy_m == pytest.approx(-3.0)
    assert summary.median_relative_displacement_error_m < 1e-10
    assert summary.gradient_frobenius_error < 1e-12
    assert summary.incorrect_triangle_orientations == 0
    assert outputs["bootstrap_summary"].iloc[0].bootstrap_replicates_valid > 0


def test_affine_gradient_and_deformation_components_are_recovered():
    truth_gradient = np.array([[0.01, 0.02], [-0.01, 0.03]])
    estimate_gradient = np.array([[0.015, 0.01], [-0.005, 0.025]])
    outputs = analyze(
        fixture(
            truth_gradient=truth_gradient,
            estimate_gradient=estimate_gradient,
            estimate_translation=(100.0, -50.0),
        ),
        bootstrap_replicates=0,
    )
    summary = outputs["pair_summary"].iloc[0]

    assert summary.gradient_frobenius_error == pytest.approx(
        np.linalg.norm(estimate_gradient - truth_gradient)
    )
    assert summary.truth_divergence_per_day == pytest.approx(0.04)
    assert summary.estimated_divergence_per_day == pytest.approx(0.04)
    assert summary.divergence_per_day_error == pytest.approx(0.0, abs=1e-14)
    assert summary.shear_per_day_error != 0.0


def test_missing_buoy_is_retained_in_endpoint_pair_and_triangle_denominators():
    frame = fixture()
    frame.loc[0, ["available", "estimated_dx_m", "estimated_dy_m"]] = [
        False,
        np.nan,
        np.nan,
    ]
    outputs = analyze(frame, bootstrap_replicates=0)
    summary = outputs["pair_summary"].iloc[0]

    assert summary.expected_buoys == 5
    assert summary.available_buoys == 4
    assert summary.coverage_fraction == pytest.approx(0.8)
    assert summary.expected_buoy_pairs == 10
    assert summary.available_buoy_pairs == 6
    assert summary.buoy_pair_coverage_fraction == pytest.approx(0.6)
    assert summary.expected_triangles > summary.available_triangles
    assert summary.affine_available


def test_truth_must_match_between_methods_and_use_projected_crs():
    first = fixture()
    second = fixture().assign(method="baseline")
    second.loc[0, "source_x"] += 1e-8
    validate_input(pd.concat([first, second], ignore_index=True))

    second.loc[0, "truth_dx_m"] += 1.0
    with pytest.raises(ValueError, match="truth metadata differs"):
        validate_input(pd.concat([first, second], ignore_index=True))

    invalid_crs = fixture().assign(analysis_crs="EPSG:4326")
    with pytest.raises(ValueError, match="EPSG:3413"):
        validate_input(invalid_crs)
