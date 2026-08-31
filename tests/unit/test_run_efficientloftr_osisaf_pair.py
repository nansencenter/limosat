import numpy as np
import pytest

from experiments.run_efficientloftr_osisaf_pair import external_routing_shifts


def test_external_routing_uses_osi_then_finite_same_center_fallback():
    shifts, sources, clipped = external_routing_shifts(
        np.array([[3_000.0, 4_000.0], [np.nan, np.nan]]),
        np.array([True, False]),
        np.zeros(2),
        maximum_displacement_m=10_000.0,
    )

    assert shifts.tolist() == [[3_000.0, 4_000.0], [0.0, 0.0]]
    assert sources.tolist() == ["osi455", "same_center_fallback"]
    assert clipped.tolist() == [False, False]


def test_external_routing_clips_osi_and_labels_phase_fallback():
    shifts, sources, clipped = external_routing_shifts(
        np.array([[30_000.0, 40_000.0], [np.nan, np.nan]]),
        np.array([True, False]),
        np.array([6_000.0, 8_000.0]),
        maximum_displacement_m=20_000.0,
    )

    assert shifts[0] == pytest.approx([12_000.0, 16_000.0])
    assert shifts[1] == pytest.approx([6_000.0, 8_000.0])
    assert sources.tolist() == ["osi455_clipped", "phase_fallback"]
    assert clipped.tolist() == [True, False]
