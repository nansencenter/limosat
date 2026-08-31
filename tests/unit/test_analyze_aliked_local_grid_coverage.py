import numpy as np
import pandas as pd

from experiments.analyze_aliked_local_grid_coverage import evaluate_grid, summarize


def test_uniform_vectors_cover_grid_with_zero_local_roughness():
    offsets = (-4000.0, 0.0, 4000.0)
    vectors = pd.DataFrame(
        [
            {
                "transition_id": "case-1",
                "source_x": x,
                "source_y": y,
                "dx_m": 1200.0,
                "dy_m": -300.0,
                "speed_m_per_day": 5000.0,
                "lightglue_score": 0.9,
            }
            for y in offsets
            for x in offsets
        ]
    )
    transitions = pd.DataFrame(
        [
            {
                "transition_id": "case-1",
                "representative_panel": True,
                "source_x": 0.0,
                "source_y": 0.0,
            }
        ]
    )

    grid, neighbours = evaluate_grid(
        transitions,
        vectors,
        grid_half_extent_m=4000.0,
        grid_spacing_m=4000.0,
        tight_radius_m=2000.0,
        consensus_radius_m=1000.0,
        maximum_speed_m_per_day=30000.0,
    )
    result = summarize(grid, neighbours)

    assert result["queries"] == 9
    assert result["covered_queries"] == 9
    assert result["neighbour_pairs"] == 12
    assert result["overall_coverage_fraction"] == 1.0
    assert np.allclose(neighbours["vector_difference_m"], 0.0)
