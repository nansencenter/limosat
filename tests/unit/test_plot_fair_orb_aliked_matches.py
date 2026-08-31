from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.plot_fair_orb_aliked_matches import (
    pair_bounds,
    spatially_balanced_sample,
)


def match_rows(count: int) -> pd.DataFrame:
    side = int(np.ceil(np.sqrt(count)))
    x, y = np.meshgrid(np.arange(side) * 1000.0, np.arange(side) * 1000.0)
    return pd.DataFrame(
        {
            "source_x": x.ravel()[:count],
            "source_y": y.ravel()[:count],
            "target_x": x.ravel()[:count] + 100.0,
            "target_y": y.ravel()[:count] - 50.0,
            "dx_m": np.full(count, 100.0),
            "dy_m": np.full(count, -50.0),
        }
    )


def test_spatially_balanced_sample_is_bounded_and_deterministic():
    rows = match_rows(900)
    bounds = (0.0, 30_000.0, 0.0, 30_000.0)

    first = spatially_balanced_sample(rows, bounds, maximum=100)
    second = spatially_balanced_sample(rows, bounds, maximum=100)

    assert len(first) <= 100
    pd.testing.assert_frame_equal(first, second)


def test_pair_bounds_include_display_scaled_targets():
    rows = match_rows(4)
    bounds = pair_bounds({"ORB": rows, "ALIKED": rows}, display_scale=6.0)

    assert bounds[0] < rows["source_x"].min()
    assert bounds[1] > (rows["source_x"] + 6.0 * rows["dx_m"]).max()
    assert bounds[2] < (rows["source_y"] + 6.0 * rows["dy_m"]).min()
    assert bounds[3] > rows["source_y"].max()
