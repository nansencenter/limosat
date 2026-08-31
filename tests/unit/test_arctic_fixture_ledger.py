import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_ledger_module():
    path = Path(__file__).resolve().parents[2] / "experiments" / "build_arctic_fixture_ledger.py"
    spec = importlib.util.spec_from_file_location("build_arctic_fixture_ledger", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_projected_interpolation_is_exact_and_rejects_extrapolation():
    ledger = load_ledger_module()
    track = pd.DataFrame(
        {
            "timestamp": ["2020-01-01T00:00:00Z", "2020-01-01T02:00:00Z"],
            "x": [0.0, 2000.0],
            "y": [1000.0, -1000.0],
        }
    )

    np.testing.assert_allclose(
        ledger.interpolate_projected(track, "2020-01-01T01:00:00Z"),
        (1000.0, 0.0),
        atol=1.0e-6,
    )

    try:
        ledger.interpolate_projected(track, "2019-12-31T23:59:59Z")
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("Temporal extrapolation must be rejected")
