import numpy as np
import pandas as pd

from experiments.run_aliked_selected_sequence import sequential_matching_prior


def accepted_field(dx=1000.0, dy=-500.0, count=12):
    return pd.DataFrame(
        {
            "available": np.ones(count, dtype=bool),
            "proposal_dx_m": np.full(count, dx),
            "proposal_dy_m": np.full(count, dy),
        }
    )


def test_sequential_prior_falls_back_when_absent():
    prior, audit = sequential_matching_prior(None, None, 2.0, True)
    assert prior is None
    assert audit["reason"] == "prior_absent"
    assert audit["fallback"] is True


def test_sequential_prior_falls_back_when_stale():
    prior, audit = sequential_matching_prior(accepted_field(), 1.0, 2.0, False)
    assert prior is None
    assert audit["reason"] == "prior_stale_noncontiguous_chain"


def test_sequential_prior_falls_back_when_inconsistent():
    prior, audit = sequential_matching_prior(accepted_field(count=7), 1.0, 2.0, True)
    assert prior is None
    assert audit["reason"] == "prior_inconsistent_insufficient_accepted_field"


def test_sequential_prior_scales_preceding_field_velocity():
    prior, audit = sequential_matching_prior(accepted_field(), 0.5, 2.0, True)
    assert prior == (4000.0, -2000.0)
    assert audit["fallback"] is False
    assert audit["velocity_dx_m_per_day"] == 2000.0
    assert audit["velocity_dy_m_per_day"] == -1000.0
