import pandas as pd
import pytest

from experiments.build_within_dataset_buoy_splits import (
    FOLDS,
    assign_buoys,
    build_group_features,
    build_splits,
)


def fixture_tables():
    observations = []
    transitions = []
    image_id = 0
    for month_index, month in enumerate(("2020-01", "2020-02", "2020-03")):
        for buoy_index in range(6):
            buoy_id = f"{month_index}{buoy_index}"
            trajectory_id = f"buoy_{buoy_id}_{month}"
            for step in range(3):
                image_id += 1
                observations.append(
                    {
                        "buoy_id": buoy_id,
                        "image_id": image_id,
                        "acquisition_pass_id": f"pass_{month}_{step}",
                        "month": month,
                        "sic_regime": "pack" if buoy_index % 2 else "marginal",
                        "spatial_block": f"block_{buoy_index % 3}",
                        "experiment_trajectory_id": trajectory_id,
                        "eligible_tracking_observation": True,
                    }
                )
                if step:
                    transitions.append(
                        {
                            "buoy_id": buoy_id,
                            "month": month,
                            "cadence_band": "12_to_30h",
                            "source_sic_regime": "pack",
                        }
                    )
    return pd.DataFrame(observations), pd.DataFrame(transitions)


def test_whole_buoy_splits_are_deterministic_and_disjoint():
    observations, transitions = fixture_tables()
    first = build_splits(observations, transitions, seed=17, restarts=24)
    shuffled = build_splits(
        observations.sample(frac=1.0, random_state=3),
        transitions.sample(frac=1.0, random_state=4),
        seed=17,
        restarts=24,
    )

    first_assignments = first[2].set_index("buoy_id")["within_dataset_split"]
    shuffled_assignments = shuffled[2].set_index("buoy_id")[
        "within_dataset_split"
    ]
    pd.testing.assert_series_equal(
        first_assignments.sort_index(), shuffled_assignments.sort_index()
    )
    assert set(first_assignments.unique()) == set(FOLDS)
    assert first[0].groupby("buoy_id")["within_dataset_split"].nunique().max() == 1
    assert first[1].groupby("buoy_id")["within_dataset_split"].nunique().max() == 1


def test_assignment_balances_core_counts_on_symmetric_fixture():
    observations, transitions = fixture_tables()
    features, _ = build_group_features(observations, transitions)
    labels, _, _ = assign_buoys(features, seed=21, restarts=24)
    counts = labels.value_counts()

    assert counts.max() - counts.min() <= 1
    split_months = observations.assign(
        split=observations["buoy_id"].map(labels)
    ).groupby("split")["month"].nunique()
    assert (split_months == 3).all()


def test_unknown_transition_buoy_is_rejected():
    observations, transitions = fixture_tables()
    transitions.loc[len(transitions)] = {
        "buoy_id": "not_in_observations",
        "month": "2020-01",
        "cadence_band": "12_to_30h",
        "source_sic_regime": "pack",
    }

    with pytest.raises(ValueError, match="absent from eligible observations"):
        build_splits(observations, transitions, seed=1, restarts=2)
