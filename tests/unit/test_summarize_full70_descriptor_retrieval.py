import pandas as pd

from experiments.summarize_full70_descriptor_retrieval import summarize_retrieval


def test_retrieval_summary_keeps_unavailable_pair_in_denominator():
    pairs = pd.DataFrame(
        {
            "pair_id": [0, 1],
            "source_experiment_split": ["evaluation", "evaluation"],
            "source_month_exclusive_buoy": [True, True],
        }
    )
    results = pd.DataFrame(
        {
            "pair_id": [0, 1],
            "method": ["orb_geo_hamming", "orb_geo_hamming"],
            "gate": ["physics_50km_day", "unavailable"],
            "accepted": [True, False],
            "endpoint_error_m": [1000.0, float("nan")],
            "truth_descriptor_rank": [1, float("nan")],
            "normalized_truth_descriptor_distance": [0.1, float("nan")],
            "candidate_quantization_error_m": [500.0, float("nan")],
        }
    )

    summary = summarize_retrieval(pairs, results)
    row = summary[
        summary["evaluation_subset"].eq("all_temporal")
        & summary["gate"].eq("physics_50km_day")
    ].iloc[0]

    assert row.expected_pairs == 2
    assert row.retrieved_pairs == 1
    assert row.within_2km_fraction_all == 0.5
