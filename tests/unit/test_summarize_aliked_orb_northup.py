import pandas as pd

from experiments.summarize_aliked_orb_northup import (
    paired_exact_p_value,
    paired_summary,
)


def test_paired_summary_uses_image_pairs_as_bootstrap_clusters():
    frame = pd.DataFrame(
        {
            "source_image_id": [1, 1, 3],
            "target_image_id": [2, 2, 4],
            "left_correct": [True, True, False],
            "right_correct": [False, True, True],
        }
    )

    result = paired_summary(frame, "test", "left", "right")

    assert result["unique_image_pairs"] == 2
    assert result["left_only_correct"] == 1
    assert result["right_only_correct"] == 1
    assert result["equal_image_pair_difference_percentage_points"] == -25.0


def test_exact_paired_test_counts_only_discordant_cases():
    assert paired_exact_p_value(7, 3) == 0.34375

