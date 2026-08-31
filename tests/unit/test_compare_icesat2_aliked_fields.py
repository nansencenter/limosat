import argparse

import numpy as np
import pandas as pd
import pytest

from experiments.compare_icesat2_aliked_fields import (
    compare_method_bins,
    parse_labeled_path,
)


def method_bins(label, shear, divergence):
    return pd.DataFrame(
        {
            "beam": ["gt1l"] * len(shear),
            "track_bin": np.arange(len(shear)),
            f"{label}_shear_per_day": shear,
            f"{label}_divergence_per_day": divergence,
        }
    )


def test_parse_labeled_path_rejects_reserved_and_malformed_labels():
    assert parse_labeled_path("full=/tmp/field.csv")[0] == "full"
    with pytest.raises(argparse.ArgumentTypeError):
        parse_labeled_path("orb=/tmp/field.csv")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_labeled_path("not-a-label=/tmp/field.csv")


def test_compare_method_bins_uses_the_first_field_as_reference():
    full = method_bins("full", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    window = method_bins("window", [1.0, 2.5, 3.0], [4.0, 5.5, 6.0])

    comparison = compare_method_bins({"full": full, "window": window})["window"]

    assert comparison["reference"] == "full"
    assert comparison["bins"] == 3
    assert comparison["shear_spearman_between_fields"] == 1.0
    assert comparison["median_absolute_shear_difference_per_day"] == 0.0
