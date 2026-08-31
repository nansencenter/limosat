import pytest

from experiments.run_full70_orb_speed_sweep import parse_floats


def test_speed_sweep_parser_requires_positive_explicit_values():
    assert parse_floats("40,50,75,100") == (40.0, 50.0, 75.0, 100.0)
    with pytest.raises(ValueError):
        parse_floats("0,50")
