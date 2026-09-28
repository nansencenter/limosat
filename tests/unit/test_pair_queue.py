import pytest

from limosat.pair_queue import (
    PairOption, build_pair_request_plan, choose_pair_batch,
)


def test_count_queue_reuses_selected_pair_and_retries_only_unresolved():
    options = {
        "a": (PairOption("short", 12), PairOption("later", 36)),
        "b": (PairOption("short", 12), PairOption("later", 36)),
        "c": (PairOption("other", 8), PairOption("short", 12)),
    }

    assert choose_pair_batch(options, maximum_pairs=1) == (
        ("short", ("a", "b", "c")),
    )
    assert choose_pair_batch(
        options,
        resolved=frozenset({"a", "c"}),
        processed_pairs=frozenset({"short"}),
        maximum_pairs=1,
    ) == (("later", ("b",)),)


def test_count_queue_ties_are_deterministic_and_no_pair_is_repeated():
    options = {
        "b": (PairOption("long", 24),),
        "a": (PairOption("short", 12),),
    }
    assert choose_pair_batch(options, maximum_pairs=2) == (
        ("short", ("a",)), ("long", ("b",))
    )
    assert choose_pair_batch(
        options, processed_pairs=frozenset({"short", "long"}), maximum_pairs=2
    ) == ()
    with pytest.raises(ValueError, match="maximum_pairs"):
        choose_pair_batch(options, maximum_pairs=0)


def test_pair_batch_serializes_as_frozen_target_request():
    requests = build_pair_request_plan(
        (("a__b", ("parcel-1", "parcel-2")),),
        run_id="march", config_sha256="config", primary_composition_manifest_sha256="primary",
    )
    assert requests["pairs"] == [{
        "pair_id": "a__b", "trajectory_ids": ["parcel-1", "parcel-2"]
    }]
    assert requests["primary_composition_manifest_sha256"] == "primary"
