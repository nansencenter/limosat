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


def test_loss_targeted_pairs_nominate_shortest_and_drop_redundant_pairs():
    from limosat.pair_queue import loss_targeted_pairs

    selected = loss_targeted_pairs([
        ("long", 72 * 3600.0, ["a", "b"]),
        ("short", 12 * 3600.0, ["a"]),
        ("middle", 24 * 3600.0, ["a", "b"]),
        ("only_c", 96 * 3600.0, ["c"]),
    ])
    # a -> short, b -> middle, c -> only_c; long adds no new loss.
    assert selected == frozenset({"short", "middle", "only_c"})


def test_loss_targeted_pairs_break_ties_by_pair_id_and_reject_bad_elapsed():
    import pytest
    from limosat.pair_queue import loss_targeted_pairs

    assert loss_targeted_pairs([("b", 1.0, ["x"]), ("a", 1.0, ["x"])]) == frozenset({"a"})
    with pytest.raises(ValueError):
        loss_targeted_pairs([("a", 0.0, ["x"])])
