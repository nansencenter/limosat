from experiments.analyze_full70_graph_vs_one_step import outcome_label


def test_outcome_label_separates_descriptor_and_graph_failure():
    assert outcome_label(1000.0, 1000.0) == "both_within_2km"
    assert outcome_label(1000.0, 3000.0) == "one_step_succeeds_graph_fails"
    assert outcome_label(3000.0, 1000.0) == "graph_rescues_one_step_failure"
    assert outcome_label(3000.0, 3000.0) == "both_fail_2km"
