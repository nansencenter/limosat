import torch

from experiments.pilot_learned_sar_features import dedode_dual_softmax_indices


def test_dedode_dual_softmax_recovers_mutual_identity_matches():
    source = torch.eye(3)
    target = torch.eye(3)

    indices, scores = dedode_dual_softmax_indices(
        source, target, inverse_temperature=20.0, threshold=0.1
    )

    assert indices.tolist() == [[0, 0], [1, 1], [2, 2]]
    assert torch.all(scores > 0.1)
