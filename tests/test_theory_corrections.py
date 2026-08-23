from __future__ import annotations

import math

import torch

from csm import FP64GaussMarkovMemory

DTYPE = torch.float64


def test_unwritten_direction_variance_includes_query_norm() -> None:
    memory = FP64GaussMarkovMemory(3, 1, epsilon=0.25)
    query = torch.tensor([3.0, 4.0, 0.0], dtype=DTYPE)
    observed = memory.confidence(memory.initial_state(), query)
    expected = query.square().sum() / memory.epsilon
    torch.testing.assert_close(observed, expected)


def test_equal_precision_duplicate_smoothing_matches_csm_mean_limit() -> None:
    memory = FP64GaussMarkovMemory(2, 1, epsilon=1e-12)
    key = torch.tensor([1.0, 0.0], dtype=DTYPE)
    keys = key.expand(4, -1)
    values = torch.tensor([[1.0], [5.0], [-2.0], [4.0]], dtype=DTYPE)
    state = memory.run(keys, values, torch.ones(4, dtype=DTYPE), torch.ones(4, dtype=DTYPE))
    ordinary_mean = values.mean(0)
    torch.testing.assert_close(
        memory.read(state, key), ordinary_mean, rtol=1e-11, atol=1e-11
    )


def test_precision_weighting_beats_evidence_blind_mean_when_precisions_differ() -> None:
    beta = torch.tensor([1.0, 4.0, 16.0], dtype=DTYPE)
    optimal_variance = 1.0 / beta.sum()
    evidence_blind_variance = (1.0 / beta).sum() / beta.numel() ** 2
    assert evidence_blind_variance > optimal_variance


def test_normalized_values_do_not_imply_ricochet_contraction() -> None:
    epsilon = 0.1
    memory = FP64GaussMarkovMemory(2, 2, epsilon=epsilon)
    keys = torch.eye(2, dtype=DTYPE)
    # Both normalized value columns point to the same successor.
    values = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=DTYPE)
    state = memory.run(
        keys, values, torch.ones(2, dtype=DTYPE), torch.ones(2, dtype=DTYPE)
    )
    operator = state.C @ torch.linalg.solve(
        memory.system_matrix(state), torch.eye(2, dtype=DTYPE)
    )
    observed_norm = torch.linalg.matrix_norm(operator, ord=2).item()
    expected_norm = math.sqrt(2.0) / (1.0 + epsilon)
    assert math.isclose(observed_norm, expected_norm, rel_tol=1e-12)
    assert observed_norm > 1.0

