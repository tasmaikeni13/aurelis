from __future__ import annotations

import torch

from csm import FP64GaussMarkovMemory, direct_inverse_oracle, recompute_state

DTYPE = torch.float64


def test_single_observation_has_closed_form_read_and_confidence() -> None:
    epsilon = 0.125
    beta_value = 3.0
    memory = FP64GaussMarkovMemory(4, 3, epsilon=epsilon)
    key = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE)
    value = torch.tensor([2.0, -1.0, 4.0], dtype=DTYPE)
    state = memory.write(memory.initial_state(), key, value, beta_value, 1.0)

    expected_read = value * beta_value / (beta_value + epsilon)
    expected_confidence = torch.tensor(1.0 / (beta_value + epsilon), dtype=DTYPE)
    torch.testing.assert_close(memory.read(state, key), expected_read)
    torch.testing.assert_close(memory.confidence(state, key), expected_confidence)


def test_repeated_keys_return_precision_weighted_ridge_mean() -> None:
    epsilon = 1e-5
    memory = FP64GaussMarkovMemory(2, 1, epsilon=epsilon)
    keys = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=DTYPE)
    values = torch.tensor([[1.0], [5.0], [-2.0]], dtype=DTYPE)
    beta = torch.tensor([1.0, 2.0, 4.0], dtype=DTYPE)
    decay = torch.ones(3, dtype=DTYPE)
    state = memory.run(keys, values, beta, decay)

    weighted_sum = (beta * values.squeeze(-1)).sum()
    expected = weighted_sum / (beta.sum() + epsilon)
    torch.testing.assert_close(memory.read(state, keys[0]).squeeze(), expected)


def test_nearly_collinear_keys_remain_consistent_with_oracle() -> None:
    epsilon = 1e-10
    memory = FP64GaussMarkovMemory(2, 2, epsilon=epsilon)
    delta = 1e-7
    keys = torch.tensor([[1.0, 0.0], [1.0, delta]], dtype=DTYPE)
    keys = keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)
    values = torch.tensor([[1.0, -1.0], [-2.0, 3.0]], dtype=DTYPE)
    beta = torch.tensor([0.7, 4.2], dtype=DTYPE)
    decay = torch.ones(2, dtype=DTYPE)
    state = memory.run(keys, values, beta, decay)

    for query in keys:
        oracle_read, _ = direct_inverse_oracle(state, query, epsilon)
        torch.testing.assert_close(
            memory.read(state, query), oracle_read, rtol=2e-6, atol=2e-6
        )
    assert torch.linalg.eigvalsh(state.S).min() >= -1e-15


def test_beta_zero_skips_write_when_decay_is_one() -> None:
    memory = FP64GaussMarkovMemory(3, 2)
    state = memory.initial_state()
    key = torch.randn(3, dtype=DTYPE)
    value = torch.randn(2, dtype=DTYPE)
    skipped = memory.write(state, key, value, beta=0.0, decay=1.0)
    torch.testing.assert_close(skipped.S, state.S)
    torch.testing.assert_close(skipped.C, state.C)


def test_lambda_one_is_undiscounted_sum() -> None:
    memory = FP64GaussMarkovMemory(3, 2)
    keys = torch.randn(5, 3, dtype=DTYPE)
    values = torch.randn(5, 2, dtype=DTYPE)
    beta = torch.rand(5, dtype=DTYPE)
    decay = torch.ones(5, dtype=DTYPE)
    state = memory.run(keys, values, beta, decay)
    expected_S = torch.einsum("t,ti,tj->ij", beta, keys, keys)
    expected_C = torch.einsum("t,tv,tk->vk", beta, values, keys)
    torch.testing.assert_close(state.S, expected_S)
    torch.testing.assert_close(state.C, expected_C)


def test_lambda_less_than_one_applies_suffix_decay() -> None:
    memory = FP64GaussMarkovMemory(2, 1)
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    values = torch.tensor([[2.0], [7.0]], dtype=DTYPE)
    beta = torch.ones(2, dtype=DTYPE)
    decay = torch.tensor([0.2, 0.25], dtype=DTYPE)
    state = memory.run(keys, values, beta, decay)
    expected_S = torch.diag(torch.tensor([0.25, 1.0], dtype=DTYPE))
    expected_C = torch.tensor([[0.5, 7.0]], dtype=DTYPE)
    torch.testing.assert_close(state.S, expected_S)
    torch.testing.assert_close(state.C, expected_C)


def test_tiny_epsilon_well_conditioned_basis_interpolates() -> None:
    epsilon = 1e-12
    memory = FP64GaussMarkovMemory(4, 4, epsilon=epsilon)
    keys = torch.eye(4, dtype=DTYPE)
    values = torch.randn(4, 4, dtype=DTYPE)
    beta = torch.ones(4, dtype=DTYPE)
    decay = torch.ones(4, dtype=DTYPE)
    state = memory.run(keys, values, beta, decay)
    for index in range(4):
        torch.testing.assert_close(
            memory.read(state, keys[index]), values[index], rtol=2e-12, atol=2e-12
        )


def test_very_large_beta_is_finite_and_matches_recomputation() -> None:
    memory = FP64GaussMarkovMemory(4, 2, epsilon=1e-6)
    keys = torch.randn(3, 4, dtype=DTYPE)
    values = torch.randn(3, 2, dtype=DTYPE)
    beta = torch.tensor([1e150, 1e149, 1e148], dtype=DTYPE)
    decay = torch.tensor([1.0, 0.9, 0.8], dtype=DTYPE)
    state = memory.run(keys, values, beta, decay)
    recomputed = recompute_state(keys, values, beta, decay)
    assert torch.isfinite(state.S).all()
    assert torch.isfinite(state.C).all()
    torch.testing.assert_close(state.S, recomputed.S, rtol=2e-15, atol=0)
    torch.testing.assert_close(state.C, recomputed.C, rtol=2e-15, atol=0)


def test_zero_values_produce_zero_reads() -> None:
    memory = FP64GaussMarkovMemory(5, 3)
    keys = torch.randn(7, 5, dtype=DTYPE)
    values = torch.zeros(7, 3, dtype=DTYPE)
    beta = torch.rand(7, dtype=DTYPE)
    decay = 0.8 + 0.2 * torch.rand(7, dtype=DTYPE)
    state = memory.run(keys, values, beta, decay)
    query = torch.randn(5, dtype=DTYPE)
    torch.testing.assert_close(memory.read(state, query), torch.zeros(3, dtype=DTYPE))


def test_evidence_reduces_variance_along_written_direction() -> None:
    memory = FP64GaussMarkovMemory(3, 1, epsilon=0.2)
    query = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)
    value = torch.tensor([4.0], dtype=DTYPE)
    empty = memory.initial_state()
    once = memory.write(empty, query, value, beta=1.0, decay=1.0)
    twice = memory.write(once, query, value, beta=1.0, decay=1.0)
    assert memory.confidence(empty, query) > memory.confidence(once, query)
    assert memory.confidence(once, query) > memory.confidence(twice, query)

