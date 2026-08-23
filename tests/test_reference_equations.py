from __future__ import annotations

import pytest
import torch

from csm import (
    FP64GaussMarkovMemory,
    GaussMarkovMemory,
    direct_inverse_oracle,
    recompute_state,
)

DTYPE = torch.float64


def random_problem(d_key: int, d_value: int, steps: int = 11):
    keys = torch.randn(steps, d_key, dtype=DTYPE)
    keys = keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)
    values = torch.randn(steps, d_value, dtype=DTYPE)
    beta = 0.05 + 2.0 * torch.rand(steps, dtype=DTYPE)
    decay = 0.7 + 0.3 * torch.rand(steps, dtype=DTYPE)
    query = torch.randn(d_key, dtype=DTYPE)
    return keys, values, beta, decay, query


@pytest.mark.parametrize("d_key", [2, 4, 8, 16, 32])
def test_sequential_matches_independent_recomputation(d_key: int) -> None:
    d_value = max(2, d_key // 2)
    memory = FP64GaussMarkovMemory(d_key, d_value, epsilon=3e-3)
    keys, values, beta, decay, query = random_problem(d_key, d_value)

    sequential = memory.run(keys, values, beta, decay)
    recomputed = recompute_state(keys, values, beta, decay)

    torch.testing.assert_close(sequential.S, recomputed.S, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(sequential.C, recomputed.C, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(
        memory.read(sequential, query),
        memory.read(recomputed, query),
        rtol=5e-13,
        atol=5e-13,
    )
    torch.testing.assert_close(
        memory.confidence(sequential, query),
        memory.confidence(recomputed, query),
        rtol=5e-13,
        atol=5e-13,
    )


@pytest.mark.parametrize("d_key", [2, 4, 8, 16, 32])
def test_cholesky_solve_and_tiny_inverse_oracle_agree(d_key: int) -> None:
    d_value = 3
    epsilon = 1e-2
    memory = FP64GaussMarkovMemory(d_key, d_value, epsilon=epsilon)
    keys, values, beta, decay, query = random_problem(d_key, d_value)
    state = memory.run(keys, values, beta, decay)

    read, confidence = memory.read_with_confidence(state, query)
    oracle_read, oracle_confidence = direct_inverse_oracle(state, query, epsilon)

    torch.testing.assert_close(read, oracle_read, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(
        confidence, oracle_confidence, rtol=2e-12, atol=2e-12
    )


def test_direct_inverse_oracle_refuses_non_tiny_matrix() -> None:
    memory = FP64GaussMarkovMemory(33, 2)
    state = memory.initial_state()
    with pytest.raises(ValueError, match="limited"):
        direct_inverse_oracle(state, torch.ones(33, dtype=DTYPE), memory.epsilon)


def test_cholesky_and_general_solve_paths_agree() -> None:
    keys, values, beta, decay, query = random_problem(8, 5)
    cholesky = FP64GaussMarkovMemory(8, 5, epsilon=7e-4)
    general = FP64GaussMarkovMemory(8, 5, epsilon=7e-4, solve_method="solve")
    state = cholesky.run(keys, values, beta, decay)
    torch.testing.assert_close(
        cholesky.read(state, query), general.read(state, query), rtol=2e-12, atol=2e-12
    )
    torch.testing.assert_close(
        cholesky.confidence(state, query),
        general.confidence(state, query),
        rtol=2e-12,
        atol=2e-12,
    )


def test_random_epsilon_values_preserve_recurrence_agreement() -> None:
    for seed in range(5):
        torch.manual_seed(10_000 + seed)
        log10_epsilon = -8.0 + 7.0 * torch.rand((), dtype=DTYPE).item()
        epsilon = 10.0**log10_epsilon
        memory = FP64GaussMarkovMemory(8, 3, epsilon=epsilon)
        keys, values, beta, decay, query = random_problem(8, 3, steps=9)
        sequential = memory.run(keys, values, beta, decay)
        recomputed = recompute_state(keys, values, beta, decay)
        torch.testing.assert_close(sequential.S, recomputed.S, rtol=2e-14, atol=2e-14)
        torch.testing.assert_close(sequential.C, recomputed.C, rtol=2e-14, atol=2e-14)
        torch.testing.assert_close(
            memory.read(sequential, query),
            memory.read(recomputed, query),
            rtol=2e-10,
            atol=2e-10,
        )


@pytest.mark.parametrize("d_key", [2, 4, 8, 16, 32])
def test_S_is_numerically_positive_semidefinite(d_key: int) -> None:
    memory = FP64GaussMarkovMemory(d_key, 4)
    keys, values, beta, decay, _ = random_problem(d_key, 4, steps=29)
    state = memory.run(keys, values, beta, decay)

    symmetry_error = torch.linalg.matrix_norm(state.S - state.S.mT).item()
    minimum_eigenvalue = torch.linalg.eigvalsh(state.S).min().item()
    scale = max(torch.linalg.matrix_norm(state.S, ord=2).item(), 1.0)
    assert symmetry_error <= 1e-13 * scale
    assert minimum_eigenvalue >= -1e-13 * scale


def test_fp64_reference_is_explicit() -> None:
    memory = FP64GaussMarkovMemory(4, 3)
    state = memory.initial_state()
    assert memory.dtype is torch.float64
    assert state.S.dtype is torch.float64
    assert state.C.dtype is torch.float64
    with pytest.raises(TypeError, match="torch.float64"):
        memory.write(
            state,
            torch.ones(4, dtype=torch.float32),
            torch.ones(3, dtype=torch.float32),
            1.0,
            1.0,
        )


def test_generic_reference_can_run_fp32_for_precision_experiments() -> None:
    memory = GaussMarkovMemory(4, 3, dtype=torch.float32)
    state = memory.initial_state()
    key = torch.ones(4, dtype=torch.float32) / 2
    value = torch.ones(3, dtype=torch.float32)
    written = memory.write(state, key, value, 1.0, 1.0)
    assert memory.read(written, key).dtype is torch.float32
