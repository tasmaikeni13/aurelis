from __future__ import annotations

import pytest
import torch

from csm import FP64GaussMarkovMemory, KEY_REGIMES, VALUE_REGIMES, make_keys, make_values

DTYPE = torch.float64


def test_vectorized_undiscounted_state_matches_sequential_writes() -> None:
    memory = FP64GaussMarkovMemory(8, 5, epsilon=1e-4)
    keys = torch.randn(13, 8, dtype=DTYPE)
    values = torch.randn(13, 5, dtype=DTYPE)
    beta = 0.1 + torch.rand(13, dtype=DTYPE)
    decay = torch.ones(13, dtype=DTYPE)
    sequential = memory.run(keys, values, beta, decay)
    vectorized = memory.undiscounted_state(keys, values, beta)
    torch.testing.assert_close(vectorized.S, sequential.S, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(vectorized.C, sequential.C, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("solve_method", ["cholesky", "solve"])
def test_batched_reads_match_single_query_path(solve_method: str) -> None:
    memory = FP64GaussMarkovMemory(6, 4, epsilon=3e-3, solve_method=solve_method)
    keys = torch.randn(9, 6, dtype=DTYPE)
    values = torch.randn(9, 4, dtype=DTYPE)
    state = memory.undiscounted_state(keys, values)
    queries = torch.randn(7, 6, dtype=DTYPE)
    reads, uncertainty = memory.read_many_with_confidence(state, queries)
    expected_reads = torch.stack([memory.read(state, query) for query in queries])
    expected_uncertainty = torch.stack(
        [memory.confidence(state, query) for query in queries]
    )
    torch.testing.assert_close(reads, expected_reads, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(
        uncertainty, expected_uncertainty, rtol=2e-13, atol=2e-13
    )


@pytest.mark.parametrize("regime", KEY_REGIMES)
def test_key_generators_are_normalized_and_deterministic(regime: str) -> None:
    first = make_keys(
        regime,
        6,
        8,
        generator=torch.Generator().manual_seed(11),
        device="cpu",
    )
    second = make_keys(
        regime,
        6,
        8,
        generator=torch.Generator().manual_seed(11),
        device="cpu",
    )
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        torch.linalg.vector_norm(first, dim=1), torch.ones(6, dtype=DTYPE)
    )


def test_orthogonal_and_duplicate_regime_invariants() -> None:
    orthogonal = make_keys(
        "orthogonal",
        6,
        8,
        generator=torch.Generator().manual_seed(12),
        device="cpu",
    )
    torch.testing.assert_close(orthogonal @ orthogonal.mT, torch.eye(6, dtype=DTYPE))
    duplicate = make_keys(
        "duplicate",
        6,
        8,
        generator=torch.Generator().manual_seed(12),
        device="cpu",
    )
    assert torch.unique(duplicate, dim=0).shape[0] < duplicate.shape[0]


@pytest.mark.parametrize("regime", VALUE_REGIMES)
def test_value_generators_have_expected_shape_and_are_finite(regime: str) -> None:
    values = make_values(
        regime,
        7,
        5,
        generator=torch.Generator().manual_seed(13),
        device="cpu",
    )
    assert values.shape == (7, 5)
    assert torch.isfinite(values).all()
    if regime == "one_hot":
        torch.testing.assert_close(values.sum(1), torch.ones(7, dtype=DTYPE))

