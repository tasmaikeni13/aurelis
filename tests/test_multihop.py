from __future__ import annotations

import torch

from csm import (
    FP64GaussMarkovMemory,
    chase_indices,
    csm_chained_reads,
    explicit_pair_state,
    nearest_code,
    prepared_read_operator,
    softmax_chained_reads,
)


DTYPE = torch.float64


def test_chained_csm_reads_use_one_unchanged_state() -> None:
    dimension = 6
    codes = torch.eye(dimension, dtype=DTYPE)
    successors = torch.tensor([1, 2, 3, 4, 5, 0])
    values = codes[successors]
    memory = FP64GaussMarkovMemory(dimension, dimension, epsilon=1e-8)
    state = memory.undiscounted_state(codes, values)
    starts = codes[torch.tensor([0, 2, 5])]
    chain = csm_chained_reads(memory, state, starts, 4)
    expected = chase_indices(successors, torch.tensor([0, 2, 5]), 4)
    assert chain.queries.shape == (5, 3, dimension)
    assert chain.confidence is not None and chain.confidence.shape == (4, 3)
    for hop in range(1, 5):
        torch.testing.assert_close(
            nearest_code(codes, chain.queries[hop]), expected[hop]
        )


def test_repeated_softmax_needs_one_adaptive_access_per_hop() -> None:
    codes = torch.eye(5, dtype=DTYPE)
    successors = torch.tensor([1, 2, 3, 4, 0])
    starts = codes[[0]]
    chain = softmax_chained_reads(
        explicit_pair_state(codes, codes[successors]), starts, 4, 1e-3
    )
    expected = chase_indices(successors, torch.tensor([0]), 4)
    for hop in range(1, 5):
        assert nearest_code(codes, chain.queries[hop]).item() == expected[hop].item()


def test_operator_norm_exposes_many_to_one_amplification() -> None:
    codes = torch.eye(4, dtype=DTYPE)
    values = codes[torch.zeros(4, dtype=torch.long)]
    memory = FP64GaussMarkovMemory(4, 4, epsilon=1e-8)
    state = memory.undiscounted_state(codes, values)
    operator = prepared_read_operator(memory, state)
    assert torch.linalg.matrix_norm(operator, ord=2).item() > 1.9


def test_nearest_code_uses_direction_not_vector_norm() -> None:
    codes = torch.eye(3, dtype=DTYPE)
    vectors = torch.tensor([[0.0, 1e-12, 0.0], [0.0, 0.0, 9.0]], dtype=DTYPE)
    torch.testing.assert_close(nearest_code(codes, vectors), torch.tensor([1, 2]))
