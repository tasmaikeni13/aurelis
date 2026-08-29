from __future__ import annotations

import pytest
import torch

from aurelis import consume, initial_state, occurrence_partition


@pytest.mark.parametrize("window", [1, 2, 5])
@pytest.mark.parametrize("length", [0, 1, 2, 5, 11])
def test_occurrence_ids_are_disjoint_and_exhaustive(window: int, length: int) -> None:
    state = initial_state(2, 3, 4, 2, window)
    for step in range(length):
        key = torch.randn(2, 3, 4)
        value = torch.randn(2, 3, 2)
        evidence = torch.rand(2, 3) + 0.1
        previous_precision = state.precision.clone()
        previous_ids = state.cache_ids.clone()
        next_state = consume(state, key, value, evidence, occurrence_id=step)
        assert torch.equal(state.precision, previous_precision)
        assert torch.equal(state.cache_ids, previous_ids)
        state = next_state

        remote, recent = occurrence_partition(state)
        assert set(remote).isdisjoint(recent)
        assert remote + recent == tuple(range(step + 1))
        assert len(recent) <= window


def test_ring_handoff_updates_exact_statistics() -> None:
    state = initial_state(1, 2, 3, 4, 2, prior=0.25)
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []
    for step in range(8):
        key = torch.randn(1, 2, 3)
        value = torch.randn(1, 2, 4)
        beta = torch.rand(1, 2) + 0.2
        keys.append(key)
        values.append(value)
        betas.append(beta)
        state = consume(state, key, value, beta)
        remote_count = max(0, step + 1 - 2)
        expected_p = 0.25 * torch.eye(3).view(1, 1, 3, 3).expand(1, 2, 3, 3)
        expected_c = torch.zeros(1, 2, 4, 3)
        for index in range(remote_count):
            expected_p = expected_p + torch.einsum(
                "bh,bhi,bhj->bhij", betas[index], keys[index], keys[index]
            )
            expected_c = expected_c + torch.einsum(
                "bh,bhv,bhd->bhvd", betas[index], values[index], keys[index]
            )
        torch.testing.assert_close(state.precision, expected_p)
        torch.testing.assert_close(state.cross, expected_c)
        torch.testing.assert_close(state.factor @ state.factor.mT, state.precision)
