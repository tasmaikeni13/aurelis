from __future__ import annotations

import torch

from csm import (
    explicit_pair_state,
    hebbian_read_many,
    hebbian_state,
    least_squares_read_many,
    least_squares_state,
    linear_attention_read_many,
    linear_attention_state,
    maximum_pairs_for_budget,
    softmax_read_many,
)
from csm.baselines import csm_state_bytes, evenly_spaced_subset, positive_feature

DTYPE = torch.float64


def test_hebbian_baseline_is_declared_outer_product() -> None:
    keys = torch.randn(7, 5, dtype=DTYPE)
    values = torch.randn(7, 3, dtype=DTYPE)
    queries = torch.randn(4, 5, dtype=DTYPE)
    state = hebbian_state(keys, values)
    torch.testing.assert_close(state.C, values.mT @ keys)
    torch.testing.assert_close(hebbian_read_many(state, queries), queries @ state.C.mT)


def test_softmax_baseline_is_a_convex_combination() -> None:
    keys = torch.randn(6, 5, dtype=DTYPE)
    values = torch.randn(6, 4, dtype=DTYPE)
    queries = torch.randn(3, 5, dtype=DTYPE)
    reads, weights = softmax_read_many(explicit_pair_state(keys, values), queries, 0.2)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3, dtype=DTYPE))
    assert (weights >= 0).all()
    torch.testing.assert_close(reads, weights @ values)


def test_linear_attention_baseline_uses_positive_normalized_weights() -> None:
    keys = torch.randn(6, 5, dtype=DTYPE)
    values = torch.randn(6, 4, dtype=DTYPE)
    queries = torch.randn(3, 5, dtype=DTYPE)
    state = linear_attention_state(keys, values)
    reads = linear_attention_read_many(state, queries)
    query_features = positive_feature(queries)
    key_features = positive_feature(keys)
    raw = query_features @ key_features.mT
    weights = raw / raw.sum(dim=1, keepdim=True)
    assert (weights >= 0).all()
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3, dtype=DTYPE))
    torch.testing.assert_close(reads, weights @ values)


def test_least_squares_oracle_interpolates_independent_keys() -> None:
    keys = torch.linalg.qr(torch.randn(8, 5, dtype=DTYPE)).Q.mT
    values = torch.randn(5, 4, dtype=DTYPE)
    reads = least_squares_read_many(least_squares_state(keys, values), keys)
    torch.testing.assert_close(reads, values, rtol=1e-12, atol=1e-12)


def test_equal_budget_pair_capacity_and_subset() -> None:
    budget = csm_state_bytes(32, 8)
    assert maximum_pairs_for_budget(budget, 32, 8) == 32
    indices = evenly_spaced_subset(64, 32, torch.device("cpu"))
    assert indices.shape == (32,)
    assert torch.unique(indices).shape == indices.shape
    assert indices.min() == 0 and indices.max() < 64
