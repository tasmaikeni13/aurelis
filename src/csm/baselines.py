"""Small, auditable associative-memory baselines for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class HebbianState:
    C: Tensor


@dataclass(frozen=True)
class ExplicitPairState:
    keys: Tensor
    values: Tensor


@dataclass(frozen=True)
class LinearAttentionState:
    C: Tensor
    normalizer: Tensor


@dataclass(frozen=True)
class LeastSquaresState:
    key_pseudoinverse: Tensor
    values: Tensor


def _validate_pairs(keys: Tensor, values: Tensor) -> None:
    if keys.ndim != 2 or values.ndim != 2:
        raise ValueError("keys and values must be matrices")
    if keys.shape[0] != values.shape[0]:
        raise ValueError("keys and values must have the same association count")
    if keys.dtype != values.dtype or keys.device != values.device:
        raise TypeError("keys and values must share dtype and device")


def hebbian_state(keys: Tensor, values: Tensor) -> HebbianState:
    """Store ``C = sum_i v_i k_i^T``."""

    _validate_pairs(keys, values)
    return HebbianState(C=values.mT @ keys)


def hebbian_read_many(state: HebbianState, queries: Tensor) -> Tensor:
    return queries @ state.C.mT


def explicit_pair_state(keys: Tensor, values: Tensor) -> ExplicitPairState:
    _validate_pairs(keys, values)
    return ExplicitPairState(keys=keys, values=values)


def softmax_read_many(
    state: ExplicitPairState, queries: Tensor, temperature: float
) -> tuple[Tensor, Tensor]:
    """Normalized dot-product smoothing over explicitly retained pairs."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = queries @ state.keys.mT / temperature
    weights = torch.softmax(logits, dim=-1)
    return weights @ state.values, weights


def positive_feature(tensor: Tensor) -> Tensor:
    """The positive ``ELU(x)+1`` feature map used by the linear baseline."""

    return F.elu(tensor) + 1.0


def linear_attention_state(keys: Tensor, values: Tensor) -> LinearAttentionState:
    """Store positive-feature numerator and normalizer sufficient statistics."""

    _validate_pairs(keys, values)
    features = positive_feature(keys)
    return LinearAttentionState(
        C=values.mT @ features,
        normalizer=features.sum(dim=0),
    )


def linear_attention_read_many(
    state: LinearAttentionState, queries: Tensor
) -> Tensor:
    query_features = positive_feature(queries)
    numerator = query_features @ state.C.mT
    denominator = query_features @ state.normalizer
    return numerator / denominator.unsqueeze(-1).clamp_min(
        torch.finfo(queries.dtype).tiny
    )


def least_squares_state(keys: Tensor, values: Tensor) -> LeastSquaresState:
    """Precompute the Moore–Penrose oracle for minimum-norm coefficients."""

    _validate_pairs(keys, values)
    return LeastSquaresState(
        key_pseudoinverse=torch.linalg.pinv(keys),
        values=values,
    )


def least_squares_read_many(state: LeastSquaresState, queries: Tensor) -> Tensor:
    coefficients = queries @ state.key_pseudoinverse
    return coefficients @ state.values


def csm_state_bytes(d_key: int, d_value: int, element_size: int = 8) -> int:
    return element_size * (d_key * d_key + d_value * d_key)


def hebbian_state_bytes(d_key: int, d_value: int, element_size: int = 8) -> int:
    return element_size * d_key * d_value


def linear_attention_state_bytes(
    d_key: int, d_value: int, element_size: int = 8
) -> int:
    return element_size * (d_key * d_value + d_key)


def explicit_pair_state_bytes(
    associations: int, d_key: int, d_value: int, element_size: int = 8
) -> int:
    return element_size * associations * (d_key + d_value)


def maximum_pairs_for_budget(
    budget_bytes: int, d_key: int, d_value: int, element_size: int = 8
) -> int:
    bytes_per_pair = element_size * (d_key + d_value)
    return max(0, budget_bytes // bytes_per_pair)


def evenly_spaced_subset(count: int, retained: int, device: torch.device) -> Tensor:
    """Deterministically retain associations across the complete write order."""

    if count < 1 or retained < 1:
        raise ValueError("count and retained must be positive")
    if retained >= count:
        return torch.arange(count, device=device)
    return torch.div(
        torch.arange(retained, device=device) * count,
        retained,
        rounding_mode="floor",
    )


def estimated_flops_per_query(
    method: str, d_key: int, d_value: int, associations: int
) -> int:
    """Leading-operation estimate with all prepared state/factors excluded."""

    if method == "csm":
        return 2 * d_key * d_key + 2 * d_key * d_value
    if method == "hebbian":
        return 2 * d_key * d_value
    if method == "linear_attention":
        return 2 * d_key * d_value + 3 * d_key + d_value
    if method == "softmax":
        return 2 * associations * (d_key + d_value) + 5 * associations
    if method == "least_squares":
        return 2 * associations * (d_key + d_value)
    raise ValueError(f"unknown method: {method}")
