"""Deterministic synthetic associative-memory datasets for Phases 2 and 3."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

KeyRegime = Literal[
    "orthogonal",
    "random_gaussian",
    "correlated",
    "near_collinear",
    "duplicate",
]
ValueRegime = Literal["gaussian", "one_hot", "binary"]

KEY_REGIMES: tuple[KeyRegime, ...] = (
    "orthogonal",
    "random_gaussian",
    "correlated",
    "near_collinear",
    "duplicate",
)
VALUE_REGIMES: tuple[ValueRegime, ...] = ("gaussian", "one_hot", "binary")


def normalize_rows(tensor: Tensor) -> Tensor:
    return tensor / torch.linalg.vector_norm(tensor, dim=1, keepdim=True).clamp_min(
        torch.finfo(tensor.dtype).tiny
    )


def make_keys(
    regime: KeyRegime,
    associations: int,
    dimension: int,
    *,
    generator: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
    correlation: float = 0.8,
    near_collinear_noise: float = 1e-3,
) -> Tensor:
    if associations < 1 or dimension < 1:
        raise ValueError("associations and dimension must be positive")
    if not 0.0 <= correlation < 1.0:
        raise ValueError("correlation must be in [0,1)")
    if near_collinear_noise <= 0:
        raise ValueError("near_collinear_noise must be positive")

    def randn(*shape: int) -> Tensor:
        return torch.randn(
            *shape, generator=generator, device=device, dtype=dtype
        )

    if regime == "orthogonal":
        if associations <= dimension:
            # Columns of Q are orthonormal; rows are the stored keys.
            return torch.linalg.qr(randn(dimension, associations), mode="reduced").Q.mT
        return normalize_rows(randn(associations, dimension))
    if regime == "random_gaussian":
        return normalize_rows(randn(associations, dimension))
    if regime == "correlated":
        common = normalize_rows(randn(1, dimension))
        keys = math.sqrt(correlation) * common + math.sqrt(1.0 - correlation) * randn(
            associations, dimension
        ) / math.sqrt(dimension)
        return normalize_rows(keys)
    if regime == "near_collinear":
        common = normalize_rows(randn(1, dimension))
        return normalize_rows(
            common + near_collinear_noise * randn(associations, dimension)
        )
    if regime == "duplicate":
        unique_count = max(1, (associations + 1) // 2)
        unique = normalize_rows(randn(unique_count, dimension))
        return unique[torch.arange(associations, device=device) % unique_count]
    raise ValueError(f"unknown key regime: {regime}")


def make_values(
    regime: ValueRegime,
    associations: int,
    dimension: int,
    *,
    generator: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    if associations < 1 or dimension < 1:
        raise ValueError("associations and dimension must be positive")
    if regime == "gaussian":
        return torch.randn(
            associations,
            dimension,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    if regime == "one_hot":
        indices = torch.randint(
            dimension,
            (associations,),
            generator=generator,
            device=device,
        )
        return torch.nn.functional.one_hot(indices, num_classes=dimension).to(dtype)
    if regime == "binary":
        bits = torch.randint(
            0,
            2,
            (associations, dimension),
            generator=generator,
            device=device,
        )
        return (2 * bits.to(dtype) - 1) / math.sqrt(dimension)
    raise ValueError(f"unknown value regime: {regime}")


def dataset_seed(
    base_seed: int,
    dimension: int,
    associations: int,
    key_regime_index: int,
    value_regime_index: int = 0,
) -> int:
    """Stable collision-free-enough integer mixing for experiment generators."""

    modulus = 2**63 - 1
    return (
        base_seed * 1_000_003
        + dimension * 10_007
        + associations * 101
        + key_regime_index * 17
        + value_regime_index
    ) % modulus

