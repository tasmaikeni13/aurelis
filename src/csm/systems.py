"""ROCm-friendly batched primitives for the affine CSM recurrence.

The Phase 1 implementation is intentionally scalar and transparent.  This
module changes evaluation order, batching, and precision policy only.  A token
is represented by the affine action ``x -> decay * x + update``; composing
those actions gives an associative scan for both sufficient statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AffineSummary:
    """An affine recurrence segment for batched, multi-head CSM state.

    ``decay`` has shape ``[..., segments]`` (or ``[...]`` for one summary),
    while ``S`` and ``C`` append matrix dimensions ``[d_k,d_k]`` and
    ``[d_v,d_k]`` respectively.
    """

    decay: Tensor
    S: Tensor
    C: Tensor


def _validate_inputs(
    keys: Tensor, values: Tensor, beta: Tensor, decay: Tensor
) -> tuple[int, int]:
    if keys.ndim < 2 or values.ndim != keys.ndim:
        raise ValueError("keys and values must have matching rank >= 2")
    if keys.shape[:-1] != values.shape[:-1]:
        raise ValueError("keys and values must share all non-feature dimensions")
    if beta.shape != keys.shape[:-1] or decay.shape != keys.shape[:-1]:
        raise ValueError("beta and decay must match keys without its feature axis")
    if not keys.dtype.is_floating_point or not values.dtype.is_floating_point:
        raise TypeError("keys and values must be floating point")
    return keys.shape[-1], values.shape[-1]


def token_summaries(
    keys: Tensor,
    values: Tensor,
    beta: Tensor,
    decay: Tensor,
    *,
    accumulation_dtype: torch.dtype = torch.float32,
) -> AffineSummary:
    """Construct one affine summary per token without reducing time."""

    _validate_inputs(keys, values, beta, decay)
    k = keys.to(accumulation_dtype)
    v = values.to(accumulation_dtype)
    b = beta.to(accumulation_dtype)
    lam = decay.to(accumulation_dtype)
    return AffineSummary(
        decay=lam,
        S=b[..., None, None] * k[..., :, None] * k[..., None, :],
        C=b[..., None, None] * v[..., :, None] * k[..., None, :],
    )


def compose_affine(earlier: AffineSummary, later: AffineSummary) -> AffineSummary:
    """Compose chronological segments: apply ``earlier`` and then ``later``."""

    if earlier.decay.shape != later.decay.shape:
        raise ValueError("summary decay shapes must match")
    if earlier.S.shape != later.S.shape or earlier.C.shape != later.C.shape:
        raise ValueError("summary state shapes must match")
    scale = later.decay[..., None, None]
    return AffineSummary(
        decay=later.decay * earlier.decay,
        S=scale * earlier.S + later.S,
        C=scale * earlier.C + later.C,
    )


def summarize_segment(
    keys: Tensor,
    values: Tensor,
    beta: Tensor,
    decay: Tensor,
    *,
    accumulation_dtype: torch.dtype = torch.float32,
) -> AffineSummary:
    """Vectorize the exact final summary over the penultimate (time) axis."""

    _validate_inputs(keys, values, beta, decay)
    k = keys.to(accumulation_dtype)
    v = values.to(accumulation_dtype)
    b = beta.to(accumulation_dtype)
    lam = decay.to(accumulation_dtype)
    steps = keys.shape[-2]
    if steps == 0:
        return AffineSummary(
            decay=lam.new_ones(lam.shape[:-1]),
            S=k.new_zeros((*k.shape[:-2], k.shape[-1], k.shape[-1])),
            C=k.new_zeros((*k.shape[:-2], v.shape[-1], k.shape[-1])),
        )
    inclusive_suffix = torch.cumprod(torch.flip(lam, (-1,)), dim=-1).flip(-1)
    exclusive_suffix = torch.cat(
        (inclusive_suffix[..., 1:], torch.ones_like(inclusive_suffix[..., :1])),
        dim=-1,
    )
    weights = b * exclusive_suffix
    S = torch.einsum("...td,...te,...t->...de", k, k, weights)
    C = torch.einsum("...tv,...td,...t->...vd", v, k, weights)
    return AffineSummary(decay=lam.prod(dim=-1), S=S, C=C)


def summarize_chunks(
    keys: Tensor,
    values: Tensor,
    beta: Tensor,
    decay: Tensor,
    chunk_size: int,
    *,
    accumulation_dtype: torch.dtype = torch.float32,
) -> AffineSummary:
    """Summarize fixed-size chunks and compose their affine actions."""

    d_key, d_value = _validate_inputs(keys, values, beta, decay)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    result = AffineSummary(
        decay=torch.ones(keys.shape[:-2], dtype=accumulation_dtype, device=keys.device),
        S=torch.zeros(
            (*keys.shape[:-2], d_key, d_key),
            dtype=accumulation_dtype,
            device=keys.device,
        ),
        C=torch.zeros(
            (*keys.shape[:-2], d_value, d_key),
            dtype=accumulation_dtype,
            device=keys.device,
        ),
    )
    for start in range(0, keys.shape[-2], chunk_size):
        stop = min(start + chunk_size, keys.shape[-2])
        current = summarize_segment(
            keys[..., start:stop, :],
            values[..., start:stop, :],
            beta[..., start:stop],
            decay[..., start:stop],
            accumulation_dtype=accumulation_dtype,
        )
        result = compose_affine(result, current)
    return result


def associative_prefix_scan(summary: AffineSummary) -> AffineSummary:
    """Inclusive Hillis--Steele scan over the last summary axis.

    This uses logarithmic launch depth.  It deliberately returns all prefix
    matrices, making its ``O(T d_k (d_k+d_v))`` activation cost explicit.
    """

    if summary.S.ndim != summary.decay.ndim + 2:
        raise ValueError("S must append two matrix axes to decay")
    if summary.C.ndim != summary.decay.ndim + 2:
        raise ValueError("C must append two matrix axes to decay")
    if summary.S.shape[:-2] != summary.decay.shape:
        raise ValueError("S prefix axes must match decay")
    if summary.C.shape[:-2] != summary.decay.shape:
        raise ValueError("C prefix axes must match decay")
    output = summary
    steps = summary.decay.shape[-1]
    stride = 1
    while stride < steps:
        left = AffineSummary(
            decay=output.decay[..., :-stride],
            S=output.S[..., :-stride, :, :],
            C=output.C[..., :-stride, :, :],
        )
        right = AffineSummary(
            decay=output.decay[..., stride:],
            S=output.S[..., stride:, :, :],
            C=output.C[..., stride:, :, :],
        )
        combined = compose_affine(left, right)
        output = AffineSummary(
            decay=torch.cat((output.decay[..., :stride], combined.decay), dim=-1),
            S=torch.cat((output.S[..., :stride, :, :], combined.S), dim=-3),
            C=torch.cat((output.C[..., :stride, :, :], combined.C), dim=-3),
        )
        stride *= 2
    return output


def prefix_states(
    keys: Tensor,
    values: Tensor,
    beta: Tensor,
    decay: Tensor,
    *,
    accumulation_dtype: torch.dtype = torch.float32,
    unit_decay_fast_path: bool = True,
) -> AffineSummary:
    """Return every inclusive prefix state.

    Unit decay admits a single cumulative sum and is the economical training
    path used in Phase 9.  General token-local decay uses the associative scan.
    """

    tokens = token_summaries(
        keys,
        values,
        beta,
        decay,
        accumulation_dtype=accumulation_dtype,
    )
    if unit_decay_fast_path and bool(torch.all(decay == 1)):
        return AffineSummary(
            decay=torch.cumprod(tokens.decay, dim=-1),
            S=torch.cumsum(tokens.S, dim=-3),
            C=torch.cumsum(tokens.C, dim=-3),
        )
    return associative_prefix_scan(tokens)


def read_prefix_states(
    summary: AffineSummary,
    queries: Tensor,
    epsilon: float,
    *,
    output_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Read all prefix states with batched fp32/fp64 Cholesky solves."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if queries.shape[:-1] != summary.decay.shape:
        raise ValueError("queries must match the summary prefix axes")
    if queries.shape[-1] != summary.S.shape[-1]:
        raise ValueError("query key dimension does not match S")
    identity = torch.eye(
        summary.S.shape[-1], dtype=summary.S.dtype, device=summary.S.device
    )
    system = summary.S + epsilon * identity
    query = queries.to(summary.S.dtype)
    factor = torch.linalg.cholesky(system)
    solved = torch.cholesky_solve(query.unsqueeze(-1), factor).squeeze(-1)
    reads = torch.einsum("...vk,...k->...v", summary.C, solved)
    uncertainty = torch.einsum("...k,...k->...", query, solved)
    dtype = output_dtype or queries.dtype
    return reads.to(dtype), uncertainty


def sequential_decode(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    beta: Tensor,
    decay: Tensor,
    epsilon: float,
    *,
    accumulation_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor, AffineSummary]:
    """Transparent batched inclusive write/read loop for timing and validation."""

    d_key, d_value = _validate_inputs(keys, values, beta, decay)
    if queries.shape != keys.shape:
        raise ValueError("queries must have the same shape as keys")
    prefix = keys.shape[:-2]
    S = torch.zeros((*prefix, d_key, d_key), dtype=accumulation_dtype, device=keys.device)
    C = torch.zeros((*prefix, d_value, d_key), dtype=accumulation_dtype, device=keys.device)
    cumulative_decay = torch.ones(prefix, dtype=accumulation_dtype, device=keys.device)
    reads: list[Tensor] = []
    uncertainties: list[Tensor] = []
    identity = torch.eye(d_key, dtype=accumulation_dtype, device=keys.device)
    k = keys.to(accumulation_dtype)
    v = values.to(accumulation_dtype)
    q = queries.to(accumulation_dtype)
    b = beta.to(accumulation_dtype)
    lam = decay.to(accumulation_dtype)
    for index in range(keys.shape[-2]):
        scale = lam[..., index, None, None]
        S = scale * S + b[..., index, None, None] * (
            k[..., index, :, None] * k[..., index, None, :]
        )
        C = scale * C + b[..., index, None, None] * (
            v[..., index, :, None] * k[..., index, None, :]
        )
        cumulative_decay = cumulative_decay * lam[..., index]
        factor = torch.linalg.cholesky(S + epsilon * identity)
        solved = torch.cholesky_solve(q[..., index, :, None], factor).squeeze(-1)
        reads.append(torch.einsum("...vk,...k->...v", C, solved))
        uncertainties.append(torch.einsum("...k,...k->...", q[..., index, :], solved))
    dtype = output_dtype or queries.dtype
    return (
        torch.stack(reads, dim=-2).to(dtype),
        torch.stack(uncertainties, dim=-1),
        AffineSummary(decay=cumulative_decay, S=S, C=C),
    )


def csm_state_bytes(
    batch_size: int,
    heads: int,
    d_key: int,
    d_value: int,
    *,
    bytes_per_element: int = 4,
) -> int:
    """Persistent recurrent-state bytes, excluding transient prefix tensors."""

    return batch_size * heads * d_key * (d_key + d_value) * bytes_per_element


def csm_leading_flops_per_token(heads: int, d_key: int, d_value: int) -> int:
    """Leading multiply/add count for one write plus one prepared read."""

    return heads * (2 * d_key * d_key + 4 * d_value * d_key)
