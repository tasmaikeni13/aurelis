"""Immutable sequential decode with exact delayed ring-buffer handoff."""

from __future__ import annotations

import torch
from torch import Tensor

from .functional import aurelis_read
from .types import ReadOutput, StreamingState


def initial_state(
    batch: int,
    heads: int,
    d_key: int,
    d_value: int,
    window: int,
    *,
    prior: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> StreamingState:
    if min(batch, heads, d_key, d_value, window) <= 0:
        raise ValueError("batch, heads, dimensions, and window must be positive")
    if prior <= 0:
        raise ValueError("prior must be strictly positive")
    eye = torch.eye(d_key, dtype=dtype, device=device)
    precision = eye.expand(batch, heads, d_key, d_key).clone() * prior
    cross = torch.zeros(batch, heads, d_value, d_key, dtype=dtype, device=device)
    factor = torch.linalg.cholesky(precision)
    return StreamingState(
        precision=precision,
        cross=cross,
        factor=factor,
        cache_keys=torch.zeros(batch, heads, window, d_key, dtype=dtype, device=device),
        cache_values=torch.zeros(batch, heads, window, d_value, dtype=dtype, device=device),
        cache_evidence=torch.zeros(batch, heads, window, dtype=dtype, device=device),
        cache_ids=torch.full((window,), -1, dtype=torch.long, device=device),
        cache_start=0,
        cache_size=0,
        remote_ids=(),
        next_id=0,
    )

def _validate_item(state: StreamingState, key: Tensor, value: Tensor, evidence: Tensor) -> None:
    if key.shape != state.cache_keys.shape[:2] + state.cache_keys.shape[-1:]:
        raise ValueError("key must have shape [batch, heads, d_key]")
    if value.shape != state.cache_values.shape[:2] + state.cache_values.shape[-1:]:
        raise ValueError("value must have shape [batch, heads, d_value]")
    if evidence.shape != state.cache_evidence.shape[:2]:
        raise ValueError("evidence must have shape [batch, heads]")
    if bool(torch.any(evidence <= 0).detach().cpu()):
        raise ValueError("evidence precision must be strictly positive")


def consume(
    state: StreamingState,
    key: Tensor,
    value: Tensor,
    evidence: Tensor,
    *,
    occurrence_id: int | None = None,
) -> StreamingState:
    """Return a new state after consuming one occurrence exactly once."""

    _validate_item(state, key, value, evidence)
    item_id = state.next_id if occurrence_id is None else occurrence_id
    if item_id != state.next_id:
        raise ValueError(f"expected occurrence_id {state.next_id}, got {item_id}")

    keys = state.cache_keys.clone()
    values = state.cache_values.clone()
    betas = state.cache_evidence.clone()
    ids = state.cache_ids.clone()
    precision = state.precision
    cross = state.cross
    remote_ids = state.remote_ids
    start = state.cache_start
    size = state.cache_size
    window = keys.shape[-2]

    if size < window:
        write_index = (start + size) % window
        new_size = size + 1
        new_start = start
    else:
        write_index = start
        evicted_key = keys[:, :, write_index, :]
        evicted_value = values[:, :, write_index, :]
        evicted_evidence = betas[:, :, write_index]
        precision = precision + torch.einsum(
            "bh,bhi,bhj->bhij", evicted_evidence, evicted_key, evicted_key
        )
        cross = cross + torch.einsum(
            "bh,bhv,bhd->bhvd", evicted_evidence, evicted_value, evicted_key
        )
        remote_ids = remote_ids + (int(ids[write_index].detach().cpu()),)
        new_start = (start + 1) % window
        new_size = size

    keys[:, :, write_index, :] = key
    values[:, :, write_index, :] = value
    betas[:, :, write_index] = evidence
    ids[write_index] = item_id
    # A fresh factorization is the Phase 0 stability oracle. Later optimized
    # paths may replace it with rank-one cholupdate plus periodic refactors.
    factor = torch.linalg.cholesky(precision)
    return StreamingState(
        precision=precision,
        cross=cross,
        factor=factor,
        cache_keys=keys,
        cache_values=values,
        cache_evidence=betas,
        cache_ids=ids,
        cache_start=new_start,
        cache_size=new_size,
        remote_ids=remote_ids,
        next_id=state.next_id + 1,
    )


def chronological_cache(state: StreamingState) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Materialize the live ring slots from oldest to newest."""

    window = state.cache_keys.shape[-2]
    indices = [((state.cache_start + offset) % window) for offset in range(state.cache_size)]
    index = torch.tensor(indices, dtype=torch.long, device=state.cache_keys.device)
    return (
        state.cache_keys.index_select(-2, index),
        state.cache_values.index_select(-2, index),
        state.cache_evidence.index_select(-1, index),
        state.cache_ids.index_select(0, index),
    )


def occurrence_partition(state: StreamingState) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return remote and recent occurrence IDs for direct invariant checks."""

    *_, recent_ids = chronological_cache(state)
    return state.remote_ids, tuple(int(value) for value in recent_ids.detach().cpu())


def read(
    state: StreamingState,
    query: Tensor,
    *,
    temperature: Tensor | float = 1.0,
    episodic_responsibility: Tensor | float = 0.0,
) -> ReadOutput:
    keys, values, evidence, _ = chronological_cache(state)
    return aurelis_read(
        state.precision,
        state.cross,
        keys,
        values,
        evidence,
        query,
        temperature=temperature,
        episodic_responsibility=episodic_responsibility,
        factor=state.factor,
    )
