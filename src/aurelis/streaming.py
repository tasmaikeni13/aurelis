"""Stateful streaming processor and causal handoff for AURELIS."""

from __future__ import annotations

import random
from typing import Any, Literal, Optional

import torch
from torch import Tensor

from .archive import Archive, archive_reference_read
from .functional import bounded_read, gated_delta_update
from .types import DeltaState, ReadResult, StateSnapshot


def initial_state(
    d_key: int,
    d_value: int,
    window: int,
    dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
) -> DeltaState:
    """Create an empty initial DeltaState."""
    dt = dtype if dtype is not None else torch.get_default_dtype()
    return DeltaState(
        S=torch.zeros((d_value, d_key), dtype=dt, device=device),
        cache_keys=torch.zeros((window, d_key), dtype=dt, device=device),
        cache_values=torch.zeros((window, d_value), dtype=dt, device=device),
        cache_alphas=torch.ones((window,), dtype=dt, device=device),
        cache_betas=torch.ones((window,), dtype=dt, device=device),
        cache_ids=(),
        cache_start=0,
        cache_size=0,
        window=window,
        evicted_ids=(),
        next_id=0,
    )


def consume(
    state: DeltaState,
    key: Tensor,
    value: Tensor,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    occurrence_id: int | None = None,
    archive: Optional[Archive] = None,
) -> DeltaState:
    """Consume a newly observed key-value pair, updating the local ring cache and recurrence.

    If the cache is full (cache_size == window), the oldest observation is evicted
    and written to the recurrent state S exactly once. If archive is provided,
    the evicted observation is appended to the archive.
    """
    oid = occurrence_id if occurrence_id is not None else state.next_id
    next_next_id = max(state.next_id, oid + 1)

    new_S = state.S
    new_evicted_ids = state.evicted_ids

    # Check if an observation must be evicted from the window
    if state.cache_size == state.window:
        evict_idx = state.cache_start
        evicted_k = state.cache_keys[evict_idx]
        evicted_v = state.cache_values[evict_idx]
        evicted_a = state.cache_alphas[evict_idx]
        evicted_b = state.cache_betas[evict_idx]
        evicted_oid = state.cache_ids[0]

        # Update recurrent state S with the evicted association (Eq. 2)
        new_S = gated_delta_update(
            state.S, evicted_k, evicted_v, alpha=evicted_a, beta=evicted_b
        )
        new_evicted_ids = (*state.evicted_ids, evicted_oid)

        # Append to archive if enabled
        if archive is not None:
            archive.append(
                occurrence_id=evicted_oid,
                key=evicted_k,
                value=evicted_v,
                alpha=evicted_a,
                beta=evicted_b,
                position=evicted_oid,
            )

        # Slide ring buffer
        new_start = (state.cache_start + 1) % state.window
        new_size = state.window - 1
        new_cache_ids = state.cache_ids[1:]
    else:
        new_start = state.cache_start
        new_size = state.cache_size
        new_cache_ids = state.cache_ids

    # Append new observation to ring buffer
    insert_idx = (new_start + new_size) % state.window

    new_keys = state.cache_keys.clone()
    new_values = state.cache_values.clone()
    new_alphas = state.cache_alphas.clone()
    new_betas = state.cache_betas.clone()

    new_keys[insert_idx] = key
    new_values[insert_idx] = value
    new_alphas[insert_idx] = torch.as_tensor(alpha, dtype=state.S.dtype, device=state.S.device)
    new_betas[insert_idx] = torch.as_tensor(beta, dtype=state.S.dtype, device=state.S.device)

    return DeltaState(
        S=new_S,
        cache_keys=new_keys,
        cache_values=new_values,
        cache_alphas=new_alphas,
        cache_betas=new_betas,
        cache_ids=(*new_cache_ids, oid),
        cache_start=new_start,
        cache_size=new_size + 1,
        window=state.window,
        evicted_ids=new_evicted_ids,
        next_id=next_next_id,
    )


def read(
    state: DeltaState,
    query: Tensor,
    archive: Optional[Archive] = None,
    mode: Literal["bounded", "archive"] = "bounded",
    kappa: float = 1.0,
    epsilon: float = 1e-3,
    max_pages: Optional[int] = None,
    read_all_pages: bool = False,
    expected_archive_length: Optional[int] = None,
    expected_index_version: Optional[int] = None,
) -> ReadResult:
    """Execute read under either bounded or archive contract.

    In bounded mode:
        Fast transported estimate r(q) = vbar_L + S_t(q - kbar_L) (Eq. 3).
    In archive mode:
        Exact archive reference with certified stopping or full softmax recovery.
    """
    # Extract active cache items in chronological order
    if state.cache_size == 0:
        active_k = state.cache_keys[:0]
        active_v = state.cache_values[:0]
    else:
        indices = [(state.cache_start + i) % state.window for i in range(state.cache_size)]
        active_k = state.cache_keys[indices]
        active_v = state.cache_values[indices]

    if mode == "bounded":
        out = bounded_read(query, active_k, active_v, state.S, kappa=kappa)
        return ReadResult(
            output=out,
            mode="bounded",
            status="approximate",
            certificate=None,
            pages_read=0,
            bytes_read=0,
        )

    # Archive mode
    causal_pos = state.next_id - 1 if state.next_id > 0 else 0
    exp_len = expected_archive_length if expected_archive_length is not None else (len(archive) if archive is not None else 0)
    exp_ver = expected_index_version if expected_index_version is not None else (archive.index_version if archive is not None else 0)

    return archive_reference_read(
        query=query,
        recent_keys=active_k,
        recent_values=active_v,
        S=state.S,
        archive=archive,
        causal_position=causal_pos,
        kappa=kappa,
        epsilon=epsilon,
        max_pages=max_pages,
        read_all_pages=read_all_pages,
        expected_archive_length=exp_len,
        expected_index_version=exp_ver,
    )


def occurrence_partition(state: DeltaState) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (evicted_ids, active_cache_ids) verifying disjoint causal partition."""
    return state.evicted_ids, state.cache_ids


def create_snapshot(
    state: DeltaState,
    archive: Optional[Archive] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> StateSnapshot:
    """Capture complete state checkpoint tuple."""
    return StateSnapshot(
        S=state.S.clone(),
        cache_keys=state.cache_keys.clone(),
        cache_values=state.cache_values.clone(),
        cache_alphas=state.cache_alphas.clone(),
        cache_betas=state.cache_betas.clone(),
        cache_ids=state.cache_ids,
        cache_start=state.cache_start,
        cache_size=state.cache_size,
        window=state.window,
        evicted_ids=state.evicted_ids,
        position=state.next_id,
        archive_length=len(archive) if archive is not None else 0,
        index_version=archive.index_version if archive is not None else 0,
        torch_rng_state=torch.get_rng_state(),
        python_rng_state=random.getstate(),
        metadata=dict(metadata or {}),
    )


def restore_from_snapshot(
    snapshot: StateSnapshot,
    archive: Optional[Archive] = None,
) -> DeltaState:
    """Restore state exactly from snapshot tuple."""
    if archive is not None:
        archive.rollback(
            target_length=snapshot.archive_length,
            target_version=snapshot.index_version,
        )

    torch.set_rng_state(snapshot.torch_rng_state)
    random.setstate(snapshot.python_rng_state)

    return DeltaState(
        S=snapshot.S.clone(),
        cache_keys=snapshot.cache_keys.clone(),
        cache_values=snapshot.cache_values.clone(),
        cache_alphas=snapshot.cache_alphas.clone(),
        cache_betas=snapshot.cache_betas.clone(),
        cache_ids=snapshot.cache_ids,
        cache_start=snapshot.cache_start,
        cache_size=snapshot.cache_size,
        window=snapshot.window,
        evicted_ids=snapshot.evicted_ids,
        next_id=snapshot.position,
    )
