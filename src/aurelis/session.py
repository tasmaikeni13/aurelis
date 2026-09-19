"""High-level stateful session and execution lifecycle for AURELIS."""

from __future__ import annotations

import random
from typing import Any, Literal, Optional

import torch
from torch import Tensor

from .archive import Archive, archive_reference_read
from .functional import full_history_softmax
from .streaming import consume, initial_state, occurrence_partition, read
from .types import (
    ArchiveEntry,
    CertificateBound,
    DeltaState,
    MemoryProfile,
    ReadResult,
    ReadStatus,
    StateSnapshot,
)


class AurelisSession:
    """Stateful streaming and execution manager for AURELIS.

    Exposes reset/snapshot/restore/fork/cancel and exact read statuses.
    Supports populated-cache autoregressive decode, multi-token prefill,
    and continuation with bit/numerical equivalence across bounded and archive modes.
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        window: int,
        mode: Literal["bounded", "archive"] = "bounded",
        page_size: int = 16,
        dtype: torch.dtype | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.d_key = d_key
        self.d_value = d_value
        self.window = window
        self.mode: Literal["bounded", "archive"] = mode
        self.page_size = page_size
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.device = device

        self.state = initial_state(
            d_key=d_key,
            d_value=d_value,
            window=window,
            dtype=self.dtype,
            device=self.device,
        )
        self.archive: Optional[Archive] = (
            Archive(page_size=page_size) if mode == "archive" else None
        )
        self._history_keys: list[Tensor] = []
        self._history_values: list[Tensor] = []
        self._history_ids: list[int] = []
        self._last_snapshot: Optional[StateSnapshot] = None
        self._step_counter = 0

    def reset(self) -> None:
        """Reset session to empty initial state."""
        self.state = initial_state(
            d_key=self.d_key,
            d_value=self.d_value,
            window=self.window,
            dtype=self.dtype,
            device=self.device,
        )
        self.archive = (
            Archive(page_size=self.page_size) if self.mode == "archive" else None
        )
        self._history_keys.clear()
        self._history_values.clear()
        self._history_ids.clear()
        self._last_snapshot = None
        self._step_counter = 0

    def snapshot(self, metadata: Optional[dict[str, Any]] = None) -> StateSnapshot:
        """Capture complete immutable checkpoint of state, cache, archive, and RNG."""
        archive_len = len(self.archive) if self.archive is not None else 0
        idx_ver = self.archive.index_version if self.archive is not None else 0
        meta = dict(metadata or {})
        meta["step_counter"] = self._step_counter
        meta["history_len"] = len(self._history_keys)

        snap = StateSnapshot(
            S=self.state.S.clone(),
            cache_keys=self.state.cache_keys.clone(),
            cache_values=self.state.cache_values.clone(),
            cache_alphas=self.state.cache_alphas.clone(),
            cache_betas=self.state.cache_betas.clone(),
            cache_ids=self.state.cache_ids,
            cache_start=self.state.cache_start,
            cache_size=self.state.cache_size,
            window=self.state.window,
            evicted_ids=self.state.evicted_ids,
            position=self.state.next_id,
            archive_length=archive_len,
            index_version=idx_ver,
            torch_rng_state=torch.get_rng_state(),
            python_rng_state=random.getstate(),
            metadata=meta,
        )
        self._last_snapshot = snap
        return snap

    def restore(self, snapshot: StateSnapshot) -> None:
        """Restore session exactly to a previously captured snapshot."""
        self.state = DeltaState(
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

        if self.archive is not None:
            self.archive.rollback(
                target_length=snapshot.archive_length,
                target_version=snapshot.index_version,
            )

        if "history_len" in snapshot.metadata:
            hlen = snapshot.metadata["history_len"]
            self._history_keys = self._history_keys[:hlen]
            self._history_values = self._history_values[:hlen]
            self._history_ids = self._history_ids[:hlen]

        if "step_counter" in snapshot.metadata:
            self._step_counter = snapshot.metadata["step_counter"]

        torch.set_rng_state(snapshot.torch_rng_state)
        random.setstate(snapshot.python_rng_state)

    def fork(self) -> AurelisSession:
        """Fork an independent copy of this session for branching or speculative evaluation."""
        forked = AurelisSession(
            d_key=self.d_key,
            d_value=self.d_value,
            window=self.window,
            mode=self.mode,
            page_size=self.page_size,
            dtype=self.dtype,
            device=self.device,
        )
        forked.state = DeltaState(
            S=self.state.S.clone(),
            cache_keys=self.state.cache_keys.clone(),
            cache_values=self.state.cache_values.clone(),
            cache_alphas=self.state.cache_alphas.clone(),
            cache_betas=self.state.cache_betas.clone(),
            cache_ids=self.state.cache_ids,
            cache_start=self.state.cache_start,
            cache_size=self.state.cache_size,
            window=self.state.window,
            evicted_ids=self.state.evicted_ids,
            next_id=self.state.next_id,
        )
        forked.archive = self.archive.clone() if self.archive is not None else None
        forked._history_keys = [k.clone() for k in self._history_keys]
        forked._history_values = [v.clone() for v in self._history_values]
        forked._history_ids = list(self._history_ids)
        forked._step_counter = self._step_counter
        return forked

    def cancel(self, target_snapshot: Optional[StateSnapshot] = None) -> None:
        """Rollback speculative or rejected tokens to a known checkpoint."""
        target = target_snapshot if target_snapshot is not None else self._last_snapshot
        if target is None:
            raise RuntimeError("Cannot cancel without a recorded snapshot")
        self.restore(target)

    def consume(
        self,
        key: Tensor,
        value: Tensor,
        alpha: float | Tensor = 1.0,
        beta: float | Tensor = 1.0,
        occurrence_id: Optional[int] = None,
    ) -> None:
        """Consume a newly observed key-value pair, updating recurrence and archive on eviction."""
        oid = occurrence_id if occurrence_id is not None else self.state.next_id

        # Track history reference
        self._history_keys.append(key.detach().clone())
        self._history_values.append(value.detach().clone())
        self._history_ids.append(oid)

        # Update streaming state and archive
        self.state = consume(
            state=self.state,
            key=key,
            value=value,
            alpha=alpha,
            beta=beta,
            occurrence_id=oid,
            archive=self.archive,
        )
        self._step_counter += 1

    def read(
        self,
        query: Tensor,
        kappa: float = 1.0,
        epsilon: float = 1e-3,
        max_pages: Optional[int] = None,
        read_all_pages: bool = False,
    ) -> ReadResult:
        """Perform read on current state."""
        return read(
            state=self.state,
            query=query,
            archive=self.archive,
            mode=self.mode,
            kappa=kappa,
            epsilon=epsilon,
            max_pages=max_pages,
            read_all_pages=read_all_pages,
        )

    def step(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        alpha: float | Tensor = 1.0,
        beta: float | Tensor = 1.0,
        occurrence_id: Optional[int] = None,
        kappa: float = 1.0,
        epsilon: float = 1e-3,
        max_pages: Optional[int] = None,
        read_all_pages: bool = False,
    ) -> ReadResult:
        """Populated-cache autoregressive decode step: consumes (k, v) and reads at q."""
        self.consume(key=key, value=value, alpha=alpha, beta=beta, occurrence_id=occurrence_id)
        return self.read(
            query=query,
            kappa=kappa,
            epsilon=epsilon,
            max_pages=max_pages,
            read_all_pages=read_all_pages,
        )

    def decode_step(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        alpha: float | Tensor = 1.0,
        beta: float | Tensor = 1.0,
        occurrence_id: Optional[int] = None,
        kappa: float = 1.0,
        epsilon: float = 1e-3,
        max_pages: Optional[int] = None,
        read_all_pages: bool = False,
    ) -> ReadResult:
        """Populated-cache autoregressive step alias."""
        return self.step(
            query=query,
            key=key,
            value=value,
            alpha=alpha,
            beta=beta,
            occurrence_id=occurrence_id,
            kappa=kappa,
            epsilon=epsilon,
            max_pages=max_pages,
            read_all_pages=read_all_pages,
        )

    def prefill(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        alphas: Optional[Tensor] = None,
        betas: Optional[Tensor] = None,
        occurrence_ids: Optional[list[int]] = None,
        kappa: float = 1.0,
        epsilon: float = 1e-3,
        max_pages: Optional[int] = None,
        read_all_pages: bool = False,
    ) -> list[ReadResult]:
        """Multi-token causal prefill processing a sequence of tokens."""
        t_len = queries.shape[0]
        results: list[ReadResult] = []

        for i in range(t_len):
            q = queries[i]
            k = keys[i]
            v = values[i]
            a = alphas[i] if alphas is not None else 1.0
            b = betas[i] if betas is not None else 1.0
            oid = occurrence_ids[i] if occurrence_ids is not None else None

            res = self.step(
                query=q,
                key=k,
                value=v,
                alpha=a,
                beta=b,
                occurrence_id=oid,
                kappa=kappa,
                epsilon=epsilon,
                max_pages=max_pages,
                read_all_pages=read_all_pages,
            )
            results.append(res)

        return results

    def full_history_reference(
        self, query: Tensor, kappa: float = 1.0, cutoff: Optional[int] = None
    ) -> Tensor:
        """Compute exact full-history softmax attention across all consumed tokens up to cutoff."""
        if not self._history_keys:
            return torch.zeros(self.d_value, dtype=self.dtype, device=self.device)
        keys = self._history_keys[:cutoff] if cutoff is not None else self._history_keys
        values = self._history_values[:cutoff] if cutoff is not None else self._history_values
        if not keys:
            return torch.zeros(self.d_value, dtype=self.dtype, device=self.device)
        k_stack = torch.stack(keys)
        v_stack = torch.stack(values)
        return full_history_softmax(k_stack, v_stack, query, kappa=kappa)

    def get_memory_profile(self) -> MemoryProfile:
        """Measure live tensor memory consumption labeled by tier."""
        elem_size = self.state.S.element_size()
        working_bytes = (
            self.state.S.numel() * elem_size
            + self.state.cache_keys.numel() * elem_size
            + self.state.cache_values.numel() * elem_size
            + self.state.cache_alphas.numel() * elem_size
            + self.state.cache_betas.numel() * elem_size
        )

        archive_raw_bytes = 0
        archive_index_bytes = 0
        if self.archive is not None:
            bd = self.archive.memory_breakdown()
            archive_raw_bytes = bd["raw_kv_bytes"]
            archive_index_bytes = bd["index_bytes"]

        total_bytes = working_bytes + archive_raw_bytes + archive_index_bytes
        tier_breakdown = {
            "tier1_working_ram": working_bytes,
            "tier2_host_archive": archive_raw_bytes,
            "tier3_cold_index": archive_index_bytes,
        }

        return MemoryProfile(
            working_state_bytes=working_bytes,
            archive_raw_bytes=archive_raw_bytes,
            archive_index_bytes=archive_index_bytes,
            total_bytes=total_bytes,
            tier_breakdown=tier_breakdown,
            item_count=self._step_counter,
            window_size=self.window,
            archive_length=len(self.archive) if self.archive is not None else 0,
        )
