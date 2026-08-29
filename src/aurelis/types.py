"""Immutable public result and state records for AURELIS."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class ReadDiagnostics:
    """Quantities needed to audit one AURELIS read."""

    attention: Tensor
    kbar: Tensor
    vbar: Tensor
    innovation: Tensor
    h: Tensor
    V_R: Tensor
    V_H: Tensor
    K_RH: Tensor
    g_raw: Tensor
    g_B: Tensor
    g_E: Tensor
    solve_residual_q: Tensor
    solve_residual_kbar: Tensor


@dataclass(frozen=True)
class ReadOutput:
    """The four manuscript endpoints and their diagnostics."""

    remote: Tensor
    full_residual: Tensor
    bayes: Tensor
    episodic: Tensor
    diagnostics: ReadDiagnostics


@dataclass(frozen=True)
class StreamingState:
    """Immutable fixed-capacity decode state.

    Cache tensors are physical ring buffers. ``cache_start`` points at the
    oldest live slot and ``cache_size`` records the live prefix. Occurrence IDs
    are metadata used to verify exact delayed handoff.
    """

    precision: Tensor
    cross: Tensor
    factor: Tensor
    cache_keys: Tensor
    cache_values: Tensor
    cache_evidence: Tensor
    cache_ids: Tensor
    cache_start: int
    cache_size: int
    remote_ids: tuple[int, ...]
    next_id: int


@dataclass(frozen=True)
class SequenceOutput:
    """Vectorized exact-training outputs at every causal boundary."""

    remote: Tensor
    full_residual: Tensor
    bayes: Tensor
    episodic: Tensor
    attention: Tensor
    precision: Tensor
    cross: Tensor
    h: Tensor
    V_R: Tensor
    V_H: Tensor
    K_RH: Tensor
    g_raw: Tensor
    g_B: Tensor
    g_E: Tensor
