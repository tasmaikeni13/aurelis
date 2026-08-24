"""Auditable chained-read primitives for functional-graph experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .baselines import ExplicitPairState, softmax_read_many
from .memory import CSMState, GaussMarkovMemory


@dataclass(frozen=True)
class ChainedReads:
    """Queries after each adaptive access and optional CSM uncertainty."""

    queries: Tensor
    confidence: Tensor | None


def csm_chained_reads(
    memory: GaussMarkovMemory,
    state: CSMState,
    starts: Tensor,
    hops: int,
) -> ChainedReads:
    """Apply ``hops`` adaptive reads to one unchanged CSM state."""

    if hops < 1:
        raise ValueError("hops must be positive")
    if starts.ndim != 2 or starts.shape[1] != memory.d_key:
        raise ValueError(f"starts must have shape [batch, {memory.d_key}]")
    current = starts
    queries = [current]
    confidence = []
    for _ in range(hops):
        current, uncertainty = memory.read_many_with_confidence(state, current)
        queries.append(current)
        confidence.append(uncertainty)
    return ChainedReads(
        queries=torch.stack(queries), confidence=torch.stack(confidence)
    )


def softmax_chained_reads(
    state: ExplicitPairState,
    starts: Tensor,
    hops: int,
    temperature: float,
) -> ChainedReads:
    """Apply the same number of adaptive explicit-softmax accesses."""

    if hops < 1:
        raise ValueError("hops must be positive")
    current = starts
    queries = [current]
    for _ in range(hops):
        current, _ = softmax_read_many(state, current, temperature)
        queries.append(current)
    return ChainedReads(queries=torch.stack(queries), confidence=None)


def nearest_code(codes: Tensor, vectors: Tensor) -> Tensor:
    """Decode vectors by maximum cosine similarity to a node-code table."""

    if codes.ndim != 2 or vectors.ndim != 2 or codes.shape[1] != vectors.shape[1]:
        raise ValueError("codes and vectors must be compatible matrices")
    tiny = torch.finfo(vectors.dtype).tiny
    normalized_codes = codes / torch.linalg.vector_norm(
        codes, dim=1, keepdim=True
    ).clamp_min(tiny)
    normalized_vectors = vectors / torch.linalg.vector_norm(
        vectors, dim=1, keepdim=True
    ).clamp_min(tiny)
    return torch.argmax(normalized_vectors @ normalized_codes.mT, dim=1)


def chase_indices(successors: Tensor, starts: Tensor, hops: int) -> Tensor:
    """Return discrete graph locations after steps ``0..hops``."""

    if successors.ndim != 1 or starts.ndim != 1:
        raise ValueError("successors and starts must be vectors")
    if hops < 1:
        raise ValueError("hops must be positive")
    current = starts
    locations = [current]
    for _ in range(hops):
        current = successors[current]
        locations.append(current)
    return torch.stack(locations)


def prepared_read_operator(memory: GaussMarkovMemory, state: CSMState) -> Tensor:
    """Materialize the small reference read operator for norm diagnostics only."""

    identity = torch.eye(
        memory.d_key, dtype=state.S.dtype, device=state.S.device
    )
    solved = torch.linalg.solve(memory.system_matrix(state), identity)
    return state.C @ solved
