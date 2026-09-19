"""Immutable public result and state records for AURELIS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from torch import Tensor

ReadStatus = Literal[
    "approximate",
    "certified",
    "full_read",
    "budget_exhausted",
    "invalid_state",
    "archive_error",
]


@dataclass(frozen=True)
class CertificateBound:
    """Quantitative evaluation of retrieval approximation error."""

    bound: float
    selected_mass: float
    estimated_unread_mass: float
    unread_mass_lower: float
    unread_mass_upper: float
    residual_bound: float
    denominator_floor: float


@dataclass(frozen=True)
class ReadResult:
    """The outcome of an AURELIS read operation."""

    output: Tensor
    mode: Literal["bounded", "archive"]
    status: ReadStatus
    certificate: Optional[CertificateBound]
    pages_read: int
    bytes_read: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PageDescriptor:
    """Bounding metadata for a paged chunk of remote key-value associations."""

    page_id: int
    count: int
    key_min: Tensor
    key_max: Tensor
    value_center: Tensor
    value_radius: float
    occurrence_ids: tuple[int, ...] = ()
    start_pos: int = 0
    end_pos: int = 0
    sealed: bool = True


@dataclass(frozen=True)
class ArchiveEntry:
    """Immutable record of an archived remote key-value observation."""

    occurrence_id: int
    key: Tensor
    value: Tensor
    alpha: float
    beta: float
    position: int


@dataclass(frozen=True)
class DeltaState:
    """Fixed-capacity solve-free recurrent state with local cache ring buffer."""

    S: Tensor
    cache_keys: Tensor
    cache_values: Tensor
    cache_alphas: Tensor
    cache_betas: Tensor
    cache_ids: tuple[int, ...]
    cache_start: int
    cache_size: int
    window: int
    evicted_ids: tuple[int, ...]
    next_id: int


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable checkpoint of AURELIS recurrent state, cache, archive state, and RNG."""

    S: Tensor
    cache_keys: Tensor
    cache_values: Tensor
    cache_alphas: Tensor
    cache_betas: Tensor
    cache_ids: tuple[int, ...]
    cache_start: int
    cache_size: int
    window: int
    evicted_ids: tuple[int, ...]
    position: int
    archive_length: int
    index_version: int
    torch_rng_state: Tensor
    python_rng_state: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryProfile:
    """Live tensor and metadata memory accounting across storage tiers."""

    working_state_bytes: int
    archive_raw_bytes: int
    archive_index_bytes: int
    total_bytes: int
    tier_breakdown: dict[str, int]
    item_count: int
    window_size: int
    archive_length: int

