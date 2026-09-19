"""Immutable public result and state records for AURELIS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

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


@dataclass(frozen=True)
class PageDescriptor:
    """Bounding metadata for a paged chunk of remote key-value associations."""

    page_id: int
    count: int
    key_min: Tensor
    key_max: Tensor
    value_center: Tensor
    value_radius: float


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
