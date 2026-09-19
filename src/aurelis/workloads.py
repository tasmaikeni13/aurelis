"""Evaluation workloads and synthetic regimes for AURELIS Phase 4.

Implements all 10+ required experiment sweeps:
1. structured_linear: Known structured linear relations mixed with rare exceptions.
2. random_incompressible: Independent random Gaussian keys and values (capacity limit).
3. diffuse_attention: Uniform or low-temperature attention spread widely across history.
4. large_value_outliers: Remote observations with extreme value magnitudes.
5. repeated_keys: Identical keys appearing with differing values across time.
6. near_collisions: Distinct keys with very small Euclidean distance but divergent values.
7. delayed_disambiguation: Keys sharing coordinate prefix, disambiguated by late coordinates.
8. abrupt_drift: Generating linear map abruptly switches midway through context.
9. multihop_query: Chained key-value pointers requiring relational retrieval.
10. boundary_retrieval: Targets situated exactly at the local window or page boundary.
11. metadata_dominant: Short contexts / tiny pages where summary scans exceed dense read cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import torch
from torch import Tensor

from .archive import Archive
from .streaming import consume, initial_state
from .types import ArchiveEntry, DeltaState, PageDescriptor


@dataclass(frozen=True)
class WorkloadInstance:
    """A self-contained evaluation workload instance with stream state, archive, and queries."""

    name: str
    context_length: int
    window_size: int
    page_size: int
    d_k: int
    d_v: int
    queries: list[Tensor]
    all_keys: list[Tensor]
    all_values: list[Tensor]
    recent_keys: Tensor
    recent_values: Tensor
    S: Tensor
    archive: Archive
    archive_entries: list[ArchiveEntry]
    page_descriptors: list[PageDescriptor]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_page_entries(self, page_id: int) -> list[ArchiveEntry]:
        """Fetch entries for a specific page."""
        return self.archive.get_page_entries(page_id)


def _build_stream_workload(
    name: str,
    keys: list[Tensor],
    values: list[Tensor],
    queries: list[Tensor],
    window_size: int = 16,
    page_size: int = 8,
    alphas: Optional[list[float]] = None,
    betas: Optional[list[float]] = None,
    dtype: torch.dtype = torch.float64,
    metadata: Optional[dict[str, Any]] = None,
) -> WorkloadInstance:
    """Feed sequence of keys and values through streaming processor to build realistic state and archive."""
    t_len = len(keys)
    d_k = keys[0].shape[-1]
    d_v = values[0].shape[-1]

    archive = Archive(page_size=page_size)
    state = initial_state(d_key=d_k, d_value=d_v, window=window_size, dtype=dtype)

    for i in range(t_len):
        k = keys[i].to(dtype=dtype)
        v = values[i].to(dtype=dtype)
        a = alphas[i] if alphas is not None else 1.0
        b = betas[i] if betas is not None else 1.0

        state = consume(
            state=state,
            key=k,
            value=v,
            alpha=a,
            beta=b,
            occurrence_id=i + 1,
            archive=archive,
        )

    # Active recent window cache in chronological order
    if state.cache_size == 0:
        recent_k = torch.empty((0, d_k), dtype=dtype)
        recent_v = torch.empty((0, d_v), dtype=dtype)
    else:
        indices = [(state.cache_start + i) % state.window for i in range(state.cache_size)]
        recent_k = state.cache_keys[indices]
        recent_v = state.cache_values[indices]

    entries = archive.get_entries()
    pages = archive.get_pages_and_partial()

    return WorkloadInstance(
        name=name,
        context_length=t_len,
        window_size=window_size,
        page_size=page_size,
        d_k=d_k,
        d_v=d_v,
        queries=[q.to(dtype=dtype) for q in queries],
        all_keys=[k.to(dtype=dtype) for k in keys],
        all_values=[v.to(dtype=dtype) for v in values],
        recent_keys=recent_k,
        recent_values=recent_v,
        S=state.S.clone(),
        archive=archive,
        archive_entries=entries,
        page_descriptors=pages,
        metadata=dict(metadata or {}),
    )


def generate_structured_linear_workload(
    context_length: int = 64,
    window_size: int = 16,
    page_size: int = 8,
    d_k: int = 16,
    d_v: int = 16,
    noise_std: float = 0.02,
    exception_rate: float = 0.05,
    seed: int = 42,
    dtype: torch.dtype = torch.float64,
) -> WorkloadInstance:
    """1. Known structured linear relations mixed with rare exceptions.

    v_i = W k_i + noise, with rare outlier exceptions.
    Here S learns W, allowing r(q) to drastically reduce unread value residuals.
    """
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d_v, d_k, generator=g, dtype=dtype) / math.sqrt(d_k)

    keys = []
    values = []
    for i in range(context_length):
        k = torch.randn(d_k, generator=g, dtype=dtype)
        k = k / max(1.0, float(torch.linalg.vector_norm(k).item()))
        if torch.rand(1, generator=g).item() < exception_rate:
            # Rare exception
            v = torch.randn(d_v, generator=g, dtype=dtype) * 3.0
        else:
            # Structured linear relation
            v = torch.matmul(W, k) + torch.randn(d_v, generator=g, dtype=dtype) * noise_std
        keys.append(k)
        values.append(v)

    queries = []
    for _ in range(4):
        q = torch.randn(d_k, generator=g, dtype=dtype)
        q = q / max(1.0, float(torch.linalg.vector_norm(q).item()))
        queries.append(q)

    return _build_stream_workload(
        name="structured_linear",
        keys=keys,
        values=values,
        queries=queries,
        window_size=window_size,
        page_size=page_size,
        dtype=dtype,
        metadata={"noise_std": noise_std, "exception_rate": exception_rate, "W_norm": float(torch.norm(W).item())},
    )


def generate_random_incompressible_workload(
    context_length: int = 64,
    window_size: int = 16,
    page_size: int = 8,
    d_k: int = 16,
    d_v: int = 16,
    seed: int = 43,
    dtype: torch.dtype = torch.float64,
) -> WorkloadInstance:
    """2. Random incompressible associations (capacity limit).

    Independent Gaussian keys and values with no low-rank or linear correlation.
    Represents the theoretical finite-state memory limit where S cannot compress data.
    """
    g = torch.Generator().manual_seed(seed)
    keys = []
    values = []
    for _ in range(context_length):
        k = torch.randn(d_k, generator=g, dtype=dtype)
        k = k / max(1.0, float(torch.linalg.vector_norm(k).item()))
        v = torch.randn(d_v, generator=g, dtype=dtype)
        keys.append(k)
        values.append(v)

    queries = []
    for _ in range(4):
        q = torch.randn(d_k, generator=g, dtype=dtype)
        q = q / max(1.0, float(torch.linalg.vector_norm(q).item()))
        queries.append(q)

    return _build_stream_workload(
        name="random_incompressible",
        keys=keys,
        values=values,
        queries=queries,
        window_size=window_size,
        page_size=page_size,
        dtype=dtype,
    )


def generate_diffuse_attention_workload(
    context_length: int = 64,
    window_size: int = 16,
    page_size: int = 8,
    d_k: int = 16,
    d_v: int = 16,
    seed: int = 44,
    dtype: torch.dtype = torch.float64,
) -> WorkloadInstance:
    """3. Diffuse attention across history.

    Keys are tightly clustered around a single common direction with small spread,
    causing softmax attention mass to spread uniformly across all pages.
    """
    g = torch.Generator().manual_seed(seed)
    base_k = torch.randn(d_k, generator=g, dtype=dtype)
    base_k = base_k / torch.linalg.vector_norm(base_k)

    keys = []
    values = []
    for _ in range(context_length):
        k = base_k + torch.randn(d_k, generator=g, dtype=dtype) * 0.05
        k = k / max(1.0, float(torch.linalg.vector_norm(k).item()))
        v = torch.randn(d_v, generator=g, dtype=dtype)
        keys.append(k)
        values.append(v)

    queries = [base_k + torch.randn(d_k, generator=g, dtype=dtype) * 0.02 for _ in range(4)]
    queries = [q / max(1.0, float(torch.linalg.vector_norm(q).item())) for q in queries]

    return _build_stream_workload(
        name="diffuse_attention",
        keys=keys,
        values=values,
        queries=queries,
        window_size=window_size,
        page_size=page_size,
        dtype=dtype,
    )


def generate_large_value_outliers_workload(
    context_length: int = 64,
    window_size: int = 16,
    page_size: int = 8,
    d_k: int = 16,
    d_v: int = 16,
    outlier_scale: float = 50.0,
    seed: int = 45,
    dtype: torch.dtype = torch.float64,
) -> WorkloadInstance:
    """4. Large value outliers.

    Most values have unit variance, but an unread page contains huge outlier vectors.
    Tests value-sensitive retrieval vs value-blind mass selection.
    """
    g = torch.Generator().manual_seed(seed)
    keys = []
    values = []
    outlier_idx = 10  # Placed inside remote archive

    for i in range(context_length):
        k = torch.randn(d_k, generator=g, dtype=dtype)
        k = k / max(1.0, float(torch.linalg.vector_norm(k).item()))
        if i == outlier_idx:
            v = torch.ones(d_v, dtype=dtype) * outlier_scale
        else:
            v = torch.randn(d_v, generator=g, dtype=dtype)
        keys.append(k)
        values.append(v)

    # Query targeting near the outlier key
    q = keys[outlier_idx] + torch.randn(d_k, generator=g, dtype=dtype) * 0.1
    q = q / max(1.0, float(torch.linalg.vector_norm(q).item()))

    return _build_stream_workload(
        name="large_value_outliers",
        keys=keys,
        values=values,
        queries=[q],
        window_size=window_size,
        page_size=page_size,
        dtype=dtype,
        metadata={"outlier_idx": outlier_idx, "outlier_scale": outlier_scale},
    )


def generate_repeated_keys_workload(
    context_length: int = 64,
    window_size: int = 16,
    page_size: int = 8,
    d_k: int = 16,
    d_v: int = 16,
    seed: int = 46,
    dtype: torch.dtype = torch.float64,
) -> WorkloadInstance:
    """5. Repeated keys with different values across time.

    The same key vector is emitted at early and late positions with opposing values.
    Tests overwriting in delta update and temporal disambiguation.
    """
    g = torch.Generator().manual_seed(seed)
    repeated_k = torch.randn(d_k, generator=g, dtype=dtype)
    repeated_k = repeated_k / torch.linalg.vector_norm(repeated_k)

    early_v = torch.ones(d_v, dtype=dtype) * 2.0
    late_v = -torch.ones(d_v, dtype=dtype) * 2.0

    keys = []
    values = []
    for i in range(context_length):
        if i == 5:
            keys.append(repeated_k)
            values.append(early_v)
        elif i == context_length - 5:
            keys.append(repeated_k)
            values.append(late_v)
        else:
            k = torch.randn(d_k, generator=g, dtype=dtype)
            k = k / max(1.0, float(torch.linalg.vector_norm(k).item()))
            v = torch.randn(d_v, generator=g, dtype=dtype)
            keys.append(k)
            values.append(v)

    return _build_stream_workload(
        name="repeated_keys",
        keys=keys,
        values=values,
        queries=[repeated_k.clone()],
        window_size=window_size,
        page_size=page_size,
        dtype=dtype,
    )


def generate_near_collisions_workload(
    context_length: int = 64,
    window_size: int = 16,
    page_size: int = 8,
    d_k: int = 16,
    d_v: int = 16,
    epsilon_dist: float = 0.02,
    seed: int = 47,
    dtype: torch.dtype = torch.float64,
) -> WorkloadInstance:
    """6. Near-collisions in key space.

    Two keys k1, k2 satisfy ||k1 - k2|| <= epsilon_dist but have radically different values.
    Tests sharpness of certificate error bounds.
    """
    g = torch.Generator().manual_seed(seed)
    base_k = torch.randn(d_k, generator=g, dtype=dtype)
    base_k = base_k / torch.linalg.vector_norm(base_k)

    k1 = base_k
    k2 = base_k + torch.randn(d_k, generator=g, dtype=dtype) * (epsilon_dist / 2.0)
    k2 = k2 / torch.linalg.vector_norm(k2)

    v1 = torch.ones(d_v, dtype=dtype) * 5.0
    v2 = -torch.ones(d_v, dtype=dtype) * 5.0

    keys = []
    values = []
    for i in range(context_length):
        if i == 4:
            keys.append(k1)
            values.append(v1)
        elif i == 12:
            keys.append(k2)
            values.append(v2)
        else:
            k = torch.randn(d_k, generator=g, dtype=dtype)
            k = k / max(1.0, float(torch.linalg.vector_norm(k).item()))
            v = torch.randn(d_v, generator=g, dtype=dtype)
            keys.append(k)
            values.append(v)

    return _build_stream_workload(
        name="near_collisions",
        keys=keys,
        values=values,
        queries=[k1.clone(), k2.clone()],
        window_size=window_size,
        page_size=page_size,
        dtype=dtype,
        metadata={"epsilon_dist": epsilon_dist},
    )


def generate_delayed_disambiguation_workload(
    context_length: int = 64,
    window_size: int = 16,
    page_size: int = 8,
    d_k: int = 16,
    d_v: int = 16,
    seed: int = 48,
    dtype: torch.dtype = torch.float64,
) -> WorkloadInstance:
    """7. Delayed disambiguation.

    Keys share d_k - 2 identical coordinates; only the last 2 coordinates distinguish them.
    Tests envelope bounding box tightness.
    """
    g = torch.Generator().manual_seed(seed)
    shared_prefix = torch.randn(d_k - 2, generator=g, dtype=dtype)

    keys = []
    values = []
    for i in range(context_length):
        suffix = torch.randn(2, generator=g, dtype=dtype) * 0.1
        k = torch.cat([shared_prefix, suffix])
        k = k / max(1.0, float(torch.linalg.vector_norm(k).item()))
        v = torch.randn(d_v, generator=g, dtype=dtype)
        keys.append(k)
        values.append(v)

    target_suffix = torch.randn(2, generator=g, dtype=dtype) * 0.1
    q = torch.cat([shared_prefix, target_suffix])
    q = q / max(1.0, float(torch.linalg.vector_norm(q).item()))

    return _build_stream_workload(
        name="delayed_disambiguation",
        keys=keys,
        values=values,
        queries=[q],
        window_size=window_size,
        page_size=page_size,
        dtype=dtype,
    )


def generate_abrupt_drift_workload(
    context_length: int = 64,
    window_size: int = 16,
    page_size: int = 8,
    d_k: int = 16,
    d_v: int = 16,
    seed: int = 49,
    dtype: torch.dtype = torch.float64,
) -> WorkloadInstance:
    """8. Abrupt drift in generating relationship.

    First half of history follows v = W1 k; second half switches to v = W2 k.
    Tests decay gate alpha adaptation and old-state interference.
    """
    g = torch.Generator().manual_seed(seed)
    W1 = torch.randn(d_v, d_k, generator=g, dtype=dtype) / math.sqrt(d_k)
    W2 = -W1 + torch.randn(d_v, d_k, generator=g, dtype=dtype) * 0.1

    keys = []
    values = []
    midpoint = context_length // 2

    for i in range(context_length):
        k = torch.randn(d_k, generator=g, dtype=dtype)
        k = k / max(1.0, float(torch.linalg.vector_norm(k).item()))
        W = W1 if i < midpoint else W2
        v = torch.matmul(W, k) + torch.randn(d_v, generator=g, dtype=dtype) * 0.01
        keys.append(k)
        values.append(v)

    q = torch.randn(d_k, generator=g, dtype=dtype)
    q = q / max(1.0, float(torch.linalg.vector_norm(q).item()))

    return _build_stream_workload(
        name="abrupt_drift",
        keys=keys,
        values=values,
        queries=[q],
        window_size=window_size,
        page_size=page_size,
        dtype=dtype,
        metadata={"midpoint": midpoint},
    )


def generate_multihop_query_workload(
    context_length: int = 64,
    window_size: int = 16,
    page_size: int = 8,
    d_k: int = 16,
    d_v: int = 16,
    seed: int = 50,
    dtype: torch.dtype = torch.float64,
) -> WorkloadInstance:
    """9. Multi-hop chained queries.

    q retrieves k1 -> v1, where v1 serves as key k2 for final target v2.
    """
    g = torch.Generator().manual_seed(seed)
    # Ensure d_k == d_v for chaining
    dim = min(d_k, d_v)

    k1 = torch.randn(dim, generator=g, dtype=dtype)
    k1 = k1 / torch.linalg.vector_norm(k1)
    k2 = torch.randn(dim, generator=g, dtype=dtype)
    k2 = k2 / torch.linalg.vector_norm(k2)
    final_v = torch.ones(dim, dtype=dtype) * 10.0

    keys = []
    values = []
    for i in range(context_length):
        if i == 6:
            keys.append(k1)
            values.append(k2.clone())  # First hop points to second key
        elif i == 14:
            keys.append(k2)
            values.append(final_v)     # Second hop points to target
        else:
            k = torch.randn(dim, generator=g, dtype=dtype)
            k = k / max(1.0, float(torch.linalg.vector_norm(k).item()))
            v = torch.randn(dim, generator=g, dtype=dtype)
            keys.append(k)
            values.append(v)

    return _build_stream_workload(
        name="multihop_query",
        keys=keys,
        values=values,
        queries=[k1.clone()],
        window_size=window_size,
        page_size=page_size,
        dtype=dtype,
    )


def generate_boundary_retrieval_workload(
    context_length: int = 64,
    window_size: int = 16,
    page_size: int = 8,
    d_k: int = 16,
    d_v: int = 16,
    seed: int = 51,
    dtype: torch.dtype = torch.float64,
) -> WorkloadInstance:
    """10. Retrieval at window/page boundaries.

    Targets situated exactly at the window boundary (t - w) and page boundaries (k * p).
    Tests boundary conditions and causal handoff correctness.
    """
    g = torch.Generator().manual_seed(seed)
    boundary_pos = context_length - window_size - 1  # Exactly oldest evicted token

    target_k = torch.randn(d_k, generator=g, dtype=dtype)
    target_k = target_k / torch.linalg.vector_norm(target_k)
    target_v = torch.ones(d_v, dtype=dtype) * 7.7

    keys = []
    values = []
    for i in range(context_length):
        if i == boundary_pos:
            keys.append(target_k)
            values.append(target_v)
        else:
            k = torch.randn(d_k, generator=g, dtype=dtype)
            k = k / max(1.0, float(torch.linalg.vector_norm(k).item()))
            v = torch.randn(d_v, generator=g, dtype=dtype)
            keys.append(k)
            values.append(v)

    return _build_stream_workload(
        name="boundary_retrieval",
        keys=keys,
        values=values,
        queries=[target_k.clone()],
        window_size=window_size,
        page_size=page_size,
        dtype=dtype,
        metadata={"boundary_pos": boundary_pos},
    )


def generate_metadata_dominant_workload(
    context_length: int = 24,
    window_size: int = 8,
    page_size: int = 2,
    d_k: int = 16,
    d_v: int = 16,
    seed: int = 52,
    dtype: torch.dtype = torch.float64,
) -> WorkloadInstance:
    """11. Metadata-dominant contexts.

    Very short context with tiny page sizes (page_size=2) yielding many small pages.
    Tests the regime where scanning metadata costs more than doing a dense read.
    """
    g = torch.Generator().manual_seed(seed)
    keys = []
    values = []
    for _ in range(context_length):
        k = torch.randn(d_k, generator=g, dtype=dtype)
        k = k / max(1.0, float(torch.linalg.vector_norm(k).item()))
        v = torch.randn(d_v, generator=g, dtype=dtype)
        keys.append(k)
        values.append(v)

    q = torch.randn(d_k, generator=g, dtype=dtype)
    q = q / max(1.0, float(torch.linalg.vector_norm(q).item()))

    return _build_stream_workload(
        name="metadata_dominant",
        keys=keys,
        values=values,
        queries=[q],
        window_size=window_size,
        page_size=page_size,
        dtype=dtype,
    )
