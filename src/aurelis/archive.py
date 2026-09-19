"""Append-only raw archive and exact archive reference retrieval for AURELIS."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from .functional import (
    bounded_read,
    completed_read,
    full_history_softmax,
    local_attention,
    residual_certificate,
)
from .policy import RetrievalPolicy
from .types import ArchiveEntry, CertificateBound, PageDescriptor, ReadResult, ReadStatus


class Archive:
    """Append-only paged raw storage for evicted remote observations.

    Retains causal occurrence IDs and original tensor encodings.
    Builds page coordinate and residual envelopes with per-query causal masking.
    """

    def __init__(self, page_size: int = 16) -> None:
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")
        self.page_size = page_size
        self._entries: list[ArchiveEntry] = []
        self._sealed_pages: list[PageDescriptor] = []
        self._index_version: int = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def index_version(self) -> int:
        return self._index_version

    def append(
        self,
        occurrence_id: int,
        key: Tensor,
        value: Tensor,
        alpha: float | Tensor = 1.0,
        beta: float | Tensor = 1.0,
        position: int = 0,
    ) -> None:
        """Append a newly evicted observation retaining causal ID and original encoding."""
        a_val = float(alpha.item()) if isinstance(alpha, Tensor) else float(alpha)
        b_val = float(beta.item()) if isinstance(beta, Tensor) else float(beta)

        entry = ArchiveEntry(
            occurrence_id=occurrence_id,
            key=key.detach().clone(),
            value=value.detach().clone(),
            alpha=a_val,
            beta=b_val,
            position=position,
        )
        self._entries.append(entry)
        self._index_version += 1

        # Check if a new page should be sealed
        if len(self._entries) % self.page_size == 0:
            page_id = len(self._sealed_pages)
            start_idx = page_id * self.page_size
            end_idx = start_idx + self.page_size
            page_entries = self._entries[start_idx:end_idx]
            self._sealed_pages.append(self._build_page_descriptor(page_id, page_entries, sealed=True))

    @staticmethod
    def _build_page_descriptor(
        page_id: int, entries: list[ArchiveEntry], sealed: bool = True
    ) -> PageDescriptor:
        """Construct bounding metadata for a chunk of entries."""
        if not entries:
            raise ValueError("Cannot build PageDescriptor from empty entries list")

        keys = torch.stack([e.key for e in entries])
        values = torch.stack([e.value for e in entries])

        k_min = torch.min(keys, dim=0).values
        k_max = torch.max(keys, dim=0).values

        center = torch.mean(values.to(dtype=torch.float64), dim=0).to(dtype=values.dtype)
        diffs = torch.linalg.vector_norm(
            values.to(dtype=torch.float64) - center.to(dtype=torch.float64), dim=-1
        )
        radius = float(torch.max(diffs).item())

        occ_ids = tuple(e.occurrence_id for e in entries)
        start_pos = entries[0].position
        end_pos = entries[-1].position

        return PageDescriptor(
            page_id=page_id,
            count=len(entries),
            key_min=k_min,
            key_max=k_max,
            value_center=center,
            value_radius=radius,
            occurrence_ids=occ_ids,
            start_pos=start_pos,
            end_pos=end_pos,
            sealed=sealed,
        )

    def get_entries(self, causal_cutoff: Optional[int] = None) -> list[ArchiveEntry]:
        """Return entries causally constrained by occurrence_id <= causal_cutoff."""
        if causal_cutoff is None:
            return list(self._entries)
        return [e for e in self._entries if e.occurrence_id <= causal_cutoff]

    def get_page_entries(
        self, page_id: int, causal_cutoff: Optional[int] = None
    ) -> list[ArchiveEntry]:
        """Return entries for a specific page, causally filtered."""
        start_idx = page_id * self.page_size
        end_idx = min(start_idx + self.page_size, len(self._entries))
        if start_idx >= len(self._entries):
            return []
        entries = self._entries[start_idx:end_idx]
        if causal_cutoff is not None:
            entries = [e for e in entries if e.occurrence_id <= causal_cutoff]
        return entries

    def get_pages_and_partial(
        self, causal_cutoff: Optional[int] = None
    ) -> list[PageDescriptor]:
        """Return all sealed page descriptors plus any unsealed partial page.

        Applies per-query causal mask: observations > causal_cutoff are omitted,
        ensuring index construction never leaks later observations.
        """
        descriptors: list[PageDescriptor] = []

        # Number of sealed pages
        num_sealed = len(self._sealed_pages)
        for p in self._sealed_pages:
            # Check if sealed page is entirely or partially within causal cutoff
            if causal_cutoff is not None:
                valid_entries = self.get_page_entries(p.page_id, causal_cutoff=causal_cutoff)
                if not valid_entries:
                    continue
                if len(valid_entries) < p.count:
                    # Partial view of this sealed page under causal mask
                    descriptors.append(
                        self._build_page_descriptor(p.page_id, valid_entries, sealed=False)
                    )
                    continue
            descriptors.append(p)

        # Check for unsealed partial page at the tail
        tail_start = num_sealed * self.page_size
        if tail_start < len(self._entries):
            tail_entries = self._entries[tail_start:]
            if causal_cutoff is not None:
                tail_entries = [e for e in tail_entries if e.occurrence_id <= causal_cutoff]
            if tail_entries:
                descriptors.append(
                    self._build_page_descriptor(num_sealed, tail_entries, sealed=False)
                )

        return descriptors

    def verify_integrity(
        self,
        expected_length: Optional[int] = None,
        expected_version: Optional[int] = None,
    ) -> bool:
        """Verify archive data invariants, sequence ordering, and version matching."""
        if expected_length is not None and len(self._entries) != expected_length:
            return False
        if expected_version is not None and self._index_version != expected_version:
            return False

        # Verify occurrence IDs strictly ascending
        for i in range(1, len(self._entries)):
            if self._entries[i].occurrence_id <= self._entries[i - 1].occurrence_id:
                return False

        # Verify sealed page metadata integrity
        for p in self._sealed_pages:
            start_idx = p.page_id * self.page_size
            end_idx = start_idx + p.count
            page_entries = self._entries[start_idx:end_idx]
            if len(page_entries) != p.count:
                return False
            for e in page_entries:
                # Key must lie within key_min and key_max (with small float tolerance)
                if torch.any(e.key < p.key_min - 1e-6) or torch.any(e.key > p.key_max + 1e-6):
                    return False
                dist = float(
                    torch.linalg.vector_norm(
                        e.value.to(torch.float64) - p.value_center.to(torch.float64)
                    ).item()
                )
                if dist > p.value_radius + 1e-6:
                    return False

        return True

    def clone(self) -> Archive:
        """Deep copy of the archive for session forking and snapshot isolation."""
        cloned = Archive(page_size=self.page_size)
        cloned._entries = [
            ArchiveEntry(
                occurrence_id=e.occurrence_id,
                key=e.key.clone(),
                value=e.value.clone(),
                alpha=e.alpha,
                beta=e.beta,
                position=e.position,
            )
            for e in self._entries
        ]
        cloned._sealed_pages = [
            PageDescriptor(
                page_id=p.page_id,
                count=p.count,
                key_min=p.key_min.clone(),
                key_max=p.key_max.clone(),
                value_center=p.value_center.clone(),
                value_radius=p.value_radius,
                occurrence_ids=p.occurrence_ids,
                start_pos=p.start_pos,
                end_pos=p.end_pos,
                sealed=p.sealed,
            )
            for p in self._sealed_pages
        ]
        cloned._index_version = self._index_version
        return cloned

    def rollback(self, target_length: int, target_version: int) -> None:
        """Rollback archive to a previous state checkpoint."""
        if target_length > len(self._entries):
            raise ValueError(
                f"Cannot rollback to length {target_length} greater than current {len(self._entries)}"
            )
        self._entries = self._entries[:target_length]
        # Recompute sealed pages up to target_length // page_size
        num_sealed = target_length // self.page_size
        self._sealed_pages = self._sealed_pages[:num_sealed]
        self._index_version = target_version

    def memory_breakdown(self) -> dict[str, int]:
        """Calculate memory consumption in bytes labeled by tier."""
        raw_kv_bytes = 0
        for e in self._entries:
            raw_kv_bytes += e.key.numel() * e.key.element_size()
            raw_kv_bytes += e.value.numel() * e.value.element_size()
            raw_kv_bytes += 32  # Metadata overhead (occurrence_id, position, gates)

        index_bytes = 0
        for p in self._sealed_pages:
            index_bytes += p.key_min.numel() * p.key_min.element_size()
            index_bytes += p.key_max.numel() * p.key_max.element_size()
            index_bytes += p.value_center.numel() * p.value_center.element_size()
            index_bytes += 16  # radius, count, page_id

        return {
            "raw_kv_bytes": raw_kv_bytes,
            "index_bytes": index_bytes,
            "tier2_host_archive": raw_kv_bytes,
            "tier3_cold_index": index_bytes,
            "total_bytes": raw_kv_bytes + index_bytes,
        }


def archive_reference_read(
    query: Tensor,
    recent_keys: Tensor,
    recent_values: Tensor,
    S: Tensor,
    archive: Optional[Archive],
    causal_position: Optional[int] = None,
    kappa: float = 1.0,
    epsilon: float = 1e-3,
    max_pages: Optional[int] = None,
    max_bytes: Optional[int] = None,
    read_all_pages: bool = False,
    expected_archive_length: Optional[int] = None,
    expected_index_version: Optional[int] = None,
    cost_fn: Optional[Callable[[PageDescriptor], float]] = None,
    timeout_seconds: Optional[float] = None,
    simulate_timeout: bool = False,
    inject_corrupt_bounds: bool = False,
    inject_missing_page: bool = False,
    inject_unsound_scale: float = 1.0,
) -> ReadResult:
    """Execute exact archive reference read with deterministic certification or full read.

    Invariants:
    1. Reading pages NEVER writes S.
    2. Retrying a read NEVER duplicates an observation.
    3. Wrong archive length / index version fails with 'invalid_state' rather than returning a certificate.
    4. Reading all observations recovers full softmax on current query/keys/values.
    5. Nonfinite inputs (NaN, Inf) return 'invalid_interval'.
    6. Timeouts return 'archive_unavailable'.
    """
    d_v = S.shape[0]

    # Nonfinite tensor check
    if (
        torch.isnan(query).any()
        or torch.isinf(query).any()
        or torch.isnan(recent_keys).any()
        or torch.isinf(recent_keys).any()
        or torch.isnan(recent_values).any()
        or torch.isinf(recent_values).any()
        or torch.isnan(S).any()
        or torch.isinf(S).any()
    ):
        return ReadResult(
            output=torch.zeros(d_v, dtype=query.dtype, device=query.device),
            mode="archive",
            status="invalid_interval",
            certificate=None,
            pages_read=0,
            bytes_read=0,
            details={"error": "NaN or Inf detected in input tensors"},
        )

    # Gate: Integrity checks
    if archive is not None:
        if (
            expected_archive_length is not None
            and len(archive) != expected_archive_length
        ):
            return ReadResult(
                output=torch.zeros(d_v, dtype=query.dtype, device=query.device),
                mode="archive",
                status="invalid_state",
                certificate=None,
                pages_read=0,
                bytes_read=0,
                details={"error": f"Archive length mismatch: expected {expected_archive_length}, got {len(archive)}"},
            )
        if (
            expected_index_version is not None
            and archive.index_version != expected_index_version
        ):
            return ReadResult(
                output=torch.zeros(d_v, dtype=query.dtype, device=query.device),
                mode="archive",
                status="invalid_state",
                certificate=None,
                pages_read=0,
                bytes_read=0,
                details={"error": f"Archive index version mismatch: expected {expected_index_version}, got {archive.index_version}"},
            )
        if not archive.verify_integrity():
            return ReadResult(
                output=torch.zeros(d_v, dtype=query.dtype, device=query.device),
                mode="archive",
                status="archive_error",
                certificate=None,
                pages_read=0,
                bytes_read=0,
                details={"error": "Archive integrity verification failed"},
            )

    # Empty history
    if recent_keys.shape[0] == 0:
        return ReadResult(
            output=torch.zeros(d_v, dtype=query.dtype, device=query.device),
            mode="archive",
            status="full_read",
            certificate=None,
            pages_read=0,
            bytes_read=0,
            details={"summary_visits": 0, "selection_cost_ops": 0},
        )

    # Delegate to validated retrieval policy
    policy = RetrievalPolicy(cost_fn=cost_fn, timeout_seconds=timeout_seconds)

    entries = []
    descriptors = []
    if archive is not None and len(archive) > 0:
        entries = archive.get_entries(causal_cutoff=causal_position)
        descriptors = archive.get_pages_and_partial(causal_cutoff=causal_position)

    def get_entries_fn(pid: int) -> list[ArchiveEntry]:
        if archive is None:
            return []
        return archive.get_page_entries(pid, causal_cutoff=causal_position)

    return policy.execute_retrieval(
        query=query,
        recent_keys=recent_keys,
        recent_values=recent_values,
        S=S,
        archive_entries=entries,
        page_descriptors=descriptors,
        get_page_entries_fn=get_entries_fn,
        causal_position=causal_position,
        kappa=kappa,
        epsilon=epsilon,
        max_pages=max_pages,
        max_bytes=max_bytes,
        read_all_pages=read_all_pages,
        dtype=query.dtype,
        inject_corrupt_bounds=inject_corrupt_bounds,
        inject_missing_page=inject_missing_page,
        inject_unsound_scale=inject_unsound_scale,
        simulate_timeout=simulate_timeout,
    )
