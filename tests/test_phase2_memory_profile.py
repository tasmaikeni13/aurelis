"""Phase 2 tests: Live tensor state measurement, bounded memory plateau, and archive byte growth by tier."""

from __future__ import annotations

import pytest
import torch

from aurelis import AurelisSession


def test_bounded_state_memory_plateau():
    """Bounded working state live tensor memory must plateau once context fills window w."""
    torch.manual_seed(7001)
    d_k, d_v, window = 16, 16, 8
    sess = AurelisSession(d_key=d_k, d_value=d_v, window=window, mode="bounded")

    working_memory_history = []
    steps = [1, 4, 8, 12, 16, 32, 64, 128]

    current_t = 0
    for target_t in steps:
        while current_t < target_t:
            sess.step(torch.randn(d_k), torch.randn(d_k), torch.randn(d_v))
            current_t += 1
        prof = sess.get_memory_profile()
        working_memory_history.append((target_t, prof.working_state_bytes))

    # All steps where t >= window (steps >= 8) must have identical working state memory
    plateau_bytes = [mem for t, mem in working_memory_history if t >= window]
    assert len(plateau_bytes) >= 5
    assert len(set(plateau_bytes)) == 1, f"Expected memory to plateau, but got: {plateau_bytes}"


def test_archive_memory_growth_and_tier_labeling():
    """Archive bytes must grow as predicted and be properly labeled across storage tiers."""
    torch.manual_seed(8002)
    d_k, d_v, window, page_size = 8, 8, 4, 4
    sess = AurelisSession(d_key=d_k, d_value=d_v, window=window, mode="archive", page_size=page_size)

    # Initial profile before evictions (t <= window)
    for _ in range(window):
        sess.step(torch.randn(d_k), torch.randn(d_k), torch.randn(d_v))

    prof_init = sess.get_memory_profile()
    assert prof_init.archive_raw_bytes == 0
    assert prof_init.archive_index_bytes == 0
    assert prof_init.tier_breakdown["tier2_host_archive"] == 0

    # Advance 16 more tokens (so 16 tokens evicted -> 4 sealed pages)
    for _ in range(16):
        sess.step(torch.randn(d_k), torch.randn(d_k), torch.randn(d_v))

    prof_post = sess.get_memory_profile()
    # Bounded working state RAM remains constant
    assert prof_post.working_state_bytes == prof_init.working_state_bytes
    assert prof_post.tier_breakdown["tier1_working_ram"] == prof_init.tier_breakdown["tier1_working_ram"]

    # Archive raw bytes grew
    assert prof_post.archive_raw_bytes > 0
    assert prof_post.tier_breakdown["tier2_host_archive"] == prof_post.archive_raw_bytes

    # Index bytes grew to accommodate 4 sealed pages
    assert prof_post.archive_index_bytes > 0
    assert prof_post.tier_breakdown["tier3_cold_index"] == prof_post.archive_index_bytes
    assert prof_post.archive_length == 16
