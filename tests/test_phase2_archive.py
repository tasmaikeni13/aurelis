"""Phase 2 tests: Paged raw archive, causal envelopes, full-history softmax recovery, and integrity."""

from __future__ import annotations

import pytest
import torch

from aurelis import (
    Archive,
    archive_reference_read,
    consume,
    full_history_softmax,
    initial_state,
    read,
)


def test_archive_empty_remote_state():
    """When t <= w, remote archive is empty and read returns full_read with zero pages read."""
    torch.manual_seed(101)
    d_k, d_v, window = 8, 8, 4
    archive = Archive(page_size=4)
    state = initial_state(d_k, d_v, window)

    for t in range(3):
        k = torch.randn(d_k)
        v = torch.randn(d_v)
        state = consume(state, k, v, occurrence_id=t, archive=archive)

    q = torch.randn(d_k)
    res = read(state, q, archive=archive, mode="archive")

    assert res.mode == "archive"
    assert res.status == "full_read"
    assert res.certificate is None
    assert res.pages_read == 0
    assert res.bytes_read == 0

    # Compare with exact local attention
    active_k = state.cache_keys[:3]
    active_v = state.cache_values[:3]
    exact_out = full_history_softmax(active_k, active_v, q)
    torch.testing.assert_close(res.output, exact_out)


def test_archive_multiple_page_boundaries():
    """Observations cross multiple page boundaries and seal descriptors accurately."""
    torch.manual_seed(202)
    d_k, d_v, window, page_size = 6, 6, 2, 4
    archive = Archive(page_size=page_size)
    state = initial_state(d_k, d_v, window)

    # 18 tokens: 2 in window, 16 evicted into archive -> exactly 4 sealed pages
    for t in range(18):
        k = torch.randn(d_k)
        v = torch.randn(d_v)
        state = consume(state, k, v, occurrence_id=t, archive=archive)

    assert len(archive) == 16
    assert len(archive._sealed_pages) == 4
    assert archive.verify_integrity()

    # Verify each page contains exactly page_size items with correct occurrence IDs
    for p_id, page in enumerate(archive._sealed_pages):
        assert page.page_id == p_id
        assert page.count == page_size
        expected_ids = tuple(p_id * page_size + i for i in range(page_size))
        assert page.occurrence_ids == expected_ids


def test_archive_full_softmax_recovery():
    """Reference archive read with read_all_pages=True recovers full softmax in real arithmetic."""
    torch.manual_seed(303)
    d_k, d_v, window, page_size = 8, 8, 3, 4
    archive = Archive(page_size=page_size)
    state = initial_state(d_k, d_v, window, dtype=torch.float64)

    all_k = []
    all_v = []
    for t in range(15):
        k = torch.randn(d_k, dtype=torch.float64)
        v = torch.randn(d_v, dtype=torch.float64)
        all_k.append(k)
        all_v.append(v)
        state = consume(state, k, v, occurrence_id=t, archive=archive)

    q = torch.randn(d_k, dtype=torch.float64)
    res = read(state, q, archive=archive, mode="archive", read_all_pages=True)

    assert res.mode == "archive"
    assert res.status == "full_read"
    assert res.pages_read > 0
    assert res.bytes_read > 0

    # Explicit full softmax over all 15 observations
    k_stack = torch.stack(all_k)
    v_stack = torch.stack(all_v)
    exact_out = full_history_softmax(k_stack, v_stack, q)

    torch.testing.assert_close(res.output, exact_out, atol=1e-12, rtol=1e-10)


def test_reading_pages_never_writes_S():
    """Invariant: Reading pages from archive NEVER updates recurrent state S."""
    torch.manual_seed(404)
    d_k, d_v, window, page_size = 8, 8, 3, 4
    archive = Archive(page_size=page_size)
    state = initial_state(d_k, d_v, window)

    for t in range(12):
        k = torch.randn(d_k)
        v = torch.randn(d_v)
        state = consume(state, k, v, occurrence_id=t, archive=archive)

    S_before = state.S.clone()

    q = torch.randn(d_k)
    # Read with full pages read
    res1 = read(state, q, archive=archive, mode="archive", read_all_pages=True)
    assert res1.status == "full_read"
    # S must be bit-identical
    assert torch.equal(state.S, S_before)

    # Read with certified mode
    res2 = read(state, q, archive=archive, mode="archive", epsilon=1e-3)
    assert torch.equal(state.S, S_before)


def test_retrying_read_never_duplicates_observation():
    """Invariant: Calling read repeatedly does not mutate state or duplicate mass."""
    torch.manual_seed(505)
    d_k, d_v, window = 8, 8, 3
    archive = Archive(page_size=4)
    state = initial_state(d_k, d_v, window)

    for t in range(10):
        state = consume(state, torch.randn(d_k), torch.randn(d_v), occurrence_id=t, archive=archive)

    q = torch.randn(d_k)
    res1 = read(state, q, archive=archive, mode="archive", epsilon=1e-2)
    res2 = read(state, q, archive=archive, mode="archive", epsilon=1e-2)

    torch.testing.assert_close(res1.output, res2.output)
    assert res1.status == res2.status
    assert res1.pages_read == res2.pages_read
    assert res1.bytes_read == res2.bytes_read
    if res1.certificate is not None:
        assert res1.certificate.selected_mass == res2.certificate.selected_mass
        assert res1.certificate.bound == res2.certificate.bound


def test_archive_per_query_causal_mask_partial_pages():
    """Per-query causal mask in partial pages ensures no leak of observations past query position."""
    torch.manual_seed(606)
    d_k, d_v, page_size = 4, 4, 5
    archive = Archive(page_size=page_size)

    # Append 8 observations: page 0 (ids 0-4, sealed), page 1 (ids 5-7, partial)
    for t in range(8):
        k = torch.randn(d_k)
        v = torch.randn(d_v)
        archive.append(occurrence_id=t, key=k, value=v, position=t)

    # Query at causal position 2 (only first 3 items visible)
    pages_at_2 = archive.get_pages_and_partial(causal_cutoff=2)
    assert len(pages_at_2) == 1
    assert pages_at_2[0].count == 3
    assert pages_at_2[0].occurrence_ids == (0, 1, 2)

    # Query at causal position 6 (first sealed page + partial with 2 items)
    pages_at_6 = archive.get_pages_and_partial(causal_cutoff=6)
    assert len(pages_at_6) == 2
    assert pages_at_6[0].count == 5
    assert pages_at_6[1].count == 2
    assert pages_at_6[1].occurrence_ids == (5, 6)


def test_archive_wrong_length_or_version_fails_integrity():
    """A wrong archive length or index version must fail with 'invalid_state', never returning a certificate."""
    torch.manual_seed(707)
    d_k, d_v, window = 6, 6, 2
    archive = Archive(page_size=4)
    state = initial_state(d_k, d_v, window)

    for t in range(8):
        state = consume(state, torch.randn(d_k), torch.randn(d_v), occurrence_id=t, archive=archive)

    q = torch.randn(d_k)

    # 1. Wrong archive length
    res_bad_len = read(
        state,
        q,
        archive=archive,
        mode="archive",
        expected_archive_length=999,  # Wrong!
    )
    assert res_bad_len.status == "invalid_state"
    assert res_bad_len.certificate is None
    assert "Archive length mismatch" in res_bad_len.details["error"]

    # 2. Wrong index version
    res_bad_ver = read(
        state,
        q,
        archive=archive,
        mode="archive",
        expected_index_version=999,  # Wrong!
    )
    assert res_bad_ver.status == "invalid_state"
    assert res_bad_ver.certificate is None
    assert "Archive index version mismatch" in res_bad_ver.details["error"]
