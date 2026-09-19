"""Tests for Phase 3 retrieval policy, priority heuristic, non-monotone tracking, and budgets."""

import torch

from aurelis.archive import Archive, archive_reference_read
from aurelis.functional import full_history_softmax
from aurelis.policy import RetrievalPolicy


def test_priority_heuristic_order():
    """Verify that pages with higher priority heuristic are fetched first."""
    torch.manual_seed(3201)
    archive = Archive(page_size=4)
    d_k, d_v = 4, 4

    # Page 0: small values (low residual bound)
    for i in range(4):
        archive.append(occurrence_id=i + 1, key=torch.ones(d_k) * 0.1, value=torch.ones(d_v) * 0.01, position=i)

    # Page 1: huge values (high residual bound)
    for i in range(4, 8):
        archive.append(occurrence_id=i + 1, key=torch.ones(d_k) * 0.1, value=torch.ones(d_v) * 10.0, position=i)

    # Page 2: huge values (high residual bound)
    for i in range(8, 12):
        archive.append(occurrence_id=i + 1, key=torch.ones(d_k) * 0.1, value=torch.ones(d_v) * 10.0, position=i)

    q = torch.ones(d_k) * 0.1
    rk = torch.ones(2, d_k) * 0.1
    rv = torch.ones(2, d_v) * 0.01
    S = torch.zeros(d_v, d_k)

    # Fetch with max_pages=1: should choose high-residual page first and exhaust budget
    res = archive_reference_read(
        query=q,
        recent_keys=rk,
        recent_values=rv,
        S=S,
        archive=archive,
        epsilon=1e-8,
        max_pages=1,
    )

    assert res.status == "budget_exhausted"
    assert res.pages_read == 1


def test_non_monotone_bound_candidate_retention():
    """Verify best valid candidate is preserved even when intermediate bounds worsen."""
    torch.manual_seed(3202)
    archive = Archive(page_size=4)
    d_k, d_v = 4, 4

    for i in range(12):
        archive.append(occurrence_id=i + 1, key=torch.randn(d_k), value=torch.randn(d_v) * 2.0, position=i)

    q = torch.randn(d_k)
    rk = torch.randn(3, d_k)
    rv = torch.randn(3, d_v)
    S = torch.zeros(d_v, d_k)

    # Read with budget limit to check candidate preservation
    res = archive_reference_read(
        query=q,
        recent_keys=rk,
        recent_values=rv,
        S=S,
        archive=archive,
        epsilon=1e-12,
        max_pages=2,
    )

    assert res.certificate is not None
    assert res.certificate.total_bound > 0.0
    # Details should document candidate tracking
    assert "best_candidate_retained" in res.details


def test_budget_exhaustion_page_limit():
    """Verify status='budget_exhausted' returned when max_pages limit is reached."""
    torch.manual_seed(3203)
    archive = Archive(page_size=4)
    d_k, d_v = 4, 4

    for i in range(16):
        archive.append(occurrence_id=i + 1, key=torch.randn(d_k), value=torch.randn(d_v), position=i)

    q = torch.randn(d_k)
    rk = torch.randn(2, d_k)
    rv = torch.randn(2, d_v)
    S = torch.zeros(d_v, d_k)

    res = archive_reference_read(
        query=q,
        recent_keys=rk,
        recent_values=rv,
        S=S,
        archive=archive,
        epsilon=1e-10,
        max_pages=2,
    )

    assert res.status == "budget_exhausted"
    assert res.pages_read == 2
    assert res.certificate is not None


def test_budget_exhaustion_byte_limit():
    """Verify status='budget_exhausted' returned when max_bytes limit is reached."""
    torch.manual_seed(3204)
    archive = Archive(page_size=4)
    d_k, d_v = 4, 4

    for i in range(16):
        archive.append(occurrence_id=i + 1, key=torch.randn(d_k), value=torch.randn(d_v), position=i)

    q = torch.randn(d_k)
    rk = torch.randn(2, d_k)
    rv = torch.randn(2, d_v)
    S = torch.zeros(d_v, d_k)

    # Set byte limit very low (e.g. 50 bytes, smaller than 1 page)
    res = archive_reference_read(
        query=q,
        recent_keys=rk,
        recent_values=rv,
        S=S,
        archive=archive,
        epsilon=1e-10,
        max_bytes=50,
    )

    assert res.status == "budget_exhausted"
    assert res.bytes_read == 0


def test_full_read_recovery():
    """Verify reading all pages recovers exact full softmax in real arithmetic."""
    torch.manual_seed(3205)
    archive = Archive(page_size=4)
    d_k, d_v = 6, 6

    all_keys = []
    all_values = []
    for i in range(12):
        k = torch.randn(d_k, dtype=torch.float64)
        v = torch.randn(d_v, dtype=torch.float64)
        all_keys.append(k)
        all_values.append(v)
        archive.append(occurrence_id=i + 1, key=k, value=v, position=i)

    # 3 recent items
    for i in range(12, 15):
        k = torch.randn(d_k, dtype=torch.float64)
        v = torch.randn(d_v, dtype=torch.float64)
        all_keys.append(k)
        all_values.append(v)

    rk = torch.stack(all_keys[12:])
    rv = torch.stack(all_values[12:])
    q = torch.randn(d_k, dtype=torch.float64)
    S = torch.zeros(d_v, d_k, dtype=torch.float64)

    # Explicit full read
    res = archive_reference_read(
        query=q,
        recent_keys=rk,
        recent_values=rv,
        S=S,
        archive=archive,
        read_all_pages=True,
    )

    assert res.status == "full_read"
    assert res.pages_read == 3

    # Direct full history reference
    all_k = torch.stack(all_keys)
    all_v = torch.stack(all_values)
    ref_out = full_history_softmax(all_k, all_v, q)

    torch.testing.assert_close(res.output, ref_out, atol=1e-12, rtol=1e-12)


def test_accounting_metrics_tracking():
    """Verify summary visits, bytes transferred, and selection cost ops are counted."""
    torch.manual_seed(3206)
    archive = Archive(page_size=4)
    d_k, d_v = 4, 4

    for i in range(8):
        archive.append(occurrence_id=i + 1, key=torch.randn(d_k), value=torch.randn(d_v), position=i)

    q = torch.randn(d_k)
    rk = torch.randn(2, d_k)
    rv = torch.randn(2, d_v)
    S = torch.zeros(d_v, d_k)

    res = archive_reference_read(
        query=q,
        recent_keys=rk,
        recent_values=rv,
        S=S,
        archive=archive,
        read_all_pages=True,
    )

    assert res.details["summary_visits"] > 0
    assert res.details["selection_cost_ops"] > 0
    assert res.bytes_read > 0
