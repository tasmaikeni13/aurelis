"""Tests for Phase 3 stress conditions, pathologies, and failure semantics.

Verifies:
1. Underflow (extreme negative scores)
2. Overflow (extreme positive scores)
3. Nearly cancelled numerators
4. Zero unread mass
5. Partial pages
6. Negative query coordinates
7. Deliberate corrupt bounds injection (validation detects, rejects certification)
8. Missing pages (returns archive_error)
9. NaNs in inputs (returns invalid_interval)
10. I/O timeout (returns archive_unavailable)
"""

import math
import pytest
import torch

from aurelis.archive import Archive, archive_reference_read
from aurelis.types import ReadResult


def test_stress_underflow():
    """Verify common shift prevents underflow/subnormal collapse on extreme negative scores."""
    torch.manual_seed(3301)
    archive = Archive(page_size=4)
    d_k, d_v = 4, 4

    for i in range(8):
        archive.append(occurrence_id=i + 1, key=torch.ones(d_k) * 100.0, value=torch.randn(d_v), position=i)

    # Extreme negative query: scores will be around -500.0
    q = -torch.ones(d_k) * 5.0
    rk = torch.ones(2, d_k) * 100.0
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

    assert res.status == "full_read"
    assert torch.all(torch.isfinite(res.output))
    assert not torch.any(torch.isnan(res.output))


def test_stress_overflow():
    """Verify common log-sum-exp shift prevents overflow on extreme positive scores."""
    torch.manual_seed(3302)
    archive = Archive(page_size=4)
    d_k, d_v = 4, 4

    for i in range(8):
        archive.append(occurrence_id=i + 1, key=torch.ones(d_k) * 100.0, value=torch.randn(d_v), position=i)

    # Extreme positive query: scores around +500.0
    q = torch.ones(d_k) * 5.0
    rk = torch.ones(2, d_k) * 100.0
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

    assert res.status == "full_read"
    assert torch.all(torch.isfinite(res.output))
    assert not torch.any(torch.isnan(res.output))


def test_stress_nearly_cancelled_numerators():
    """Verify cancellation stability when opposing values nearly sum to zero."""
    torch.manual_seed(3303)
    archive = Archive(page_size=4)
    d_k, d_v = 4, 4

    # Half values +10.0, half values -10.0
    for i in range(4):
        archive.append(occurrence_id=i + 1, key=torch.ones(d_k) * 0.1, value=torch.ones(d_v) * 10.0, position=i)
    for i in range(4, 8):
        archive.append(occurrence_id=i + 1, key=torch.ones(d_k) * 0.1, value=-torch.ones(d_v) * 10.0, position=i)

    q = torch.ones(d_k) * 0.1
    rk = torch.ones(2, d_k) * 0.1
    rv = torch.zeros(2, d_v)
    S = torch.zeros(d_v, d_k)

    res = archive_reference_read(
        query=q,
        recent_keys=rk,
        recent_values=rv,
        S=S,
        archive=archive,
        read_all_pages=True,
    )

    assert res.status == "full_read"
    assert torch.all(torch.isfinite(res.output))
    # Due to symmetry, output should be close to 0.0
    assert torch.linalg.vector_norm(res.output).item() < 1e-4


def test_stress_zero_unread_mass():
    """Verify clean full_read status when remote archive is empty."""
    torch.manual_seed(3304)
    d_k, d_v = 4, 4
    rk = torch.randn(4, d_k)
    rv = torch.randn(4, d_v)
    q = torch.randn(d_k)
    S = torch.zeros(d_v, d_k)

    res = archive_reference_read(
        query=q,
        recent_keys=rk,
        recent_values=rv,
        S=S,
        archive=None,  # No remote archive
    )

    assert res.status == "full_read"
    assert res.pages_read == 0
    assert res.certificate is None


def test_stress_partial_pages():
    """Verify handling of odd history lengths with partial pages at tail."""
    torch.manual_seed(3305)
    archive = Archive(page_size=4)
    d_k, d_v = 4, 4

    # 13 items = 3 sealed pages + 1 partial page of 1 item
    for i in range(13):
        archive.append(occurrence_id=i + 1, key=torch.randn(d_k), value=torch.randn(d_v), position=i)

    q = torch.randn(d_k)
    rk = torch.randn(3, d_k)
    rv = torch.randn(3, d_v)
    S = torch.zeros(d_v, d_k)

    res = archive_reference_read(
        query=q,
        recent_keys=rk,
        recent_values=rv,
        S=S,
        archive=archive,
        read_all_pages=True,
    )

    assert res.status == "full_read"
    assert res.pages_read == 4  # 3 sealed + 1 partial


def test_stress_negative_query_coordinates():
    """Verify key boxes handle negative query coordinates correctly (swapping bounds)."""
    torch.manual_seed(3306)
    archive = Archive(page_size=4)
    d_k, d_v = 4, 4

    for i in range(8):
        archive.append(occurrence_id=i + 1, key=torch.randn(d_k), value=torch.randn(d_v), position=i)

    # Strictly negative query
    q = -torch.abs(torch.randn(d_k)) - 0.5
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

    assert res.status == "full_read"
    assert torch.all(torch.isfinite(res.output))


def test_stress_corrupt_bounds_injection():
    """Deliberately inject unsound bounds; verify validation detects them and rejects certified status."""
    torch.manual_seed(3307)
    archive = Archive(page_size=4)
    d_k, d_v = 4, 4

    for i in range(8):
        archive.append(occurrence_id=i + 1, key=torch.randn(d_k), value=torch.randn(d_v), position=i)

    q = torch.randn(d_k)
    rk = torch.randn(2, d_k)
    rv = torch.randn(2, d_v)
    S = torch.zeros(d_v, d_k)

    # Inject corrupt bounds where L > U
    res = archive_reference_read(
        query=q,
        recent_keys=rk,
        recent_values=rv,
        S=S,
        archive=archive,
        epsilon=1.0,  # Even with large epsilon, MUST NOT certify!
        inject_corrupt_bounds=True,
    )

    assert res.status != "certified", "Corrupted bounds must NEVER receive certified status!"
    assert res.status in ("invalid_interval", "invalid_state")


def test_stress_missing_pages():
    """Simulate missing page from archive storage; verify returns archive_error."""
    torch.manual_seed(3308)
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
        epsilon=1e-10,
        inject_missing_page=True,
    )

    assert res.status == "archive_error"
    assert res.certificate is not None
    assert "missing" in res.details["error"]


def test_stress_nans_and_infs():
    """Verify NaN or Inf in inputs triggers invalid_interval and refuses certification."""
    d_k, d_v = 4, 4
    archive = Archive(page_size=4)
    for i in range(4):
        archive.append(occurrence_id=i + 1, key=torch.randn(d_k), value=torch.randn(d_v), position=i)

    nan_q = torch.tensor([float("nan"), 1.0, 0.0, 0.0])
    rk = torch.randn(2, d_k)
    rv = torch.randn(2, d_v)
    S = torch.zeros(d_v, d_k)

    res = archive_reference_read(
        query=nan_q,
        recent_keys=rk,
        recent_values=rv,
        S=S,
        archive=archive,
        epsilon=1.0,
    )

    assert res.status == "invalid_interval"
    assert res.certificate is None


def test_stress_io_timeout():
    """Verify simulated I/O timeout returns archive_unavailable and never certified."""
    torch.manual_seed(3309)
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
        epsilon=1e-10,
        simulate_timeout=True,
    )

    assert res.status == "archive_unavailable"
    assert "timeout" in res.details["error"].lower()
