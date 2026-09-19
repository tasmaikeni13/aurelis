"""Tests for Phase 3: computable page envelopes, key boxes, value centers, radii, and unread cover."""

import math
import pytest
import torch

from aurelis.archive import Archive
from aurelis.certificate import (
    compute_outward_page_envelope,
    compute_outward_score_interval,
    validate_page_summary,
    validate_unread_cover,
)
from aurelis.types import ArchiveEntry, PageDescriptor


def test_key_box_coordinate_containment():
    """Verify that every entry key strictly lies within key_min and key_max."""
    torch.manual_seed(3001)
    archive = Archive(page_size=4)
    d_k, d_v = 8, 8

    # Add 12 observations (3 pages)
    for i in range(12):
        k = torch.randn(d_k)
        v = torch.randn(d_v)
        archive.append(occurrence_id=i + 1, key=k, value=v, position=i)

    pages = archive.get_pages_and_partial()
    assert len(pages) == 3

    for p in pages:
        entries = archive.get_page_entries(p.page_id)
        assert len(entries) == 4
        is_valid, reason = validate_page_summary(p, entries)
        assert is_valid, f"Validation failed: {reason}"

        for e in entries:
            assert torch.all(e.key >= p.key_min - 1e-6)
            assert torch.all(e.key <= p.key_max + 1e-6)


def test_value_center_and_radius_containment():
    """Verify that every entry value lies within value_radius of value_center."""
    torch.manual_seed(3002)
    archive = Archive(page_size=4)
    d_k, d_v = 8, 8

    for i in range(8):
        k = torch.randn(d_k)
        v = torch.randn(d_v) * 2.0
        archive.append(occurrence_id=i + 1, key=k, value=v, position=i)

    pages = archive.get_pages_and_partial()
    for p in pages:
        entries = archive.get_page_entries(p.page_id)
        for e in entries:
            dist = float(torch.linalg.vector_norm(e.value.double() - p.value_center.double()).item())
            assert dist <= p.value_radius + 1e-6


def test_outward_score_interval_enclosure():
    """Verify outward score bounds strictly enclose exact dot products for all keys in box."""
    torch.manual_seed(3003)
    d_k = 16
    q = torch.randn(d_k, dtype=torch.float64)

    # Construct random keys and form bounding box
    keys = torch.randn(10, d_k, dtype=torch.float64)
    k_min = torch.min(keys, dim=0).values
    k_max = torch.max(keys, dim=0).values

    ell_out, u_out, delta_dot = compute_outward_score_interval(q, k_min, k_max, kappa=1.5, dtype=torch.float64)

    assert delta_dot >= 0.0
    for i in range(10):
        exact_score = float((1.5 * torch.dot(q, keys[i])).item())
        assert ell_out <= exact_score + 1e-15, f"Lower bound violated: {ell_out} > {exact_score}"
        assert exact_score <= u_out + 1e-15, f"Upper bound violated: {exact_score} > {u_out}"


def test_outward_page_envelope_shifted_exponentials():
    """Verify shifted L_j, U_j, b_j are positive, finite, and L_j <= U_j."""
    torch.manual_seed(3004)
    archive = Archive(page_size=4)
    d_k, d_v = 8, 8

    for i in range(4):
        archive.append(occurrence_id=i + 1, key=torch.randn(d_k), value=torch.randn(d_v), position=i)

    pages = archive.get_pages_and_partial()
    page = pages[0]
    q = torch.randn(d_k)
    prior = torch.randn(d_v)

    env = compute_outward_page_envelope(page, q, prior, kappa=1.0, m_shift=0.5)

    assert env["Lj"] > 0.0
    assert env["Uj"] >= env["Lj"]
    assert env["bj"] >= 0.0
    assert math.isfinite(env["Lj"])
    assert math.isfinite(env["Uj"])
    assert math.isfinite(env["bj"])


def test_partial_page_unsealed_handling():
    """Verify that unsealed tail entries are packaged into a partial page descriptor."""
    archive = Archive(page_size=4)
    d_k, d_v = 4, 4

    # Append 6 entries: 1 sealed page of 4, plus 1 partial page of 2
    for i in range(6):
        archive.append(occurrence_id=i + 1, key=torch.randn(d_k), value=torch.randn(d_v), position=i)

    pages = archive.get_pages_and_partial()
    assert len(pages) == 2
    assert pages[0].sealed is True
    assert pages[0].count == 4
    assert pages[1].sealed is False
    assert pages[1].count == 2
    assert pages[1].occurrence_ids == (5, 6)

    # Summary validation on partial page
    entries = archive.get_page_entries(pages[1].page_id)
    is_valid, reason = validate_page_summary(pages[1], entries)
    assert is_valid, reason


def test_partial_page_per_query_causal_mask():
    """Verify that per-query causal cutoff filters sealed pages into causal views."""
    archive = Archive(page_size=4)
    d_k, d_v = 4, 4

    # Append 8 entries (2 sealed pages)
    for i in range(8):
        archive.append(occurrence_id=i + 1, key=torch.randn(d_k), value=torch.randn(d_v), position=i)

    # Cutoff at occurrence_id=6 (includes page 0 fully, page 1 partially with 2 entries)
    pages = archive.get_pages_and_partial(causal_cutoff=6)
    assert len(pages) == 2
    assert pages[0].count == 4
    assert pages[0].sealed is True
    assert pages[1].count == 2
    assert pages[1].sealed is False
    assert pages[1].occurrence_ids == (5, 6)


def test_summary_validator_detects_corrupt_summary():
    """Verify validate_page_summary detects artificially contracted box or radius."""
    archive = Archive(page_size=4)
    d_k, d_v = 4, 4

    for i in range(4):
        archive.append(occurrence_id=i + 1, key=torch.randn(d_k), value=torch.randn(d_v), position=i)

    page = archive.get_pages_and_partial()[0]
    entries = archive.get_page_entries(page.page_id)

    # 1. Corrupt key bounds: inverted (min > max)
    corrupt_page_inverted = PageDescriptor(
        page_id=page.page_id,
        count=page.count,
        key_min=page.key_max + 1.0,  # Unsound: min > max
        key_max=page.key_max,
        value_center=page.value_center,
        value_radius=page.value_radius,
        occurrence_ids=page.occurrence_ids,
        start_pos=page.start_pos,
        end_pos=page.end_pos,
        sealed=page.sealed,
    )
    is_valid, reason = validate_page_summary(corrupt_page_inverted, entries)
    assert not is_valid
    assert "key_min > key_max" in reason

    # 2. Corrupt key bounds: valid ordering (min <= max) but shifted so keys escape
    corrupt_page_box = PageDescriptor(
        page_id=page.page_id,
        count=page.count,
        key_min=page.key_min + 10.0,
        key_max=page.key_max + 10.0,
        value_center=page.value_center,
        value_radius=page.value_radius,
        occurrence_ids=page.occurrence_ids,
        start_pos=page.start_pos,
        end_pos=page.end_pos,
        sealed=page.sealed,
    )
    is_valid, reason = validate_page_summary(corrupt_page_box, entries)
    assert not is_valid
    assert "escapes bounding box" in reason

    # 2. Corrupt radius by halving it
    corrupt_page_radius = PageDescriptor(
        page_id=page.page_id,
        count=page.count,
        key_min=page.key_min,
        key_max=page.key_max,
        value_center=page.value_center,
        value_radius=page.value_radius * 0.1,  # Unsound!
        occurrence_ids=page.occurrence_ids,
        start_pos=page.start_pos,
        end_pos=page.end_pos,
        sealed=page.sealed,
    )
    is_valid, reason = validate_page_summary(corrupt_page_radius, entries)
    assert not is_valid
    assert "exceeds radius" in reason


def test_unread_cover_disjoint_and_complete():
    """Verify validate_unread_cover checks disjointness and completeness."""
    archive = Archive(page_size=4)
    d_k, d_v = 4, 4

    for i in range(12):
        archive.append(occurrence_id=i + 1, key=torch.randn(d_k), value=torch.randn(d_v), position=i)

    pages = archive.get_pages_and_partial()
    expected_ids = list(range(1, 13))

    # Valid cover
    is_valid, reason = validate_unread_cover(pages, expected_ids)
    assert is_valid, reason

    # Missing ID
    is_valid, reason = validate_unread_cover(pages[:2], expected_ids)
    assert not is_valid
    assert "missing=" in reason

    # Overlapping duplicate ID
    duplicate_pages = list(pages) + [pages[0]]
    is_valid, reason = validate_unread_cover(duplicate_pages, expected_ids)
    assert not is_valid
    assert "Duplicate occurrence ID" in reason


def test_key_ball_containment_and_tightness():
    """Verify Revision 1.1 key ball score bounds contain true scores and tighten box intervals."""
    torch.manual_seed(3005)
    d_k = 16
    keys = torch.randn(8, d_k, dtype=torch.float64)
    q = torch.randn(d_k, dtype=torch.float64)

    k_min = torch.min(keys, dim=0).values
    k_max = torch.max(keys, dim=0).values
    k_cen = torch.mean(keys, dim=0)
    k_rad = float(torch.max(torch.linalg.vector_norm(keys - k_cen, dim=-1)).item())

    # Box-only interval
    ell_box, u_box, _ = compute_outward_score_interval(q, k_min, k_max, dtype=torch.float64)

    # Combined box + ball interval
    ell_comb, u_comb, _ = compute_outward_score_interval(
        q, k_min, k_max, dtype=torch.float64, key_center=k_cen, key_radius=k_rad
    )

    # True inner products
    true_scores = [float(torch.dot(q, k).item()) for k in keys]
    min_s, max_s = min(true_scores), max(true_scores)

    # 1. Containment holds for both
    assert ell_box <= min_s + 1e-12 and u_box >= max_s - 1e-12
    assert ell_comb <= min_s + 1e-12 and u_comb >= max_s - 1e-12

    # 2. Combined is at least as tight as box
    assert ell_comb >= ell_box - 1e-12
    assert u_comb <= u_box + 1e-12
    assert (u_comb - ell_comb) <= (u_box - ell_box) + 1e-12

