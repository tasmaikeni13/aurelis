"""Tests for Phase 3 enclosure validity: fp64 reference outputs lie within returned enclosures."""

import pytest
import torch

from aurelis.archive import Archive, archive_reference_read
from aurelis.functional import full_history_softmax


@pytest.mark.parametrize("d_k,d_v,page_size,num_entries", [
    (8, 8, 4, 12),
    (16, 16, 8, 24),
    (32, 16, 4, 16),
])
def test_dense_fp64_reference_within_returned_enclosure(d_k: int, d_v: int, page_size: int, num_entries: int):
    """Dense fp64 reference outputs must lie within valid returned enclosures: ||y_* - y_hat|| <= E_A + delta_num."""
    torch.manual_seed(3401 + num_entries)
    archive = Archive(page_size=page_size)

    all_keys = []
    all_values = []
    for i in range(num_entries):
        k = torch.randn(d_k, dtype=torch.float64)
        v = torch.randn(d_v, dtype=torch.float64)
        all_keys.append(k)
        all_values.append(v)
        archive.append(occurrence_id=i + 1, key=k, value=v, position=i)

    # 4 recent tokens
    for i in range(num_entries, num_entries + 4):
        k = torch.randn(d_k, dtype=torch.float64)
        v = torch.randn(d_v, dtype=torch.float64)
        all_keys.append(k)
        all_values.append(v)

    rk = torch.stack(all_keys[num_entries:])
    rv = torch.stack(all_values[num_entries:])
    q = torch.randn(d_k, dtype=torch.float64)
    S = torch.zeros(d_v, d_k, dtype=torch.float64)

    # Compute dense fp64 reference output y_*
    all_k = torch.stack(all_keys)
    all_v = torch.stack(all_values)
    y_star = full_history_softmax(all_k, all_v, q)

    # Test at several epsilon levels and page budgets
    for max_p in [0, 1, 2]:
        res = archive_reference_read(
            query=q,
            recent_keys=rk,
            recent_values=rv,
            S=S,
            archive=archive,
            epsilon=1e-3,
            max_pages=max_p,
        )

        if res.certificate is not None:
            actual_error = float(torch.linalg.vector_norm(y_star - res.output).item())
            total_bound = res.certificate.total_bound

            # Strict enclosure test!
            assert actual_error <= total_bound + 1e-12, (
                f"Enclosure violated! actual_error={actual_error:.6e} > total_bound={total_bound:.6e} "
                f"(E_A={res.certificate.approximation_bound:.6e}, delta_num={res.certificate.delta_num:.6e})"
            )


def test_bound_to_actual_error_ratio_greater_equal_one():
    """Verify that bound-to-actual error ratio is >= 1.0 on all certified approximations."""
    torch.manual_seed(3402)
    d_k, d_v, page_size = 12, 12, 4
    archive = Archive(page_size=page_size)

    all_keys = []
    all_values = []
    for i in range(16):
        k = torch.randn(d_k, dtype=torch.float64)
        v = torch.randn(d_v, dtype=torch.float64)
        all_keys.append(k)
        all_values.append(v)
        archive.append(occurrence_id=i + 1, key=k, value=v, position=i)

    rk = torch.randn(4, d_k, dtype=torch.float64)
    rv = torch.randn(4, d_v, dtype=torch.float64)
    all_keys.extend(list(rk))
    all_values.extend(list(rv))

    all_k = torch.stack(all_keys)
    all_v = torch.stack(all_values)

    q = torch.randn(d_k, dtype=torch.float64)
    S = torch.zeros(d_v, d_k, dtype=torch.float64)
    y_star = full_history_softmax(all_k, all_v, q)

    res = archive_reference_read(
        query=q,
        recent_keys=rk,
        recent_values=rv,
        S=S,
        archive=archive,
        epsilon=1e-2,
        max_pages=2,
    )

    if res.certificate is not None:
        actual_error = float(torch.linalg.vector_norm(y_star - res.output).item())
        if actual_error > 1e-12:
            ratio = res.certificate.total_bound / actual_error
            assert ratio >= 1.0, f"Bound/actual error ratio < 1.0: {ratio}"


def test_separate_reporting_of_approximation_and_numerical_allowance():
    """Verify that approximation bound and numerical allowance are reported separately."""
    torch.manual_seed(3403)
    d_k, d_v = 8, 8
    archive = Archive(page_size=4)
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
        epsilon=1e-2,
        max_pages=1,
    )

    assert res.certificate is not None
    assert hasattr(res.certificate, "approximation_bound")
    assert hasattr(res.certificate, "delta_num")
    assert hasattr(res.certificate, "total_bound")

    assert res.certificate.approximation_bound > 0.0
    assert res.certificate.delta_num > 0.0
    assert abs(res.certificate.total_bound - (res.certificate.approximation_bound + res.certificate.delta_num)) < 1e-12
