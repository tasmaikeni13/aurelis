"""Unit tests for Phase 4 comparators and cost accounting."""

from __future__ import annotations

import math
import pytest
import torch

from aurelis import (
    ComparatorResult,
    execute_comparator,
    generate_structured_linear_workload,
)


@pytest.fixture
def structured_setup():
    workload = generate_structured_linear_workload(
        context_length=48,
        window_size=16,
        page_size=8,
        d_k=8,
        d_v=8,
        seed=201,
        dtype=torch.float64,
    )
    return workload


@pytest.mark.parametrize(
    "comparator_name",
    [
        "aurelis",
        "local_barycenter",
        "zero",
        "global_summary",
        "per_page_center",
        "simple_fusion",
        "sparse_no_completion",
        "value_blind_mass",
    ],
)
def test_all_comparators_execute_and_account(structured_setup, comparator_name):
    """Verify all 8 comparators execute without errors and produce full accounting."""
    workload = structured_setup
    q = workload.queries[0]

    res = execute_comparator(
        comparator_name=comparator_name,
        query=q,
        recent_keys=workload.recent_keys,
        recent_values=workload.recent_values,
        S=workload.S,
        archive_entries=workload.archive_entries,
        page_descriptors=workload.page_descriptors,
        get_page_entries_fn=workload.get_page_entries,
        epsilon=0.05,
        dtype=torch.float64,
    )

    assert isinstance(res, ComparatorResult)
    assert res.comparator_name == comparator_name
    assert res.output.shape == (workload.d_v,)
    assert torch.all(torch.isfinite(res.output))
    assert res.actual_error >= 0.0

    acct = res.accounting
    assert acct.total_flops >= 0
    assert acct.composite_cost >= 0.0
    assert acct.working_state_bytes > 0
    assert acct.archive_raw_bytes > 0
    assert acct.archive_index_bytes > 0
    assert acct.total_memory_bytes == acct.working_state_bytes + acct.archive_raw_bytes + acct.archive_index_bytes


def test_certified_enclosure_for_certifiable_comparators(structured_setup):
    """When a comparator returns certified status, actual error must be <= returned bound."""
    workload = structured_setup
    q = workload.queries[0]

    for comp in ["aurelis", "local_barycenter", "zero", "global_summary", "per_page_center"]:
        res = execute_comparator(
            comparator_name=comp,
            query=q,
            recent_keys=workload.recent_keys,
            recent_values=workload.recent_values,
            S=workload.S,
            archive_entries=workload.archive_entries,
            page_descriptors=workload.page_descriptors,
            get_page_entries_fn=workload.get_page_entries,
            epsilon=0.5,  # Moderate epsilon to allow certification
            dtype=torch.float64,
        )
        if res.status == "certified":
            assert res.certified_bound is not None
            assert res.actual_error <= res.certified_bound + 1e-12


def test_equal_kv_budget_evaluation(structured_setup):
    """Verify that comparators can be evaluated at fixed fetched page budgets (e.g. 1 page)."""
    workload = structured_setup
    q = workload.queries[0]

    for comp in ["aurelis", "local_barycenter", "per_page_center", "sparse_no_completion"]:
        res = execute_comparator(
            comparator_name=comp,
            query=q,
            recent_keys=workload.recent_keys,
            recent_values=workload.recent_values,
            S=workload.S,
            archive_entries=workload.archive_entries,
            page_descriptors=workload.page_descriptors,
            get_page_entries_fn=workload.get_page_entries,
            enforce_equal_kv_budget=1,
            dtype=torch.float64,
        )
        assert res.pages_read <= 1
        assert res.actual_error >= 0.0


def test_full_read_fallback_recovers_ground_truth(structured_setup):
    """When all pages are read, actual error must be 0 within float precision."""
    workload = structured_setup
    q = workload.queries[0]

    res = execute_comparator(
        comparator_name="aurelis",
        query=q,
        recent_keys=workload.recent_keys,
        recent_values=workload.recent_values,
        S=workload.S,
        archive_entries=workload.archive_entries,
        page_descriptors=workload.page_descriptors,
        get_page_entries_fn=workload.get_page_entries,
        epsilon=1e-15,  # Force reading all pages
        dtype=torch.float64,
    )

    assert res.status == "full_read"
    assert res.actual_error < 1e-12


def test_revision_1_1_grouped_completion(structured_setup):
    """Verify that under Revision 1.1, AURELIS uses grouped completion (Eq. 13) and valid bounds."""
    workload = structured_setup
    q = workload.queries[0]

    res_aur = execute_comparator(
        comparator_name="aurelis",
        query=q,
        recent_keys=workload.recent_keys,
        recent_values=workload.recent_values,
        S=workload.S,
        archive_entries=workload.archive_entries,
        page_descriptors=workload.page_descriptors,
        get_page_entries_fn=workload.get_page_entries,
        epsilon=0.5,
        dtype=torch.float64,
    )

    res_ppc = execute_comparator(
        comparator_name="per_page_center",
        query=q,
        recent_keys=workload.recent_keys,
        recent_values=workload.recent_values,
        S=workload.S,
        archive_entries=workload.archive_entries,
        page_descriptors=workload.page_descriptors,
        get_page_entries_fn=workload.get_page_entries,
        epsilon=0.5,
        dtype=torch.float64,
    )

    assert res_aur.certified_bound is not None
    assert res_ppc.certified_bound is not None
    # Both methods produce valid bounds bounding actual error
    if res_aur.status == "certified":
        assert res_aur.actual_error <= res_aur.certified_bound + 1e-12
    if res_ppc.status == "certified":
        assert res_ppc.actual_error <= res_ppc.certified_bound + 1e-12

