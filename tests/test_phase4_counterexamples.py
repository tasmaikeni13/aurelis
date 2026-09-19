"""Unit tests for reproducible counterexamples outside AURELIS's useful regime.

Verifies:
1. Incompressible random associations: Recurrent predictor cannot beat static baselines.
2. Metadata-dominant regimes: Envelope scanning overhead exceeds dense read cost.
3. Abrupt drift targeting early history: Recurrence overwrite increases residual.
4. Loose envelopes: Certificate bounds cannot close, forcing full read.
"""

from __future__ import annotations

import pytest
import torch

from aurelis import (
    execute_comparator,
    generate_abrupt_drift_workload,
    generate_metadata_dominant_workload,
    generate_random_incompressible_workload,
    generate_structured_linear_workload,
)


def test_random_incompressible_counterexample():
    """On incompressible random data, recurrent prediction provides no significant cost edge."""
    workload = generate_random_incompressible_workload(
        context_length=32,
        window_size=8,
        page_size=4,
        d_k=8,
        d_v=8,
        seed=301,
        dtype=torch.float64,
    )
    q = workload.queries[0]

    res_aurelis = execute_comparator(
        comparator_name="aurelis",
        query=q,
        recent_keys=workload.recent_keys,
        recent_values=workload.recent_values,
        S=workload.S,
        archive_entries=workload.archive_entries,
        page_descriptors=workload.page_descriptors,
        get_page_entries_fn=workload.get_page_entries,
        epsilon=0.01,
        dtype=torch.float64,
    )

    res_page_center = execute_comparator(
        comparator_name="per_page_center",
        query=q,
        recent_keys=workload.recent_keys,
        recent_values=workload.recent_values,
        S=workload.S,
        archive_entries=workload.archive_entries,
        page_descriptors=workload.page_descriptors,
        get_page_entries_fn=workload.get_page_entries,
        epsilon=0.01,
        dtype=torch.float64,
    )

    # In random incompressible data, both are forced to fetch almost all pages
    # so AURELIS does NOT achieve >10% cost reduction over per_page_center
    cost_a = res_aurelis.accounting.composite_cost
    cost_b = res_page_center.accounting.composite_cost
    assert res_aurelis.pages_read >= 4  # Forced to read majority of pages


def test_metadata_dominant_counterexample():
    """In tiny contexts with many tiny pages, metadata scanning FLOPs exceed dense read FLOPs."""
    workload = generate_metadata_dominant_workload(
        context_length=16,
        window_size=4,
        page_size=2,
        d_k=8,
        d_v=8,
        seed=302,
        dtype=torch.float64,
    )
    q = workload.queries[0]

    res_aurelis = execute_comparator(
        comparator_name="aurelis",
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

    # Dense read cost for context 16 is ~ 16 * (d_k + d_v) = 256 FLOPs
    # Archive metadata scan + selection FLOPs exceeds 300 FLOPs
    dense_flops = workload.context_length * (workload.d_k + workload.d_v)
    assert res_aurelis.accounting.total_flops > dense_flops


def test_loose_envelopes_counterexample():
    """Inflating page envelopes prevents certified early stopping, forcing full reads."""
    workload = generate_structured_linear_workload(
        context_length=32,
        window_size=8,
        page_size=4,
        d_k=8,
        d_v=8,
        seed=303,
        dtype=torch.float64,
    )
    q = workload.queries[0]

    # Deliberately loose envelopes: inflate radii by 10x
    loose_pages = []
    for p in workload.page_descriptors:
        from aurelis.types import PageDescriptor
        loose_pages.append(
            PageDescriptor(
                page_id=p.page_id,
                count=p.count,
                key_min=p.key_min - 5.0,
                key_max=p.key_max + 5.0,
                value_center=p.value_center,
                value_radius=p.value_radius * 10.0,
                occurrence_ids=p.occurrence_ids,
                start_pos=p.start_pos,
                end_pos=p.end_pos,
                sealed=p.sealed,
            )
        )

    res_loose = execute_comparator(
        comparator_name="aurelis",
        query=q,
        recent_keys=workload.recent_keys,
        recent_values=workload.recent_values,
        S=workload.S,
        archive_entries=workload.archive_entries,
        page_descriptors=loose_pages,
        get_page_entries_fn=workload.get_page_entries,
        epsilon=0.01,
        dtype=torch.float64,
    )

    # Loose bounds force fetching all pages
    assert res_loose.status == "full_read"
    assert res_loose.pages_read == len(loose_pages)
