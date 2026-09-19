"""Unit tests for Phase 4 evaluation workloads."""

from __future__ import annotations

import pytest
import torch

from aurelis import (
    generate_abrupt_drift_workload,
    generate_boundary_retrieval_workload,
    generate_delayed_disambiguation_workload,
    generate_diffuse_attention_workload,
    generate_large_value_outliers_workload,
    generate_metadata_dominant_workload,
    generate_multihop_query_workload,
    generate_near_collisions_workload,
    generate_random_incompressible_workload,
    generate_repeated_keys_workload,
    generate_structured_linear_workload,
)


@pytest.mark.parametrize(
    "generator_fn,name",
    [
        (generate_structured_linear_workload, "structured_linear"),
        (generate_random_incompressible_workload, "random_incompressible"),
        (generate_diffuse_attention_workload, "diffuse_attention"),
        (generate_large_value_outliers_workload, "large_value_outliers"),
        (generate_repeated_keys_workload, "repeated_keys"),
        (generate_near_collisions_workload, "near_collisions"),
        (generate_delayed_disambiguation_workload, "delayed_disambiguation"),
        (generate_abrupt_drift_workload, "abrupt_drift"),
        (generate_multihop_query_workload, "multihop_query"),
        (generate_boundary_retrieval_workload, "boundary_retrieval"),
        (generate_metadata_dominant_workload, "metadata_dominant"),
    ],
)
def test_all_workload_generators(generator_fn, name):
    """Verify all 11 workload generators produce valid WorkloadInstance objects."""
    workload = generator_fn()
    assert workload.name == name
    assert workload.context_length > 0
    assert len(workload.queries) > 0
    assert len(workload.all_keys) == workload.context_length
    assert len(workload.all_values) == workload.context_length
    assert workload.recent_keys.shape[0] <= workload.window_size
    assert workload.recent_values.shape[0] <= workload.window_size
    assert workload.S.shape == (workload.d_v, workload.d_k)
    assert workload.archive.verify_integrity()

    # Check page descriptors
    if workload.context_length > workload.window_size:
        assert len(workload.page_descriptors) > 0
        assert len(workload.archive_entries) == workload.context_length - workload.recent_keys.shape[0]
        # Test get_page_entries
        entries = workload.get_page_entries(0)
        assert len(entries) > 0
