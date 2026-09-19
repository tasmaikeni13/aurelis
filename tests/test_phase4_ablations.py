"""Unit tests for Phase 4 ablations."""

from __future__ import annotations

import math
import pytest
import torch

from aurelis import (
    AblationRecord,
    run_delayed_vs_immediate_ablation,
    run_mass_correction_ablation,
    run_recurrence_dimension_ablation,
    run_residual_selection_ablation,
    run_summary_quality_ablation,
    run_transport_vs_local_ablation,
)


def test_delayed_vs_immediate_ablation():
    rec = run_delayed_vs_immediate_ablation(context_length=32, window_size=8, page_size=4)
    assert isinstance(rec, AblationRecord)
    assert rec.metric_name == "composite_cost"
    assert rec.value_a >= 0.0
    assert rec.value_b >= 0.0


def test_transport_vs_local_ablation():
    rec = run_transport_vs_local_ablation(context_length=32, window_size=8, page_size=4)
    assert isinstance(rec, AblationRecord)
    assert rec.metric_name == "composite_cost"
    assert rec.value_a >= 0.0
    assert rec.value_b >= 0.0
    assert math.isfinite(rec.diff)


def test_recurrence_dimension_ablation():
    records = run_recurrence_dimension_ablation(dimensions=(8, 16), context_length=32, window_size=8, page_size=4)
    assert len(records) == 2
    assert records[0]["dimension"] == 8
    assert records[1]["dimension"] == 16
    assert records[1]["working_bytes"] > records[0]["working_bytes"]


def test_mass_correction_ablation():
    rec = run_mass_correction_ablation(context_length=32, window_size=8, page_size=4)
    assert isinstance(rec, AblationRecord)
    assert rec.metric_name == "actual_l2_error"
    assert rec.value_a < rec.value_b  # Corrected completion has strictly lower error than unweighted sum


def test_residual_selection_ablation():
    rec = run_residual_selection_ablation(context_length=32, window_size=8, page_size=4)
    assert isinstance(rec, AblationRecord)
    assert rec.metric_name == "composite_cost"


def test_summary_quality_ablation():
    rec = run_summary_quality_ablation(context_length=32, window_size=8, page_size=4)
    assert isinstance(rec, AblationRecord)
    assert rec.metric_name == "composite_cost"
    assert rec.value_a <= rec.value_b  # Tight summary costs less or equal to loose
