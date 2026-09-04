"""Unit and property tests for Phase 4 nonstationarity, composition, and capacity."""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from aurelis.nn_phase4 import DriftAwareAurelisBlock, MultiHopPointerChaser, Phase4SequenceModel
from aurelis.phase4_suites import Phase4SuiteGenerator


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_stationary_control_equivalence(device: torch.device) -> None:
    """When cue=0 and decay=1, drift-aware block matches stationary behavior exactly."""
    torch.manual_seed(42)
    B, L, D = 4, 24, 16
    d_model, heads, d_k, d_v, w = 32, 2, 8, 8, 8

    block = DriftAwareAurelisBlock(d_model, heads, d_k, d_v, w, shared_charts=True, learned_evidence=True).to(device)
    hidden = torch.randn(B, L, d_model, device=device)

    # Forward without cue
    out_stat, diag_stat = block(hidden, cue=None)
    # Forward with zero cue
    out_zero, diag_zero = block(hidden, cue=torch.zeros(B, L, device=device))

    assert torch.allclose(out_stat, out_zero, atol=1e-5)
    assert torch.allclose(diag_stat["g_B"], diag_zero["g_B"], atol=1e-5)
    assert torch.allclose(diag_stat["precision"], diag_zero["precision"], atol=1e-5)


def test_observable_drift_flushes_remote_state(device: torch.device) -> None:
    """A changepoint pulse (cue=1) discounts pre-changepoint precision."""
    torch.manual_seed(42)
    B, L, D = 2, 32, 16
    d_model, heads, d_k, d_v, w = 32, 2, 8, 8, 8

    block = DriftAwareAurelisBlock(d_model, heads, d_k, d_v, w, gamma_min=0.01).to(device)
    hidden = torch.randn(B, L, d_model, device=device)

    cue = torch.zeros(B, L, device=device)
    cue[:, 16] = 1.0  # Changepoint pulse at step 16

    out_drift, diag_drift = block(hidden, cue=cue)
    out_stat, diag_stat = block(hidden, cue=None)

    # After step 16, precision in drift-aware block should be smaller because older state was flushed
    prec_drift_norm = torch.linalg.matrix_norm(diag_drift["precision"][:, :, 20], ord="fro")
    prec_stat_norm = torch.linalg.matrix_norm(diag_stat["precision"][:, :, 20], ord="fro")
    assert bool(torch.all(prec_drift_norm < prec_stat_norm).item())


def test_evidence_weighting_and_tempering(device: torch.device) -> None:
    """Tempering clamps extreme precisions; valid precision decreases local noise variance h."""
    torch.manual_seed(42)
    B, L, D = 2, 16, 16
    d_model, heads, d_k, d_v, w = 16, 2, 4, 4, 6

    block = DriftAwareAurelisBlock(d_model, heads, d_k, d_v, w, beta_min=0.05, beta_max=50.0).to(device)
    hidden = torch.randn(B, L, d_model, device=device)

    # Test with extreme precision
    extreme_ev = torch.full((B, heads, L), 1000.0, device=device)
    _, diag_high = block(hidden, override_evidence=extreme_ev)
    assert bool(torch.all(diag_high["evidence"] <= 50.0).item())

    small_ev = torch.full((B, heads, L), 0.001, device=device)
    _, diag_low = block(hidden, override_evidence=small_ev)
    assert bool(torch.all(diag_low["evidence"] >= 0.05).item())

    # High precision should yield smaller local noise h than low precision
    assert bool(torch.all(diag_high["h"] < diag_low["h"]).item())


def test_multi_hop_pointer_chaser_rounds_and_latencies(device: torch.device) -> None:
    """MultiHopPointerChaser records round counts, vector outputs, confidences, and latencies."""
    torch.manual_seed(42)
    B, L = 2, 24
    d_model, heads, d_k, d_v, w = 16, 2, 4, 4, 6

    block = DriftAwareAurelisBlock(d_model, heads, d_k, d_v, w).to(device)
    hidden = torch.randn(B, L, d_model, device=device)
    q0 = torch.randn(B, d_k, device=device)

    chaser = MultiHopPointerChaser(block, d_k, d_v)
    res = chaser.chase_pointers(hidden, q0, max_hops=4, adaptive=False)

    assert len(res["hop_outputs"]) == 4
    assert len(res["hop_latencies_ms"]) == 4
    assert len(res["hop_confidences"]) == 4
    assert res["rounds_taken"] == 4
    assert res["total_latency_ms"] > 0
    assert res["operator_norm"] >= 0


def test_multi_hop_adaptive_stopping(device: torch.device) -> None:
    """Adaptive pointer chasing terminates when step change is below tolerance."""
    torch.manual_seed(42)
    B, L = 2, 20
    d_model, heads, d_k, d_v, w = 16, 2, 4, 4, 6

    block = DriftAwareAurelisBlock(d_model, heads, d_k, d_v, w).to(device)
    hidden = torch.randn(B, L, d_model, device=device)
    q0 = torch.randn(B, d_k, device=device)

    chaser = MultiHopPointerChaser(block, d_k, d_v)
    # High tolerance to guarantee early stopping
    res = chaser.chase_pointers(hidden, q0, max_hops=8, adaptive=True, tol=10.0)
    assert res["rounds_taken"] < 8


def test_phase4_suite_generator(device: torch.device) -> None:
    """Phase4SuiteGenerator produces well-formed batches across suites."""
    gen = Phase4SuiteGenerator(d_in=20, d_out=8, d_feat=8, default_window=8, seed=42, device=device)

    b_drift = gen.generate_operator_drift(4, length=32, drift_type="abrupt", observable=True)
    assert b_drift.x.shape == (4, 32, 20)
    assert b_drift.y.shape == (4, 32, 8)
    assert b_drift.cue is not None and b_drift.cue.shape == (4, 32)

    b_het = gen.generate_heterogeneous_precision(4, length=24, noise_distribution="heteroscedastic", corruption_type="inverted")
    assert b_het.true_precisions.shape == (4, 24)
    assert b_het.corrupted_precisions.shape == (4, 24)

    b_over = gen.generate_repeated_overrides(4, length=32, override_type="cache_override")
    assert b_over.x.shape == (4, 32, 20)

    b_cap = gen.generate_adversarial_capacity(4, length=32, num_associations=12)
    assert b_cap.x.shape == (4, 32, 20)

    hidden, q0, targets = gen.generate_pointer_chasing(4, length=32, hops=4)
    assert hidden.shape == (4, 32, 20)
    assert q0.shape == (4, 8)
    assert len(targets) == 4
    assert targets[0].shape == (4, 8)

    hidden_m, q0_m, targets_m = gen.generate_mixed_chain(4, length=32, pattern=["C", "R", "C", "R"])
    assert hidden_m.shape == (4, 32, 20)
    assert len(targets_m) == 4
    assert targets_m[0].shape == (4, 8)

