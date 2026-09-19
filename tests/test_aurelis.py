"""Comprehensive unit tests for standalone AURELIS architecture."""

from __future__ import annotations

import pytest
import torch

from aurelis import (
    bounded_read,
    completed_read,
    consume,
    gated_delta_update,
    initial_state,
    local_attention,
    occurrence_partition,
    pointwise_envelope,
    read,
    residual_certificate,
)


def test_gated_delta_update_exact_write():
    """Unit key write reproduction: beta=1 on unit key writes the value at key (Eq. 2)."""
    torch.manual_seed(42)
    d_k, d_v = 16, 8
    S = torch.randn(d_v, d_k)
    key = torch.randn(d_k)
    key = key / torch.norm(key)  # Unit norm
    val = torch.randn(d_v)

    S_next = gated_delta_update(S, key, val, alpha=1.0, beta=1.0)
    pred = S_next @ key
    torch.testing.assert_close(pred, val, atol=1e-5, rtol=1e-5)


def test_gated_delta_update_linearity():
    """Query linearity: evaluating S^+ on q1 + q2 equals eval(q1) + eval(q2)."""
    torch.manual_seed(137)
    d_k, d_v = 16, 8
    S = torch.randn(d_v, d_k)
    key = torch.randn(d_k)
    val = torch.randn(d_v)
    q1 = torch.randn(d_k)
    q2 = torch.randn(d_k)

    S_next = gated_delta_update(S, key, val, alpha=0.9, beta=0.8)
    out_sum = S_next @ (q1 + q2)
    expected = (S_next @ q1) + (S_next @ q2)
    torch.testing.assert_close(out_sum, expected, atol=1e-6, rtol=1e-6)


def test_delta_transition_nonexpansion():
    """Rank-one transition nonexpansiveness when beta >= 0 and beta ||k||^2 <= 2 (Eq. 5)."""
    torch.manual_seed(2026)
    d_k = 32
    key = torch.randn(d_k)
    key = key / torch.norm(key)  # ||k|| = 1
    beta = 1.5  # beta * ||k||^2 = 1.5 <= 2

    # For any row error x, ||x - beta (k^T x) k|| <= ||x||
    for _ in range(20):
        x = torch.randn(d_k)
        x_trans = x - beta * torch.dot(key, x) * key
        assert torch.norm(x_trans) <= torch.norm(x) + 1e-6


def test_bounded_read_linear_reproduction():
    """Linear reproduction (Eq. 4): when S = W and v_i = W k_i, r(q) = W q exactly."""
    torch.manual_seed(101)
    d_k, d_v = 12, 12
    W = torch.randn(d_v, d_k)
    window = 5
    keys = torch.randn(window, d_k)
    values = keys @ W.T  # v_i = W k_i

    query = torch.randn(d_k)
    r_q = bounded_read(query, keys, values, S=W, kappa=1.0)
    expected = W @ query
    torch.testing.assert_close(r_q, expected, atol=1e-5, rtol=1e-5)


def test_completed_read_balance():
    """Completed read balance (Eq. 8): (z_S + z_O) y_hat = n_S + z_O prior."""
    torch.manual_seed(99)
    d_v = 8
    zs = 2.5
    zh = 1.5
    ns = torch.randn(d_v)
    prior = torch.randn(d_v)

    y_hat = completed_read(zs, zh, ns, prior)
    lhs = (zs + zh) * y_hat
    rhs = ns + zh * prior
    torch.testing.assert_close(lhs, rhs, atol=1e-6, rtol=1e-6)


def test_completed_read_full_softmax_recovery():
    """Empty unread endpoint: when estimated unread mass is 0, y_hat = n_S / z_S."""
    torch.manual_seed(77)
    d_v = 8
    zs = 3.2
    ns = torch.randn(d_v)
    prior = torch.randn(d_v)

    y_hat = completed_read(zs, 0.0, ns, prior)
    expected = ns / zs
    torch.testing.assert_close(y_hat, expected, atol=1e-6, rtol=1e-6)


def test_residual_certificate_bound():
    """Deterministic residual certificate (Eq. 10): bound >= ||truth - approx||."""
    torch.manual_seed(333)
    d_v = 6
    zs = 4.0
    zo_true = 2.0
    ns = torch.randn(d_v)
    no_true = torch.randn(d_v)
    prior = torch.randn(d_v)

    # Truth = (ns + no_true) / (zs + zo_true)
    y_true = (ns + no_true) / (zs + zo_true)

    # Conservative unread mass bounds
    lo = 1.5
    hi = 2.5
    zh_est = (lo + hi) / 2.0  # 2.0
    res_true = torch.norm(no_true - zo_true * prior).item()
    res_bound = res_true + 0.5  # Valid upper bound

    y_approx = completed_read(zs, zh_est, ns, prior)
    cert = residual_certificate(
        selected_mass=zs,
        unread_lower=lo,
        unread_upper=hi,
        residual_bound=res_bound,
        prior=prior,
        completed=y_approx,
    )

    actual_err = torch.norm(y_true - y_approx).item()
    assert cert.bound >= actual_err - 1e-6


def test_streaming_state_partition_and_handoff():
    """Disjoint causal partitioning and recurrence handoff."""
    torch.manual_seed(444)
    d_k, d_v, window = 8, 8, 4
    state = initial_state(d_k, d_v, window)

    for step in range(10):
        k = torch.randn(d_k)
        v = torch.randn(d_v)
        state = consume(state, k, v, occurrence_id=step)

        evicted, recent = occurrence_partition(state)
        # Disjointness
        assert set(evicted).isdisjoint(set(recent))
        # Total set covers all history up to step
        assert sorted(evicted + recent) == list(range(step + 1))
        # Window size invariant
        assert len(recent) <= window
        if step + 1 <= window:
            assert len(evicted) == 0
        else:
            assert len(evicted) == (step + 1) - window


def test_streaming_read():
    """Streaming bounded read returns well-formed ReadResult."""
    d_k, d_v, window = 8, 8, 3
    state = initial_state(d_k, d_v, window)
    for i in range(5):
        state = consume(state, torch.randn(d_k), torch.randn(d_v))

    res = read(state, torch.randn(d_k))
    assert res.mode == "bounded"
    assert res.status == "approximate"
    assert res.output.shape == (d_v,)
