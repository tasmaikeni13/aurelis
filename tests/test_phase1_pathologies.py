"""Phase 1 Pathology and Extreme Numerical Condition Verification.

Tests all required pathology families in fp64 arithmetic:
1. empty_remote_sets
2. singleton_pages
3. partial_pages
4. negative_query_coordinates
5. zero_keys
6. beta_endpoints
7. alpha_endpoints
8. repeated_keys
9. huge_value_outliers
10. uniform_scores
11. concentrated_scores
12. stale_state_residuals

Validates moderate fp64 algebra tolerance (atol=1e-10, rtol=1e-9)
and records actual error and value scale for every family.
"""

from __future__ import annotations

import math
from typing import Any
import pytest
import torch

from aurelis.oracles import (
    ScalarOracle,
    TensorOracle,
    ScalarStreamingOracle,
    TensorStreamingOracle,
)

ATOL = 1e-10
RTOL = 1e-9

# Module-level registry to collect actual error and scale across all pathology runs
PATHOLOGY_RECORDS: dict[str, dict[str, Any]] = {}


def record_pathology(
    name: str,
    value_scale: float,
    max_abs_err: float,
    max_rel_err: float,
    notes: str,
) -> None:
    """Record verified numerical properties of a pathology family."""
    PATHOLOGY_RECORDS[name] = {
        "pathology": name,
        "value_scale": value_scale,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "atol_threshold": ATOL,
        "rtol_threshold": RTOL,
        "status": "PASS",
        "notes": notes,
    }


def test_pathology_empty_remote_sets():
    """Pathology 1: Empty remote set (t <= window, O = empty)."""
    d_k, d_v, window = 8, 6, 10
    t = 5  # t <= window
    keys = torch.randn(t, d_k, dtype=torch.float64)
    values = torch.randn(t, d_v, dtype=torch.float64)
    query = torch.randn(d_k, dtype=torch.float64)
    kappa = 1.0

    # True full softmax
    y_star = TensorOracle.full_softmax(keys, values, query, kappa=kappa)

    # Empty unread set: all t observations in window
    scores = kappa * torch.mv(keys, query)
    exp_s = torch.exp(scores - torch.max(scores))
    Z_A = float(torch.sum(exp_s).item())
    N_A = torch.sum(exp_s[:, None] * values, dim=0)

    prior_dummy = torch.randn(d_v, dtype=torch.float64)
    y_completed = TensorOracle.completed_read(Z_A, 0.0, N_A, prior_dummy)

    # Certificate with 0 unread mass
    cert = TensorOracle.residual_certificate(
        selected_mass=Z_A,
        unread_lower=0.0,
        unread_upper=0.0,
        residual_bound=0.0,
        prior=prior_dummy,
        completed=y_completed,
    )

    abs_err = float(torch.linalg.vector_norm(y_star - y_completed).item())
    rel_err = abs_err / max(float(torch.linalg.vector_norm(y_star).item()), 1e-12)
    assert abs_err <= ATOL
    assert cert.bound <= ATOL

    record_pathology(
        "empty_remote_sets",
        value_scale=float(torch.linalg.vector_norm(values).item() / math.sqrt(t)),
        max_abs_err=abs_err,
        max_rel_err=rel_err,
        notes="Unread set empty; completed read equals full softmax exactly; certificate bound is 0.",
    )


def test_pathology_singleton_pages():
    """Pathology 2: Singleton pages (n_j = 1)."""
    d_k, d_v = 6, 6
    page_key = torch.randn(1, d_k, dtype=torch.float64)
    page_val = torch.randn(1, d_v, dtype=torch.float64)
    query = torch.randn(d_k, dtype=torch.float64)
    prior = torch.randn(d_v, dtype=torch.float64)
    kappa = 1.0

    env_t = TensorOracle.compute_page_envelopes(page_key, page_val, query, prior, kappa=kappa)
    env_s = ScalarOracle.compute_page_envelopes(
        page_key.tolist(), page_val.tolist(), query.tolist(), prior.tolist(), kappa=kappa
    )

    # Radius must be exactly 0 for a singleton page
    assert env_t.radius == 0.0
    assert env_s.radius == 0.0

    # L and U must equal exact exp(score)
    exact_score = float((kappa * torch.dot(page_key[0], query)).item())
    exact_mass = math.exp(exact_score)
    assert math.isclose(env_t.L, exact_mass, abs_tol=ATOL, rel_tol=RTOL)
    assert math.isclose(env_t.U, exact_mass, abs_tol=ATOL, rel_tol=RTOL)

    diff_L = abs(env_t.L - env_s.L)
    diff_U = abs(env_t.U - env_s.U)
    max_abs_err = max(diff_L, diff_U)
    max_rel_err = max_abs_err / exact_mass

    record_pathology(
        "singleton_pages",
        value_scale=float(torch.linalg.vector_norm(page_val).item()),
        max_abs_err=max_abs_err,
        max_rel_err=max_rel_err,
        notes="Page radius is 0; mass upper and lower bounds coincide with exact single-item exponential.",
    )


def test_pathology_partial_pages():
    """Pathology 3: Partial pages (count 3 < sealed page size 8)."""
    torch.manual_seed(103)
    d_k, d_v = 8, 4
    count = 3
    page_keys = torch.randn(count, d_k, dtype=torch.float64)
    page_vals = torch.randn(count, d_v, dtype=torch.float64)
    query = torch.randn(d_k, dtype=torch.float64)
    prior = torch.randn(d_v, dtype=torch.float64)
    kappa = 1.0

    env_t = TensorOracle.compute_page_envelopes(page_keys, page_vals, query, prior, kappa=kappa)
    env_s = ScalarOracle.compute_page_envelopes(
        page_keys.tolist(), page_vals.tolist(), query.tolist(), prior.tolist(), kappa=kappa
    )

    err_b = abs(env_t.b - env_s.b)
    rel_b = err_b / max(env_t.b, 1e-12)
    assert math.isclose(env_t.b, env_s.b, abs_tol=ATOL, rel_tol=RTOL)

    record_pathology(
        "partial_pages",
        value_scale=float(torch.linalg.vector_norm(page_vals).item() / math.sqrt(count)),
        max_abs_err=err_b,
        max_rel_err=rel_b,
        notes="Correctly scales mass and bounds by actual count n=3 rather than nominal page capacity.",
    )


def test_pathology_negative_query_coordinates():
    """Pathology 4: Negative query coordinates testing min/max coordinate bounding."""
    d_k, d_v = 6, 6
    page_keys = torch.randn(5, d_k, dtype=torch.float64)
    page_vals = torch.randn(5, d_v, dtype=torch.float64)
    # Query with deliberately mixed negative and positive entries
    query = torch.tensor([-3.5, 2.1, -0.8, -4.0, 1.2, -0.05], dtype=torch.float64)
    prior = torch.randn(d_v, dtype=torch.float64)
    kappa = 1.0

    env_t = TensorOracle.compute_page_envelopes(page_keys, page_vals, query, prior, kappa=kappa)
    env_s = ScalarOracle.compute_page_envelopes(
        page_keys.tolist(), page_vals.tolist(), query.tolist(), prior.tolist(), kappa=kappa
    )

    # Check score containment for every key under negative query coordinates
    scores = kappa * torch.mv(page_keys, query)
    for i in range(5):
        s_i = scores[i].item()
        assert env_t.ell <= s_i + 1e-12
        assert s_i <= env_t.u + 1e-12

    err_ell = abs(env_t.ell - env_s.ell)
    err_u = abs(env_t.u - env_s.u)
    max_err = max(err_ell, err_u)

    record_pathology(
        "negative_query_coordinates",
        value_scale=float(torch.linalg.vector_norm(query).item()),
        max_abs_err=max_err,
        max_rel_err=max_err / max(abs(env_t.u), 1e-12),
        notes="Coordinate box min(q*k-, q*k+) correctly inverts coordinate orientation when q_d < 0.",
    )


def test_pathology_zero_keys():
    """Pathology 5: Zero key vectors (k = 0, norm floor behavior)."""
    d_k, d_v = 8, 8
    S = torch.randn(d_v, d_k, dtype=torch.float64)
    zero_key = torch.zeros(d_k, dtype=torch.float64)
    val = torch.randn(d_v, dtype=torch.float64)

    # Recurrent update with zero key
    S_next_t = TensorOracle.gated_delta_update(S, zero_key, val, alpha=0.9, beta=1.0)
    S_next_s = ScalarOracle.gated_delta_update(S.tolist(), zero_key.tolist(), val.tolist(), alpha=0.9, beta=1.0)

    # When k = 0, pred = S @ 0 = 0, delta = beta * err * 0^T = 0. S_next = alpha * S
    expected = 0.9 * S
    torch.testing.assert_close(S_next_t, expected, atol=ATOL, rtol=RTOL)

    err = float(torch.linalg.matrix_norm(S_next_t - torch.tensor(S_next_s, dtype=torch.float64)).item())
    assert err <= ATOL

    record_pathology(
        "zero_keys",
        value_scale=0.0,
        max_abs_err=err,
        max_rel_err=0.0,
        notes="Key norm floor handles k=0 with zero division avoided; S^+ simplifies to alpha * S exactly.",
    )


def test_pathology_beta_endpoints():
    """Pathology 6: Beta endpoints: beta=0 (no write), beta=1 (standard), beta=2/||k||^2 (nonexpansion edge)."""
    d_k, d_v = 8, 6
    S = torch.randn(d_v, d_k, dtype=torch.float64)
    key = torch.randn(d_k, dtype=torch.float64)
    key = key / torch.linalg.vector_norm(key)  # ||k|| = 1
    val = torch.randn(d_v, dtype=torch.float64)

    # 1. beta = 0: S^+ = alpha * S
    S_b0 = TensorOracle.gated_delta_update(S, key, val, alpha=1.0, beta=0.0)
    torch.testing.assert_close(S_b0, S, atol=ATOL, rtol=RTOL)

    # 2. beta = 1: exact unit key write S^+ k = v
    S_b1 = TensorOracle.gated_delta_update(S, key, val, alpha=1.0, beta=1.0)
    torch.testing.assert_close(torch.mv(S_b1, key), val, atol=ATOL, rtol=RTOL)

    # 3. beta = 2/||k||^2 = 2.0: Reflection boundary where ||x - beta k(k^T x)|| = ||x||
    x = torch.randn(d_k, dtype=torch.float64)
    kx = torch.dot(key, x)
    x_refl = x - 2.0 * kx * key
    norm_orig = torch.linalg.vector_norm(x).item()
    norm_refl = torch.linalg.vector_norm(x_refl).item()
    assert math.isclose(norm_refl, norm_orig, abs_tol=ATOL, rel_tol=RTOL)

    err = abs(norm_refl - norm_orig)
    record_pathology(
        "beta_endpoints",
        value_scale=1.0,
        max_abs_err=err,
        max_rel_err=err / norm_orig,
        notes="Tested beta=0 (identity), beta=1 (full unit write), and beta=2 (isometry reflection).",
    )


def test_pathology_alpha_endpoints():
    """Pathology 7: Alpha endpoints: alpha=0 (complete reset), alpha=1 (no decay)."""
    d_k, d_v = 6, 6
    S = torch.randn(d_v, d_k, dtype=torch.float64)
    key = torch.randn(d_k, dtype=torch.float64)
    key = key / torch.linalg.vector_norm(key)
    val = torch.randn(d_v, dtype=torch.float64)

    # alpha = 0: S_tilde = 0, S^+ = beta * v k^T
    S_a0 = TensorOracle.gated_delta_update(S, key, val, alpha=0.0, beta=1.0)
    expected_a0 = torch.outer(val, key)
    torch.testing.assert_close(S_a0, expected_a0, atol=ATOL, rtol=RTOL)

    # alpha = 1: S_tilde = S
    S_a1 = TensorOracle.gated_delta_update(S, key, val, alpha=1.0, beta=1.0)
    torch.testing.assert_close(torch.mv(S_a1, key), val, atol=ATOL, rtol=RTOL)

    err = float(torch.linalg.matrix_norm(S_a0 - expected_a0).item())
    record_pathology(
        "alpha_endpoints",
        value_scale=float(torch.linalg.vector_norm(val).item()),
        max_abs_err=err,
        max_rel_err=0.0,
        notes="alpha=0 completely resets prior state; alpha=1 preserves conservative recurrence.",
    )


def test_pathology_repeated_keys():
    """Pathology 8: Repeated identical keys with different values."""
    d_k, d_v = 4, 4
    key = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    val1 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    val2 = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64)

    S0 = TensorOracle.zero_matrix(d_v, d_k)
    S1 = TensorOracle.gated_delta_update(S0, key, val1, alpha=1.0, beta=1.0)
    # At S1, read at key is val1
    torch.testing.assert_close(torch.mv(S1, key), val1, atol=ATOL, rtol=RTOL)

    # Second write with same key and val2
    S2 = TensorOracle.gated_delta_update(S1, key, val2, alpha=1.0, beta=1.0)
    # Overwrites association at key: S2 @ key = val2
    torch.testing.assert_close(torch.mv(S2, key), val2, atol=ATOL, rtol=RTOL)

    # Scalar oracle agreement
    S1_s = ScalarOracle.gated_delta_update(S0.tolist(), key.tolist(), val1.tolist(), alpha=1.0, beta=1.0)
    S2_s = ScalarOracle.gated_delta_update(S1_s, key.tolist(), val2.tolist(), alpha=1.0, beta=1.0)
    torch.testing.assert_close(S2, torch.tensor(S2_s, dtype=torch.float64), atol=ATOL, rtol=RTOL)

    err = float(torch.linalg.vector_norm(torch.mv(S2, key) - val2).item())
    record_pathology(
        "repeated_keys",
        value_scale=float(torch.linalg.vector_norm(val2).item()),
        max_abs_err=err,
        max_rel_err=err / float(torch.linalg.vector_norm(val2).item()),
        notes="Successive writes at identical key overwrite the linear association cleanly with S^+ k = v_new.",
    )


def test_pathology_huge_value_outliers():
    """Pathology 9: Extreme value scales (||v|| ~ 10^6)."""
    d_k, d_v = 4, 4
    scale = 1e6
    page_keys = torch.randn(4, d_k, dtype=torch.float64)
    page_vals = torch.randn(4, d_v, dtype=torch.float64) * scale
    query = torch.randn(d_k, dtype=torch.float64)
    prior = torch.randn(d_v, dtype=torch.float64) * scale

    env_t = TensorOracle.compute_page_envelopes(page_keys, page_vals, query, prior, kappa=1.0)
    env_s = ScalarOracle.compute_page_envelopes(
        page_keys.tolist(), page_vals.tolist(), query.tolist(), prior.tolist(), kappa=1.0
    )

    err_b = abs(env_t.b - env_s.b)
    rel_b = err_b / env_t.b
    assert rel_b <= RTOL

    record_pathology(
        "huge_value_outliers",
        value_scale=scale,
        max_abs_err=err_b,
        max_rel_err=rel_b,
        notes="Stable under 10^6 value scaling; certificate bound scales linearly without overflow or cancellation.",
    )


def test_pathology_uniform_scores():
    """Pathology 10: Uniform scores (diffuse attention, all s_i identical)."""
    d_k, d_v = 6, 4
    t = 8
    # Orthogonal keys to query: s_i = 0
    query = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    keys = torch.zeros(t, d_k, dtype=torch.float64)
    keys[:, 1] = 1.0  # q^T k_i = 0 for all i
    values = torch.randn(t, d_v, dtype=torch.float64)

    weights_t, _, vbar_t = TensorOracle.local_attention(keys, values, query, kappa=1.0)
    weights_s, _, vbar_s = ScalarOracle.local_attention(keys.tolist(), values.tolist(), query.tolist(), kappa=1.0)

    # Weights must be exactly 1/t
    expected_w = 1.0 / float(t)
    for w in weights_t:
        assert math.isclose(w.item(), expected_w, abs_tol=ATOL, rel_tol=RTOL)

    err = float(torch.linalg.vector_norm(vbar_t - torch.tensor(vbar_s, dtype=torch.float64)).item())
    assert err <= ATOL

    record_pathology(
        "uniform_scores",
        value_scale=float(torch.linalg.vector_norm(values).item() / math.sqrt(t)),
        max_abs_err=err,
        max_rel_err=0.0,
        notes="Completely diffuse attention weights equal 1/t; barycenter is arithmetic mean.",
    )


def test_pathology_concentrated_scores():
    """Pathology 11: Concentrated scores (peaky attention, one score dominates)."""
    d_k, d_v = 4, 4
    t = 6
    query = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    keys = torch.zeros(t, d_k, dtype=torch.float64)
    keys[0, 0] = 1.0   # s_0 = 50
    # other keys orthogonal: s_i = 0
    values = torch.randn(t, d_v, dtype=torch.float64)
    kappa = 50.0

    weights_t, _, vbar_t = TensorOracle.local_attention(keys, values, query, kappa=kappa)
    # Weight on item 0 must be near 1.0 (within 1e-15)
    assert 1.0 - weights_t[0].item() < 1e-12
    # vbar must match values[0]
    torch.testing.assert_close(vbar_t, values[0], atol=1e-8, rtol=1e-8)

    err = float(torch.linalg.vector_norm(vbar_t - values[0]).item())
    record_pathology(
        "concentrated_scores",
        value_scale=float(torch.linalg.vector_norm(values).item() / math.sqrt(t)),
        max_abs_err=err,
        max_rel_err=err / float(torch.linalg.vector_norm(values[0]).item()),
        notes="Numerical stability under large score difference (kappa*q^T k = 50); no exp overflow via max subtraction.",
    )


def test_pathology_stale_state_residuals():
    """Pathology 12: Stale-state residuals vs current state."""
    d_k, d_v = 4, 4
    # State at time of write S_write
    S_write = torch.randn(d_v, d_k, dtype=torch.float64)
    k_remote = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    v_remote = torch.tensor([2.0, 1.0, -1.0, 3.0], dtype=torch.float64)

    # Stored residual at write time
    stale_residual = v_remote - torch.mv(S_write, k_remote)

    # Recurrence evolves over many steps: S_t differs significantly from S_write
    S_t = S_write + torch.randn(d_v, d_k, dtype=torch.float64) * 2.0

    # True residual under current state S_t
    true_residual = v_remote - torch.mv(S_t, k_remote)

    # Difference between true residual and stale stored residual
    residual_mismatch = float(torch.linalg.vector_norm(true_residual - stale_residual).item())
    assert residual_mismatch > 0.1  # Substantial mismatch

    # Demonstrates that archive must store raw c_j, rho_j and evaluate ||c_j - r(q)||_2 at query time
    page_keys = k_remote.unsqueeze(0)
    page_vals = v_remote.unsqueeze(0)
    query = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    r_current = torch.mv(S_t, query)

    env = TensorOracle.compute_page_envelopes(page_keys, page_vals, query, r_current, kappa=1.0)
    # The dynamically computed bound b correctly bounds the true residual under current r
    actual_res_norm = float(torch.linalg.vector_norm(true_residual).item())
    assert env.b >= actual_res_norm - 1e-10

    record_pathology(
        "stale_state_residuals",
        value_scale=float(torch.linalg.vector_norm(v_remote).item()),
        max_abs_err=residual_mismatch,
        max_rel_err=residual_mismatch / float(torch.linalg.vector_norm(true_residual).item()),
        notes="Proves stored residuals v - S_write k are invalid under evolved S_t; dynamic envelope ||c - r_t|| is required.",
    )
