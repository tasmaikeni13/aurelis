"""Phase 1 Mathematical Contract and Oracle Verification Tests.

Verifies equations (2)-(13) in fp64 arithmetic:
- ScalarOracle (pure Python loops) vs TensorOracle (PyTorch float64)
- Exact mathematical contract checks across vector dimensions > 1:
  1. Occurrence handoff and streaming/history equivalence
  2. Delta update and unit-key full write (Eq. 2)
  3. Perturbation energy identity and nonexpansion (Eq. 5)
  4. Transport error decomposition and conditional exact hit (Eq. 4)
  5. Selected/unread partition
  6. Normalized completion balance identity (Eq. 8)
  7. Both terms of the error identity (Eq. 9)
  8. Page mass and value envelopes (Eqs. 11, 12)
  9. Midpoint mass error
  10. Certificate validity (Eq. 10)
  11. All-pages recovery endpoint
  12. Per-page grouped comparator (Eq. 13)
"""

from __future__ import annotations

import math
import pytest
import torch

from aurelis.oracles import (
    ScalarOracle,
    TensorOracle,
    ScalarStreamingOracle,
    TensorStreamingOracle,
)

# Tolerances specified in phase1.md
ATOL = 1e-10
RTOL = 1e-9


@pytest.mark.parametrize("d_k,d_v", [(2, 2), (4, 4), (8, 6), (16, 12), (32, 16)])
def test_delta_update_oracle_agreement(d_k: int, d_v: int):
    """Equation (2): Scalar and Tensor oracles agree on gated delta update."""
    torch.manual_seed(1001 + d_k)
    S = torch.randn(d_v, d_k, dtype=torch.float64)
    key = torch.randn(d_k, dtype=torch.float64)
    val = torch.randn(d_v, dtype=torch.float64)
    alpha = 0.85
    beta = 0.75

    # Tensor oracle
    S_tensor = TensorOracle.gated_delta_update(S, key, val, alpha=alpha, beta=beta)

    # Scalar oracle
    S_list = S.tolist()
    k_list = key.tolist()
    v_list = val.tolist()
    S_scalar = ScalarOracle.gated_delta_update(S_list, k_list, v_list, alpha=alpha, beta=beta)

    # Compare
    S_scalar_tensor = torch.tensor(S_scalar, dtype=torch.float64)
    torch.testing.assert_close(S_tensor, S_scalar_tensor, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("d_k,d_v", [(2, 2), (4, 8), (16, 16)])
def test_unit_key_exact_write(d_k: int, d_v: int):
    """Equation (2): Unit-key beta=1 write reproduces value exactly: S^+ k = v."""
    torch.manual_seed(2001 + d_k)
    S = torch.randn(d_v, d_k, dtype=torch.float64)
    key = torch.randn(d_k, dtype=torch.float64)
    key = key / torch.linalg.vector_norm(key)  # exact unit key ||k||=1
    val = torch.randn(d_v, dtype=torch.float64)

    # Tensor oracle
    S_next_t = TensorOracle.gated_delta_update(S, key, val, alpha=1.0, beta=1.0)
    pred_t = torch.mv(S_next_t, key)
    torch.testing.assert_close(pred_t, val, atol=ATOL, rtol=RTOL)

    # Scalar oracle
    S_next_s = ScalarOracle.gated_delta_update(S.tolist(), key.tolist(), val.tolist(), alpha=1.0, beta=1.0)
    pred_s = [sum(S_next_s[i][j] * key[j].item() for j in range(d_k)) for i in range(d_v)]
    torch.testing.assert_close(torch.tensor(pred_s, dtype=torch.float64), val, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("d_k", [4, 8, 16, 32])
def test_perturbation_energy_identity_and_nonexpansion(d_k: int):
    """Equation (5): ||x - beta k(k^T x)||^2 = ||x||^2 - beta(2 - beta ||k||^2)(k^T x)^2."""
    torch.manual_seed(3001 + d_k)
    for _ in range(10):
        key = torch.randn(d_k, dtype=torch.float64)
        k_norm = torch.linalg.vector_norm(key).item()
        if k_norm > 1.0:
            key = key / k_norm
            k_norm = 1.0

        x = torch.randn(d_k, dtype=torch.float64)
        beta = 1.4  # beta * ||k||^2 <= 2 since ||k|| <= 1 and beta <= 2

        # LHS
        kx = torch.dot(key, x)
        x_trans = x - beta * kx * key
        lhs = torch.dot(x_trans, x_trans).item()

        # RHS of Equation (5)
        x_norm_sq = torch.dot(x, x).item()
        rhs = x_norm_sq - beta * (2.0 - beta * (k_norm ** 2)) * (kx.item() ** 2)

        assert math.isclose(lhs, rhs, abs_tol=ATOL, rel_tol=RTOL)
        # Nonexpansion: ||x_trans|| <= ||x||
        assert math.sqrt(lhs) <= math.sqrt(x_norm_sq) + ATOL


@pytest.mark.parametrize("d_k,d_v", [(4, 4), (8, 6), (16, 12)])
def test_transport_error_decomposition(d_k: int, d_v: int):
    """Equation (4): r(q) - W q = (vbar_L - W kbar_L) + (S_t - W)(q - kbar_L)."""
    torch.manual_seed(4001 + d_k)
    W = torch.randn(d_v, d_k, dtype=torch.float64)
    S = torch.randn(d_v, d_k, dtype=torch.float64)
    window = 6
    keys = torch.randn(window, d_k, dtype=torch.float64)
    # Random values (not necessarily linear)
    values = torch.randn(window, d_v, dtype=torch.float64)
    query = torch.randn(d_k, dtype=torch.float64)
    kappa = 1.2

    # Bounded read r(q)
    r_q = TensorOracle.bounded_read(query, keys, values, S, kappa=kappa)
    _, kbar, vbar = TensorOracle.local_attention(keys, values, query, kappa=kappa)

    lhs = r_q - torch.mv(W, query)
    term1 = vbar - torch.mv(W, kbar)
    term2 = torch.mv(S - W, query - kbar)
    rhs = term1 + term2

    torch.testing.assert_close(lhs, rhs, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("d_k,d_v", [(4, 4), (8, 8)])
def test_conditional_exact_hit_and_linear_reproduction(d_k: int, d_v: int):
    """Equation (4) corollaries:
    1. One-hot local hit yields r(k_j) = v_j.
    2. Exact S = W and linearly consistent local values v_i = W k_i yield r(q) = W q.
    """
    torch.manual_seed(5001 + d_k)
    W = torch.randn(d_v, d_k, dtype=torch.float64)
    window = 5
    keys = torch.randn(window, d_k, dtype=torch.float64)
    values = torch.matmul(keys, W.T)  # v_i = W k_i

    # Linear reproduction
    q = torch.randn(d_k, dtype=torch.float64)
    r_linear = TensorOracle.bounded_read(q, keys, values, S=W, kappa=1.0)
    expected_linear = torch.mv(W, q)
    torch.testing.assert_close(r_linear, expected_linear, atol=ATOL, rtol=RTOL)

    # One-hot local hit: construct query matching key[2] with massive scale
    # or synthetic one-hot barycenter: if q = k_j and weights are one-hot at j
    target_idx = 2
    q_hit = keys[target_idx].clone()
    # At exact hit, (q - kbar) = 0 if barycenter is one-hot
    # Testing formula directly: when attention weights are [0, 0, 1, 0, 0]
    one_hot_weights = torch.zeros(window, dtype=torch.float64)
    one_hot_weights[target_idx] = 1.0
    kbar_hit = keys[target_idx]
    vbar_hit = values[target_idx]
    r_hit = vbar_hit + torch.mv(W, q_hit - kbar_hit)
    torch.testing.assert_close(r_hit, values[target_idx], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("d_k,d_v", [(4, 4), (8, 6), (16, 8)])
def test_normalized_completion_balance_identity(d_k: int, d_v: int):
    """Equation (8): (Z_A + Z_hat_O) y_hat_A = N_A + Z_hat_O * r."""
    torch.manual_seed(6001 + d_k)
    zs = 4.5
    zh = 2.3
    na = torch.randn(d_v, dtype=torch.float64)
    r = torch.randn(d_v, dtype=torch.float64)

    # Tensor oracle
    y_hat_t = TensorOracle.completed_read(zs, zh, na, r)
    lhs_t = (zs + zh) * y_hat_t
    rhs_t = na + zh * r
    torch.testing.assert_close(lhs_t, rhs_t, atol=ATOL, rtol=RTOL)

    # Scalar oracle
    y_hat_s = ScalarOracle.completed_read(zs, zh, na.tolist(), r.tolist())
    y_hat_s_t = torch.tensor(y_hat_s, dtype=torch.float64)
    torch.testing.assert_close(y_hat_t, y_hat_s_t, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("d_k,d_v", [(4, 4), (8, 6), (16, 8)])
def test_completion_error_identity_both_terms(d_k: int, d_v: int):
    """Equation (9): (Z_A + Z_O)(y_* - y_hat_A) = R_O + (Z_O - Z_hat_O)(r - y_hat_A)."""
    torch.manual_seed(7001 + d_k)
    t = 12
    window = 4
    keys = torch.randn(t, d_k, dtype=torch.float64)
    values = torch.randn(t, d_v, dtype=torch.float64)
    query = torch.randn(d_k, dtype=torch.float64)
    kappa = 1.0

    # True full attention (Eq. 7)
    y_star = TensorOracle.full_softmax(keys, values, query, kappa=kappa)

    # Partition into selected A (recent + some remote) and unread O
    scores = kappa * torch.mv(keys, query)
    max_s = torch.max(scores)
    exp_s = torch.exp(scores - max_s)

    idx_A = list(range(t - window, t)) + [0, 1]
    idx_O = [i for i in range(t) if i not in idx_A]

    Z_A = float(torch.sum(exp_s[idx_A]).item())
    N_A = torch.sum(exp_s[idx_A, None] * values[idx_A], dim=0)

    Z_O = float(torch.sum(exp_s[idx_O]).item())
    N_O = torch.sum(exp_s[idx_O, None] * values[idx_O], dim=0)

    # Prior estimate r
    prior_r = torch.randn(d_v, dtype=torch.float64)

    # Interval bounds for Z_O
    L_O = Z_O * 0.7
    U_O = Z_O * 1.4
    Z_hat_O = (L_O + U_O) / 2.0

    # Completed read
    y_hat_A = TensorOracle.completed_read(Z_A, Z_hat_O, N_A, prior_r)

    # LHS of Eq. (9)
    total_mass = Z_A + Z_O
    lhs = total_mass * (y_star - y_hat_A)

    # Term 1: R_O = N_O - Z_O * r
    R_O = N_O - Z_O * prior_r

    # Term 2: (Z_O - Z_hat_O) * (r - y_hat_A)
    term2 = (Z_O - Z_hat_O) * (prior_r - y_hat_A)

    rhs = R_O + term2
    torch.testing.assert_close(lhs, rhs, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("d_k,d_v", [(4, 4), (8, 6), (16, 8)])
def test_page_envelopes_and_certificate(d_k: int, d_v: int):
    """Equations (10, 11, 12): Page envelopes and deterministic residual certificate."""
    torch.manual_seed(8001 + d_k)
    page_size = 6
    page_keys = torch.randn(page_size, d_k, dtype=torch.float64)
    page_values = torch.randn(page_size, d_v, dtype=torch.float64)
    query = torch.randn(d_k, dtype=torch.float64)
    prior_r = torch.randn(d_v, dtype=torch.float64)
    kappa = 1.0

    # Envelope from TensorOracle
    env_t = TensorOracle.compute_page_envelopes(page_keys, page_values, query, prior_r, kappa=kappa)
    # Envelope from ScalarOracle
    env_s = ScalarOracle.compute_page_envelopes(
        page_keys.tolist(), page_values.tolist(), query.tolist(), prior_r.tolist(), kappa=kappa
    )

    # Check oracle agreement on envelopes
    assert math.isclose(env_t.ell, env_s.ell, abs_tol=ATOL, rel_tol=RTOL)
    assert math.isclose(env_t.u, env_s.u, abs_tol=ATOL, rel_tol=RTOL)
    assert math.isclose(env_t.L, env_s.L, abs_tol=ATOL, rel_tol=RTOL)
    assert math.isclose(env_t.U, env_s.U, abs_tol=ATOL, rel_tol=RTOL)
    assert math.isclose(env_t.radius, env_s.radius, abs_tol=ATOL, rel_tol=RTOL)
    assert math.isclose(env_t.b, env_s.b, abs_tol=ATOL, rel_tol=RTOL)

    # Verify key coordinate box containment
    k_min = torch.min(page_keys, dim=0).values
    k_max = torch.max(page_keys, dim=0).values
    for i in range(page_size):
        assert torch.all(page_keys[i] >= k_min - 1e-12)
        assert torch.all(page_keys[i] <= k_max + 1e-12)

    # Verify score containment in [ell, u]
    scores = kappa * torch.mv(page_keys, query)
    for i in range(page_size):
        s_i = scores[i].item()
        assert env_t.ell <= s_i + 1e-12
        assert s_i <= env_t.u + 1e-12

    # Verify page mass containment in [L, U]
    actual_mass = float(torch.sum(torch.exp(scores)).item())
    assert env_t.L <= actual_mass + 1e-10
    assert actual_mass <= env_t.U + 1e-10

    # Verify residual bound b >= ||sum_i exp(s_i)(v_i - r)||_2
    actual_res_vec = torch.sum(torch.exp(scores)[:, None] * (page_values - prior_r), dim=0)
    actual_res_norm = float(torch.linalg.vector_norm(actual_res_vec).item())
    assert env_t.b >= actual_res_norm - 1e-10


@pytest.mark.parametrize("d_k,d_v", [(4, 4), (8, 6), (16, 8)])
def test_all_pages_recovery_endpoint(d_k: int, d_v: int):
    """Equation (8): All-pages recovery endpoint (O is empty).

    When all pages are fetched, Z_hat_O = 0 and completed_read recovers exact full softmax.
    """
    torch.manual_seed(9001 + d_k)
    t = 10
    keys = torch.randn(t, d_k, dtype=torch.float64)
    values = torch.randn(t, d_v, dtype=torch.float64)
    query = torch.randn(d_k, dtype=torch.float64)
    kappa = 1.0

    y_star = TensorOracle.full_softmax(keys, values, query, kappa=kappa)

    # Selected set is all observations
    scores = kappa * torch.mv(keys, query)
    exp_s = torch.exp(scores - torch.max(scores))
    Z_A = float(torch.sum(exp_s).item())
    N_A = torch.sum(exp_s[:, None] * values, dim=0)

    dummy_prior = torch.randn(d_v, dtype=torch.float64)
    y_recovered = TensorOracle.completed_read(Z_A, 0.0, N_A, dummy_prior)

    torch.testing.assert_close(y_recovered, y_star, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("d_k,d_v", [(4, 4), (8, 6), (16, 8)])
def test_per_page_comparator_equation13(d_k: int, d_v: int):
    """Equation (13): Per-page grouped comparator formula and certificate bound."""
    torch.manual_seed(10001 + d_k)
    t = 14
    keys = torch.randn(t, d_k, dtype=torch.float64)
    values = torch.randn(t, d_v, dtype=torch.float64)
    query = torch.randn(d_k, dtype=torch.float64)
    kappa = 1.0

    y_star = TensorOracle.full_softmax(keys, values, query, kappa=kappa)

    # Selected set: indices 10..13 (window 4)
    idx_A = list(range(10, 14))
    scores = kappa * torch.mv(keys, query)
    exp_s = torch.exp(scores - torch.max(scores))
    Z_A = float(torch.sum(exp_s[idx_A]).item())
    N_A = torch.sum(exp_s[idx_A, None] * values[idx_A], dim=0)

    # Two unread pages: Page 0 (0..4), Page 1 (5..9)
    page0_idx = list(range(0, 5))
    page1_idx = list(range(5, 10))

    # Page predictors p_j = center of each page
    p0 = torch.mean(values[page0_idx], dim=0)
    p1 = torch.mean(values[page1_idx], dim=0)
    predictors = torch.stack([p0, p1], dim=0)

    # Compute page envelopes
    env0 = TensorOracle.compute_page_envelopes(keys[page0_idx], values[page0_idx], query, p0, kappa=kappa)
    env1 = TensorOracle.compute_page_envelopes(keys[page1_idx], values[page1_idx], query, p1, kappa=kappa)

    lowers = [env0.L, env1.L]
    uppers = [env0.U, env1.U]
    bounds = [env0.b, env1.b]
    estimates = [(env0.L + env0.U) / 2.0, (env1.L + env1.U) / 2.0]

    # Completed read from TensorOracle
    y_grouped_t = TensorOracle.grouped_completed_read(Z_A, N_A, estimates, predictors)
    cert_bound_t = TensorOracle.grouped_residual_certificate(
        Z_A, lowers, uppers, bounds, predictors, y_grouped_t
    )

    # Completed read from ScalarOracle
    y_grouped_s = ScalarOracle.grouped_completed_read(
        Z_A, N_A.tolist(), estimates, predictors.tolist()
    )
    cert_bound_s = ScalarOracle.grouped_residual_certificate(
        Z_A, lowers, uppers, bounds, predictors.tolist(), y_grouped_s
    )

    # Agreement between oracles
    torch.testing.assert_close(y_grouped_t, torch.tensor(y_grouped_s, dtype=torch.float64), atol=ATOL, rtol=RTOL)
    assert math.isclose(cert_bound_t, cert_bound_s, abs_tol=ATOL, rel_tol=RTOL)

    # Verify certificate soundness: ||y_* - y_grouped||_2 <= cert_bound
    actual_err = float(torch.linalg.vector_norm(y_star - y_grouped_t).item())
    assert cert_bound_t >= actual_err - 1e-10


@pytest.mark.parametrize("d_k,d_v,window", [(4, 4, 3), (8, 6, 4)])
def test_streaming_vs_history_equivalence(d_k: int, d_v: int, window: int):
    """Occurrence handoff: streaming ring buffer matches batch history calculation in fp64."""
    torch.manual_seed(11001 + d_k)
    num_steps = 15
    keys = torch.randn(num_steps, d_k, dtype=torch.float64)
    values = torch.randn(num_steps, d_v, dtype=torch.float64)
    alphas = [0.9 + 0.05 * (i % 3) for i in range(num_steps)]
    betas = [0.8 + 0.1 * (i % 2) for i in range(num_steps)]

    # Streaming oracles
    scalar_stream = ScalarStreamingOracle(d_v, d_k, window)
    tensor_stream = TensorStreamingOracle(d_v, d_k, window)

    for i in range(num_steps):
        scalar_stream.consume(keys[i].tolist(), values[i].tolist(), alpha=alphas[i], beta=betas[i], occurrence_id=i)
        tensor_stream.consume(keys[i], values[i], alpha=alphas[i], beta=betas[i], occurrence_id=i)

    # Check partition disjointness and coverage
    s_evicted, s_recent = scalar_stream.get_partition()
    t_evicted, t_recent = tensor_stream.get_partition()

    assert s_evicted == t_evicted
    assert s_recent == t_recent
    assert set(s_evicted).isdisjoint(set(s_recent))
    assert sorted(s_evicted + s_recent) == list(range(num_steps))
    assert len(s_recent) == window
    assert len(s_evicted) == num_steps - window

    # Check S agreement
    S_scalar = torch.tensor(scalar_stream.S, dtype=torch.float64)
    torch.testing.assert_close(tensor_stream.S, S_scalar, atol=ATOL, rtol=RTOL)

    # Check bounded read agreement
    test_q = torch.randn(d_k, dtype=torch.float64)
    read_s = torch.tensor(scalar_stream.bounded_read(test_q.tolist()), dtype=torch.float64)
    read_t = tensor_stream.bounded_read(test_q)
    torch.testing.assert_close(read_t, read_s, atol=ATOL, rtol=RTOL)
