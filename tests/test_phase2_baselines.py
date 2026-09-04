"""Equation tests on tiny hand-computable cases for all Phase 2 baselines."""

from __future__ import annotations

import math
import pytest
import torch

from aurelis import (
    aurelis_read,
    baseline_flops,
    baseline_parameter_count,
    baseline_state_bytes,
    cumulative_least_squares_mesa,
    delta_rule_memory,
    full_residual_fixed_gate,
    global_linear_attention,
    independent_inverse_variance_fusion,
    learned_local_remote_concat,
    learned_local_remote_sum,
    local_softmax_attention,
    native_hybrid_attention,
    remote_bayes_ridge,
)


def test_local_softmax_attention_hand_computable() -> None:
    # d_k=2, d_v=2, cache=2
    # k_1 = [1, 0], v_1 = [1, 0]
    # k_2 = [0, 1], v_2 = [0, 1]
    # q = [1, 0], tau = 1.0
    # dot(k_1, q) = 1, dot(k_2, q) = 0
    # weights = [e/(e+1), 1/(e+1)]
    # vbar = [e/(e+1), 1/(e+1)]
    # kbar = [e/(e+1), 1/(e+1)]
    keys = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    values = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    query = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)

    vbar, weights, kbar = local_softmax_attention(keys, values, query, temperature=1.0)

    e = math.e
    expected_w1 = e / (e + 1.0)
    expected_w2 = 1.0 / (e + 1.0)
    assert torch.allclose(weights[0, 0], torch.tensor([expected_w1, expected_w2], dtype=torch.float64), atol=1e-15)
    assert torch.allclose(vbar[0, 0], torch.tensor([expected_w1, expected_w2], dtype=torch.float64), atol=1e-15)
    assert torch.allclose(kbar[0, 0], torch.tensor([expected_w1, expected_w2], dtype=torch.float64), atol=1e-15)


def test_remote_bayes_ridge_hand_computable() -> None:
    # d_k=1, d_v=1
    # 1 observation: k=2, v=6, beta=1. Prior=2.0
    # P = 2.0 + 1 * (2*2) = 6.0
    # C = 1 * (6*2) = 12.0
    # M = C / P = 12 / 6 = 2.0
    # q = 3.0 => y = M * q = 2.0 * 3.0 = 6.0
    P = torch.tensor([[[[6.0]]]], dtype=torch.float64)
    C = torch.tensor([[[[12.0]]]], dtype=torch.float64)
    q = torch.tensor([[[3.0]]], dtype=torch.float64)

    y = remote_bayes_ridge(P, C, q)
    assert torch.allclose(y[0, 0], torch.tensor([6.0], dtype=torch.float64), atol=1e-15)


def test_global_linear_attention_hand_computable() -> None:
    # d_k=1, d_v=1
    # k_1 = [0.0], v_1 = [3.0] -> phi(0) = elu(0) + 1 = 1.0
    # k_2 = [0.0], v_2 = [7.0] -> phi(0) = 1.0
    # S = 3*1 + 7*1 = 10.0
    # z = 1 + 1 = 2.0
    # q = [0.0] -> phi(0) = 1.0
    # y = S * 1 / (z * 1) = 10 / 2 = 5.0
    keys = torch.tensor([[[[0.0], [0.0]]]], dtype=torch.float64)
    values = torch.tensor([[[[3.0], [7.0]]]], dtype=torch.float64)
    q = torch.tensor([[[0.0]]], dtype=torch.float64)

    y = global_linear_attention(keys, values, q)
    assert torch.allclose(y[0, 0], torch.tensor([5.0], dtype=torch.float64), atol=1e-14)


def test_delta_rule_memory_hand_computable() -> None:
    # d_k=1, d_v=1, beta=1.0, decay=1.0
    # Step 1: k_1=[1.0], v_1=[4.0]. S_0=0 => v_hat=0, err=4.0 => S_1 = 0 + 1*4*1 = 4.0
    # Step 2: k_2=[1.0], v_2=[10.0]. v_hat=4*1=4.0, err=6.0 => S_2 = 4 + 1*6*1 = 10.0
    # Query q=[1.0] => y = S_2 * 1.0 = 10.0
    keys = torch.tensor([[[[1.0], [1.0]]]], dtype=torch.float64)
    values = torch.tensor([[[[4.0], [10.0]]]], dtype=torch.float64)
    q = torch.tensor([[[1.0]]], dtype=torch.float64)

    y = delta_rule_memory(keys, values, q, beta=1.0, decay=1.0)
    assert torch.allclose(y[0, 0], torch.tensor([10.0], dtype=torch.float64), atol=1e-15)


def test_cumulative_least_squares_mesa_hand_computable() -> None:
    # d_k=1, d_v=1, prior=1.0
    # obs 1: k_1=[1.0], v_1=[3.0], beta=1
    # obs 2: k_2=[2.0], v_2=[5.0], beta=1
    # P = 1.0 + 1*(1) + 1*(4) = 6.0
    # C = 1*(3*1) + 1*(5*2) = 13.0
    # M = 13 / 6
    # q = [6.0] => y = (13/6)*6 = 13.0
    keys = torch.tensor([[[[1.0], [2.0]]]], dtype=torch.float64)
    values = torch.tensor([[[[3.0], [5.0]]]], dtype=torch.float64)
    evidence = torch.tensor([[[1.0, 1.0]]], dtype=torch.float64)
    q = torch.tensor([[[6.0]]], dtype=torch.float64)

    y = cumulative_least_squares_mesa(keys, values, evidence, q, prior=1.0)
    assert torch.allclose(y[0, 0], torch.tensor([13.0], dtype=torch.float64), atol=1e-15)


def test_learned_local_remote_sum_and_concat_hand_computable() -> None:
    y_local = torch.tensor([[[10.0]]], dtype=torch.float64)
    y_remote = torch.tensor([[[4.0]]], dtype=torch.float64)

    # alpha = 0.25 => (1-0.25)*4 + 0.25*10 = 3 + 2.5 = 5.5
    y_sum = learned_local_remote_sum(y_local, y_remote, alpha=0.25)
    assert torch.allclose(y_sum[0, 0], torch.tensor([5.5], dtype=torch.float64), atol=1e-15)

    # Concat with weights [0.5, 1.5] => 10*0.5 + 4*1.5 = 5 + 6 = 11.0
    weight = torch.tensor([[[0.5, 1.5]]], dtype=torch.float64)
    y_concat = learned_local_remote_concat(y_local, y_remote, weight)
    assert torch.allclose(y_concat[0, 0], torch.tensor([11.0], dtype=torch.float64), atol=1e-15)


def test_full_residual_fixed_gate_hand_computable() -> None:
    # d_k=1, d_v=1
    # P = [[2.0]], C = [[2.0]] => M = 1.0
    # Cache has 1 item: k=[1.0], v=[2.0], evidence=[1.0]
    # q = [2.0]
    # kbar = 1.0, vbar = 2.0, residual_query = 2.0 - 1.0 = 1.0
    # M(q - kbar) = 1.0 * 1.0 = 1.0
    # y = vbar + M(q - kbar) = 2.0 + 1.0 = 3.0
    P = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    C = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    keys = torch.tensor([[[[1.0]]]], dtype=torch.float64)
    values = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    evidence = torch.tensor([[[1.0]]], dtype=torch.float64)
    q = torch.tensor([[[2.0]]], dtype=torch.float64)

    y, _ = full_residual_fixed_gate(P, C, keys, values, evidence, q)
    assert torch.allclose(y[0, 0], torch.tensor([3.0], dtype=torch.float64), atol=1e-15)


def test_independent_inverse_variance_fusion_and_bayes_comparison() -> None:
    # P = [[2.0]], C = [[2.0]] => M = 1.0
    # Cache: 1 item k=[1.0], v=[2.0], beta=1.0 => h = 1.0/1.0 = 1.0
    # Query q = [2.0]
    # kbar = 1.0, vbar = 2.0
    # r = q - kbar = 1.0
    # p_q = 2.0 / 2.0 = 1.0
    # p_k = 1.0 / 2.0 = 0.5
    # p_r = 1.0 / 2.0 = 0.5
    # V_R = q * p_q = 2.0
    # V_H = h + r * p_r = 1.0 + 1.0 * 0.5 = 1.5
    # K_RH = q * p_r = 2.0 * 0.5 = 1.0
    #
    # Bayes gate:
    # D = h + kbar * p_k = 1.0 + 0.5 = 1.5
    # num = q * p_k = 2.0 * 0.5 = 1.0
    # g_B = 1.0 / 1.5 = 2/3
    #
    # Independent gate:
    # g_indep = V_R / (V_R + V_H) = 2.0 / 3.5 = 4/7
    P = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    C = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    keys = torch.tensor([[[[1.0]]]], dtype=torch.float64)
    values = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    evidence = torch.tensor([[[1.0]]], dtype=torch.float64)
    q = torch.tensor([[[2.0]]], dtype=torch.float64)

    y_indep, g_indep, diag = independent_inverse_variance_fusion(
        P, C, keys, values, evidence, q
    )
    assert torch.allclose(g_indep[0, 0], torch.tensor(4.0 / 7.0, dtype=torch.float64), atol=1e-15)
    # y_indep = M*q + g_indep * (vbar - M*kbar) = 1*2 + (4/7)*(2 - 1) = 18/7
    assert torch.allclose(y_indep[0, 0], torch.tensor([18.0 / 7.0], dtype=torch.float64), atol=1e-15)

    aurelis_out = aurelis_read(P, C, keys, values, evidence, q)
    assert torch.allclose(aurelis_out.diagnostics.g_B[0, 0], torch.tensor(2.0 / 3.0, dtype=torch.float64), atol=1e-15)
    # y_B = 2 + (2/3)*1 = 8/3
    assert torch.allclose(aurelis_out.bayes[0, 0], torch.tensor([8.0 / 3.0], dtype=torch.float64), atol=1e-15)

    # Theoretical conditional variance:
    # V(g) = (1-g)^2 V_R + g^2 V_H + 2 g (1-g) K_RH
    V_R = 2.0
    V_H = 1.5
    K_RH = 1.0
    var_bayes = (1 - 2/3)**2 * V_R + (2/3)**2 * V_H + 2 * (2/3) * (1/3) * K_RH  # 4/3 = 1.3333333333
    var_indep = (1 - 4/7)**2 * V_R + (4/7)**2 * V_H + 2 * (4/7) * (3/7) * K_RH  # 66/49 = 1.3469387755
    assert var_bayes < var_indep
    assert math.isclose(var_bayes, 4.0 / 3.0, abs_tol=1e-14)
    assert math.isclose(var_indep, 66.0 / 49.0, abs_tol=1e-14)


def test_native_hybrid_attention_hand_computable() -> None:
    # m=1, w=1, d_k=1, d_v=1
    # recurrent: k_rec=[1.0], v_rec=[10.0]
    # local: k_loc=[2.0], v_loc=[20.0]
    # q=[1.0], tau=1.0
    # scores: [1.0, 2.0]
    # softmax: [1/(1+e), e/(1+e)]
    # y = 10/(1+e) + 20e/(1+e) = (10 + 20e)/(1+e)
    rec_k = torch.tensor([[[[1.0]]]], dtype=torch.float64)
    rec_v = torch.tensor([[[[10.0]]]], dtype=torch.float64)
    loc_k = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    loc_v = torch.tensor([[[[20.0]]]], dtype=torch.float64)
    q = torch.tensor([[[1.0]]], dtype=torch.float64)

    y, weights = native_hybrid_attention(rec_k, rec_v, loc_k, loc_v, q, temperature=1.0)
    e = math.e
    expected_y = (10.0 + 20.0 * e) / (1.0 + e)
    assert torch.allclose(y[0, 0], torch.tensor([expected_y], dtype=torch.float64), atol=1e-14)


def test_cost_models_positive_and_consistent() -> None:
    d_k, d_v, w, m = 16, 16, 32, 8
    models = [
        "local_softmax",
        "remote_bayes",
        "global_linear",
        "delta_rule",
        "mesa",
        "learned_sum",
        "learned_concat",
        "independent_fusion",
        "full_residual",
        "aurelis",
        "native_hybrid",
    ]
    for model in models:
        state_b = baseline_state_bytes(model, d_k, d_v, w, recurrent_slots=m, dtype=torch.float64)
        params = baseline_parameter_count(model, d_k, d_v)
        fl = baseline_flops(model, d_k, d_v, w, recurrent_slots=m)
        assert state_b > 0
        assert params >= 0
        assert fl > 0
