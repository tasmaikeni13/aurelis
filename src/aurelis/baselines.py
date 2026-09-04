"""Transparent fp64 implementations and cost models for Phase 2 baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from .functional import aurelis_read, local_causal_softmax
from .types import ReadOutput


@dataclass(frozen=True)
class BaselineOutput:
    """Standardized output container for baseline readers."""

    name: str
    output: Tensor
    diagnostics: dict[str, Any]


def local_softmax_attention(
    keys: Tensor,
    values: Tensor,
    query: Tensor,
    *,
    temperature: Tensor | float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pure local causal softmax attention over the recent window."""

    cache_size = keys.shape[-2]
    batch_shape = query.shape[:-1]
    d_value = values.shape[-1]
    if cache_size == 0:
        return (
            values.new_zeros((*batch_shape, d_value)),
            keys.new_empty((*batch_shape, 0)),
            torch.zeros_like(query),
        )

    tau = torch.as_tensor(temperature, dtype=query.dtype, device=query.device)
    while tau.ndim < query.ndim - 1:
        tau = tau.unsqueeze(0)
    scores = torch.einsum("...nd,...d->...n", keys, query) * tau.unsqueeze(-1)
    weights = torch.softmax(scores, dim=-1)
    vbar = torch.einsum("...n,...nv->...v", weights, values)
    kbar = torch.einsum("...n,...nd->...d", weights, keys)
    return vbar, weights, kbar


def remote_bayes_ridge(
    precision: Tensor,
    cross: Tensor,
    query: Tensor,
    *,
    factor: Tensor | None = None,
) -> Tensor:
    """Remote Bayesian ridge linear map: y = C P^{-1} q."""

    if factor is None:
        factor = torch.linalg.cholesky(precision)
    p_query = torch.cholesky_solve(query.unsqueeze(-1), factor).squeeze(-1)
    return torch.einsum("...vd,...d->...v", cross, p_query)


def elu_plus_one(x: Tensor) -> Tensor:
    """Standard positive feature map: elu(x) + 1 >= 0."""
    return F.elu(x) + 1.0


def global_linear_attention(
    history_keys: Tensor,
    history_values: Tensor,
    query: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Global positive-feature linear attention over the entire history."""

    length = history_keys.shape[-2]
    batch_shape = query.shape[:-1]
    d_value = history_values.shape[-1]
    if length == 0:
        return history_values.new_zeros((*batch_shape, d_value))

    phi_k = elu_plus_one(history_keys)  # [..., length, d_k]
    phi_q = elu_plus_one(query)         # [..., d_k]

    # S = sum_t v_t phi(k_t)^T  [..., d_v, d_k]
    S = torch.einsum("...nv,...nd->...vd", history_values, phi_k)
    # z = sum_t phi(k_t)         [..., d_k]
    z = phi_k.sum(dim=-2)

    numerator = torch.einsum("...vd,...d->...v", S, phi_q)
    denominator = torch.einsum("...d,...d->...", z, phi_q).unsqueeze(-1)
    safe_denominator = torch.where(
        denominator.abs() > eps,
        denominator,
        torch.full_like(denominator, eps),
    )
    return numerator / safe_denominator


def delta_rule_memory(
    history_keys: Tensor,
    history_values: Tensor,
    query: Tensor,
    *,
    beta: float = 1.0,
    decay: float = 1.0,
) -> Tensor:
    """Sequential Gated-Delta / DeltaNet recurrent memory state."""

    length = history_keys.shape[-2]
    batch_shape = query.shape[:-1]
    d_key = query.shape[-1]
    d_value = history_values.shape[-1]
    if length == 0:
        return history_values.new_zeros((*batch_shape, d_value))

    # S starts at zero [..., d_value, d_key]
    S = history_values.new_zeros((*batch_shape, d_value, d_key))
    for t in range(length):
        k_t = history_keys[..., t, :]
        v_t = history_values[..., t, :]
        norm_k = torch.linalg.vector_norm(k_t, dim=-1, keepdim=True).clamp_min(1e-12)
        k_hat = k_t / norm_k
        v_pred = torch.einsum("...vd,...d->...v", S, k_hat)
        err = v_t - v_pred
        S = decay * S + beta * torch.einsum("...v,...d->...vd", err, k_hat)

    return torch.einsum("...vd,...d->...v", S, query)


def cumulative_least_squares_mesa(
    history_keys: Tensor,
    history_values: Tensor,
    history_evidence: Tensor,
    query: Tensor,
    *,
    prior: float = 1.0,
) -> Tensor:
    """Cumulative least squares (Mesa-style) over the entire prefix."""

    batch_shape = query.shape[:-1]
    d_key = query.shape[-1]
    d_value = history_values.shape[-1]
    length = history_keys.shape[-2]

    eye = torch.eye(d_key, dtype=query.dtype, device=query.device)
    precision = eye.expand(*batch_shape, d_key, d_key).clone() * prior
    if length > 0:
        precision = precision + torch.einsum(
            "...n,...ni,...nj->...ij", history_evidence, history_keys, history_keys
        )
        cross = torch.einsum(
            "...n,...nv,...nd->...vd", history_evidence, history_values, history_keys
        )
    else:
        cross = history_values.new_zeros((*batch_shape, d_value, d_key))

    factor = torch.linalg.cholesky(precision)
    p_query = torch.cholesky_solve(query.unsqueeze(-1), factor).squeeze(-1)
    return torch.einsum("...vd,...d->...v", cross, p_query)


def learned_local_remote_sum(
    y_local: Tensor,
    y_remote: Tensor,
    alpha: float = 0.5,
) -> Tensor:
    """Convex sum of local attention and remote ridge: (1-alpha)*remote + alpha*local."""
    return (1.0 - alpha) * y_remote + alpha * y_local


def learned_local_remote_concat(
    y_local: Tensor,
    y_remote: Tensor,
    weight: Tensor,
) -> Tensor:
    """Concatenation of local and remote predictions projected by weight."""
    # weight shape: [..., d_value, 2 * d_value]
    concat = torch.cat((y_local, y_remote), dim=-1)
    return torch.einsum("...c,...vc->...v", concat, weight)


def independent_inverse_variance_fusion(
    precision: Tensor,
    cross: Tensor,
    keys: Tensor,
    values: Tensor,
    evidence: Tensor,
    query: Tensor,
    *,
    temperature: Tensor | float = 1.0,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Inverse-variance fusion that incorrectly assumes endpoint independence (K_RH=0)."""

    weights, kbar, vbar, h = local_causal_softmax(
        keys, values, evidence, query, temperature
    )
    residual_query = query - kbar
    rhs = torch.stack((query, kbar, residual_query), dim=-1)
    factor = torch.linalg.cholesky(precision)
    solved = torch.cholesky_solve(rhs, factor)
    p_query, p_kbar, p_residual = solved.unbind(dim=-1)

    remote = torch.einsum("...vd,...d->...v", cross, p_query)
    mapped_kbar = torch.einsum("...vd,...d->...v", cross, p_kbar)
    innovation = vbar - mapped_kbar
    full_residual = remote + innovation

    V_R = torch.sum(query * p_query, dim=-1)
    V_H = h + torch.sum(residual_query * p_residual, dim=-1)
    K_RH = torch.sum(query * p_residual, dim=-1)

    # Heuristic: assumes K_RH = 0, so g_indep = V_R / (V_R + V_H)
    indep_denom = V_R + V_H
    safe_indep_denom = torch.where(
        torch.isfinite(indep_denom) & (indep_denom > 0),
        indep_denom,
        torch.ones_like(indep_denom),
    )
    cache_present = keys.shape[-2] > 0
    if cache_present:
        g_indep = torch.clamp(V_R / safe_indep_denom, 0.0, 1.0)
    else:
        g_indep = torch.zeros_like(V_R)

    output = remote + g_indep.unsqueeze(-1) * innovation
    diag = {
        "V_R": V_R,
        "V_H": V_H,
        "K_RH": K_RH,
        "g_indep": g_indep,
        "innovation": innovation,
        "h": h,
    }
    return output, g_indep, diag


def full_residual_fixed_gate(
    precision: Tensor,
    cross: Tensor,
    keys: Tensor,
    values: Tensor,
    evidence: Tensor,
    query: Tensor,
    *,
    temperature: Tensor | float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Full-residual output with g fixed to 1.0: y = vbar + M(q - kbar)."""

    weights, kbar, vbar, h = local_causal_softmax(
        keys, values, evidence, query, temperature
    )
    residual_query = query - kbar
    factor = torch.linalg.cholesky(precision)
    p_residual = torch.cholesky_solve(residual_query.unsqueeze(-1), factor).squeeze(-1)
    mapped_residual = torch.einsum("...vd,...d->...v", cross, p_residual)
    output = vbar + mapped_residual
    return output, {"vbar": vbar, "kbar": kbar, "residual_query": residual_query}


def native_hybrid_attention(
    recurrent_keys: Tensor,
    recurrent_values: Tensor,
    local_keys: Tensor,
    local_values: Tensor,
    query: Tensor,
    *,
    temperature: Tensor | float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Native-Hybrid-style joint softmax over recurrent slots and recent tokens."""

    if recurrent_keys.shape[-2] == 0 and local_keys.shape[-2] == 0:
        batch_shape = query.shape[:-1]
        d_val = local_values.shape[-1]
        return local_values.new_zeros((*batch_shape, d_val)), query.new_empty((*batch_shape, 0))

    if recurrent_keys.shape[-2] == 0:
        joint_keys = local_keys
        joint_values = local_values
    elif local_keys.shape[-2] == 0:
        joint_keys = recurrent_keys
        joint_values = recurrent_values
    else:
        joint_keys = torch.cat((recurrent_keys, local_keys), dim=-2)
        joint_values = torch.cat((recurrent_values, local_values), dim=-2)

    tau = torch.as_tensor(temperature, dtype=query.dtype, device=query.device)
    while tau.ndim < query.ndim - 1:
        tau = tau.unsqueeze(0)
    scores = torch.einsum("...nd,...d->...n", joint_keys, query) * tau.unsqueeze(-1)
    weights = torch.softmax(scores, dim=-1)
    output = torch.einsum("...n,...nv->...v", weights, joint_values)
    return output, weights


def baseline_state_bytes(
    baseline_name: str,
    d_key: int,
    d_value: int,
    window: int,
    recurrent_slots: int = 0,
    dtype: torch.dtype = torch.float64,
) -> int:
    """Return exact live-state bytes for each baseline."""

    b = torch.tensor([], dtype=dtype).element_size()
    name = baseline_name.lower()

    if name in ("local_softmax", "local_only"):
        return b * window * (d_key + d_value)
    if name in ("remote_bayes", "remote_only"):
        return b * (d_key * d_key + d_value * d_key)
    if name in ("global_linear", "linear_attention"):
        return b * (d_value * d_key + d_key)
    if name in ("delta_rule", "gated_delta"):
        return b * (d_value * d_key)
    if name in ("mesa", "cumulative_least_squares"):
        return b * (d_key * d_key + d_value * d_key)
    if name in ("learned_sum", "learned_concat"):
        return b * (d_key * d_key + d_value * d_key + window * (d_key + d_value))
    if name in ("independent_fusion", "independent_inverse_variance"):
        return b * (d_key * d_key + d_value * d_key + window * (d_key + d_value + 1))
    if name in ("full_residual", "full_residual_fixed"):
        return b * (d_key * d_key + d_value * d_key + window * (d_key + d_value))
    if name in ("aurelis_b", "aurelis_e", "aurelis"):
        return b * (d_key * d_key + d_value * d_key + window * (d_key + d_value + 1))
    if name in ("native_hybrid", "native_hybrid_attention"):
        return b * (recurrent_slots + window) * (d_key + d_value)
    raise ValueError(f"Unknown baseline name: {baseline_name}")


def baseline_parameter_count(
    baseline_name: str,
    d_key: int,
    d_value: int,
) -> int:
    """Return extra parameters beyond standard projections."""

    name = baseline_name.lower()
    if name in ("learned_sum",):
        return 1
    if name in ("learned_concat",):
        return 2 * d_value * d_value
    return 0


def baseline_flops(
    baseline_name: str,
    d_key: int,
    d_value: int,
    window: int,
    recurrent_slots: int = 0,
) -> int:
    """Return theoretical arithmetic FLOPs per query evaluation."""

    name = baseline_name.lower()
    if name in ("local_softmax", "local_only"):
        return 2 * window * d_key + 3 * window + 2 * window * d_value

    if name in ("remote_bayes", "remote_only"):
        return 2 * (d_key**2) + 2 * d_value * d_key

    if name in ("global_linear", "linear_attention"):
        return 3 * d_key + 2 * d_value * d_key + d_value

    if name in ("delta_rule", "gated_delta"):
        return 2 * d_value * d_key

    if name in ("mesa", "cumulative_least_squares"):
        return 2 * (d_key**2) + 2 * d_value * d_key

    if name in ("learned_sum",):
        local_fl = 2 * window * d_key + 3 * window + 2 * window * d_value
        remote_fl = 2 * (d_key**2) + 2 * d_value * d_key
        return local_fl + remote_fl + 3 * d_value

    if name in ("learned_concat",):
        local_fl = 2 * window * d_key + 3 * window + 2 * window * d_value
        remote_fl = 2 * (d_key**2) + 2 * d_value * d_key
        return local_fl + remote_fl + 2 * (2 * d_value) * d_value

    if name in ("independent_fusion", "independent_inverse_variance"):
        local_fl = 4 * window * d_key + 6 * window + 2 * window * d_value
        solve_fl = 6 * (d_key**2) + 4 * d_value * d_key
        return local_fl + solve_fl + 4 * d_key + 4 * d_value

    if name in ("full_residual", "full_residual_fixed"):
        local_fl = 4 * window * d_key + 6 * window + 2 * window * d_value
        solve_fl = 2 * (d_key**2) + 2 * d_value * d_key
        return local_fl + solve_fl + 2 * d_value

    if name in ("aurelis_b", "aurelis_e", "aurelis"):
        local_fl = 4 * window * d_key + 6 * window + 2 * window * d_value
        solve_fl = 6 * (d_key**2) + 4 * d_value * d_key
        quad_fl = 6 * d_key + 4 * d_value
        return local_fl + solve_fl + quad_fl

    if name in ("native_hybrid", "native_hybrid_attention"):
        tot = recurrent_slots + window
        return 2 * tot * d_key + 3 * tot + 2 * tot * d_value

    raise ValueError(f"Unknown baseline: {baseline_name}")
