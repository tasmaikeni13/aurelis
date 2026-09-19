"""Pure reference mathematics for AURELIS operations."""

from __future__ import annotations

import math
import torch
from torch import Tensor

from .types import CertificateBound


def gated_delta_update(
    S: Tensor,
    key: Tensor,
    value: Tensor,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
) -> Tensor:
    """Evaluate solve-free gated delta recurrent update (Eq. 2).

    S^+ = alpha * S + beta * (v - alpha * S @ k) k^T

    Args:
        S: Recurrent state tensor [..., d_value, d_key]
        key: Input key vector [..., d_key]
        value: Input value vector [..., d_value]
        alpha: Decay factor in [0, 1]
        beta: Write gate in [0, 1]

    Returns:
        Updated state tensor S^+ of same shape as S.
    """
    alpha_t = torch.as_tensor(alpha, dtype=S.dtype, device=S.device)
    beta_t = torch.as_tensor(beta, dtype=S.dtype, device=S.device)

    # Key normalization check / projection
    norm_sq = torch.sum(key * key, dim=-1, keepdim=True)
    scale = torch.clamp(torch.sqrt(norm_sq), min=1.0)
    k = key / scale

    # S_tilde = alpha * S
    S_tilde = alpha_t * S

    # Prediction at k: pred = S_tilde @ k [..., d_value]
    pred = torch.einsum("...vk,...k->...v", S_tilde, k)

    # Innovation error: err = v - pred [..., d_value]
    error = value - pred

    # Delta write: delta = beta * error k^T [..., d_value, d_key]
    delta = beta_t * torch.einsum("...v,...k->...vk", error, k)

    return S_tilde + delta


def local_attention(
    keys: Tensor,
    values: Tensor,
    query: Tensor,
    kappa: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute local window softmax attention and key/value barycenters.

    Args:
        keys: Local keys [..., window, d_key]
        values: Local values [..., window, d_value]
        query: Query vector [..., d_key]
        kappa: Score scale factor > 0

    Returns:
        (weights, kbar, vbar)
    """
    if keys.shape[-2] == 0:
        batch_shape = query.shape[:-1]
        weights = keys.new_empty((*batch_shape, 0))
        kbar = torch.zeros_like(query)
        vbar = values.new_zeros((*batch_shape, values.shape[-1]))
        return weights, kbar, vbar

    # Dot-product scores s_i = kappa * q^T k_i
    scores = kappa * torch.einsum("...d,...wd->...w", query, keys)
    weights = torch.softmax(scores, dim=-1)

    # Barycenters
    kbar = torch.einsum("...w,...wd->...d", weights, keys)
    vbar = torch.einsum("...w,...wv->...v", weights, values)

    return weights, kbar, vbar


def bounded_read(
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    S: Tensor,
    kappa: float = 1.0,
) -> Tensor:
    """Evaluate bounded mode read with local attention and linear transport (Eq. 3).

    r(q) = vbar_L + S (q - kbar_L)

    Args:
        query: Query vector [..., d_key]
        keys: Local window keys [..., window, d_key]
        values: Local window values [..., window, d_value]
        S: Recurrent state matrix [..., d_value, d_key]
        kappa: Scale factor > 0

    Returns:
        Predicted output vector [..., d_value]
    """
    _, kbar, vbar = local_attention(keys, values, query, kappa=kappa)
    diff = query - kbar
    transport = torch.einsum("...vk,...k->...v", S, diff)
    return vbar + transport


def completed_read(
    selected_mass: float | Tensor,
    estimated_unread_mass: float | Tensor,
    selected_numerator: Tensor,
    prior: Tensor,
) -> Tensor:
    """Evaluate mass-consistent archive completion (Eq. 8).

    y_hat = (selected_numerator + estimated_unread_mass * prior) / (selected_mass + estimated_unread_mass)
    """
    zs = torch.as_tensor(selected_mass, dtype=prior.dtype, device=prior.device)
    zh = torch.as_tensor(estimated_unread_mass, dtype=prior.dtype, device=prior.device)
    total_mass = zs + zh
    return (selected_numerator + zh * prior) / total_mass


def residual_certificate(
    selected_mass: float,
    unread_lower: float,
    unread_upper: float,
    residual_bound: float,
    prior: Tensor,
    completed: Tensor,
) -> CertificateBound:
    """Evaluate deterministic residual certificate (Eq. 10).

    bound = (residual_bound + delta_z * ||prior - completed||) / (selected_mass + unread_lower)
    """
    delta_z = (unread_upper - unread_lower) / 2.0
    est_unread = (unread_lower + unread_upper) / 2.0
    prior_diff_norm = float(torch.norm(prior - completed, p=2).item())

    denom_floor = selected_mass + unread_lower
    if denom_floor <= 0.0:
        bound_val = float("inf")
    else:
        bound_val = (residual_bound + delta_z * prior_diff_norm) / denom_floor

    return CertificateBound(
        bound=bound_val,
        selected_mass=selected_mass,
        estimated_unread_mass=est_unread,
        unread_mass_lower=unread_lower,
        unread_mass_upper=unread_upper,
        residual_bound=residual_bound,
        denominator_floor=denom_floor,
    )


def pointwise_envelope(
    keys: Tensor,
    query: Tensor,
    kappa: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Compute upper and lower score envelopes for coordinate key boxes (Eq. 11).

    Returns:
        (dot_lower, dot_upper) bounds on kappa * q^T k.
    """
    # Assuming keys has [..., count, d_key]
    scores = kappa * torch.einsum("...d,...cd->...c", query, keys)
    dot_lower = torch.min(scores, dim=-1).values
    dot_upper = torch.max(scores, dim=-1).values
    return dot_lower, dot_upper
