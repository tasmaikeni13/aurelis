"""Vectorized exact all-boundary AURELIS training reference."""

from __future__ import annotations

import torch
from torch import Tensor

from .types import SequenceOutput


def vectorized_reference(
    keys: Tensor,
    values: Tensor,
    evidence: Tensor,
    queries: Tensor,
    *,
    window: int,
    prior: float = 1.0,
    temperature: Tensor | float = 1.0,
    episodic_responsibility: Tensor | float = 0.0,
) -> SequenceOutput:
    """Evaluate every causal boundary with exact prefix sums and dense factors."""

    if keys.ndim != 4:
        raise ValueError("keys must have shape [batch, heads, time, d_key]")
    batch, heads, length, d_key = keys.shape
    if values.shape[:3] != (batch, heads, length):
        raise ValueError("values prefix dimensions do not match")
    if evidence.shape != (batch, heads, length) or queries.shape != keys.shape:
        raise ValueError("evidence or queries have incompatible shapes")
    if window <= 0 or prior <= 0:
        raise ValueError("window and prior must be strictly positive")
    if length == 0:
        raise ValueError("vectorized reference requires at least one boundary")

    outer_precision = torch.einsum("bhti,bhtj,bht->bhtij", keys, keys, evidence)
    outer_cross = torch.einsum("bhtv,bhtd,bht->bhtvd", values, keys, evidence)
    prefix_precision = torch.cumsum(outer_precision, dim=2)
    prefix_cross = torch.cumsum(outer_cross, dim=2)
    if window >= length:
        remote_precision = torch.zeros_like(prefix_precision)
        remote_cross = torch.zeros_like(prefix_cross)
    else:
        remote_precision = torch.cat(
            (torch.zeros_like(prefix_precision[:, :, :window]), prefix_precision[:, :, :-window]),
            dim=2,
        )
        remote_cross = torch.cat(
            (torch.zeros_like(prefix_cross[:, :, :window]), prefix_cross[:, :, :-window]),
            dim=2,
        )
    eye = torch.eye(d_key, dtype=keys.dtype, device=keys.device)
    precision = remote_precision + prior * eye.view(1, 1, 1, d_key, d_key)
    cross = remote_cross

    tau = torch.as_tensor(temperature, dtype=queries.dtype, device=queries.device)
    while tau.ndim < 3:
        tau = tau.unsqueeze(0)
    scores = torch.einsum("bhtd,bhsd->bhts", queries, keys) * tau.unsqueeze(-1)
    positions = torch.arange(length, device=keys.device)
    q_position = positions[:, None]
    k_position = positions[None, :]
    local_mask = (k_position <= q_position) & (k_position > q_position - window)
    scores = scores.masked_fill(~local_mask.view(1, 1, length, length), -torch.inf)
    attention = torch.softmax(scores, dim=-1)
    kbar = torch.einsum("bhts,bhsd->bhtd", attention, keys)
    vbar = torch.einsum("bhts,bhsv->bhtv", attention, values)
    h = torch.sum(attention.square() / evidence.unsqueeze(-2), dim=-1)

    residual_query = queries - kbar
    rhs = torch.stack((queries, kbar, residual_query), dim=-1)
    factors = torch.linalg.cholesky(precision)
    solved = torch.cholesky_solve(rhs, factors)
    p_query, p_kbar, p_residual = solved.unbind(dim=-1)
    remote = torch.einsum("bhtvd,bhtd->bhtv", cross, p_query)
    mapped_kbar = torch.einsum("bhtvd,bhtd->bhtv", cross, p_kbar)
    innovation = vbar - mapped_kbar
    full_residual = remote + innovation
    V_R = torch.sum(queries * p_query, dim=-1)
    V_H = h + torch.sum(residual_query * p_residual, dim=-1)
    K_RH = torch.sum(queries * p_residual, dim=-1)
    denominator = h + torch.sum(kbar * p_kbar, dim=-1)
    g_raw = torch.sum(queries * p_kbar, dim=-1) / denominator
    g_B = torch.clamp(g_raw, 0.0, 1.0)
    responsibility = torch.broadcast_to(
        torch.as_tensor(episodic_responsibility, dtype=queries.dtype, device=queries.device),
        g_B.shape,
    )
    g_E = torch.maximum(g_B, responsibility)
    return SequenceOutput(
        remote=remote,
        full_residual=full_residual,
        bayes=remote + g_B.unsqueeze(-1) * innovation,
        episodic=remote + g_E.unsqueeze(-1) * innovation,
        attention=attention,
        precision=precision,
        cross=cross,
        h=h,
        V_R=V_R,
        V_H=V_H,
        K_RH=K_RH,
        g_raw=g_raw,
        g_B=g_B,
        g_E=g_E,
    )
