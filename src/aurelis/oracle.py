"""Independent historical fp64 oracle assembled directly from a full prefix."""

from __future__ import annotations

import torch
from torch import Tensor

from .types import ReadDiagnostics, ReadOutput


def historical_oracle(
    history_keys: Tensor,
    history_values: Tensor,
    history_evidence: Tensor,
    query: Tensor,
    *,
    window: int,
    prior: float = 1.0,
    temperature: Tensor | float = 1.0,
    episodic_responsibility: Tensor | float = 0.0,
) -> ReadOutput:
    """Rebuild one boundary from full history without streaming update code."""

    if history_keys.ndim != 4:
        raise ValueError("history_keys must have shape [batch, heads, time, d_key]")
    batch, heads, length, d_key = history_keys.shape
    if window <= 0 or prior <= 0:
        raise ValueError("window and prior must be strictly positive")
    if history_values.shape[:3] != (batch, heads, length):
        raise ValueError("history_values prefix dimensions do not match")
    if history_evidence.shape != (batch, heads, length):
        raise ValueError("history_evidence must have shape [batch, heads, time]")
    if length and bool(torch.any(history_evidence <= 0).detach().cpu()):
        raise ValueError("evidence precision must be strictly positive")

    remote_end = max(0, length - window)
    remote_keys = history_keys[:, :, :remote_end, :]
    remote_values = history_values[:, :, :remote_end, :]
    remote_evidence = history_evidence[:, :, :remote_end]
    eye = torch.eye(d_key, dtype=history_keys.dtype, device=history_keys.device)
    precision = eye.expand(batch, heads, d_key, d_key).clone() * prior
    if remote_end:
        precision = precision + torch.einsum(
            "bhn,bhni,bhnj->bhij", remote_evidence, remote_keys, remote_keys
        )
        cross = torch.einsum(
            "bhn,bhnv,bhnd->bhvd", remote_evidence, remote_values, remote_keys
        )
    else:
        cross = history_values.new_zeros((batch, heads, history_values.shape[-1], d_key))

    local_keys = history_keys[:, :, remote_end:, :]
    local_values = history_values[:, :, remote_end:, :]
    local_evidence = history_evidence[:, :, remote_end:]
    if length:
        tau = torch.as_tensor(temperature, dtype=query.dtype, device=query.device)
        while tau.ndim < query.ndim - 1:
            tau = tau.unsqueeze(0)
        scores = torch.einsum("bhnd,bhd->bhn", local_keys, query) * tau.unsqueeze(-1)
        attention = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
        attention = attention / attention.sum(dim=-1, keepdim=True)
        kbar = torch.einsum("bhn,bhnd->bhd", attention, local_keys)
        vbar = torch.einsum("bhn,bhnv->bhv", attention, local_values)
        h = torch.sum(attention.square() / local_evidence, dim=-1)
    else:
        attention = history_keys.new_empty((batch, heads, 0))
        kbar = torch.zeros_like(query)
        vbar = history_values.new_zeros((batch, heads, history_values.shape[-1]))
        h = query.new_full((batch, heads), torch.inf)

    residual_query = query - kbar
    rhs = torch.stack((query, kbar, residual_query), dim=-1)
    solved = torch.linalg.solve(precision, rhs)
    p_query, p_kbar, p_residual = solved.unbind(dim=-1)
    remote = torch.einsum("bhvd,bhd->bhv", cross, p_query)
    mapped_kbar = torch.einsum("bhvd,bhd->bhv", cross, p_kbar)
    innovation = vbar - mapped_kbar
    full_residual = remote + innovation
    V_R = torch.sum(query * p_query, dim=-1)
    V_H = h + torch.sum(residual_query * p_residual, dim=-1)
    K_RH = torch.sum(query * p_residual, dim=-1)
    denominator = h + torch.sum(kbar * p_kbar, dim=-1)
    safe_denominator = torch.where(
        torch.isfinite(denominator) & (denominator > 0),
        denominator,
        torch.ones_like(denominator),
    )
    g_raw = torch.sum(query * p_kbar, dim=-1) / safe_denominator
    if length == 0:
        g_raw = torch.zeros_like(g_raw)
    g_B = torch.clamp(g_raw, 0.0, 1.0)
    responsibility = torch.broadcast_to(
        torch.as_tensor(episodic_responsibility, dtype=query.dtype, device=query.device),
        g_B.shape,
    )
    if length == 0 and bool(torch.any(responsibility != 0).detach().cpu()):
        raise ValueError("empty cache cannot carry episodic responsibility")
    g_E = torch.maximum(g_B, responsibility)
    residual_norm = torch.linalg.vector_norm(precision @ solved - rhs, dim=-2)
    return ReadOutput(
        remote=remote,
        full_residual=full_residual,
        bayes=remote + g_B.unsqueeze(-1) * innovation,
        episodic=remote + g_E.unsqueeze(-1) * innovation,
        diagnostics=ReadDiagnostics(
            attention=attention,
            kbar=kbar,
            vbar=vbar,
            innovation=innovation,
            h=h,
            V_R=V_R,
            V_H=V_H,
            K_RH=K_RH,
            g_raw=g_raw,
            g_B=g_B,
            g_E=g_E,
            solve_residual_q=residual_norm[..., 0],
            solve_residual_kbar=residual_norm[..., 1],
        ),
    )
