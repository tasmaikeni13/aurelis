"""Transparent fp64 reference equations for one AURELIS read."""

from __future__ import annotations

import torch
from torch import Tensor

from .types import ReadDiagnostics, ReadOutput


def _validate_read_shapes(
    precision: Tensor,
    cross: Tensor,
    keys: Tensor,
    values: Tensor,
    evidence: Tensor,
    query: Tensor,
) -> None:
    if precision.ndim != 4 or precision.shape[-1] != precision.shape[-2]:
        raise ValueError("precision must have shape [batch, heads, d_key, d_key]")
    batch, heads, d_key, _ = precision.shape
    if cross.ndim != 4 or cross.shape[:2] != (batch, heads) or cross.shape[-1] != d_key:
        raise ValueError("cross must have shape [batch, heads, d_value, d_key]")
    if keys.ndim != 4 or keys.shape[:2] != (batch, heads) or keys.shape[-1] != d_key:
        raise ValueError("keys must have shape [batch, heads, cache, d_key]")
    if values.shape[:3] != keys.shape[:3] or values.shape[-1] != cross.shape[-2]:
        raise ValueError("values must have shape [batch, heads, cache, d_value]")
    if evidence.shape != keys.shape[:3]:
        raise ValueError("evidence must have shape [batch, heads, cache]")
    if query.shape != (batch, heads, d_key):
        raise ValueError("query must have shape [batch, heads, d_key]")


def local_causal_softmax(
    keys: Tensor,
    values: Tensor,
    evidence: Tensor,
    query: Tensor,
    temperature: Tensor | float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return shared local weights, key/value barycenters, and local noise.

    The cache is already the causal recent window. Empty caches are defined to
    contribute zero barycenters and infinite local noise, forcing ``g_B=0``.
    """

    cache_size = keys.shape[-2]
    if cache_size == 0:
        batch_shape = query.shape[:-1]
        weights = keys.new_empty((*batch_shape, 0))
        kbar = torch.zeros_like(query)
        vbar = values.new_zeros((*batch_shape, values.shape[-1]))
        h = query.new_full(batch_shape, torch.inf)
        return weights, kbar, vbar, h

    if bool(torch.any(evidence <= 0).detach().cpu()):
        raise ValueError("evidence precision must be strictly positive")
    tau = torch.as_tensor(temperature, dtype=query.dtype, device=query.device)
    while tau.ndim < query.ndim - 1:
        tau = tau.unsqueeze(0)
    logits = torch.einsum("bhnd,bhd->bhn", keys, query) * tau.unsqueeze(-1)
    weights = torch.softmax(logits, dim=-1)
    kbar = torch.einsum("bhn,bhnd->bhd", weights, keys)
    vbar = torch.einsum("bhn,bhnv->bhv", weights, values)
    h = torch.sum(weights.square() / evidence, dim=-1)
    return weights, kbar, vbar, h


def _solve(
    precision: Tensor,
    rhs: Tensor,
    *,
    method: str,
    factor: Tensor | None,
    inverse_cap: int,
) -> tuple[Tensor, Tensor]:
    if method == "cholesky":
        used_factor = torch.linalg.cholesky(precision) if factor is None else factor
        solution = torch.cholesky_solve(rhs, used_factor)
        return solution, used_factor
    if method == "dense":
        return torch.linalg.solve(precision, rhs), torch.linalg.cholesky(precision)
    if method == "inverse":
        if precision.shape[-1] > inverse_cap:
            raise ValueError(
                f"explicit inverse oracle is capped at d_key <= {inverse_cap}"
            )
        return torch.linalg.inv(precision) @ rhs, torch.linalg.cholesky(precision)
    raise ValueError(f"unknown solve method: {method}")


def aurelis_read(
    precision: Tensor,
    cross: Tensor,
    keys: Tensor,
    values: Tensor,
    evidence: Tensor,
    query: Tensor,
    *,
    temperature: Tensor | float = 1.0,
    episodic_responsibility: Tensor | float = 0.0,
    factor: Tensor | None = None,
    solve_method: str = "cholesky",
    inverse_cap: int = 16,
) -> ReadOutput:
    """Evaluate remote, full-residual, AURELIS-B, and AURELIS-E outputs.

    ``precision`` is ``P`` (including the positive prior) and ``cross`` is
    ``C``. No explicit inverse is used unless ``solve_method='inverse'`` is
    deliberately selected for the tiny test oracle.
    """

    _validate_read_shapes(precision, cross, keys, values, evidence, query)
    weights, kbar, vbar, h = local_causal_softmax(
        keys, values, evidence, query, temperature
    )
    residual_query = query - kbar
    rhs = torch.stack((query, kbar, residual_query), dim=-1)
    solved, _ = _solve(
        precision,
        rhs,
        method=solve_method,
        factor=factor,
        inverse_cap=inverse_cap,
    )
    p_query, p_kbar, p_residual = solved.unbind(dim=-1)
    remote = torch.einsum("bhvd,bhd->bhv", cross, p_query)
    mapped_kbar = torch.einsum("bhvd,bhd->bhv", cross, p_kbar)
    innovation = vbar - mapped_kbar
    full_residual = remote + innovation

    V_R = torch.sum(query * p_query, dim=-1)
    V_H = h + torch.sum(residual_query * p_residual, dim=-1)
    K_RH = torch.sum(query * p_residual, dim=-1)
    numerator = torch.sum(query * p_kbar, dim=-1)
    denominator = h + torch.sum(kbar * p_kbar, dim=-1)
    cache_present = keys.shape[-2] > 0
    safe_denominator = torch.where(
        torch.isfinite(denominator) & (denominator > 0),
        denominator,
        torch.ones_like(denominator),
    )
    g_raw = numerator / safe_denominator
    if not cache_present:
        g_raw = torch.zeros_like(g_raw)
    g_B = torch.clamp(g_raw, 0.0, 1.0)
    responsibility = torch.as_tensor(
        episodic_responsibility, dtype=query.dtype, device=query.device
    )
    responsibility = torch.broadcast_to(responsibility, g_B.shape)
    if bool(torch.any((responsibility < 0) | (responsibility > 1)).detach().cpu()):
        raise ValueError("episodic responsibility must lie in [0, 1]")
    if not cache_present and bool(torch.any(responsibility != 0).detach().cpu()):
        raise ValueError("empty cache cannot carry episodic responsibility")
    g_E = torch.maximum(g_B, responsibility)
    bayes = remote + g_B.unsqueeze(-1) * innovation
    episodic = remote + g_E.unsqueeze(-1) * innovation

    reconstructed = precision @ solved
    solve_residual = torch.linalg.vector_norm(reconstructed - rhs, dim=-2)
    diagnostics = ReadDiagnostics(
        attention=weights,
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
        solve_residual_q=solve_residual[..., 0],
        solve_residual_kbar=solve_residual[..., 1],
    )
    return ReadOutput(remote, full_residual, bayes, episodic, diagnostics)


def explicit_inverse_read(*args: Tensor, inverse_cap: int = 16, **kwargs: object) -> ReadOutput:
    """Dimension-capped inverse path used only as an independent test oracle."""

    return aurelis_read(
        *args,
        solve_method="inverse",
        inverse_cap=inverse_cap,
        **kwargs,
    )


def prepared_aurelis_head(
    precision: Tensor,
    cross: Tensor,
    keys: Tensor,
    values: Tensor,
    evidence: Tensor,
    query: Tensor,
    temperature: Tensor,
    episodic_responsibility: Tensor,
) -> tuple[Tensor, ...]:
    """Compiler-friendly nonempty-cache AURELIS head with tensor-only inputs.

    State construction and input validation deliberately remain outside this
    prepared region. The returned tuple contains the four outputs followed by
    attention, barycenters, router terms, and solve residuals.
    """

    logits = torch.einsum("bhnd,bhd->bhn", keys, query) * temperature.unsqueeze(-1)
    attention = torch.softmax(logits, dim=-1)
    kbar = torch.einsum("bhn,bhnd->bhd", attention, keys)
    vbar = torch.einsum("bhn,bhnv->bhv", attention, values)
    h = torch.sum(attention.square() / evidence, dim=-1)
    residual_query = query - kbar
    rhs = torch.stack((query, kbar, residual_query), dim=-1)
    factor = torch.linalg.cholesky(precision)
    solved = torch.cholesky_solve(rhs, factor)
    p_query, p_kbar, p_residual = solved.unbind(dim=-1)
    remote = torch.einsum("bhvd,bhd->bhv", cross, p_query)
    mapped_kbar = torch.einsum("bhvd,bhd->bhv", cross, p_kbar)
    innovation = vbar - mapped_kbar
    full_residual = remote + innovation
    V_R = torch.sum(query * p_query, dim=-1)
    V_H = h + torch.sum(residual_query * p_residual, dim=-1)
    K_RH = torch.sum(query * p_residual, dim=-1)
    denominator = h + torch.sum(kbar * p_kbar, dim=-1)
    g_raw = torch.sum(query * p_kbar, dim=-1) / denominator
    g_B = torch.clamp(g_raw, 0.0, 1.0)
    g_E = torch.maximum(g_B, episodic_responsibility)
    bayes = remote + g_B.unsqueeze(-1) * innovation
    episodic = remote + g_E.unsqueeze(-1) * innovation
    solve_residual = torch.linalg.vector_norm(precision @ solved - rhs, dim=-2)
    return (
        remote,
        full_residual,
        bayes,
        episodic,
        attention,
        kbar,
        vbar,
        h,
        V_R,
        V_H,
        K_RH,
        g_raw,
        g_B,
        g_E,
        solve_residual[..., 0],
        solve_residual[..., 1],
    )
