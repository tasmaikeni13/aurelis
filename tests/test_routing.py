from __future__ import annotations

import torch

from aurelis import aurelis_read


def test_bayes_gate_minimizes_dense_correlated_variance() -> None:
    batch, heads, d_key, d_value, cache = 3, 2, 4, 2, 5
    matrix = torch.randn(batch, heads, d_key, d_key)
    precision = matrix @ matrix.mT + 0.5 * torch.eye(d_key)
    cross = torch.randn(batch, heads, d_value, d_key)
    keys = torch.randn(batch, heads, cache, d_key)
    values = torch.randn(batch, heads, cache, d_value)
    evidence = torch.rand(batch, heads, cache) + 0.2
    query = torch.randn(batch, heads, d_key)
    out = aurelis_read(precision, cross, keys, values, evidence, query)
    diagnostic = out.diagnostics
    grid = torch.linspace(0.0, 1.0, 20001)
    gate = grid.view(-1, 1, 1)
    variance = (
        (1 - gate).square() * diagnostic.V_R
        + gate.square() * diagnostic.V_H
        + 2 * gate * (1 - gate) * diagnostic.K_RH
    )
    dense_gate = grid[variance.argmin(dim=0)]
    torch.testing.assert_close(diagnostic.g_B, dense_gate, atol=5.1e-5, rtol=0)
    routed = (
        (1 - diagnostic.g_B).square() * diagnostic.V_R
        + diagnostic.g_B.square() * diagnostic.V_H
        + 2
        * diagnostic.g_B
        * (1 - diagnostic.g_B)
        * diagnostic.K_RH
    )
    assert torch.all(routed <= torch.minimum(diagnostic.V_R, diagnostic.V_H) + 1e-12)


def test_episodic_exact_one_hot_hit() -> None:
    d_key, d_value = 4, 3
    precision = torch.eye(d_key).view(1, 1, d_key, d_key)
    cross = torch.randn(1, 1, d_value, d_key)
    query = torch.randn(1, 1, d_key)
    target = torch.randn(1, 1, 1, d_value)
    out = aurelis_read(
        precision,
        cross,
        query.unsqueeze(-2),
        target,
        torch.ones(1, 1, 1),
        query,
        episodic_responsibility=torch.ones(1, 1),
    )
    torch.testing.assert_close(out.episodic, target.squeeze(-2), rtol=0, atol=1e-14)
    assert torch.equal(out.diagnostics.g_E, torch.ones_like(out.diagnostics.g_E))


def test_empty_cache_is_remote_only() -> None:
    d_key, d_value = 3, 2
    precision = torch.eye(d_key).view(1, 1, d_key, d_key)
    cross = torch.randn(1, 1, d_value, d_key)
    query = torch.randn(1, 1, d_key)
    out = aurelis_read(
        precision,
        cross,
        torch.empty(1, 1, 0, d_key),
        torch.empty(1, 1, 0, d_value),
        torch.empty(1, 1, 0),
        query,
    )
    torch.testing.assert_close(out.remote, out.full_residual)
    torch.testing.assert_close(out.remote, out.bayes)
    torch.testing.assert_close(out.remote, out.episodic)
    assert torch.isinf(out.diagnostics.h).all()
    assert torch.count_nonzero(out.diagnostics.g_B) == 0
