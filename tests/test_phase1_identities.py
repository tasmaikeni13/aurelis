from __future__ import annotations

import torch

from aurelis import aurelis_read


def test_general_gate_error_identity_for_unclipped_gates() -> None:
    generator = torch.Generator().manual_seed(20260829)
    d_key, d_value, window = 5, 3, 7
    truth = torch.randn(d_value, d_key, generator=generator, dtype=torch.float64)
    memory = torch.randn(d_value, d_key, generator=generator, dtype=torch.float64)
    keys = torch.randn(window, d_key, generator=generator, dtype=torch.float64)
    residuals = torch.randn(window, d_value, generator=generator, dtype=torch.float64)
    values = keys @ truth.mT + residuals
    query = torch.randn(d_key, generator=generator, dtype=torch.float64)
    weights = torch.softmax(1.7 * (keys @ query), dim=0)
    kbar = weights @ keys
    vbar = weights @ values
    delta_bar = weights @ residuals
    for gate in (-0.5, 0.0, 0.37, 1.0, 1.5):
        actual = memory @ query + gate * (vbar - memory @ kbar) - truth @ query
        expected = (memory - truth) @ query + gate * (
            delta_bar - (memory - truth) @ kbar
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-14)


def test_finite_ridge_bound_on_full_rank_noise_free_problems() -> None:
    generator = torch.Generator().manual_seed(55)
    d_key, d_value, n_remote, window = 6, 4, 18, 5
    keys = torch.randn(n_remote, d_key, generator=generator, dtype=torch.float64)
    local_keys = torch.randn(window, d_key, generator=generator, dtype=torch.float64)
    truth = torch.randn(d_value, d_key, generator=generator, dtype=torch.float64)
    query = torch.randn(d_key, generator=generator, dtype=torch.float64)
    evidence = torch.logspace(-2, 2, n_remote, dtype=torch.float64)
    gram = torch.einsum("n,ni,nj->ij", evidence, keys, keys)
    lambda_min = torch.linalg.eigvalsh(gram).amin()
    assert lambda_min > 0
    for prior in torch.logspace(-8, 2, 8, dtype=torch.float64):
        precision = gram + prior * torch.eye(d_key, dtype=torch.float64)
        cross = torch.einsum(
            "n,nv,nd->vd", evidence, keys @ truth.mT, keys
        )
        output = aurelis_read(
            precision.view(1, 1, d_key, d_key),
            cross.view(1, 1, d_value, d_key),
            local_keys.view(1, 1, window, d_key),
            (local_keys @ truth.mT).view(1, 1, window, d_value),
            torch.ones(1, 1, window, dtype=torch.float64),
            query.view(1, 1, d_key),
        )
        residual_query = query - output.diagnostics.kbar[0, 0]
        bound = (
            prior
            * torch.linalg.matrix_norm(truth, ord=2)
            / (lambda_min + prior)
            * torch.linalg.vector_norm(residual_query)
        )
        error = torch.linalg.vector_norm(output.full_residual[0, 0] - truth @ query)
        assert error <= bound + 2e-12


def test_router_clipping_exercises_all_three_regions() -> None:
    observed = []
    for query_value in (-1.0, 0.5, 2.0):
        output = aurelis_read(
            torch.ones(1, 1, 1, 1, dtype=torch.float64),
            torch.zeros(1, 1, 1, 1, dtype=torch.float64),
            torch.ones(1, 1, 1, 1, dtype=torch.float64),
            torch.zeros(1, 1, 1, 1, dtype=torch.float64),
            torch.full((1, 1, 1), 10.0, dtype=torch.float64),
            torch.tensor([[[query_value]]], dtype=torch.float64),
        )
        observed.append((float(output.diagnostics.g_raw), float(output.diagnostics.g_B)))
    assert observed[0][0] < 0 and observed[0][1] == 0
    assert 0 < observed[1][0] < 1 and observed[1][1] == observed[1][0]
    assert observed[2][0] > 1 and observed[2][1] == 1

