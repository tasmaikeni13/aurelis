from __future__ import annotations

import pytest
import torch

from aurelis import aurelis_read, explicit_inverse_read, prepared_aurelis_head


def problem(d_key: int = 6) -> tuple[torch.Tensor, ...]:
    batch, heads, cache, d_value = 2, 2, 4, 3
    matrix = torch.randn(batch, heads, d_key, d_key)
    precision = matrix @ matrix.mT + 0.2 * torch.eye(d_key)
    cross = torch.randn(batch, heads, d_value, d_key)
    keys = torch.randn(batch, heads, cache, d_key)
    values = torch.randn(batch, heads, cache, d_value)
    evidence = torch.rand(batch, heads, cache) + 0.1
    query = torch.randn(batch, heads, d_key)
    return precision, cross, keys, values, evidence, query


def test_cholesky_dense_and_capped_inverse_agree() -> None:
    args = problem()
    cholesky = aurelis_read(*args)
    dense = aurelis_read(*args, solve_method="dense")
    inverse = explicit_inverse_read(*args, inverse_cap=8)
    for field in ("remote", "full_residual", "bayes", "episodic"):
        torch.testing.assert_close(getattr(cholesky, field), getattr(dense, field))
        torch.testing.assert_close(getattr(cholesky, field), getattr(inverse, field))
    assert cholesky.diagnostics.solve_residual_q.max() < 1e-12
    assert cholesky.diagnostics.solve_residual_kbar.max() < 1e-12


def test_compiler_prepared_head_matches_validated_reference() -> None:
    args = problem()
    temperature = torch.rand(2, 2) + 0.2
    responsibility = torch.rand(2, 2)
    expected = aurelis_read(
        *args,
        temperature=temperature,
        episodic_responsibility=responsibility,
    )
    actual = prepared_aurelis_head(*args, temperature, responsibility)
    for index, field in enumerate(("remote", "full_residual", "bayes", "episodic")):
        torch.testing.assert_close(actual[index], getattr(expected, field))
    for index, field in enumerate(("h", "V_R", "V_H", "K_RH", "g_raw", "g_B", "g_E"), 7):
        torch.testing.assert_close(actual[index], getattr(expected.diagnostics, field))


def test_inverse_oracle_refuses_uncapped_dimensions() -> None:
    with pytest.raises(ValueError, match="capped"):
        explicit_inverse_read(*problem(d_key=17), inverse_cap=16)


def test_non_positive_definite_precision_is_an_expected_failure() -> None:
    precision, cross, keys, values, evidence, query = problem(d_key=3)
    precision = precision.clone()
    precision[..., -1, -1] = -1.0
    with pytest.raises(torch.linalg.LinAlgError):
        aurelis_read(precision, cross, keys, values, evidence, query)


def test_near_singular_conditioned_domain_remains_finite() -> None:
    d_key = 5
    direction = torch.randn(1, 1, d_key, 1)
    precision = direction @ direction.mT + 1e-8 * torch.eye(d_key)
    cross = torch.randn(1, 1, 2, d_key)
    keys = torch.randn(1, 1, 3, d_key)
    values = torch.randn(1, 1, 3, 2)
    evidence = torch.ones(1, 1, 3)
    query = torch.randn(1, 1, d_key)
    cholesky = aurelis_read(precision, cross, keys, values, evidence, query)
    dense = aurelis_read(
        precision, cross, keys, values, evidence, query, solve_method="dense"
    )
    torch.testing.assert_close(cholesky.bayes, dense.bayes, rtol=2e-7, atol=2e-7)
    assert torch.isfinite(cholesky.bayes).all()
