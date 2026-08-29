from __future__ import annotations

import pytest
import torch

from aurelis import (
    consume,
    historical_oracle,
    initial_state,
    read,
    vectorized_reference,
)


FIELDS = ("remote", "full_residual", "bayes", "episodic")
DIAGNOSTICS = ("h", "V_R", "V_H", "K_RH", "g_raw", "g_B", "g_E")


def assert_outputs_close(actual: object, expected: object, *, atol: float = 2e-10) -> None:
    for field in FIELDS:
        torch.testing.assert_close(
            getattr(actual, field), getattr(expected, field), rtol=2e-10, atol=atol
        )
    for field in DIAGNOSTICS:
        torch.testing.assert_close(
            getattr(actual.diagnostics, field),
            getattr(expected.diagnostics, field),
            rtol=2e-10,
            atol=atol,
            equal_nan=True,
        )


@pytest.mark.parametrize("length", [0, 1, 3, 4, 5, 9, 17])
@pytest.mark.parametrize("window", [1, 4])
def test_streaming_matches_independently_assembled_history(length: int, window: int) -> None:
    generator = torch.Generator().manual_seed(1000 + 31 * length + window)
    batch, heads, d_key, d_value = 2, 3, 5, 4
    keys = torch.randn(batch, heads, length, d_key, generator=generator)
    values = torch.randn(batch, heads, length, d_value, generator=generator)
    evidence = torch.rand(batch, heads, length, generator=generator) * 2.0 + 0.05
    query = torch.randn(batch, heads, d_key, generator=generator)
    temperature = torch.rand(heads, generator=generator) + 0.25
    responsibility = torch.rand(batch, heads, generator=generator) if length else 0.0
    state = initial_state(batch, heads, d_key, d_value, window, prior=0.15)
    for index in range(length):
        state = consume(
            state, keys[:, :, index], values[:, :, index], evidence[:, :, index]
        )
    actual = read(
        state,
        query,
        temperature=temperature,
        episodic_responsibility=responsibility,
    )
    expected = historical_oracle(
        keys,
        values,
        evidence,
        query,
        window=window,
        prior=0.15,
        temperature=temperature,
        episodic_responsibility=responsibility,
    )
    assert_outputs_close(actual, expected)


def test_repeated_keys_near_singular_and_over_capacity() -> None:
    batch, heads, length, d_key, d_value, window = 1, 2, 23, 6, 3, 3
    base = torch.randn(batch, heads, 1, d_key)
    keys = base.expand(batch, heads, length, d_key).clone()
    keys = keys + 1e-9 * torch.randn_like(keys)
    values = torch.randn(batch, heads, length, d_value)
    evidence = torch.logspace(-3, 3, length).view(1, 1, length).expand(batch, heads, length)
    query = base.squeeze(-2) + 1e-8 * torch.randn(batch, heads, d_key)
    state = initial_state(batch, heads, d_key, d_value, window, prior=1e-7)
    for index in range(length):
        state = consume(
            state, keys[:, :, index], values[:, :, index], evidence[:, :, index]
        )
    actual = read(state, query, temperature=3.0)
    expected = historical_oracle(
        keys,
        values,
        evidence,
        query,
        window=window,
        prior=1e-7,
        temperature=3.0,
    )
    assert_outputs_close(actual, expected, atol=3e-7)
    assert torch.isfinite(actual.bayes).all()


def test_vectorized_reference_matches_each_full_prefix_oracle() -> None:
    generator = torch.Generator().manual_seed(77)
    batch, heads, length, d_key, d_value = 2, 2, 8, 4, 3
    keys = torch.randn(batch, heads, length, d_key, generator=generator)
    values = torch.randn(batch, heads, length, d_value, generator=generator)
    evidence = torch.rand(batch, heads, length, generator=generator) + 0.2
    queries = torch.randn(batch, heads, length, d_key, generator=generator)
    temperature = torch.tensor([0.7, 1.3])
    responsibility = torch.rand(batch, heads, length, generator=generator)
    actual = vectorized_reference(
        keys,
        values,
        evidence,
        queries,
        window=3,
        prior=0.4,
        temperature=temperature.view(1, heads, 1),
        episodic_responsibility=responsibility,
    )
    for boundary in range(length):
        expected = historical_oracle(
            keys[:, :, : boundary + 1],
            values[:, :, : boundary + 1],
            evidence[:, :, : boundary + 1],
            queries[:, :, boundary],
            window=3,
            prior=0.4,
            temperature=temperature,
            episodic_responsibility=responsibility[:, :, boundary],
        )
        for field in FIELDS:
            torch.testing.assert_close(
                getattr(actual, field)[:, :, boundary],
                getattr(expected, field),
                rtol=2e-10,
                atol=2e-10,
            )
