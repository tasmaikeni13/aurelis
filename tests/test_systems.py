from __future__ import annotations

import pytest
import torch

from csm import (
    AffineSummary,
    FP64GaussMarkovMemory,
    associative_prefix_scan,
    compose_affine,
    prefix_states,
    read_prefix_states,
    sequential_decode,
    summarize_chunks,
    summarize_segment,
    token_summaries,
)


def problem(dtype: torch.dtype = torch.float64, steps: int = 11):
    generator = torch.Generator().manual_seed(81023)
    keys = torch.randn(2, 3, steps, 5, generator=generator, dtype=dtype)
    values = torch.randn(2, 3, steps, 4, generator=generator, dtype=dtype)
    beta = 0.1 + torch.rand(2, 3, steps, generator=generator, dtype=dtype)
    decay = 0.7 + 0.3 * torch.rand(2, 3, steps, generator=generator, dtype=dtype)
    queries = torch.randn(2, 3, steps, 5, generator=generator, dtype=dtype)
    return keys, values, beta, decay, queries


def select(summary: AffineSummary, index: int) -> AffineSummary:
    return AffineSummary(
        summary.decay[:, :, index],
        summary.S[:, :, index],
        summary.C[:, :, index],
    )


def test_affine_composition_is_associative() -> None:
    keys, values, beta, decay, _ = problem(steps=3)
    tokens = token_summaries(keys, values, beta, decay, accumulation_dtype=torch.float64)
    a, b, c = (select(tokens, i) for i in range(3))
    left = compose_affine(compose_affine(a, b), c)
    right = compose_affine(a, compose_affine(b, c))
    torch.testing.assert_close(left.decay, right.decay, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(left.S, right.S, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(left.C, right.C, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 32])
def test_vectorized_and_chunked_summaries_match_phase1_oracle(chunk_size: int) -> None:
    keys, values, beta, decay, _ = problem()
    vectorized = summarize_segment(
        keys, values, beta, decay, accumulation_dtype=torch.float64
    )
    chunked = summarize_chunks(
        keys, values, beta, decay, chunk_size, accumulation_dtype=torch.float64
    )
    torch.testing.assert_close(chunked.decay, vectorized.decay, rtol=5e-15, atol=5e-15)
    torch.testing.assert_close(chunked.S, vectorized.S, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(chunked.C, vectorized.C, rtol=2e-14, atol=2e-14)
    for batch in range(2):
        for head in range(3):
            oracle = FP64GaussMarkovMemory(5, 4).run(
                keys[batch, head], values[batch, head], beta[batch, head], decay[batch, head]
            )
            torch.testing.assert_close(vectorized.S[batch, head], oracle.S, rtol=2e-14, atol=2e-14)
            torch.testing.assert_close(vectorized.C[batch, head], oracle.C, rtol=2e-14, atol=2e-14)


def test_associative_prefix_scan_matches_every_oracle_prefix() -> None:
    keys, values, beta, decay, _ = problem(steps=9)
    tokens = token_summaries(keys, values, beta, decay, accumulation_dtype=torch.float64)
    scanned = associative_prefix_scan(tokens)
    for time in range(9):
        oracle = FP64GaussMarkovMemory(5, 4).run(
            keys[0, 0, : time + 1],
            values[0, 0, : time + 1],
            beta[0, 0, : time + 1],
            decay[0, 0, : time + 1],
        )
        torch.testing.assert_close(scanned.S[0, 0, time], oracle.S, rtol=3e-14, atol=3e-14)
        torch.testing.assert_close(scanned.C[0, 0, time], oracle.C, rtol=3e-14, atol=3e-14)


def test_parallel_and_sequential_decode_match() -> None:
    keys, values, beta, decay, queries = problem(steps=7)
    scanned = prefix_states(
        keys, values, beta, decay, accumulation_dtype=torch.float64
    )
    parallel_reads, parallel_uncertainty = read_prefix_states(
        scanned, queries, 1e-2, output_dtype=torch.float64
    )
    loop_reads, loop_uncertainty, final = sequential_decode(
        keys,
        values,
        queries,
        beta,
        decay,
        1e-2,
        accumulation_dtype=torch.float64,
        output_dtype=torch.float64,
    )
    torch.testing.assert_close(parallel_reads, loop_reads, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(
        parallel_uncertainty, loop_uncertainty, rtol=2e-13, atol=2e-13
    )
    torch.testing.assert_close(scanned.S[..., -1, :, :], final.S, rtol=2e-14, atol=2e-14)


def test_unit_decay_fast_path_and_general_scan_agree() -> None:
    keys, values, beta, decay, _ = problem(steps=8)
    decay.fill_(1)
    fast = prefix_states(keys, values, beta, decay, accumulation_dtype=torch.float64)
    scan = prefix_states(
        keys,
        values,
        beta,
        decay,
        accumulation_dtype=torch.float64,
        unit_decay_fast_path=False,
    )
    torch.testing.assert_close(fast.S, scan.S, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(fast.C, scan.C, rtol=2e-15, atol=2e-15)


def test_optimized_decode_is_differentiable() -> None:
    keys, values, beta, decay, queries = problem(torch.float64, steps=4)
    keys.requires_grad_()
    values.requires_grad_()
    queries.requires_grad_()
    beta.requires_grad_()
    decay.requires_grad_()
    summary = prefix_states(keys, values, beta, decay, accumulation_dtype=torch.float64)
    reads, uncertainty = read_prefix_states(summary, queries, 0.1)
    (reads.square().mean() + uncertainty.mean()).backward()
    for tensor in (keys, values, queries, beta, decay):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_invalid_chunk_size_is_rejected() -> None:
    keys, values, beta, decay, _ = problem()
    with pytest.raises(ValueError, match="positive"):
        summarize_chunks(keys, values, beta, decay, 0)
