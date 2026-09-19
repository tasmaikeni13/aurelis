"""Phase 2 tests: Populated-cache decode, multi-token prefill, and continuation equivalence."""

from __future__ import annotations

import time
import pytest
import torch

from aurelis import AurelisSession


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_bounded_decode_equivalence(dtype: torch.dtype):
    """Make token-by-token execution, multi-token prefill, and continuation agree in bounded mode."""
    torch.manual_seed(1001)
    d_k, d_v, window = 12, 12, 4
    t_total = 24
    atol = 1e-10 if dtype == torch.float64 else 1e-6

    queries = torch.randn(t_total, d_k, dtype=dtype)
    keys = torch.randn(t_total, d_k, dtype=dtype)
    values = torch.randn(t_total, d_v, dtype=dtype)
    alphas = torch.tensor([0.95 + 0.01 * (i % 4) for i in range(t_total)], dtype=dtype)
    betas = torch.tensor([0.85 + 0.02 * (i % 3) for i in range(t_total)], dtype=dtype)

    # 1. Token-by-token execution
    sess_step = AurelisSession(d_key=d_k, d_value=d_v, window=window, mode="bounded", dtype=dtype)
    outs_step = []
    for t in range(t_total):
        res = sess_step.step(queries[t], keys[t], values[t], alpha=alphas[t], beta=betas[t])
        outs_step.append(res.output)
    tensors_step = torch.stack(outs_step)

    # 2. Multi-token prefill
    sess_prefill = AurelisSession(d_key=d_k, d_value=d_v, window=window, mode="bounded", dtype=dtype)
    res_prefill = sess_prefill.prefill(queries, keys, values, alphas=alphas, betas=betas)
    tensors_prefill = torch.stack([r.output for r in res_prefill])

    # 3. Continuation: prefill first 10 tokens, then token-by-token for the rest
    t_split = 10
    sess_cont = AurelisSession(d_key=d_k, d_value=d_v, window=window, mode="bounded", dtype=dtype)
    res_cont_part1 = sess_cont.prefill(
        queries[:t_split], keys[:t_split], values[:t_split],
        alphas=alphas[:t_split], betas=betas[:t_split]
    )
    outs_cont = [r.output for r in res_cont_part1]
    for t in range(t_split, t_total):
        res = sess_cont.step(queries[t], keys[t], values[t], alpha=alphas[t], beta=betas[t])
        outs_cont.append(res.output)
    tensors_cont = torch.stack(outs_cont)

    # Assert mutual agreement
    torch.testing.assert_close(tensors_step, tensors_prefill, atol=atol, rtol=atol)
    torch.testing.assert_close(tensors_step, tensors_cont, atol=atol, rtol=atol)

    # Check final recurrent state S agreement
    torch.testing.assert_close(sess_step.state.S, sess_prefill.state.S, atol=atol, rtol=atol)
    torch.testing.assert_close(sess_step.state.S, sess_cont.state.S, atol=atol, rtol=atol)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_archive_decode_equivalence_and_history_agreement(dtype: torch.dtype):
    """Make token-by-token execution, multi-token prefill, and continuation agree in archive mode."""
    torch.manual_seed(2002)
    d_k, d_v, window, page_size = 8, 8, 3, 4
    t_total = 20
    atol = 1e-10 if dtype == torch.float64 else 1e-5

    queries = torch.randn(t_total, d_k, dtype=dtype)
    keys = torch.randn(t_total, d_k, dtype=dtype)
    values = torch.randn(t_total, d_v, dtype=dtype)

    # 1. Token-by-token execution
    sess_step = AurelisSession(d_key=d_k, d_value=d_v, window=window, mode="archive", page_size=page_size, dtype=dtype)
    outs_step = []
    for t in range(t_total):
        res = sess_step.step(queries[t], keys[t], values[t], read_all_pages=True)
        outs_step.append(res.output)
    tensors_step = torch.stack(outs_step)

    # 2. Multi-token prefill
    sess_prefill = AurelisSession(d_key=d_k, d_value=d_v, window=window, mode="archive", page_size=page_size, dtype=dtype)
    res_prefill = sess_prefill.prefill(queries, keys, values, read_all_pages=True)
    tensors_prefill = torch.stack([r.output for r in res_prefill])

    # 3. Continuation: prefill first 8 tokens, then step-by-step
    t_split = 8
    sess_cont = AurelisSession(d_key=d_k, d_value=d_v, window=window, mode="archive", page_size=page_size, dtype=dtype)
    res_cont_part1 = sess_cont.prefill(queries[:t_split], keys[:t_split], values[:t_split], read_all_pages=True)
    outs_cont = [r.output for r in res_cont_part1]
    for t in range(t_split, t_total):
        res = sess_cont.step(queries[t], keys[t], values[t], read_all_pages=True)
        outs_cont.append(res.output)
    tensors_cont = torch.stack(outs_cont)

    # Assert mutual agreement
    torch.testing.assert_close(tensors_step, tensors_prefill, atol=atol, rtol=atol)
    torch.testing.assert_close(tensors_step, tensors_cont, atol=atol, rtol=atol)

    # 4. Compare every step output against full-history softmax reference
    for t in range(t_total):
        ref_out = sess_step.full_history_reference(queries[t], cutoff=t + 1)
        torch.testing.assert_close(outs_step[t], ref_out, atol=atol, rtol=atol)


def test_true_cached_decode_complexity():
    """Verify that cached decode executes in constant time O(1) in bounded mode."""
    torch.manual_seed(3003)
    d_k, d_v, window = 32, 32, 16
    sess = AurelisSession(d_key=d_k, d_value=d_v, window=window, mode="bounded")

    # Warmup
    for _ in range(50):
        sess.step(torch.randn(d_k), torch.randn(d_k), torch.randn(d_v))

    # Measure step timings at t=50 vs t=150
    times_early = []
    for _ in range(20):
        t0 = time.perf_counter()
        sess.step(torch.randn(d_k), torch.randn(d_k), torch.randn(d_v))
        times_early.append(time.perf_counter() - t0)

    for _ in range(100):
        sess.step(torch.randn(d_k), torch.randn(d_k), torch.randn(d_v))

    times_late = []
    for _ in range(20):
        t0 = time.perf_counter()
        sess.step(torch.randn(d_k), torch.randn(d_k), torch.randn(d_v))
        times_late.append(time.perf_counter() - t0)

    avg_early = sum(times_early) / len(times_early)
    avg_late = sum(times_late) / len(times_late)

    # In bounded mode, execution time should remain within the same order of magnitude (< 3x)
    # confirming it does NOT grow with total sequence length (unlike quadratic full attention).
    assert avg_late < 3.0 * avg_early + 1e-4
