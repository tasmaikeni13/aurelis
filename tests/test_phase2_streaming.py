"""Phase 2 tests: Streaming state, ring buffer, causal handoff, and partitioning."""

from __future__ import annotations

import pytest
import torch

from aurelis import (
    bounded_read,
    consume,
    initial_state,
    occurrence_partition,
    read,
)


@pytest.mark.parametrize("d_k, d_v, window", [(8, 8, 4), (16, 8, 6), (32, 16, 8)])
def test_streaming_t_below_equal_above_window(d_k: int, d_v: int, window: int):
    """Exercise t below, equal to, and above window w."""
    torch.manual_seed(42 + d_k)
    state = initial_state(d_k, d_v, window, dtype=torch.float64)

    # 1. t < w
    for t in range(window - 1):
        k = torch.randn(d_k, dtype=torch.float64)
        v = torch.randn(d_v, dtype=torch.float64)
        state = consume(state, k, v, occurrence_id=t)
        evicted, recent = occurrence_partition(state)
        assert len(evicted) == 0
        assert len(recent) == t + 1
        assert torch.all(state.S == 0.0)  # No evictions yet, S remains zero

    # 2. t == w
    k_w = torch.randn(d_k, dtype=torch.float64)
    v_w = torch.randn(d_v, dtype=torch.float64)
    state = consume(state, k_w, v_w, occurrence_id=window - 1)
    evicted, recent = occurrence_partition(state)
    assert len(evicted) == 0
    assert len(recent) == window
    assert torch.all(state.S == 0.0)

    # 3. t > w (first eviction into S)
    k_post = torch.randn(d_k, dtype=torch.float64)
    v_post = torch.randn(d_v, dtype=torch.float64)
    state = consume(state, k_post, v_post, occurrence_id=window)
    evicted, recent = occurrence_partition(state)
    assert len(evicted) == 1
    assert evicted == (0,)
    assert len(recent) == window
    assert not torch.all(state.S == 0.0)  # S updated with evicted item 0

    # More steps t >> w
    for t in range(window + 1, window + 15):
        k = torch.randn(d_k, dtype=torch.float64)
        v = torch.randn(d_v, dtype=torch.float64)
        state = consume(state, k, v, occurrence_id=t)
        evicted, recent = occurrence_partition(state)
        assert set(evicted).isdisjoint(set(recent))
        assert sorted(evicted + recent) == list(range(t + 1))
        assert len(recent) == window
        assert len(evicted) == (t + 1) - window


def test_streaming_duplicate_content_distinct_ids():
    """Duplicate key/value content with distinct occurrence IDs are preserved distinctly."""
    torch.manual_seed(999)
    d_k, d_v, window = 8, 8, 3
    state = initial_state(d_k, d_v, window)

    # Fixed repeated key and value
    fixed_k = torch.ones(d_k) / (d_k ** 0.5)
    fixed_v = torch.ones(d_v) * 2.5

    for t in range(7):
        state = consume(state, fixed_k, fixed_v, occurrence_id=100 + t)
        evicted, recent = occurrence_partition(state)
        # All IDs must match exactly the range up to current step
        expected_all = [100 + i for i in range(t + 1)]
        assert sorted(evicted + recent) == expected_all
        assert set(evicted).isdisjoint(set(recent))


def test_streaming_packed_examples_reset_isolation():
    """Packed examples with boundary resets do not cross-contaminate."""
    torch.manual_seed(555)
    d_k, d_v, window = 8, 8, 4

    # Example 1
    state1 = initial_state(d_k, d_v, window)
    for t in range(10):
        state1 = consume(state1, torch.randn(d_k), torch.randn(d_v), occurrence_id=t)

    # Document reset at boundary: Example 2 starts with fresh initial state
    state2 = initial_state(d_k, d_v, window)
    for t in range(6):
        state2 = consume(state2, torch.randn(d_k), torch.randn(d_v), occurrence_id=t)

    evicted2, recent2 = occurrence_partition(state2)
    assert len(evicted2) == 2
    assert len(recent2) == window
    assert sorted(evicted2 + recent2) == list(range(6))
    # State2 has zero knowledge of Example 1 IDs
    assert not any(i >= 10 for i in evicted2 + recent2)


def test_streaming_variable_lengths_and_interleaved_requests():
    """Interleaved execution of multiple requests maintains strict isolation without contamination."""
    torch.manual_seed(777)
    d_k, d_v, window = 6, 6, 3

    # Three concurrent sessions with variable lengths: 10, 15, 20
    lengths = [10, 15, 20]
    states = [initial_state(d_k, d_v, window) for _ in range(3)]
    queries = [torch.randn(d_k) for _ in range(3)]

    # Generate distinct data per stream
    stream_keys = [[torch.randn(d_k) for _ in range(l)] for l in lengths]
    stream_vals = [[torch.randn(d_v) for _ in range(l)] for l in lengths]

    # Interleaved execution step-by-step
    max_len = max(lengths)
    for step in range(max_len):
        for s_idx in range(3):
            if step < lengths[s_idx]:
                states[s_idx] = consume(
                    states[s_idx],
                    stream_keys[s_idx][step],
                    stream_vals[s_idx][step],
                    occurrence_id=step,
                )

    # Compare against sequential execution of each stream
    for s_idx in range(3):
        seq_state = initial_state(d_k, d_v, window)
        for step in range(lengths[s_idx]):
            seq_state = consume(
                seq_state,
                stream_keys[s_idx][step],
                stream_vals[s_idx][step],
                occurrence_id=step,
            )

        # Recurrent state S must be identical
        torch.testing.assert_close(states[s_idx].S, seq_state.S)
        # Partition must match exactly
        assert states[s_idx].evicted_ids == seq_state.evicted_ids
        assert states[s_idx].cache_ids == seq_state.cache_ids
        # Bounded read output must match
        res_interleaved = read(states[s_idx], queries[s_idx])
        res_seq = read(seq_state, queries[s_idx])
        torch.testing.assert_close(res_interleaved.output, res_seq.output)
