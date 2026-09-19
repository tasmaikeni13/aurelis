"""Phase 2 tests: Session lifecycle, snapshots, rollback, forking, and speculative execution."""

from __future__ import annotations

import pytest
import torch

from aurelis import AurelisSession


def test_prefix_snapshot_restores_complete_state_tuple():
    """Prefix snapshots restore the complete state tuple (S, ring buffer, archive, position, RNG)."""
    torch.manual_seed(4001)
    d_k, d_v, window = 8, 8, 4
    sess = AurelisSession(d_key=d_k, d_value=d_v, window=window, mode="archive", page_size=4)

    # Run 10 tokens
    for t in range(10):
        sess.step(torch.randn(d_k), torch.randn(d_k), torch.randn(d_v))

    # Capture complete state tuple snapshot
    snap = sess.snapshot(metadata={"checkpoint_tag": "step_10"})
    assert snap.position == 10
    assert snap.archive_length == len(sess.archive) if sess.archive else 0

    # Advance 8 more tokens
    keys_future = [torch.randn(d_k) for _ in range(8)]
    vals_future = [torch.randn(d_v) for _ in range(8)]
    qs_future = [torch.randn(d_k) for _ in range(8)]

    outs_original = []
    for t in range(8):
        res = sess.step(qs_future[t], keys_future[t], vals_future[t], read_all_pages=True)
        outs_original.append(res.output)

    # Restore to step 10 snapshot
    sess.restore(snap)
    assert sess.state.next_id == 10
    assert len(sess.archive) == snap.archive_length
    assert sess.archive.index_version == snap.index_version
    assert torch.equal(sess.state.S, snap.S)
    assert sess.state.cache_ids == snap.cache_ids
    assert sess.state.evicted_ids == snap.evicted_ids

    # Re-run same 8 future tokens from restored state
    outs_restored = []
    for t in range(8):
        res = sess.step(qs_future[t], keys_future[t], vals_future[t], read_all_pages=True)
        outs_restored.append(res.output)

    # Verify bit-for-bit exact reproduction
    for t in range(8):
        torch.testing.assert_close(outs_restored[t], outs_original[t])


def test_speculative_decoding_rollback():
    """Rejected speculative tokens can be rolled back to pre-draft checkpoint."""
    torch.manual_seed(5002)
    d_k, d_v, window = 8, 8, 3
    sess_spec = AurelisSession(d_key=d_k, d_value=d_v, window=window, mode="bounded")
    sess_clean = AurelisSession(d_key=d_k, d_value=d_v, window=window, mode="bounded")

    # Step prompt tokens on both
    prompt_len = 6
    p_keys = [torch.randn(d_k) for _ in range(prompt_len)]
    p_vals = [torch.randn(d_v) for _ in range(prompt_len)]
    p_qs = [torch.randn(d_k) for _ in range(prompt_len)]

    for t in range(prompt_len):
        sess_spec.step(p_qs[t], p_keys[t], p_vals[t])
        sess_clean.step(p_qs[t], p_keys[t], p_vals[t])

    # Snapshot before drafting speculative tokens
    snap = sess_spec.snapshot()

    # Draft 4 speculative tokens on sess_spec
    for _ in range(4):
        sess_spec.step(torch.randn(d_k), torch.randn(d_k), torch.randn(d_v))

    # Reject speculative draft and rollback
    sess_spec.cancel(snap)

    # Now run 5 accepted true tokens on both sessions
    acc_keys = [torch.randn(d_k) for _ in range(5)]
    acc_vals = [torch.randn(d_v) for _ in range(5)]
    acc_qs = [torch.randn(d_k) for _ in range(5)]

    for t in range(5):
        res_spec = sess_spec.step(acc_qs[t], acc_keys[t], acc_vals[t])
        res_clean = sess_clean.step(acc_qs[t], acc_keys[t], acc_vals[t])
        torch.testing.assert_close(res_spec.output, res_clean.output)

    # States must be identical
    torch.testing.assert_close(sess_spec.state.S, sess_clean.state.S)
    assert sess_spec.state.cache_ids == sess_clean.state.cache_ids
    assert sess_spec.state.evicted_ids == sess_clean.state.evicted_ids


def test_session_fork_isolation():
    """Forking creates independent session with zero cross-contamination."""
    torch.manual_seed(6003)
    d_k, d_v, window = 6, 6, 3
    sess_parent = AurelisSession(d_key=d_k, d_value=d_v, window=window, mode="archive", page_size=2)

    # Run 5 tokens
    for t in range(5):
        sess_parent.step(torch.randn(d_k), torch.randn(d_k), torch.randn(d_v))

    # Fork
    sess_child = sess_parent.fork()

    # Branch A on parent
    for t in range(4):
        sess_parent.step(torch.randn(d_k), torch.randn(d_k), torch.randn(d_v))

    # Branch B on child
    for t in range(6):
        sess_child.step(torch.randn(d_k), torch.randn(d_k), torch.randn(d_v))

    # Verify positions and archives diverged independently
    assert sess_parent.state.next_id == 9
    assert sess_child.state.next_id == 11
    assert len(sess_parent.archive) == 9 - window
    assert len(sess_child.archive) == 11 - window
    assert not torch.equal(sess_parent.state.S, sess_child.state.S)


def test_session_reset():
    """Reset restores fresh initial state and empties archive."""
    d_k, d_v, window = 4, 4, 2
    sess = AurelisSession(d_key=d_key if (d_key := d_k) else 4, d_value=d_v, window=window, mode="archive")
    for _ in range(5):
        sess.step(torch.randn(d_k), torch.randn(d_k), torch.randn(d_v))

    assert sess.state.next_id == 5
    assert len(sess.archive) > 0

    sess.reset()
    assert sess.state.next_id == 0
    assert sess.state.cache_size == 0
    assert torch.all(sess.state.S == 0.0)
    assert len(sess.archive) == 0
