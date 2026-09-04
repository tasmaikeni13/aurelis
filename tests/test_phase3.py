"""Unit tests for Phase 3 neural architectures, ablations, and curriculum generators."""

from __future__ import annotations

import torch
import pytest

from aurelis.nn_phase3 import (
    LearnedAurelisBlock,
    LocalOnlyBlock,
    RemoteOnlyBlock,
    LearnedSumBlock,
    GatedDeltaBlock,
    CumulativeLeastSquaresBlock,
    Phase3SequenceModel,
    compute_effective_rank,
)
from aurelis.curriculum import CurriculumGenerator


def test_effective_rank_known_matrices() -> None:
    # Rank 1 matrix: erank should be 1.0
    u = torch.randn(10, 1)
    v = torch.randn(1, 10)
    rank1 = u @ v
    erank1 = compute_effective_rank(rank1)
    assert abs(erank1 - 1.0) < 1e-3

    # Orthogonal matrix (equal singular values): erank should be equal to dimension
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    erank_ortho = compute_effective_rank(q)
    assert abs(erank_ortho - 8.0) < 1e-3


def test_all_architectures_forward_backward() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, L, Din, Dm, Dout = 2, 16, 16, 32, 4
    x = torch.randn(B, L, Din, device=device, requires_grad=True)

    blocks = [
        ("aurelis_e", LearnedAurelisBlock(Dm, 2, 8, 8, 8, gate_mode="aurelis_e")),
        ("aurelis_b", LearnedAurelisBlock(Dm, 2, 8, 8, 8, gate_mode="aurelis_b")),
        ("local_only", LocalOnlyBlock(Dm, 2, 8, 8, 8)),
        ("remote_only", RemoteOnlyBlock(Dm, 2, 8, 8, 8)),
        ("learned_sum", LearnedSumBlock(Dm, 2, 8, 8, 8)),
        ("gated_delta", GatedDeltaBlock(Dm, 2, 8, 8)),
        ("mesa", CumulativeLeastSquaresBlock(Dm, 2, 8, 8)),
        ("indep_charts", LearnedAurelisBlock(Dm, 2, 8, 8, 8, shared_charts=False)),
        ("fixed_evidence", LearnedAurelisBlock(Dm, 2, 8, 8, 8, learned_evidence=False)),
        ("fixed_0", LearnedAurelisBlock(Dm, 2, 8, 8, 8, gate_mode="fixed_0")),
        ("fixed_1", LearnedAurelisBlock(Dm, 2, 8, 8, 8, gate_mode="fixed_1")),
        ("learned_sigmoid", LearnedAurelisBlock(Dm, 2, 8, 8, 8, gate_mode="learned_sigmoid")),
        ("cache_overlap", LearnedAurelisBlock(Dm, 2, 8, 8, 8, cache_overlap=True)),
    ]

    for name, block in blocks:
        model = Phase3SequenceModel(Din, Dm, Dout, block).to(device)
        pred, diag = model(x)
        assert pred.shape == (B, L, Dout), f"{name} failed output shape"
        loss = pred.square().mean()
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), f"{name} failed gradient propagation"
        x.grad = None


def test_episodic_override_dominates_bayes_numerically() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    block = LearnedAurelisBlock(32, 2, 8, 8, 8, gate_mode="aurelis_e").to(device)
    hidden = torch.randn(4, 16, 32, device=device)
    _, diag = block(hidden)
    g_B = diag["g_B"]
    g_E = diag["g_E"]
    e_t = diag["e_t"]

    assert (g_E >= g_B - 1e-6).all(), "g_E must dominate g_B everywhere"
    assert (g_E >= e_t - 1e-6).all(), "g_E must dominate e_t everywhere"
    assert (g_E >= 0.0).all() and (g_E <= 1.0).all(), "g_E must stay in [0, 1]"


def test_curriculum_generator_seven_tasks() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = CurriculumGenerator(d_in=16, d_out=4, d_feat=10, default_window=12, device=device)

    b1 = gen.generate_task1_noisy_linear(4, 24)
    assert b1.family_id == 1 and b1.x.shape == (4, 24, 16) and b1.y.shape == (4, 24, 4)

    b2 = gen.generate_task2_recent_copy(4, 24)
    assert b2.family_id == 2 and b2.mask[:, -1].sum().item() == 4.0

    b3 = gen.generate_task3_remote_recall(4, 24)
    assert b3.family_id == 3 and torch.isfinite(b3.y).all()

    b4 = gen.generate_task4_mixed_exception(4, 24)
    assert b4.family_id == 4 and "is_exception" in b4.metadata

    b5 = gen.generate_task5_selective_copy(4, 24)
    assert b5.family_id == 5 and b5.mask[:, -1].sum().item() == 4.0

    b6 = gen.generate_task6_cache_boundary(4, 24, age=12)
    assert b6.family_id == 6 and b6.metadata["age"] == 12

    b7 = gen.generate_task7_negatives(4, 24)
    assert b7.family_id == 7 and torch.isfinite(b7.y).all()
