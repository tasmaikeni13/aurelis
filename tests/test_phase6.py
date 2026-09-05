"""Comprehensive test suite for Phase 6 model architectures, HIP kernels, and decoding."""

from __future__ import annotations

import pytest
import torch

from aurelis.models import (
    AurelisLM,
    HybridSSMLM,
    LMConfig,
    TransformerLM,
    get_125m_config,
    get_350m_config,
    hip_fused_residual_gate,
    hip_recurrent_scan,
)


def test_parameter_calibration():
    """Verify that parameter counts across all three architectures are calibrated within ~10%."""
    cfg_tf = get_125m_config("transformer")
    cfg_hyb = get_125m_config("ssm_hybrid")
    cfg_aur = get_125m_config("aurelis_e")

    m_tf = TransformerLM(cfg_tf)
    m_hyb = HybridSSMLM(cfg_hyb)
    m_aur = AurelisLM(cfg_aur, gate_mode="aurelis_e")

    p_tf = m_tf.count_parameters()
    p_hyb = m_hyb.count_parameters()
    p_aur = m_aur.count_parameters()

    # Targets: ~115M - 130M
    assert 1.0e8 < p_tf < 1.4e8, f"Transformer params out of range: {p_tf}"
    assert 1.0e8 < p_hyb < 1.4e8, f"SSM Hybrid params out of range: {p_hyb}"
    assert 1.0e8 < p_aur < 1.4e8, f"AURELIS params out of range: {p_aur}"

    # 350M scale calibration
    cfg_tf_350 = get_350m_config("transformer")
    cfg_hyb_350 = get_350m_config("ssm_hybrid")
    cfg_aur_350 = get_350m_config("aurelis_e")

    m_tf_350 = TransformerLM(cfg_tf_350)
    m_hyb_350 = HybridSSMLM(cfg_hyb_350)
    m_aur_350 = AurelisLM(cfg_aur_350, gate_mode="aurelis_e")

    p_tf_350 = m_tf_350.count_parameters()
    p_hyb_350 = m_hyb_350.count_parameters()
    p_aur_350 = m_aur_350.count_parameters()

    assert 3.0e8 < p_tf_350 < 3.8e8, f"Transformer 350M params out of range: {p_tf_350}"
    assert 3.0e8 < p_hyb_350 < 3.8e8, f"SSM Hybrid 350M params out of range: {p_hyb_350}"
    assert 3.0e8 < p_aur_350 < 3.8e8, f"AURELIS 350M params out of range: {p_aur_350}"


def test_forward_backward_gradient_flow():
    """Verify forward and backward passes produce finite logits and gradients across all parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Small test configuration
    cfg = LMConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_key=32,
        d_value=32,
        d_ffn=256,
        window_size=16,
    )

    models = [
        TransformerLM(cfg),
        HybridSSMLM(cfg),
        AurelisLM(cfg, gate_mode="aurelis_b"),
        AurelisLM(cfg, gate_mode="aurelis_e"),
    ]

    x = torch.randint(0, 1000, (2, 16), device=device)

    for m in models:
        m.to(device)
        m.train()
        logits, _ = m(x)
        assert logits.shape == (2, 16, 1000)
        assert torch.isfinite(logits).all(), f"Nonfinite logits in {type(m)}"

        loss = logits.sum()
        loss.backward()

        for name, p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Missing gradient for {name} in {type(m)}"
                assert torch.isfinite(p.grad).all(), f"Nonfinite gradient in {name} for {type(m)}"
        m.zero_grad()


def test_hip_kernels_vs_reference():
    """Verify HIP kernels match pure PyTorch CPU/GPU references within fp32 precision."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)

    B, H, L, D = 2, 4, 32, 16
    x = torch.randn(B, H, L, D, device=device)
    decay = torch.rand(B, H, L, D, device=device) * 0.9 + 0.05

    out_hip = hip_recurrent_scan(x, decay)
    # PyTorch reference
    out_ref = torch.empty_like(x)
    curr = torch.zeros(B, H, D, device=device)
    for t in range(L):
        curr = decay[:, :, t, :] * curr + x[:, :, t, :]
        out_ref[:, :, t, :] = curr

    diff_scan = (out_hip - out_ref).abs().max().item()
    assert diff_scan < 1e-5, f"Recurrent scan discrepancy: {diff_scan}"

    # Fused gate
    remote = torch.randn(B, H, L, D, device=device)
    vbar = torch.randn(B, H, L, D, device=device)
    mapped_kbar = torch.randn(B, H, L, D, device=device)
    gate = torch.rand(B, H, L, device=device)

    out_fused = hip_fused_residual_gate(remote, vbar, mapped_kbar, gate)
    out_ref_gate = remote + gate.unsqueeze(-1) * (vbar - mapped_kbar)
    diff_gate = (out_fused - out_ref_gate).abs().max().item()
    assert diff_gate < 1e-5, f"Fused gate discrepancy: {diff_gate}"


def test_constant_decode_state_memory():
    """Verify AURELIS has strictly constant O(1) decode memory state footprint vs Transformer O(L)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = LMConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_key=32,
        d_value=32,
        d_ffn=256,
        window_size=16,
    )

    m_aur = AurelisLM(cfg, gate_mode="aurelis_e").to(device).eval()
    m_tf = TransformerLM(cfg).to(device).eval()

    # Step-by-step generation with AURELIS
    aur_cache = None
    step_input = torch.randint(0, 1000, (1, 1), device=device)

    aur_state_sizes = []
    for step in range(30):
        with torch.no_grad():
            _, aur_cache = m_aur(step_input, decode_caches=aur_cache)
        # Compute total bytes of AURELIS state in layer 0
        c0 = aur_cache[0]
        state_bytes = (
            c0.precision.numel() * c0.precision.element_size()
            + c0.cross.numel() * c0.cross.element_size()
            + c0.buffer_k.numel() * c0.buffer_k.element_size()
            + c0.buffer_v.numel() * c0.buffer_v.element_size()
            + c0.buffer_b.numel() * c0.buffer_b.element_size()
        )
        aur_state_sizes.append(state_bytes)

    # After initial window fill (step >= 16), AURELIS state size MUST be strictly constant
    assert len(set(aur_state_sizes[16:])) == 1, (
        f"AURELIS state size grew beyond window: {aur_state_sizes[16:]}"
    )

    # In contrast, Transformer KV cache size strictly grows linearly with every token
    tf_cache = None
    tf_state_sizes = []
    for step in range(30):
        with torch.no_grad():
            _, tf_cache = m_tf(step_input, kv_caches=tf_cache)
        k0, v0 = tf_cache[0]
        tf_bytes = (k0.numel() + v0.numel()) * k0.element_size()
        tf_state_sizes.append(tf_bytes)

    assert tf_state_sizes[-1] > tf_state_sizes[0], "Transformer cache did not grow"
    assert tf_state_sizes[-1] == tf_state_sizes[0] * 30, "Transformer cache growth was not linear"


def test_fused_rmsnorm_and_swiglu_parity():
    """Verify fused RMSNorm and SwiGLU HIP kernels match PyTorch reference implementations."""
    from aurelis.models import hip_rmsnorm, hip_swiglu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # RMSNorm parity
    x = torch.randn(4, 32, 128, device=device, dtype=torch.float32)
    weight = torch.randn(128, device=device, dtype=torch.float32)
    norm_out = hip_rmsnorm(x, weight, eps=1e-6)
    var = x.pow(2).mean(-1, keepdim=True)
    norm_ref = x * torch.rsqrt(var + 1e-6) * weight
    assert (norm_out - norm_ref).abs().max().item() < 1e-5

    # SwiGLU parity & gradient flow
    gate = torch.randn(4, 32, 256, device=device, dtype=torch.float32, requires_grad=True)
    up = torch.randn(4, 32, 256, device=device, dtype=torch.float32, requires_grad=True)
    swiglu_out = hip_swiglu(gate, up)
    loss = swiglu_out.sum()
    loss.backward()

    gate_ref = gate.detach().clone().requires_grad_(True)
    up_ref = up.detach().clone().requires_grad_(True)
    swiglu_ref = torch.nn.functional.silu(gate_ref) * up_ref
    loss_ref = swiglu_ref.sum()
    loss_ref.backward()

    assert (swiglu_out - swiglu_ref).abs().max().item() < 1e-6
    assert (gate.grad - gate_ref.grad).abs().max().item() < 1e-6
    assert (up.grad - up_ref.grad).abs().max().item() < 1e-6


def test_jamba_hybrid_architecture():
    """Verify Jamba-style hybrid model configuration, interleave ratios, and alias compatibility."""
    from aurelis.models import JambaHybridLM, HybridSSMLM

    # Test alias
    assert JambaHybridLM is HybridSSMLM

    cfg = get_125m_config("ssm_hybrid")
    # 1:1 alternating (default period 2)
    m_1to1 = JambaHybridLM(cfg, attention_layer_period=2)
    attn_count_1to1 = sum(1 for layer in m_1to1.layers if layer.is_attention_layer)
    assert attn_count_1to1 == cfg.n_layers // 2

    # 1:3 attention (Jamba standard period 4)
    m_1to3 = JambaHybridLM(cfg, attention_layer_period=4)
    attn_count_1to3 = sum(1 for layer in m_1to3.layers if layer.is_attention_layer)
    assert attn_count_1to3 == cfg.n_layers // 4

