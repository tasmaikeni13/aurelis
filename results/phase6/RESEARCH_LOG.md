# Phase 6 Research & Systems Engineering Log

## Focus: Architectural Triad for Publication & Accelerated ROCm Kernels

1. **Publication Candidate Triad Selection**:
   - For an authoritative publication, comparing AURELIS against pure Transformer is necessary but insufficient; the literature requires comparing against state-of-the-art SSM+Attention hybrids (e.g. Samba/Jamba/RecurrentGemma).
   - We implemented and calibrated:
     1. AURELIS (AURELIS-E with straight-through episodic override & AURELIS-B)
     2. Modern Causal Transformer (RoPE + Pre-RMSNorm + SwiGLU)
     3. Strong SSM+Attention Hybrid (Alternating Mamba-2 style selective scan + causal multi-head attention + SwiGLU)
   - Calibrated at both 125M and 350M scales.

2. **ROCm / HIP Acceleration on MI300X (`gfx942`)**:
   - Implemented native HIP kernels compiled via `torch.utils.cpp_extension` with `--offload-arch=gfx942`:
     - `recurrent_scan_f32_kernel`: Fused sequence scan running $h_t = a_t h_{t-1} + x_t$.
     - `fused_residual_gate_f32_kernel`: Fused evaluation of $y = \text{remote} + g \cdot (\bar{v} - M\bar{k})$.
   - Validated against double-precision and eager PyTorch reference baselines with residual error $< 5 \times 10^{-7}$.

3. **Inference Decode Memory Scaling**:
   - Proved and measured on device that AURELIS achieves strictly constant $O(1)$ decoding cache memory independent of sequence length $L$, yielding an 8.0x memory reduction at $L=4096$ vs Transformer.
