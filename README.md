# AURELIS

**Attention with Uncertainty-Routed Residuals over an Episodic–Long-range Inference State**

Standard causal attention and fixed-capacity recurrent memory fail in polar opposite ways. Softmax attention keeps exact observations around, but its KV cache grows linearly with context length until your GPU runs out of VRAM. Recurrent layers (SSMs, linear attention, delta nets) keep memory bounded, but they force you to compress history before future queries are even known.

AURELIS attacks this tradeoff directly inside a single attention head. We keep the most recent $w$ key–value pairs in an exact sliding-window attention cache. When tokens fall out of that window, they get handed off exactly once to a remote Bayesian ridge regression state. 

When a query $q$ arrives, the layer reads:

$$y(q) = Mq + g(q) \left[ \bar{v}(q) - M\bar{k}(q) \right]$$

Here, $M = CP^{-1}$ is the remote posterior mean map, and $\bar{k}(q)$ and $\bar{v}(q)$ are local attention barycenters. The bracketed term is an innovation residual. Our gate $g(q)$ comes from the derived cross-covariance between the remote prior and local attention errors. When a target is an exact memorized exception rather than a smooth latent relationship, an episodic override kicks in: $g_E = \max(g_B, e_t)$.

During inference decoding, AURELIS runs with strictly constant memory state: $O(d_k^2 + d_v d_k + w(d_k + d_v))$ per head. It never grows as the context stretches out.

---

## Current Status: Phase 6 PASS (Ready for Multi-GPU Scaling)

The codebase strictly adheres to [`phases/AUTONOMY_PROTOCOL.md`](phases/AUTONOMY_PROTOCOL.md). Every algebraic identity has formal Lean 4 machine proofs with zero unproven assumptions, verified against double-precision CPU oracles and benchmarked on our AMD Instinct MI300X VF accelerator under ROCm.

| Phase | Target | Scope & Milestones | Status | Artifacts & Evidence |
|---|:---:|---|:---:|---|
| **Phase 0** | 1x MI300X | Reference substrate, PyTorch ROCm/HIP audit, GEMM benchmarks | **PASS** | [`results/phase0/PASS.md`](results/phase0/PASS.md) |
| **Phase 1** | 1x MI300X | Exact identities, handoff partition, fp64 numerical oracles | **PASS** | [`results/phase1/PASS.md`](results/phase1/PASS.md) |
| **Phase 2** | 1x MI300X | Controlled baselines (Mesa, DeltaNet, Linear Attn) & falsification | **PASS** | [`results/phase2/PASS.md`](results/phase2/PASS.md) |
| **Phase 3** | 1x MI300X | Learned projections, straight-through estimator, 7 task families | **PASS** | [`results/phase3/PASS.md`](results/phase3/PASS.md) |
| **Phase 4** | 1x MI300X | Nonstationarity, changepoints, heteroscedastic noise & pointer chasing | **PASS** | [`results/phase4/PASS.md`](results/phase4/PASS.md) |
| **Phase 5** | 1x MI300X | Systems profiling, rocSOLVER solves, and fused kernel design | **PASS** | [`phases/phase5.md`](phases/phase5.md) |
| **Phase 6** | 1x MI300X | **LM Viability: AURELIS vs Transformer vs SSM Hybrid (125M & 350M)** | **PASS** | [`results/phase6/PASS.md`](results/phase6/PASS.md) |
| **Phase 7** | 8x MI300X | 125M Multi-Seed Pretraining on 1.0B FineWeb-Edu tokens | *Planned* | [`phases/phase7.md`](phases/phase7.md) |
| **Phase 8** | 8x MI300X | 350M Medium-Scale Pretraining on 3.0B FineWeb-Edu tokens | *Planned* | [`phases/phase8.md`](phases/phase8.md) |
| **Phase 9** | 8x MI300X | Clean-room reproduction, paper release audit, standalone manuscript | *Planned* | [`phases/phase9.md`](phases/phase9.md) |

---

## Phase 6 Highlights: The Publication Triad & Hardware Benchmarks

For peer-reviewed publication, comparing against a vanilla Transformer alone leaves too many open questions. We implemented, calibrated, and benchmarked three distinct model architectures at both **125M** and **350M** parameter scales:

1. **AURELIS (Candidate 1)**: Same-head sliding window + delayed Bayesian ridge regression state, straight-through episodic gating, and constant-size decode caching.
2. **Modern Causal Transformer (Candidate 2)**: Decoder-only causal multi-head self-attention with RoPE position embeddings, Pre-RMSNorm, and SwiGLU MLP.
3. **Strong SSM + Attention Hybrid (Candidate 3)**: Interleaved Mamba-2 style selective state space recurrence + causal multi-head attention blocks with Pre-RMSNorm and SwiGLU MLP.

### Parameter Calibration on Target Scales

We matched model capacity across all three architectures within $\pm 3.6\%$:

| Model Architecture | 125M Pilot Target | 350M Scaling Target | Inference Decode State Scaling |
|---|:---:|:---:|:---:|
| **AURELIS-E** | 116,694,960 | 329,075,840 | **O(1) Constant (4.50 MB per sequence)** |
| **AURELIS-B** | 116,694,960 | 329,075,840 | **O(1) Constant (4.50 MB per sequence)** |
| **Modern Causal Transformer** | 123,551,232 | 353,454,080 | $O(L)$ Linear Growth (up to 36.00 MB at 4k) |
| **SSM + Attention Hybrid** | 120,270,336 | 341,559,296 | Mixed $O(L)$ Growth (18.14 MB at 4k) |

### Hardware Acceleration with Custom HIP Kernels (AMD Instinct MI300X)

Rather than treating the MI300X like a black box, we wrote native HIP C++ kernels targeting `gfx942` using `torch.utils.cpp_extension`:
- `recurrent_scan_f32_kernel`: Fused sequence scan for state transitions: $h_t = a_t h_{t-1} + x_t$.
- `fused_residual_gate_f32_kernel`: Fused GPU evaluation of $y = \text{remote} + g \cdot (\bar{v} - M\bar{k})$.

Both kernels run with single-precision floating point parity against fp64 CPU reference paths, keeping maximum absolute errors below $9.54 \times 10^{-7}$.

### Real-World Decode Memory Savings

Because AURELIS evicts older tokens into a fixed-dimension precision matrix $P \in \mathbb{R}^{d_k \times d_k}$ and cross-covariance matrix $C \in \mathbb{R}^{d_v \times d_k}$, its inference state stops growing once the sliding window fills up:

| Context Window | Transformer KV Cache | SSM Hybrid State | AURELIS Decode Cache | AURELIS Memory Win |
|---|:---:|:---:|:---:|:---:|
| 512 tokens | 4.50 MB | 2.39 MB | 4.50 MB | 1.0x |
| 1024 tokens | 9.00 MB | 4.64 MB | 4.50 MB | **2.0x** |
| 2048 tokens | 18.00 MB | 9.14 MB | 4.50 MB | **4.0x** |
| 4096 tokens | 36.00 MB | 18.14 MB | 4.50 MB | **8.0x** |

At context length 4096, AURELIS uses **one-eighth** the active decoding memory of the Transformer. That directly translates to larger batch sizes and higher serving throughput on the accelerator.

---

## Formal Machine Proofs (Lean 4)

We formalize every core algebraic property in Lean 4 (`mathlib` 4.19.0) under namespace `Aurelis`:

- **Handoff Partition**: `handoff_partition`, `cache_overlap_redundancy`
- **Matrix Definiteness**: `precision_update_posSemidef`, `regularized_precision_posDef`, `regularized_precision_isUnit`
- **Associative Scans**: `Affine.combine_assoc`, `Affine.aggregate_correct`
- **Residual Identities**: `corrected_error_identity`, `weighted_residual_identity`, `corrected_reproduces_linear`, `corrected_exact_hit`
- **Multi-Hop Chains**: `composition_error_identity`, `composition_reproduces_linear`
- **Optimal Gating**: `routeVariance_completion`, `clippedGate_optimal`, `clippedGate_le_clippedIndependentGate`
- **Episodic Router**: `episodicGate`, `episodicGate_ge_bayes`, `episodicGate_ge_episodic`, `episodicGate_bounds`

The Lean verification runs clean with **zero `sorry`**, **zero `admit`**, and **zero custom `axiom`**. Check [`lean/PROOF_COVERAGE.md`](lean/PROOF_COVERAGE.md) for detailed mappings between theorems and mathematical statements.

---

## Quickstart & Reproduction

Everything needed to reproduce our benchmarks is checked in and scripted:

```bash
# 1. Spin up the virtual environment and install pinned ROCm wheels
./scripts/bootstrap.sh

# 2. Run the full unit and architecture test suite (69 tests)
.venv/bin/pytest tests/ -v

# 3. Run the Phase 6 benchmark suite and verify all gates on AMD Instinct MI300X
./scripts/run_phase6.sh
```

All raw metric logs, evaluation rows, and generated figures live in `results/` and `plots/`.
