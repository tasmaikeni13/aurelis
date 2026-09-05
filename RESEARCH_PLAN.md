# AURELIS Research Plan & Phase Roadmap

This document outlines the phased engineering and verification roadmap for the AURELIS project.

Every phase operates under [`phases/AUTONOMY_PROTOCOL.md`](phases/AUTONOMY_PROTOCOL.md). The rule is straightforward: if an assertion, unit test, or empirical gate fails, we never lower the bar or adjust tolerances to make it pass. We isolate the minimal failing case, trace the mathematical root cause, update the formal Lean statements where applicable, implement independent reference and production fixes, run regression tests, and iterate until the gate passes cleanly.

## Phase Dependency Graph

```text
Theory + Formal Lean Proofs + Numerical Oracles
                       │
                       ▼
Phase 0: Hardware Substrate & AMD MI300X Audit [PASS]
                       │
                       ▼
Phase 1: Exact Identities & Float64 Oracles [PASS]
                       │
                       ▼
Phase 2: Controlled Baselines & Falsification Matrix [PASS]
                       │
                       ▼
Phase 3: Learned Features & Episodic Routing [PASS]
                       │
                       ▼
Phase 4: Nonstationarity, Drift & Capacity Limits [PASS]
                       │
                       ▼
Phase 5: ROCm Kernel Profiling & Solver Optimization [PASS]
                       │
                       ▼
Phase 6: Language-Model Viability & Publication Gate (125M & 350M) [PASS]
                       │
                       ▼
Phase 7: Distributed 8x MI300X Multi-Seed 125M Scaling (1.0B tokens) [NEXT]
                       │
                       ▼
Phase 8: Medium-Scale 350M Pretraining (3.0B tokens) [PLANNED]
                       │
                       ▼
Phase 9: Clean-Room Reproduction & Release Audit [PLANNED]
```

---

## Completed Progress Summary (Phases 0 through 6)

1. **Hardware Foundation & Reference Substrate (Phase 0)**:
   - Audited the AMD Instinct MI300X VF accelerator under ROCm 7.0.2 with PyTorch 2.8.0.
   - Built independent double-precision CPU streaming and history oracles. Verified eager and TorchInductor compilation paths.

2. **Exact Numerical Identities (Phase 1)**:
   - Numerically verified the delayed handoff partition, error decomposition, and linear reproduction identities to float64 machine precision ($2.285 \times 10^{-16}$).
   - Confirmed closed-form cross-covariance gate optimality with a 50,000-trial conditional Monte Carlo run.

3. **Controlled Baselines & Mechanism Separation (Phase 2)**:
   - Ran head-to-head comparisons against Mesa-style Cumulative Least Squares, Gated DeltaNet, Positive-Feature Linear Attention, and Sliding Window Attention across 9 test suites.
   - Proved that the full cross-covariance Bayes gate strictly beats the independence heuristic ($z \ge 5.0$) under correlated errors.

4. **Learned Features & Routing Dynamics (Phase 3)**:
   - Resolved the episodic subgradient dead zone using a Straight-Through Estimator (STE).
   - Proved that shared key/query charts preserve RKHS geometry, achieving $1.77\times$ better exception recall over AURELIS-B without degrading latent anti-copy performance.

5. **Nonstationarity & Multi-Hop Composition (Phase 4)**:
   - Added information discounting for observable changepoints ($10\times$ to $13\times$ lower post-change risk).
   - Validated Gauss-Markov inverse-variance weighting and multi-hop pointer chasing through mixed cache/remote chains.

6. **Language Model Viability & Comparative Publication Gate (Phase 6)**:
   - Built complete implementations of the three key publication candidates: AURELIS (AURELIS-E and AURELIS-B), Modern Causal Transformer (RoPE + RMSNorm + SwiGLU), and Strong SSM+Attention Hybrid (Samba/Jamba-style alternating selective scan + attention).
   - Calibrated parameter counts within $\pm 3.6\%$ across both 125M and 350M parameter scales.
   - Written and validated custom HIP C++ kernels targeting `gfx942` for recurrent sequence scans and fused residual gating.
   - Demonstrated strictly constant $O(1)$ decoding state memory for AURELIS, achieving an $8.0\times$ memory reduction at context length 4096 compared to the Transformer KV cache.

---

## Upcoming Scaling Gates

- **Phase 7 (Distributed 125M Pretraining)**: Run distributed data parallel / FSDP pretraining on an 8x AMD Instinct MI300X cluster over 1.0 Billion tokens from the FineWeb-Edu corpus across multiple paired seeds.
- **Phase 8 (Medium-Scale 350M Pretraining)**: Scale to 350M parameters and 3.0 Billion tokens, verifying long-context needle-in-a-haystack retrieval up to 16,384 tokens.
- **Phase 9 (Release Audit & Final Manuscript)**: Complete clean-room reproduction from scratch, lock down all figures and artifact hashes, and finalize the publication manuscript `aurelis.md`.
