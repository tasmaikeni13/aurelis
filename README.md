# AURELIS

**Attention with Uncertainty-Routed Residuals over an Episodic–Long-range
Inference State**

AURELIS is a proposed same-head hybrid of exact local softmax attention and a
bounded remote Bayesian least-squares state. Recent associations remain in a
window; evicted associations enter the remote state exactly once. A query is
read as

```text
y(q) = Mq + g(q) [vbar(q) - M kbar(q)]
```

where the analytic Bayes gate includes the cross-covariance between the remote
and residual estimators. An explicit episodic responsibility can override that
gate when the target is an observed cached exception rather than a denoised
latent relation.

## Current Progress & Status: Phase 4 PASS

The project strictly follows [`phases/AUTONOMY_PROTOCOL.md`](phases/AUTONOMY_PROTOCOL.md) with full mathematical derivations, Lean 4 formal machine proofs, and hardware-accelerated empirical verification on **AMD Instinct MI300X VF** GPUs under ROCm.

| Phase | Description | Hardware Target | Status | Evidence Record |
|---|---|:---:|:---:|---|
| **Phase 0** | Reference Substrate & Hardware Foundation | 1x MI300X | **PASS** | [`results/phase0/PASS.md`](results/phase0/PASS.md) |
| **Phase 1** | Exact Identities & Numerical Oracles | 1x MI300X | **PASS** | [`results/phase1/PASS.md`](results/phase1/PASS.md) |
| **Phase 2** | Controlled Baselines & Falsification Matrix | 1x MI300X | **PASS** | [`results/phase2/PASS.md`](results/phase2/PASS.md) |
| **Phase 3** | Learned Features & Episodic Routing (7 Task Families) | 1x MI300X | **PASS** | [`results/phase3/PASS.md`](results/phase3/PASS.md) |
| **Phase 4** | Nonstationarity, Compositional Access & Capacity Limits | 1x MI300X | **PASS** | [`results/phase4/PASS.md`](results/phase4/PASS.md) |
| **Phase 5** | Systems Profiling, Fused Kernels & Memory Optimization | 1x MI300X | *Next* | [`phases/phase5.md`](phases/phase5.md) |
| **Phase 6** | Natural Language Pilot Runs on FineWeb-Edu | 1x MI300X | *Planned* | [`phases/phase6.md`](phases/phase6.md) |
| **Phase 7** | **125M Parameters on 1.0B FineWeb-Edu Tokens** | **8x MI300X** | *Planned* | [`phases/phase7.md`](phases/phase7.md) |
| **Phase 8** | **350M Parameters on 3.0B FineWeb-Edu Tokens** | **8x MI300X** | *Planned* | [`phases/phase8.md`](phases/phase8.md) |
| **Phase 9** | Independent Clean-Room Reproduction & Release Audit | 8x MI300X | *Planned* | [`phases/phase9.md`](phases/phase9.md) |

---

## Phase 4 Highlights: Nonstationarity, Composition & Capacity Limits

Phase 4 tests the principal ways an undiscounted remote state can fail, introducing principled dynamic Bayesian updates, heteroscedastic precision weighting, and multi-hop pointer chasing across 5 paired seeds (401–405) on the AMD Instinct MI300X:

1. **Drift-Aware Adaptation with Observable Changepoints**:
   - Integrated dynamic linear model information discounting $\gamma_t = \text{clamp}(1 - c_t(1 - \gamma_{\min}), \gamma_{\min}, 1.0)$ into the affine state updates.
   - On observable changepoints, drift-aware AURELIS achieves **$10\times$ to $13\times$ lower post-change risk** than the stationary model ($0.0807$ vs $0.8767$ MSE), while preserving fundamental theoretical limitations on unobservable shifts ($0.8593$ vs $0.8036$).
2. **Gauss-Markov Heteroscedastic Precision Weighting**:
   - Valid precision weighting $\beta_t = 1/\sigma_t^2$ reduces MSE by **$12\times$** over uniform weighting ($0.0029$ vs $0.0354$). Corrupting precision labels transparently degrades risk by **$55\times$** ($0.1603$), proving the architecture actively relies on statistical evidence quality.
3. **Multi-Hop Composition with Cache Presence Discrimination**:
   - Solved error propagation across mixed cache/remote chains using sharp sigmoid presence discrimination $c_{\text{cache}} = \sigma(20(s_{\max} - 0.70))$ and temperature scaling ($\tau = 8.0$), achieving **83%–100% decoded success** across all 2-hop and 4-hop mixed chains (gate thresholds $\ge 85\%$ / $50\%$) with maximum vector error $\le 0.0658 \ll 0.35$.
4. **Subspace Capacity Limits Monotonically Enforced**:
   - Under adversarial association stress tests, error grows strictly monotonically beyond rank $d_k=8$ ($0.3886$ at $N=2$ to $0.9661$ at $N=256$), confirming that fixed-state memory does not claim unbounded associative recall.
5. **16x Context Length Extrapolation**:
   - Condition numbers and prediction MSE remain strictly finite and stable up to $16\times$ train sequence length ($L=512$ vs $L=32$).

---

## Phase 3 Highlights: Learned Features & Routing

Phase 3 established trained neural sequence models across 7 task families and 5 paired seeds (301–305):

1. **Straight-Through Estimator (STE) for $g_E = \max(g_B, e_t)$**:
   - Resolved the subgradient dead zone $\partial_b \max(a, b) = 0$ when $b < a$ by evaluating the exact mathematical hard maximum forward while backpropagating through a smooth surrogate $g_B + (1 - g_B) e_t$.
2. **Shared Feature Charts ($W_{kq}$)**:
   - Proved and numerically verified that shared key/query charts preserve the positive semi-definite RKHS kernel geometry, outperforming independent-chart ablations ($W_k \ne W_q$) with aggregate risk **0.4625** vs **0.6029** while retaining effective rank $\text{erank} = 13.35 \ge 2.0$.
3. **Calibrated Episodic Routing**:
   - Driven purely by an observable input feature channel without leaking evaluator labels, achieving an episodic responsibility AUROC of **1.0000** and cue correlation $R^2 = \mathbf{0.9478}$ (threshold $\ge 0.80$).
4. **Exception Isolation**:
   - AURELIS-E achieves a **$1.77\times$** error reduction on memorized exceptions over AURELIS-B without degrading latent anti-copy performance.

---

## Formal Proof Coverage (Lean 4)

All mathematical lemmas are formalized in Lean 4 (mathlib 4.19.0) under namespace `Aurelis`:

- **Handoff Partition**: `handoff_partition`, `cache_overlap_redundancy`
- **Bayesian State Updates**: `precision_update_posSemidef`, `regularized_precision_posDef`, `regularized_precision_isUnit`, `leaky_precision_update_posDef`
- **Associative Scan**: `Affine.combine_assoc`, `Affine.aggregate_correct`
- **Residual Identities**: `corrected_error_identity`, `weighted_residual_identity`, `corrected_reproduces_linear`, `corrected_exact_hit`
- **Multi-Hop Composition**: `composition_error_identity`, `composition_reproduces_linear`
- **Routing Gate Optimality**: `routeVariance_completion`, `clippedGate_optimal`, `clippedGate_le_clippedIndependentGate`
- **Episodic Router**: `episodicGate`, `episodicGate_ge_bayes`, `episodicGate_ge_episodic`, `episodicGate_bounds`

The proof suite compiles cleanly with **zero `sorry`**, **zero `admit`**, and **zero custom `axiom`**. See [`lean/PROOF_COVERAGE.md`](lean/PROOF_COVERAGE.md).

---

## Multi-Stage Scaling Roadmap (8x MI300X)

Starting from Phase 7, the project scales to distributed multi-GPU nodes on an **8x AMD Instinct MI300X** cluster (Phases 0–6 executed on 1x MI300X):

1. **Stage 1 (Phase 7)**:
   - **Scale**: **125M parameters** (`d_model=768`, 12 heads, `d_k=64, d_v=64`, 12 layers, context length 2048).
   - **Corpus**: **1.0 Billion tokens** of **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`).
   - **Cluster**: Distributed 8x AMD Instinct MI300X node using PyTorch FSDP on ROCm.
2. **Stage 2 (Phase 8)**:
   - **Scale**: **350M parameters** (`d_model=1024`, 16 heads, `d_k=64, d_v=64`, 24 layers, context length 2048).
   - **Corpus**: **3.0 Billion tokens** of **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`).
   - **Cluster**: Distributed 8x AMD Instinct MI300X cluster with long-context passkey and needle-in-a-haystack verification up to 16k context.

---

## Exact Reproduction

To set up the environment and run the complete verified test and experiment pipeline:

```bash
# 1. Bootstrap environment and compile Lean proofs
./scripts/bootstrap.sh

# 2. Run Phase 0 (Reference substrate and MI300X audit)
./scripts/run_phase0.sh

# 3. Run Phase 1 (Exact mathematical identities and fp64 oracles)
./scripts/run_phase1.sh

# 4. Run Phase 2 (Controlled baseline comparison and falsification suites)
./scripts/run_phase2.sh

# 5. Run Phase 3 (Learned neural models, 7 task families, 5 paired seeds)
./scripts/run_phase3.sh

# 6. Run Phase 4 (Nonstationarity, multi-hop composition, and capacity limits)
./scripts/run_phase4.sh
```

All raw training rows, JSON metrics, evaluation tables, and generated publication plots are stored under `results/` and `plots/`.
