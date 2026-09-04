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

## Current Progress & Status: Phase 3 PASS

The project strictly follows [`phases/AUTONOMY_PROTOCOL.md`](phases/AUTONOMY_PROTOCOL.md) with full mathematical derivations, Lean 4 formal machine proofs, and hardware-accelerated empirical verification on **AMD Instinct MI300X VF** GPUs under ROCm.

| Phase | Description | Status | Evidence Record |
|---|---|:---:|---|
| **Phase 0** | Reference Substrate & Hardware Foundation | **PASS** | [`results/phase0/PASS.md`](results/phase0/PASS.md) |
| **Phase 1** | Exact Identities & Numerical Oracles | **PASS** | [`results/phase1/PASS.md`](results/phase1/PASS.md) |
| **Phase 2** | Controlled Baselines & Falsification Matrix | **PASS** | [`results/phase2/PASS.md`](results/phase2/PASS.md) |
| **Phase 3** | Learned Features & Episodic Routing (7 Task Families) | **PASS** | [`results/phase3/PASS.md`](results/phase3/PASS.md) |
| **Phase 4** | Nonstationarity, Drift & Multi-Hop Composition | *Next* | [`phases/phase4.md`](phases/phase4.md) |
| **Phase 5** | MI300X/ROCm Systems Profiling & Kernel Optimization | *Planned* | [`phases/phase5.md`](phases/phase5.md) |
| **Phase 6** | Natural Language Pilot Runs on FineWeb-Edu | *Planned* | [`phases/phase6.md`](phases/phase6.md) |
| **Phase 7** | **125M Parameters on 1.0B FineWeb-Edu Tokens** | *Planned* | [`phases/phase7.md`](phases/phase7.md) |
| **Phase 8** | Camera-Ready Replication Suite | *Planned* | [`phases/phase8.md`](phases/phase8.md) |

---

## Phase 3 Highlights: Learned Features & Routing

Phase 3 transitions from analytical verification to trained neural sequence models across 7 task families and 5 paired seeds (301–305) on the AMD Instinct MI300X:

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
- **Bayesian State Updates**: `precision_update_posSemidef`, `regularized_precision_posDef`, `regularized_precision_isUnit`
- **Associative Scan**: `Affine.combine_assoc`, `Affine.aggregate_correct`
- **Residual Identities**: `corrected_error_identity`, `weighted_residual_identity`, `corrected_reproduces_linear`, `corrected_exact_hit`
- **Routing Gate Optimality**: `routeVariance_completion`, `clippedGate_optimal`, `clippedGate_le_clippedIndependentGate`
- **Episodic Router**: `episodicGate`, `episodicGate_ge_bayes`, `episodicGate_ge_episodic`, `episodicGate_bounds`

The proof suite compiles cleanly with **zero `sorry`**, **zero `admit`**, and **zero custom `axiom`**. See [`lean/PROOF_COVERAGE.md`](lean/PROOF_COVERAGE.md).

---

## Publication Roadmap (125M / 1.0B FineWeb-Edu)

The project targets a rigorous, reproducible publication comparing AURELIS against major attention and recurrent memory architectures:

- **Target Scale**: **125M parameters** (`d_model=768`, 12 heads, `d_k=64, d_v=64`, 12 layers, context length 2048).
- **Corpus**: **1.0 Billion tokens** of **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`).
- **Matched Comparators**: Standard Transformer, Sliding-Window Transformer, Gated DeltaNet, Cumulative Least-Squares (Mesa/RWKV-style), and Native Hybrid Attention.
- **Compute Target**: 8x AMD Instinct MI300X node using distributed PyTorch FSDP on ROCm.

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
```

All raw training rows, JSON metrics, evaluation tables, and generated publication plots are stored under `results/` and `plots/`.

---

## Repository Map

| Path | Description |
|---|---|
| [`src/aurelis/`](src/aurelis/) | Core library: streaming state, history oracles, baselines, and Phase 3 neural models |
| [`lean/`](lean/) | Lean 4 formal mathematical proofs (`Aurelis.Router`, `Aurelis.Handoff`, etc.) |
| [`configs/`](configs/) | Versioned hyperparameters, task thresholds, and gate configurations |
| [`experiments/`](experiments/) | Multi-seed experiment runners for each phase |
| [`tests/`](tests/) | Unit and property test suite (60/60 passing tests) |
| [`phases/`](phases/) | Project protocol ([`AUTONOMY_PROTOCOL.md`](phases/AUTONOMY_PROTOCOL.md)) and Phase 0–8 specifications |
| [`results/`](results/) | Versioned `metrics.json`, `report.md`, and official `PASS.md` records |
| [`plots/`](plots/) | Generated publication figures for empirical regimes and gate evidence |
| [`scripts/`](scripts/) | End-to-end execution and automated gate verification scripts |
| [`CLAIMS.md`](CLAIMS.md) | Official claim-by-claim registry tracking proof and empirical status |
| [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) | Chronological research journal and execution log |
