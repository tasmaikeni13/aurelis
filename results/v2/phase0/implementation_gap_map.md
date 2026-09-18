# AURELIS-R v2 Phase 0: Implementation Gap Map

**Generated UTC:** `2026-09-18T06:45:55.318559+00:00`  
**Theory Revision:** `v2.0`  
**Target Specification:** `aurelis.md`, `IMPLEMENTATION_CONTRACT.md`

---

## 1. Module-by-Module Migration Analysis

| Module / Area | Existing v1 Implementation | v2 Contract Requirement | Equations Covered | Gap Severity | Migration Action |
|---|---|---|---|---|---|
| `src/aurelis/functional.py / oracle.py` | Standard path calls Cholesky factorization of dense key-space precision matrix P = λI + KᵀK and computes M = C P⁻¹; Bayes router computes scalar gate g_B. | Independent fp64 references for Eq. (2) gated delta recurrence, Eq. (3) bounded read r(q) = v̄_L + S_t(q - k̄_L), Eq. (7)-(8) normalized archive completion, Eq. (10) deterministic certificate, and Eq. (11)-(12) page box envelopes. | Eq. (2), (3), (7), (8), (9), (10), (11), (12) | **High (Complete redesign required)** | Preserve v1 functions under legacy names; write clean v2 fp64 independent oracles in Phase 1. |
| `src/aurelis/types.py / streaming.py` | Defines BayesianState with C, P, inv_P tensors; HandoffState; read results without error certificates or unread mass tracking. | DeltaState with S ∈ ℝ^{d_v × d_k}, ring buffer of recent keys/values and write gates (α, β), causal occurrence IDs, archive page descriptors, and explicit read result types with status ('approximate', 'certified', 'full_read', 'budget_exhausted', 'invalid_state', 'archive_error'). | Eq. (2), (3), (8), (10) | **High (New state schema and return contracts required)** | Implement v2 state classes and read result contracts in Phase 2. |
| `src/aurelis/nn.py / nn_phase3.py / nn_phase4.py` | Neural modules with Bayesian ridge solvers, straight-through maximum episodic router, and heteroscedastic weighting. | Shared query/key coordinate projection conventions, delayed gated delta recurrence, solve-free bounded read, and archive forward paths with log-sum-exp shift. | Eq. (2), (3), (8), (14) | **High (Retire Bayesian router; implement solve-free recurrence)** | Implement clean nn modules for bounded and archive modes in Phase 3 & 4. |
| `src/aurelis/models/aurelis_lm.py / jax_aurelis.py` | Constructs full L × L attention scores then applies causal mask; materializes prefix precision matrices; lacks populated-cache continuation decode. | Actual cached step decode, true local window attention without quadratic score materialization, structured chunk delta training with backward recomputation. | Eq. (2), (3), (8), (15) | **High (Quadratic intermediates must be completely removed)** | Re-implement LM architectures and true cached decode in Phase 5 & 6. |
| `src/aurelis/models/tpu_kernels.py` | Kernel stubs; precision checks executed on CPU fallback. | Explicit actual backend implementation; chunk delta scan, local attention, page gathers/reductions, and interval evaluations. | Eq. (2), (11), (12) | **Medium (Backend-specific kernel implementation)** | Implement verified CPU and accelerator kernels in Phase 5. |
| `src/aurelis/baselines.py / models/hybrid_ssm.py` | 10 synthetic baselines tuned for v1 Bayesian ridge comparison; simplified architectures. | Modern dense GQA Transformer (RoPE, RMSNorm, SwiGLU), recurrence-free archive completion, per-page predictor comparator (Eq. 13), and Mamba-2 style SSM baseline. | Eq. (7), (13), §8, §9 | **Medium (Need strong competitive baselines)** | Implement updated baselines in Phase 2 & 4. |
| `experiments/ and benchmarks/` | Diagnostic routines with formula-assigned scores and single-step decode latency without prefix caches. | Real token evaluation on held-out tasks, populated prefix cache latency benchmarks, measured cache bytes, and explicit certificate violation audits. | §7, §8, §9 | **High (Rewrite benchmark scripts with real measurements)** | Write new v2 experiment scripts in Phase 6, 7, and 8. |
| `scripts/ and tests/` | Scripts verify legacy metrics and check for retired project name strings; tests check Cholesky solvers. | V2 verification gates, real data provenance checks, formal proof coverage checks, and unit tests for delta recurrence, certificate bounds, and page intervals. | AUTONOMY_PROTOCOL.md, IMPLEMENTATION_CONTRACT.md | **Medium (New test suites for v2 contracts)** | Implement v2 test suites alongside each phase. |

---

## 2. Key Mathematical Architectural Deltas

1. **Elimination of Key-Space Solvers:** v1 computed Bayesian state $M = C P^{-1}$ requiring $O(d_k^3)$ Cholesky factorizations per step. v2 replaces this with solve-free gated delta updates (Eq. 2) and bounded read (Eq. 3), reducing persistent state update work to $O(d_v d_k + w(d_k + d_v))$.
2. **Exact Disjoint Occurrence Partition:** Cache tokens ($w$) and evicted tokens are partitioned exactly. Reading an archive page never repeats a recurrent write.
3. **Mass-Consistent Archive Completion:** Normalized completion (Eq. 8) combines selected exact tokens and predicted unread mass with deterministic residual certificate (Eq. 10).
4. **Removal of Quadratic Intermediates:** Retires full $L \times L$ score matrices in `jax_aurelis.py` in favor of true local window attention and structured delta chunk scans.

---

## 3. Implementation Gap Map Verdict

- Comprehensive mapping of all repository modules against v2 equations: **PASS**
- Migration paths defined for downstream phases: **PASS**
