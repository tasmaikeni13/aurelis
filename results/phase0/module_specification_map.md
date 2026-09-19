# AURELIS Phase 0: Module Architecture Specification Map

**Generated UTC:** `2026-09-19T07:02:05.356442+00:00`  
**Target Specification:** `aurelis.md`, `phases/IMPLEMENTATION_CONTRACT.md`

---

## 1. Module-by-Module Specification

| Module / Area | Contract Requirement | Equations Covered | Target Phase |
|---|---|---|---|
| `src/aurelis/functional.py / oracle.py` | Independent fp64 references for Eq. (2) gated delta recurrence, Eq. (3) bounded read r(q) = v̄_L + S_t(q - k̄_L), Eq. (7)-(8) normalized archive completion, Eq. (10) deterministic certificate, and Eq. (11)-(12) page box envelopes. | Eq. (2), (3), (7), (8), (9), (10), (11), (12) | Phase 1 |
| `src/aurelis/types.py / streaming.py` | DeltaState with S ∈ ℝ^{d_v × d_k}, ring buffer of recent keys/values and write gates (α, β), causal occurrence IDs, archive page descriptors, and explicit read result types with status ('approximate', 'certified', 'full_read', 'budget_exhausted', 'invalid_state', 'archive_error'). | Eq. (2), (3), (8), (10) | Phase 2 |
| `src/aurelis/nn.py` | Shared query/key coordinate projection conventions, delayed gated delta recurrence, solve-free bounded read, and archive forward paths with log-sum-exp shift. | Eq. (2), (3), (8), (14) | Phase 3 & 4 |
| `src/aurelis/models/` | Actual cached step decode, true local window attention without quadratic score materialization, structured chunk delta training with backward recomputation. | Eq. (2), (3), (8), (15) | Phase 5 & 6 |
| `src/aurelis/baselines/` | Modern dense GQA Transformer (RoPE, RMSNorm, SwiGLU), recurrence-free archive completion, per-page predictor comparator (Eq. 13), and Mamba-2 style SSM baseline. | Eq. (7), (13), §8, §9 | Phase 2 & 4 |
| `benchmarks/ and experiments/` | Real token evaluation on held-out tasks, populated prefix cache latency benchmarks, measured cache bytes, and explicit certificate violation audits. | §7, §8, §9 | Phase 6, 7, and 8 |
| `tests/` | Verification gates, real data provenance checks, formal proof coverage checks, and unit tests for delta recurrence, certificate bounds, and page intervals. | AUTONOMY_PROTOCOL.md, IMPLEMENTATION_CONTRACT.md | Phase 1-9 |


---

## 2. Architectural Principles

1. **Solve-Free Recurrent State:** Eliminates key-space matrix factorizations and inverses ($O(d_k^3)$ Cholesky factorizations), reducing persistent state update work to $O(d_v d_k + w(d_k + d_v))$.
2. **Disjoint Causal Partitioning:** Cache observations within window $w$ and evicted observations are strictly partitioned.
3. **Mass-Consistent Archive Completion:** Normalized archive completion combines selected exact observations with predicted unread mass under deterministic residual certificates.
