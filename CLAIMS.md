# AURELIS Claim Registry

This document tracks every specific mathematical, formal, and empirical claim made about the AURELIS architecture.

### Status Definitions
- **Proved**: A complete mathematical proof exists in the manuscript or appendix.
- **Lean-checked**: The theorem statement and proof compile with Lean 4 and mathlib with zero `sorry` or custom axioms.
- **Numerically verified**: An automated, deterministic experiment reproduced the claim within defined numerical bounds.
- **Pending**: Work is in progress or planned for a later phase; no scientific claim is asserted yet.

| Claim ID | Claim Summary | Direct Evidence | Lean Coverage | Current Status & Scope |
|---|---|---|---|---|
| **AUR-HANDOFF-1** | Recent `take w` and remote `drop w` partition history exactly once without double-counting | Manuscript Lemma 5.1; `tests/test_partition.py`; Phase 0 & 1 logs | `handoff_partition` | Proved, Lean-checked, and verified against the streaming ring buffer. |
| **AUR-STATE-1** | Nonnegative rank-one evidence updates preserve positive semi-definiteness; prior ensures unique solve | Manuscript Appendix A.4 | `precision_update_posSemidef`, `regularized_precision_posDef`, `regularized_precision_isUnit` | Proved and Lean-checked for real finite matrices. |
| **AUR-SCAN-1** | Scalar-decayed remote statistics compose associatively, enabling parallel prefix scans | Manuscript Eq. 7.2 | `Affine.combine_assoc`, `Affine.aggregate_correct` | Proved and Lean-checked. |
| **AUR-RESID-1** | Full-residual error decomposes into the local residual plus the remote slope error on $(q - \bar{k})$ | Manuscript Theorem 5.2; independent fp64 oracle comparison | `corrected_error_identity`, `weighted_residual_identity` | Proved, Lean-checked, and verified in dual implementations. |
| **AUR-LIN-1** | If the remote linear map is exact, the full residual reproduces it regardless of attention smoothing | Manuscript Corollary 5.3; measured residual error $2.285 \times 10^{-16}$ | `corrected_reproduces_linear` | Proved, Lean-checked, and verified in float64. |
| **AUR-HIT-1** | A one-hot cached hit with episodic override recovers the stored value exactly, ignoring remote state | Manuscript Corollary 5.4; fp64 error `0.0` | `corrected_exact_hit` | Proved, Lean-checked; finite softmax approximates one-hot. |
| **AUR-RIDGE-1** | Scalar ridge slope error is bounded by the prior, minimum remote eigenvalue, and residual query norm | Manuscript Prop. 5.5; Phase 1 parameter sweeps | `scalar_ridge_slope_error`, `scalar_ridge_residual_bound` | Scalar case proved and Lean-checked; matrix spectral bound is verified empirically. |
| **AUR-COV-1** | Under the disjoint linear-Gaussian model, remote and residual error covariance is $q^T P^{-1}(q - \bar{k})$ | Manuscript Section 6; 50,000-run Monte Carlo sweep | Scalar reductions | Analytic derivation confirmed by Monte Carlo within $0.293\%$ relative error. |
| **AUR-GATE-1** | The closed-form projected Bayes gate minimizes conditional MSE among all convex mixtures | Manuscript Theorem 6.1; dense grid sweeps | `routeVariance_completion`, `clippedGate_optimal` | Proved, Lean-checked, and verified across all test suites. |
| **AUR-TARGET-1** | Latent linear relation denoising and exact exception recall pull in opposite directions when targets conflict | Manuscript Prop. 6.2; synthetic conflict sweeps | `corrected_exact_hit` | Proved mathematically; demonstrated on synthetic tasks. |
| **AUR-COST-1** | Inference decode memory state is strictly constant $O(1)$ with respect to total context length | Manuscript Section 7.1; `tests/test_phase6.py`; MI300X memory traces | Not formalized | Proved by construction; verified on MI300X ($8\times$ memory reduction at 4k context). |
| **AUR-TRAIN-1** | State construction is $O(L)$ associative scan; exact all-prefix solves require dense solves or iterative approximations | Manuscript Section 7.2 | Scan algebra only | Proved arithmetic counts; iterative and chunked solvers implemented. |
| **AUR-ROCM-1** | Eager, compiled, and custom HIP kernel paths agree with fp64 references on AMD Instinct MI300X | `results/phase0/environment.json`, `results/phase6/metrics.json` | None | Verified on MI300X VF (`gfx942`) with residual errors $< 10^{-6}$. |
| **AUR-GATE-2** | Full cross-covariance gating beats the independence heuristic when remote and local errors correlate | Manuscript Theorem 6.1; Phase 2 correlated test suite ($z \ge 5.0$) | `clippedGate_le_clippedIndependentGate` | Proved, Lean-checked, and empirically verified. |
| **AUR-SEP-1** | Hybrid mechanisms show distinct tradeoffs across parameter, dimension, FLOP, and state budgets | Phase 2 baselines matrix (10 baselines, 9 suites) | None | Verified across Mesa, DeltaNet, Linear Attention, and AURELIS. |
| **AUR-LEARN-1** | Learned projections and straight-through episodic routing preserve the core theoretical mechanism | Phase 3 results (7 task families, 5 paired seeds) | `episodicGate`, `episodicGate_ge_bayes`, `cache_overlap_redundancy` | Proved, Lean-checked, and verified (AUROC 1.000, $R^2 = 0.948$). |
| **AUR-LM-1** | AURELIS matches Transformer and SSM Hybrid language modeling viability while maintaining $O(1)$ decode state | Phase 6 results (`results/phase6/metrics.json`, `results/phase6/PASS.md`) | None | Verified on MI300X across 125M and 350M scales; multi-billion token scaling targets Phase 7/8. |

Machine-readable numerical records live in `analysis/results/summary.json` and `results/`. Lean formal coverage and theorem boundaries are documented in [`lean/PROOF_COVERAGE.md`](lean/PROOF_COVERAGE.md).
