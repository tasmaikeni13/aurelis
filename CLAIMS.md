# AURELIS claim registry

Status vocabulary: **proved** means a mathematical proof is supplied;
**Lean-checked** means the stated formal encoding compiles; **numerically
verified** means the committed program passes its declared finite experiment;
**pending** means no empirical claim is made.

| ID | Claim | Direct evidence | Lean coverage | Status / boundary |
|---|---|---|---|---|
| AUR-HANDOFF-1 | Recent `take w` and remote `drop w` partition every history occurrence | Paper Lemma 5.1 | `handoff_partition` | Proved and Lean-checked; not yet a GPU ring-buffer test |
| AUR-STATE-1 | Nonnegative rank-one evidence updates preserve PSD, and positive prior makes the solve unique | Paper Appendix A.4 | `precision_update_posSemidef`, `regularized_precision_posDef`, `regularized_precision_isUnit` | Proved and Lean-checked over finite real matrices |
| AUR-SCAN-1 | Scalar-decayed remote statistic updates compose associatively | Paper Eq. 7.2 | `Affine.combine_assoc`, `Affine.aggregate_correct` | Proved and Lean-checked |
| AUR-RESID-1 | Full-residual error equals local residual plus remote slope error on `q-kbar` | Paper Theorem 5.2; numerical max error `9.992e-16` | `corrected_error_identity`, `weighted_residual_identity` | Proved, Lean-checked, numerically verified |
| AUR-LIN-1 | With exact remote linear map, full residual reproduces the map for arbitrary attention weights | Paper Corollary 5.3; numerical error `2.285e-16` | `corrected_reproduces_linear` | Proved, Lean-checked, numerically verified |
| AUR-HIT-1 | A one-hot cached hit with full episodic gate returns its stored value independently of remote state | Paper Corollary 5.4; fp64 error `0` | `corrected_exact_hit` | Proved, Lean-checked; finite softmax only approaches one-hot |
| AUR-RIDGE-1 | Noise-free scalar-ridge slope error is bounded by prior, remote minimum eigenvalue, and residual-query norm | Paper Proposition 5.5 | Not fully formalized | Analytic proof; full matrix norm formalization pending |
| AUR-COV-1 | Under the declared disjoint linear-Gaussian model, endpoint covariance is `q^T P^-1(q-kbar)` | Paper Eqs. 6.3–6.5; conditional Monte Carlo | Scalar reductions only | Analytic probability derivation; Monte Carlo relative errors below `0.5%` |
| AUR-GATE-1 | Projected closed-form gate minimizes conditional MSE over convex mixtures and is endpoint-noninferior in-model | Paper Theorem 6.1; dense-grid regret `0` | `routeVariance_completion`, `clippedGate_optimal`, endpoint theorems | Proved, Lean-checked, numerically verified under assumptions |
| AUR-TARGET-1 | Latent denoising and exact observed-exception copy are incompatible when their targets differ | Paper Proposition 6.2; exception sweep | Deterministic exact-hit theorem only | Proved by target inequality; learned task inference pending |
| AUR-COST-1 | Decode state/work is independent of total context at fixed `d_k,d_v,w` | Paper Section 7.1 | Not formalized | Algorithmic count; implementation measurement pending |
| AUR-TRAIN-1 | Remote statistics scan associatively, but exact all-prefix training solves are not automatically linear-work | Paper Sections 7.2–7.3 | Scan algebra only | Proved count with explicit solver alternatives; no speed claim |
| AUR-ROCM-1 | A correct and competitive MI300X/ROCm implementation exists | Phase 0/5 prompts | None | Pending; no current claim |
| AUR-LEARN-1 | Learned features preserve the theoretical mechanism | Phase 3 prompt | None | Pending; no current claim |
| AUR-LM-1 | AURELIS improves language-model quality/efficiency | Phase 6/7 prompts | None | Pending; no current claim |

Machine-readable numerical details are in
`analysis/results/summary.json`. Formal boundaries are in
`lean/PROOF_COVERAGE.md`.
