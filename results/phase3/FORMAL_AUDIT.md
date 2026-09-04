# Phase 3 formal-requirement audit

This audit compares the deterministic mathematical properties of learned AURELIS
evaluated in Phase 3 with their Lean 4 formal statements in `lean/Aurelis/`.

| Empirical / Algorithmic Claim | Lean 4 Theorem | Coverage Boundary |
|---|---|---|
| Episodic override dominance over Bayes gate | `episodicGate_ge_bayes` in `Router.lean` | Full real-algebra theorem: $g_B \le \max(g_B, e_t)$ for all $g_B, e_t \in \mathbb{R}$. |
| Episodic override boundary containment in $[0, 1]$ | `episodicGate_bounds` in `Router.lean` | Full real-algebra theorem: if $0 \le g_B \le 1$ and $0 \le e_t \le 1$, then $0 \le \max(g_B, e_t) \le 1$. |
| Cache overlap double counting redundancy | `cache_overlap_redundancy` in `Handoff.lean` | Full list theorem: $(recent\ w\ history ++ history).length > history.length$ for non-empty history, proving double counting violates token conservation. |
| Conservation of history occurrences under delayed handoff | `handoff_partition` in `Handoff.lean` | Full list identity: $recent\ w\ history ++ remote\ w\ history = history$. |
| Unconstrained analytic Bayes gate variance optimality | `routeVariance_completion`, `clippedGate_optimal` in `Router.lean` | Full real-algebra theorems: $g_B$ minimizes conditional MSE over convex combinations. |
| Posterior query/key barycenter covariance simplification | `posterior_denominator_identity`, `posterior_numerator_identity` in `Router.lean` | Full real-algebra reductions of posterior covariance terms. |

## Empirical and Machine Learning Boundaries

The following properties remain outside formal proof and are established experimentally via Phase 3 empirical evaluation:
- Neural network parameter convergence and optimization dynamics under AdamW.
- Effective rank preservation of singular value spectra ($\text{erank}(W) \ge 2.0$) on finite random seeds.
- Generalization of learned associative retrieval to held-out sequence lengths and noise distributions.
- Feature attribution and AUROC of the episodic router responding to the observable task cue.
