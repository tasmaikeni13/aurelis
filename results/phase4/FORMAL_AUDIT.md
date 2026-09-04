# Phase 4 formal-requirement audit

This audit compares the mathematical properties of nonstationarity, compositional access,
and capacity limits evaluated in Phase 4 with their Lean 4 formal statements in `lean/Aurelis/`.

| Empirical / Algorithmic Claim | Lean 4 Theorem | Coverage Boundary |
|---|---|---|
| Multi-hop linear composition error decomposition | `composition_error_identity` in `ResidualCorrection.lean` | Full linear-map identity: $(M_2 \circ M_1)(x) - (W_2 \circ W_1)(x) = (M_2 - W_2)(M_1 x) + W_2 ((M_1 - W_1) x)$. |
| Multi-hop linear composition exact reproduction | `composition_reproduces_linear` in `ResidualCorrection.lean` | Full linear-map theorem: $(W_2 \circ W_1)(x) = W_2 (W_1 x)$. |
| Positive definiteness under leaky exponential decay | `leaky_precision_update_posDef` in `MatrixState.lean` | Full finite real-matrix theorem: $(1 - \gamma) \Lambda + \gamma P + \beta k k^T \succ 0$ for $\gamma \in [0, 1)$, $\Lambda \succ 0$, $P \succeq 0$. |
| Precision update preservation of positive semidefiniteness | `precision_update_posSemidef` in `MatrixState.lean` | Full finite real-matrix theorem: $\gamma P + \beta k k^T \succeq 0$ for $\gamma \ge 0, \beta \ge 0, P \succeq 0$. |
| Associative affine scan monoid for discounted state | `Affine.combine_assoc`, `Affine.aggregate_correct` in `AffineScan.lean` | Full algebraic identity and generic recurrence theorem. |
| Invariance of delayed handoff partition under composition | `handoff_partition` in `Handoff.lean` | Full list identity: $take\ w\ history ++ drop\ w\ history = history$. |
| Analytic Bayes gate variance optimality | `routeVariance_completion`, `clippedGate_optimal` in `Router.lean` | Full real-algebra theorems: $g_B$ minimizes conditional MSE over convex combinations. |
| Episodic override bounds and dominance | `episodicGate_ge_bayes`, `episodicGate_bounds` in `Router.lean` | Full real-algebra theorems. |

## Empirical and Machine Learning Boundaries

The following properties remain outside formal proof and are established experimentally via Phase 4 empirical evaluation:
- Dynamic operator drift adaptation speeds and post-change risk convergence under finite training data.
- Empirical error propagation curves and decoded success rates across multi-hop pointer chasing chains of lengths $\{1, 2, 4, 8, 16\}$.
- Heteroscedastic noise resilience under empirical Student-$t$ ($\nu=3$) and outlier distributions.
- Measured accelerator wall-clock latency (ms) scaling as a function of adaptive hop count on AMD Instinct MI300X VF.
- Monotonic recall degradation on adversarial dictionary sweeps exceeding subspace rank capacity ($N > d_k$).
