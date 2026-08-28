# Formal proof coverage

| Lean theorem | Paper content | Coverage boundary |
|---|---|---|
| `handoff_partition` | The recent cache and remote suffix partition every history occurrence | Full list identity; not a GPU ring-buffer implementation |
| `recent_length_le_window` | Cache size is at most the fixed window | Full list theorem |
| `Affine.combine_assoc` and identities | Remote affine summaries form a scan monoid | Full algebraic identity |
| `Affine.aggregate_correct` | A finite sequential recurrence equals its aggregate action | Full generic recurrence theorem |
| `statisticsElement_action` | One matrix-statistic update is the advertised affine update | Full finite-matrix identity |
| `precision_update_posSemidef` | Nonnegative evidence preserves PSD precision statistics | Full finite real-matrix theorem |
| `regularized_precision_posDef` / `isUnit` | A positive prior makes the remote solve uniquely defined | Full finite real-matrix theorem |
| `corrected_error_identity` | Error splits into attention residual plus remote slope error | Full linear-map identity |
| `corrected_reproduces_linear` | Residual correction reproduces an exact linear operator for arbitrary local barycenter | Full linear-map theorem |
| `corrected_exact_hit` | A one-hot recent hit returns its stored target independently of remote state | Full linear-map theorem |
| `weighted_residual_identity` | Barycentric residual equals the weighted pointwise residual sum | Full finite-index theorem |
| `softmaxWeight_pos` / `softmaxWeight_sum` | Finite softmax weights are a strict probability vector | Full finite-index theorem |
| `routeVariance_completion` | The routing variance is a completed square about the analytic gate | Full real-algebra theorem |
| `rawGate_le_remote` / `rawGate_le_residual` | The unconstrained analytic gate is variance-noninferior to both endpoints | Full real-algebra theorem when the quadratic denominator is positive |
| `clippedGate_optimal` | Projecting the gate to `[0,1]` minimizes variance over all convex gates | Full real-algebra theorem when the denominator is positive |
| `posterior_denominator_identity` / `posterior_numerator_identity` | Posterior covariance terms simplify to the paper's gate formula | Full scalar reduction |

Not formalized: derivation of the conditional covariance from matrix-normal
probability, concentration bounds, the fp64 Monte Carlo experiment, Cholesky
backward error, kernel complexity, or any empirical performance claim. Those
remain analytic or experimental evidence and are not described as Lean-proved.
