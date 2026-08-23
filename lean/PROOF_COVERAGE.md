# Formal proof coverage

| Lean theorem | Mathematical content | Manuscript target | Status boundary |
|---|---|---|---|
| `Affine.combine_assoc` | affine scan composition is associative | Theorem 4.1 / Proposition 6.1 | Full algebraic identity |
| `Affine.combine_identity_left/right` | `(1,0)` is the scan identity | Theorem 4.1 / Proposition 6.1 | Full algebraic identity |
| `Affine.act_combine` | combined element acts like chronological composition | Theorem 4.1 | Full algebraic identity |
| `Affine.aggregate_correct` | an arbitrary finite recurrence equals its aggregate scan action | Definition 5.1 / Proposition 6.1 | Generic state-module recurrence |
| `statisticsElement_action` | the affine action is exactly the simultaneous matrix `(S,C)` recurrence | Definition 5.1 | Full finite-matrix identity |
| `outer_self_posSemidef` | a real rank-one outer product is PSD | Definition 5.1 write term | Full finite-matrix theorem |
| `memory_S_update_posSemidef` | nonnegative decay/evidence preserves PSD under the exact matrix update | Definition 5.1 | Full finite-matrix theorem |
| `regularized_system_posDef` | positive epsilon makes `S + epsilon I` positive definite | Definition 5.1 read system | Full finite-matrix theorem |
| `regularized_system_isUnit` | the regularized read system is invertible | Definition 5.1 read system | Full finite-matrix theorem |
| `quadraticUpdate_nonneg` | scalar quadratic-form reduction of PSD preservation | PSD invariant behind Definition 5.1 | Redundant scalar certificate |
| `regularizedQuadratic_positive` | scalar quadratic-form reduction of the epsilon floor | `A=S+epsilon I` in Definition 5.1 | Requires a nonzero query (`normSquared>0`) |
| `oneKey_read_error` | exact finite-epsilon shrinkage error for one normalized key | Theorem 5.2, one-key case | Scalar/one-key specialization |
| `oneKey_variance_positive` | the one-key posterior variance is positive | Proposition 5.6, one-key case | Scalar/one-key specialization |

Not yet claimed as formally proved: the full matrix interpolation bound in Theorem 5.2, Gauss–Markov optimality, calibration under the stochastic model, ricochet error bounds, Cholesky backward stability, or any dyadic-cascade theorem. Those require substantially larger probability/numerical-analysis developments. Their absence is not a proof failure.
