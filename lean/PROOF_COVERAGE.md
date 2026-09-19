# AURELIS Formal Proof Coverage

Pinned Lean/mathlib 4.19.0. All theorems are checked under namespace `Aurelis`. All claims are conditional on their formal theorem premises.

## Theorem Coverage

| Theorem | Paper claim | Exact scope |
|---|---|---|
| `exact_recall_injective` | §1.1 exact recall implies injective encoding | Arbitrary address/value/state types, deterministic correct decoder |
| `exact_recall_capacity` | Eq. (1), at least $m^n$ finite states | Finite cardinal inequality; logarithmic bit form is analytic |
| `deltaRead_add` | Eq. (2) is linear under query addition | Evaluation identity for a linear memory map |
| `deltaRead_exact_write` | Unit-key $\beta=1$ write returns $v$ at $k$ | Inner product $\langle k, k \rangle = 1$; does not preserve unrelated keys |
| `deltaTransition_energy` | Eq. (5) rank-one energy identity | Real inner-product space, arbitrary real $\beta$ |
| `deltaTransition_nonexpansive` | Eq. (5) contraction condition | $\beta \ge 0, \beta \|k\|^2 \le 2$; one row, fixed inputs |
| `decayed_delta_nonexpansive` | Nonnegative $\alpha$ multiplies perturbation bound | Row result; matrix Frobenius corollary and driven-state bound (6) are analytic |
| `completedRead_balance` | Eq. (8) normalization identity | Nonzero estimated total mass |
| `completedRead_full` | Empty-unread endpoint | Algebraic selected normalized sum, not an accelerator kernel equality |
| `completion_error_identity` | Eq. (9) numerator and mass-error split | Vectors in real normed spaces; both total masses nonzero |
| `residual_certificate` | Eq. (10) norm consequence | Positive denominator floor, valid residual and mass-error bounds |
| `midpoint_error` | Midpoint mass interval | Exact reals with $\text{lower} \le \text{actual} \le \text{upper}$ |
| `exp_score_interval` | Part of Eqs. (11)–(12) | Scalar exponential monotonicity; no tensor box constructor |
| `weighted_residual_bound` | Residual envelope | Finite weighted vector sum with nonnegative bounded weights and residual radii |
| `completedRead_certificate` | Eq. (10), tied to the actual completed output | Positive selected mass, nonnegative lower mass, valid upper/lower and residual norm |
| `coordinate_product_interval` / `dot_box_interval` | Coordinate key boxes imply score intervals | Exact finite sums; no mutable page/index correspondence |
| `page_mass_interval` | Page score interval implies exponential mass interval | Exact reals; no floating-point enclosure |
| `page_residual_ball` | Page value ball and score upper bound imply residual envelope | Exact finite weighted sum; page metadata validity remains external |
| `handoff_partition` | Partition of recent/remote history | List partition facts, not a mutable ring implementation |
| `recent_length_le_window` | Recent cache length bounded by window | List length bounded by $w$ |
| `remote_empty_before_window` | Remote indices empty before step reaches window | List property |
| `corrected_error_identity` | Linear transport error decomposition | Applies to any linear memory map |
| `corrected_reproduces_linear` | Exact reproduction of linear ground truth | When memory equals target map and local values are consistent |
| `corrected_exact_hit` | One-hot local hit reproduces value | Query matches local key exactly |
| `groupedCompletedRead_balance` | Eq. (13) grouped completion balance | Nonzero total mass across groups |
| `groupedCompletedRead_full` | Eq. (13) empty unread group recovery | Algebraic selected normalized sum |
| `grouped_residual_certificate` | Eq. (13) grouped certificate bound | Finite index collection, positive denominator floor, valid envelope bounds |

The certificate is derived directly from the defined normalized output; the norm theorem consumes explicit envelope premises. Proving those premises for an actual page/kernel remains an implementation obligation.

## Structural and Foundational Lemmas

- `AffineScan`: scalar-decay affine list composition.
- `Softmax`: basic finite softmax positivity and normalization properties.

## Explicitly Outside Current Formal Coverage

- Full page key-box construction and aggregate page-cover correspondence.
- Matrix chunk algorithm, driven-state (6).
- Floating-point rounding/underflow/overflow, quantization, interval kernels.
- Training gradients, learned representation, probabilistic calibration.
- Complexity on real hardware, measured latency/memory, index efficiency.
- Entire-model/logit/trajectory correctness and application safety.
- Scientific novelty, strong-baseline advantage, or industry readiness.

No admitted proof (`sorry`) or undeclared project axiom is permitted. Lean's standard foundational axioms are not omitted from the trust model. "Kernel checked" is scoped to these statements, not a claim that the entire research system is verified.
