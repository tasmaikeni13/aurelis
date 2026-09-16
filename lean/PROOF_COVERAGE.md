# Formal proof coverage — revision 2

Pinned Lean/mathlib 4.19.0. The library root imports both the new specification
and preserved legacy algebra. New names are in Aurelis.V2; reused legacy
names are in Aurelis. All claims are conditional on their theorem premises.

## V2 coverage

| Theorem | Paper claim | Exact scope |
|---|---|---|
| exact_recall_injective | §1.1 exact recall implies injective encoding | Arbitrary address/value/state types, deterministic correct decoder |
| exact_recall_capacity | (1), at least m^n finite states | Finite cardinal inequality; logarithmic bit form is analytic |
| deltaRead_add | (2) is linear under query addition | Evaluation identity for a linear memory map |
| deltaRead_exact_write | Unit-key beta=1 write returns v at k | Inner product k·k=1; does not preserve unrelated keys |
| deltaTransition_energy | (5) rank-one energy identity | Real inner-product space, arbitrary real beta |
| deltaTransition_nonexpansive | (5) contraction condition | beta≥0, beta·norm(k)²≤2; one row, fixed inputs |
| decayed_delta_nonexpansive | Nonnegative alpha multiplies perturbation bound | Row result; matrix Frobenius corollary and driven-state bound (6) are analytic |
| completedRead_balance | (8) normalization identity | Nonzero estimated total mass |
| completedRead_full | Empty-unread endpoint | Algebraic selected normalized sum, not a GPU equality |
| completion_error_identity | (9) numerator and mass-error split | Vectors in real normed spaces; both total masses nonzero |
| residual_certificate | (10) norm consequence | Positive denominator floor, valid residual and mass-error bounds |
| midpoint_error | Midpoint mass interval | Exact reals with lower≤actual≤upper |
| exp_score_interval | Part of (11)–(12) | Scalar exponential monotonicity; no tensor box constructor |
| weighted_residual_bound | Residual envelope | Finite weighted vector sum with nonnegative bounded weights and residual radii |
| completedRead_certificate | (10), tied to the actual completed output | Positive selected mass, nonnegative lower mass, valid upper/lower and residual norm |
| coordinate_product_interval / dot_box_interval | Coordinate key boxes imply score intervals | Exact finite sums; no mutable page/index correspondence |
| page_mass_interval | Page score interval implies exponential mass interval | Exact reals; no floating-point enclosure |
| page_residual_ball | Page value ball and score upper bound imply residual envelope | Exact finite weighted sum; page metadata validity remains external |

The certificate is not proved by assuming its conclusion. The vector identity
is derived from the defined normalized output; the norm theorem then consumes
explicit envelope premises. Proving those premises for an actual page/kernel
remains a separate correspondence obligation.

## Reused deterministic results

handoff_partition, recent_length_le_window, and remote_empty_before_window
prove list partition/size facts, not a mutable ring implementation.
corrected_error_identity, corrected_reproduces_linear, corrected_exact_hit,
and weighted_residual_identity apply to any linear memory, including a delta
state. A one-hot-hit theorem does not say finite softmax chooses one-hot.

## Preserved v1 algebra, not v2 architecture

MatrixState proves positive-(semi)definiteness and invertibility for the old
ridge statistics. Router proves the old scalar quadratic minimizer and clipped
gate under its premises. AffineScan proves scalar-decay affine list composition.
Softmax proves basic finite softmax positivity/normalization. ResidualCorrection
also retains scalar ridge and linear composition identities.

These remain useful mathematical history. They do not establish a v2 Bayesian
posterior, the matrix delta chunk scan, posterior-calibrated uncertainty,
production numerics, or any old empirical result.

## Explicitly outside current formal coverage

- Full page key-box construction and aggregate page-cover correspondence.
- Per-page predictor formula (13), matrix chunk algorithm, driven-state (6).
- Floating-point rounding/underflow/overflow, quantization, interval kernels.
- Training gradients, learned representation, probabilistic calibration.
- Complexity on real hardware, measured latency/memory, index efficiency.
- Entire-model/logit/trajectory correctness and application safety.
- Scientific novelty, strong-baseline advantage, or industry readiness.

No admitted proof or new project axiom is permitted. Lean's standard foundational
axioms are not omitted from the trust model. “Kernel checked” is scoped to
these statements, not a claim that the entire research system is verified.
