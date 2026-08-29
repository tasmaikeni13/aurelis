# Phase 1 formal-requirement audit

This audit compares every deterministic theorem used by the Phase 1
experiment with its actual Lean statement. Probability, floating-point error,
complexity, and implementation behavior are not promoted to Lean proofs.

| Experiment claim | Lean statement | Faithfulness / boundary |
|---|---|---|
| Delayed remote/recent occurrence partition | `handoff_partition`, `recent_length_le_window` in `Aurelis/Handoff.lean` | Full list identity and length bound; ring-buffer occurrence IDs remain Python evidence. |
| Full-residual error identity and linear reproduction | `corrected_error_identity`, `corrected_reproduces_linear`, `weighted_residual_identity` in `Aurelis/ResidualCorrection.lean` | Generic real-module/linear-map identities; numerical operator norms are empirical. |
| General-gate identity (5.3), including gates outside `[0,1]` | `gated_error_identity`, `gatedRead_one` in `Aurelis/ResidualCorrection.lean` | Added in Phase 1; no clipping or probability premise is hidden. |
| Exact one-hot cached endpoint | `corrected_exact_hit` | Exact linear-map theorem. Finite-temperature convergence and fp64 saturation are explicitly not cited as this theorem. |
| Finite-ridge reduction and bound | `scalar_ridge_slope_error`, `scalar_ridge_residual_bound` | Added in Phase 1 as a faithful scalar specialization. The paper's matrix spectral-norm bound remains analytic and is swept numerically. |
| Routing quadratic, both raw-gate reductions, clipping, and endpoint non-inferiority | `routeVariance_completion`, `posterior_denominator_identity`, `posterior_numerator_identity`, `clippedGate_optimal`, `rawGate_le_remote`, `rawGate_le_residual` in `Aurelis/Router.lean` | Full real scalar algebra with positive-denominator and convex-gate premises. Conditional covariance is not derived in Lean. |
| Positive-definite regularized state | `precision_update_posSemidef`, `regularized_precision_posDef`, `regularized_precision_isUnit` in `Aurelis/MatrixState.lean` | Finite real matrices. Cholesky success, conditioning, and nonfinite inputs remain numerical evidence. |
| Affine statistic construction used by inherited tests | `Affine.combine_assoc`, `Affine.aggregate_correct`, `statisticsElement_action` | Exact generic scan algebra; no parallel-kernel or operation-count theorem. |

The experiment's conditional Monte Carlo uses the analytic matrix-normal
posterior from the manuscript. The probability derivation, 99% intervals,
Cholesky forward/backward error, dtype behavior, state bytes, profiler counts,
and handoff gradients remain correctly classified as analytic or empirical in
`lean/PROOF_COVERAGE.md`.
