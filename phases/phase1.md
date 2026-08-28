# Phase 1 — Mathematical oracle, calibration, and pathology gate

Start only after Phase 0 PASS. Read `aurelis.md`, the Phase 0 evidence, and
`phases/AUTONOMY_PROTOCOL.md`. Execute the mandatory failure-repair loop until
all inherited and current gates pass.

This phase attempts to falsify the exact AURELIS equations before learned
features or scale can hide errors.

## Experiments

Build one pinned experiment that sweeps:

- `d_k in {2,4,8,16,32,64}`, at least three `d_v`, and
  `w in {1,2,8,32}`;
- remote load below, at, and above feature rank;
- prior precision over at least eight log-spaced values;
- attention temperature from diffuse to fp64 one-hot saturation;
- homoscedastic and heteroscedastic evidence spanning at least four orders of
  magnitude;
- random, orthogonal, correlated, duplicate, near-collision, rank-deficient,
  and zero/small-norm keys; and
- fp64/fp32/bf16 where supported, always compared on identical quantized
  inputs.

Test and record:

1. delayed-handoff streaming state versus full-history oracle at every prefix;
2. residual error identity, linear reproduction, finite-ridge bound, exact-hit
   endpoint, and general-gate identity;
3. both forms of `g_raw`, clipping cases below zero/interior/above one, dense
   grid optimum, and endpoint non-inferiority;
4. conditional Monte Carlo for `V_R,V_H,K_RH,V(g)` under the exact posterior;
5. deliberately invalid independence routing with the covariance omitted;
6. latent denoising versus episodic-copy target conflict;
7. output/gradient behavior immediately before, at, and after handoff;
8. condition number, solve residual, forward error, Cholesky status, and
   nonfinite behavior; and
9. analytic versus observed state bytes and operation counts.

Use confidence intervals and preregistered tolerances derived from Monte Carlo
standard error and floating-point conditioning. Do not choose tolerances after
viewing the worst row.

## Formal requirements

Audit every theorem used by the experiment against the Lean statement. Extend
Lean for any newly relied-upon deterministic identity, including any repair to
clipping, handoff indexing, finite-ridge scalar reductions, or bounds that can
be faithfully encoded. Preserve analytic-only probability claims as such.

## PASS gates

- Every exact fp64 identity is within its condition-aware tolerance over all
  valid rows; invalid-domain rows remain visible and correctly classified.
- Conditional predicted/empirical MSE and covariance agree within a
  preregistered 99% interval across all matched-model regimes.
- The covariance-omitting router is measurably wrong on a constructed case,
  proving the full formula is actually exercised.
- Clipped routing is never worse than both endpoints beyond Monte Carlo error
  in-model; counterexamples under misspecification are retained.
- Full-residual one-hot recall is exact in fp64; finite-temperature behavior is
  described as convergence, not exactness.
- Handoff has no occurrence loss or duplication, and boundary discontinuity is
  quantified for outputs and gradients.
- Reduced-precision error tracks condition number and never becomes an oracle.
- All Python tests, the complete numerical experiment, all inherited gates,
  and `lake build` pass.
- `results/phase1/PASS.md` satisfies the shared PASS record.
