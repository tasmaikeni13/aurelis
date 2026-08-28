# AURELIS numerical analysis

Generated deterministically by `analysis/aurelis_numerical.py` with seed
`20260828`.  These experiments test the mathematical mechanism; they
are not language-model or accelerator benchmarks.

## Gates

| Check | Result | Gate |
|---|---:|---:|
| Residual decomposition maximum absolute error | 9.992e-16 | < 1e-10 |
| Gate formula equivalence maximum absolute error | 3.331e-16 | < 1e-10 |
| Routed variance regret against dense grid | 0.000e+00 | < 1e-8 |
| Routed variance non-inferiority slack | 0.000e+00 | <= 1e-12 |
| Linear reproduction error | 2.285e-16 | < 1e-12 |
| Hard cached-hit residual error | 0.000e+00 | < 1e-12 |
| Remote uncertainty relative calibration error | 0.128% | < 3% |
| Full-residual uncertainty relative calibration error | 0.462% | < 3% |
| Routed uncertainty relative calibration error | 0.293% | < 3% |

## Principal observations

Arbitrary softmax weights incur L2 error
`0.768142` on an exactly linear map,
whereas first-moment residual correction reduces the error to
`2.285e-16`.  With a one-hot cached hit, the
same correction returns the exceptional stored value with error
`0.000e+00`.

In the conditional Gaussian experiment (50000 trials),
the analytic router used gate `0.513628`.  Its predicted and
empirical per-coordinate MSE were
`0.143750` and
`0.143329`.  The empirical improvement over
the better endpoint was `0.052914`.

Across the finite-sample sweep, routed MSE moved from
`0.278790` at 8 remote writes to
`0.003722` at 256 remote writes.
The full tables are stored beside this report; conditioning failures are
retained rather than filtered.

## Scope

The calibration certificate assumes a linear-Gaussian latent operator,
disjoint remote/local observations, fixed attention weights conditional on
keys, and exact floating-point solves up to measured numerical error.  It is
not a claim that learned features satisfy those assumptions, nor that the
Bayes gate is optimal for exact episodic-copy targets.  The hard-hit result is
a separate deterministic theorem with gate one.
