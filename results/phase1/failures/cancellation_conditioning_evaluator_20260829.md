# Phase 1 failed iteration: cancellation-blind evaluator

- Command/config/seed: pinned Phase 1 experiment, unchanged config, seed
  `20260829`.
- Frozen evidence: `results/phase1/metrics.json` from the failed run and the
  64 raw deterministic rows. Four rows failed; the other suites passed.
- Classification: evaluator / floating-point conditioning, not theorem.

## Mechanism

For zero and `1e-12`-norm keys with tiny priors, `P` itself has condition
number approximately one, but the covariance form of the router subtracts
terms of size up to `1e10` to recover a denominator or numerator many orders
smaller. Matrix condition number alone therefore does not describe the scalar
cancellation. The reduced posterior-metric form remains stable. Dense-grid
variance evaluation has the same large-term cancellation, producing absolute
roundoff around `1e-6` at variance scale `1e10`.

The official LAPACK Users' Guide, "Accuracy and Stability / How to Measure
Errors" (`https://netlib.org/lapack/lug/node75.html`, accessed 2026-08-29),
distinguishes absolute from relative scalar error and motivates scaling an
error certificate to the magnitude of the computed quantity. Goldberg,
"What Every Computer Scientist Should Know About Floating-Point Arithmetic,"
ACM Computing Surveys 23(1), DOI `10.1145/103162.103163`, supplies the standard
rounding/cancellation model.

## Repair

Retain the preregistered epsilon multiplier and admissible `kappa*epsilon`
threshold, but apply them to the actual scalar reduction condition number:
the sum of input magnitudes divided by the magnitude of the reduced result.
Compose that scalar reduction condition with the condition number of the
linear solve feeding it; the first repair iteration used only the scalar term
and therefore correctly failed on rank-deficient/near-duplicate solves. Classify
cancellation- or solve-unresolved reductions as invalid-domain rows and retain
them. Scale variance-comparison absolute tolerance to the endpoint variance
magnitude. No tolerance constant, equation, data, seed, or output is changed.
