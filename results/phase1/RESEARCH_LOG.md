# Phase 1 research and repair log

## Floating-point cancellation in covariance reductions

- LAPACK Users' Guide, "Accuracy and Stability / How to Measure Errors",
  `https://netlib.org/lapack/lug/node75.html`, accessed 2026-08-29. Design
  decision: retain absolute errors but scale the floating-point allowance to
  operand/result magnitude; do not interpret a well-conditioned matrix solve
  as proof that a subsequent subtractive scalar reduction is well-conditioned.
- David Goldberg, "What Every Computer Scientist Should Know About
  Floating-Point Arithmetic," *ACM Computing Surveys* 23(1), 1991,
  DOI `10.1145/103162.103163`, accessed 2026-08-29. Design decision: explicitly
  classify catastrophic-cancellation domains and compare the stable reduced
  gate formula with the covariance form only when the latter is resolvable at
  fp64 precision.

The failed rows and exact evaluator repair are preserved in
`results/phase1/failures/cancellation_conditioning_evaluator_20260829.md`.

