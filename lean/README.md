# AURELIS Lean 4 formalization

This project kernel-checks the deterministic algebra used by the AURELIS
paper: delayed handoff, affine scan composition, positive-definite remote
precision, residual correction, softmax barycenters, and the closed-form
uncertainty router.

Pinned versions: Lean 4.19.0 and mathlib 4.19.0.

```bash
cd lean
lake update
lake build
```

A successful build proves the statements in `Aurelis/` from their explicit
assumptions. It does not formalize the Gaussian probability model, finite
precision of a particular solver, learned-feature adequacy, runtime, or
language-model quality. The exact boundary is recorded in
[`PROOF_COVERAGE.md`](PROOF_COVERAGE.md).
