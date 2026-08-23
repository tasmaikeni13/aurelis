# Lean 4 formalization

This directory certifies the algebraic core used by Phase 0/1. It is intentionally narrower than the manuscript: empirical claims, complexity claims, statistical model adequacy, and prose-level novelty claims are not converted into theorems merely by compiling this project.

Pinned toolchain: Lean 4.19.0 and mathlib 4.19.0.

```bash
cd lean
lake update
lake build
```

## What a result means

- `lake build` success means Lean's kernel accepted the exact statements in `CSM/` from the listed assumptions.
- A build failure can be a syntax/API/proof problem and is **not by itself** evidence that the mathematical statement is false.
- A faithful statement contradicted by a Lean-checked counterexample, or a discovered missing premise that the manuscript omitted, requires correction in the theory.
- An incomplete formalization is labeled incomplete; it is never reported as a failed theorem.

See [`PROOF_COVERAGE.md`](PROOF_COVERAGE.md) for the theorem-to-manuscript map and scope.

