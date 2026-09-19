# AURELIS Formal Core

Pinned Lean 4.19.0 and mathlib v4.19.0.

To build the machine-checked proofs:

```bash
cd lean
lake build
```

`Aurelis.lean` imports the formal modules specifying finite-state recall capacity, solve-free delta memory perturbation stability, residual completion with uncertain normalizer, page envelope bounds, and causal handoff partitioning.

Real-valued definitions may be noncomputable in Lean. These are machine-checked mathematical specifications, not executable accelerator kernels. Standard Lean foundations remain; there are no project axioms or admitted proofs (`sorry`). See [PROOF_COVERAGE.md](PROOF_COVERAGE.md) for exact theorem premises and scope boundaries.

The formal verification covers algebraic and analytic properties under stated premises. It does not by itself establish hardware kernel speeds, trained sequence quality, novelty, or serving latency.
