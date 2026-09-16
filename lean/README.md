# AURELIS formal core

Pinned Lean 4.19.0 and mathlib v4.19.0. With dependencies available:

    cd lean
    lake build

Aurelis.lean imports the v2 modules Capacity, DeltaMemory, and CertifiedRead,
as well as the preserved v1 modules. The new proofs establish finite-state
recall capacity, the known delta update's row perturbation energy, and a vector
completion error bound with uncertain normalizer. The handoff and transport
algebra is reused independently of any ridge solver.

Real-valued definitions may be noncomputable in Lean. These are machine-checked
mathematical specifications, not executable accelerator kernels. Standard Lean
foundations remain; there are no project axioms or admitted proofs. See
[PROOF_COVERAGE.md](PROOF_COVERAGE.md) for premises and exclusions.

The build does not prove sound floating-point intervals, chunk-kernel equivalence,
trained quality, novelty, runtime, serving correctness, or answer safety.
Do not revive v1 empirical claims because legacy theorems still compile.
