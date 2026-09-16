# AURELIS-R

Solve-free recurrent memory with residual-certified retrieval.

This repository is being revised from the Bayesian ridge architecture (v1) to
the research specification in [aurelis.md](aurelis.md). The new design combines
a gated delta state, local residual transport, and an optional exact archive
whose reads are controlled by a derived error bound.

**Current status: theory and scoped Lean proofs; v2 implementation and
performance validation are pending.** Existing Python/JAX code, scripts,
configs, and results are v1. They were deliberately left unchanged in this
revision. You must implement the new theory following
[the implementation handoff](phases/IMPLEMENTATION_CONTRACT.md) and
[phases 0–9](phases/README.md).
The [adaptive change-impact protocol](phases/CHANGE_IMPACT_PROTOCOL.md) governs
theory repairs: it invalidates and regenerates the full dependent phase closure.

Two operating contracts are explicit:

- Bounded mode: fixed recurrent state and recent cache; approximate remote memory.
- Archive mode: growing exact storage, adaptive retrieval, and a local error
  certificate or explicit budget failure. Full reads recover softmax on the
  same current Q/K/V, subject to numerical error.

This is not a claim of bounded-memory unlimited exact recall, established
novelty, model-level safety, or demonstrated deployment speed. The research
must show recurrence earns its cost against strong sparse/hybrid baselines.

## Read first

- [Paper and equations](aurelis.md)
- [Primary-source literature and novelty boundary](research/LITERATURE_REVIEW.md)
- [V1 bottleneck and evidence audit](research/V1_AUDIT.md)
- [Claim registry](CLAIMS.md)
- [Research plan](RESEARCH_PLAN.md)
- [Revision manifest](results/v2/REVISION_MANIFEST.yaml)
- [Formal proof scope](lean/PROOF_COVERAGE.md)

The old Phase 6 evaluator contains assigned diagnostic scores and simulated
decode measurements; its PASS files do not establish model quality or deployment.
Historical artifacts remain available but are not current publication evidence.

## Formal verification

Pinned Lean/mathlib: 4.19.0. With the existing dependencies installed:

    cd lean
    lake build

New proofs cover finite-state recall capacity, delta transition stability, and
the vector residual/normalizer certificate. Formal real arithmetic does not
prove a floating-point kernel, speed, learned quality, or end-to-end safety.
See [lean/README.md](lean/README.md) for the exact boundary.
