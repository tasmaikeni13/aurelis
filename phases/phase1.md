# Phase 1 — Independent mathematics and formal correspondence

Depends on phase 0. Read the common protocol and implementation contract.
Implement new v2 CPU oracles and tests; current v1 scripts are insufficient.

## Adaptive start gate

Read the current revision manifest and change-impact closure before using any
equation. If a later repair changes the state, read, target, certificate, or a
Lean premise used here, this phase is STALE even if its old oracle passed.
Update the independent oracle and theorem ledger for the new revision, then
rerun Phase 1 before any streaming or benchmark descendant.

## Deliver

Implement independent history and streaming fp64 calculations for equations
(2)–(12). Keep a tiny explicit full softmax reference. Use direct scalar/array
formulas in one oracle and tensor operations in the other; avoid shared helpers
that make agreement circular.

Check occurrence handoff; delta update and unit-key full write; perturbation
energy/nonexpansion; transport error and conditional exact hit; selected/unread
partition; normalized completion; both terms of the error identity; page mass
and value envelopes; midpoint error; certificate; and all-pages recovery.
Cover vector dimensions >1. Include paper (13)'s per-page comparator.

Run Lean and inspect theorem assumptions. Extend formal statements for page
construction, matrix chunk algebra, and/or grouped completion when practical.
Document missing proofs faithfully rather than citing a scalar surrogate for
a matrix claim. A finite-real proof is not a floating-point implementation.

## Pathologies and gates

Use empty remote sets, singleton/partial pages, negative query coordinates,
zero keys, beta endpoints, alpha=0/1, repeated keys, huge value outliers,
uniform scores, concentrated scores, and stale-state residual examples.
Moderate fp64 algebra tolerance begins at atol=1e-10, rtol=1e-9. Record the
actual error and value scale for every family.

Construct a counterexample showing that dropping the normalizer term can
underestimate error. Demonstrate that bounded state cannot meet arbitrary
exact recall past capacity and that finite softmax is not a hard lookup.
Do not require a false property such as monotonic error reduction on each fetch.

PASS requires all deterministic reference contracts and the Lean build, no
admitted goals/custom axioms, and a theorem-to-equation ledger. Numerical
certificate checks are diagnostics only; phase 3 establishes runtime bounds.
