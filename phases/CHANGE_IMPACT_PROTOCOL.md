# AURELIS-R adaptive change-impact protocol

This protocol is part of the phase contract. It applies whenever a phase
discovers a failed premise, unsound bound, invalid evaluator, implementation
bug that changes semantics, new theory, revised equation, changed Lean theorem,
changed metric, changed baseline, changed data, or changed hardware assumption.
The example of a Phase 3 repair is not special; the same procedure applies to
every phase.

## Revision identity

Every theory and implementation generation has a monotonically increasing
revision identifier, for example v2.0, v2.1, and v3.0.

- Increment the patch number (v2.1) for a local clarification or a repair
  that preserves all semantic contracts and equations.
- Increment the minor number (v2.0 to v2.1) for a changed equation,
  invariant, theorem premise/conclusion, state transition, certificate,
  evaluator, metric, baseline, or acceptance gate.
- Increment the major number (v2.x to v3.0) when the research object or
  operating contract changes enough that prior results cannot reasonably be
  compared as the same architecture.

The revision identifier is recorded in aurelis.md, the claim registry,
lean/PROOF_COVERAGE.md, every phase report, every checkpoint, every config,
and every result directory. A result is complete only relative to its exact
revision.

## Dependency manifest

Maintain results/v2/REVISION_MANIFEST.yaml (or an equivalent machine-readable
file) with one record per revision:

- revision ID and parent revision;
- changed files/equations/theorems/contracts;
- reason and failure evidence;
- changed assumptions and expected effect;
- earliest affected phase;
- affected descendants;
- artifacts retained as independent;
- artifacts marked stale or superseded;
- required reruns and their status;
- reviewer/sign-off and UTC timestamp.

Each phase report lists its direct inputs and output artifact hashes. A phase may
consume a predecessor only when the predecessor's revision matches and its
status is PASS. An old PASS is never silently promoted into a new revision.

## Classify the change

The agent must classify each repair before implementing it:

| Change kind | Examples | Minimum invalidation |
|---|---|---|
| Semantic theory | new read/update, changed partition, changed target | earliest phase that defines or verifies that semantic contract |
| Mathematical/formal | changed equation, missing premise, invalid proof, new theorem | earliest phase using the changed equation; rerun Lean and numerical descendants |
| Numerical/certificate | unsound interval, rounding, overflow, bound or status bug | certificate phase and every phase using its output |
| Implementation semantics | cache order, rollback, causality, state layout changes output | streaming correctness phase and all descendants |
| Evaluator/metric | hard-coded score, wrong reference, changed SLO or baseline | earliest phase whose gate uses it and all descendants |
| Optimization only | kernel/layout change preserving bit/semantic contract | benchmark phase and descendants only after equivalence |
| Infrastructure | device, dtype, library, dataset, tokenizer, or serving change | earliest phase whose evidence depends on it |
| Editorial only | wording or links with no claim/contract change | no experimental invalidation, but update references |

When uncertain, choose the earlier invalidation point. “The code still runs”
is not evidence that the dependency is unchanged.

## Compute the invalidation closure

Given a change in phase N, inspect the dependency manifest and the equations,
theorems, contracts, metrics, and artifacts consumed by phases 0 through 9.

1. Find the earliest phase E that consumes the changed object. E may be earlier
   than N; a Phase 7 serving discovery can invalidate Phase 2 state semantics.
2. Mark E and every transitive descendant as STALE, even if their old tests
   passed. Mark results from the parent revision SUPERSEDED for comparison
   until reviewed.
3. Preserve old reports, raw logs, checkpoints, plots, and hashes in their
   original revision directory. Never overwrite, delete, or relabel them as
   the new revision.
4. Mark phases before E RETAINED only after checking that their inputs and
   claims are independent of the change. If any input is uncertain, mark them
   stale too.
5. Create a new revision manifest and a rerun queue beginning at E. Do not
   skip a phase merely because the repaired phase itself passes.

The transitive closure is required. If a Phase 3 certificate equation changes,
Phase 3 numerical checks, Phase 4 comparisons, Phase 5 kernels, Phase 6
training/evaluation, Phase 7 serving, Phase 8 scaling, and Phase 9 audit are
stale. If the change also alters the Phase 1 oracle or Phase 2 state semantics,
restart at that earlier phase. If only a Phase 5 kernel layout changes and a
bit-exact equivalence proof survives, Phase 0 through Phase 4 may be retained
and Phase 5 onward rerun.

## Repair and regeneration procedure

The agent must perform these steps in order:

1. Freeze the failure: save command, revision, config, seed, environment,
   device, raw trace, smallest reproduction, and affected claim.
2. State whether the failure is an implementation defect, numerical defect,
   invalid premise, evaluator defect, resource failure, or scientific
   hypothesis failure.
3. Research the mechanism and record primary sources and design consequences.
4. Write the replacement equation, invariant, metric, and a counterexample or
   domain boundary outside it.
5. Update the canonical paper, implementation contract, claims, Lean coverage,
   literature ledger, and revision manifest together.
6. Regenerate every phase prompt affected by the change. Each prompt must name
   the new revision, inputs, gates, and required artifacts; remove inherited
   instructions that refer to retired math or state.
7. Rerun Lean and independent numerical oracles before optimized code.
8. Re-run from the earliest affected phase through every dependent phase.
   Re-preregister any changed held-out metric or budget before new held-out
   results.
9. Re-run inherited gates against the new contract. Keep old failures visible.
10. Publish a new report linking old and new artifacts and explaining which
    conclusions changed.

A repair is not complete when the repaired phase passes. It is complete when
the invalidation closure has either passed on the new revision or terminated in
a documented FAILED_HYPOTHESIS or BLOCKED_RESOURCE result. A negative result
is valid completion; silently retaining downstream PASS files is not.

## Status rules

Allowed statuses are:

- NOT_STARTED: no result for this revision;
- RUNNING: work is in progress;
- PASS: all gates for this revision passed;
- FAILED_HYPOTHESIS: evidence rejects the registered scientific claim;
- BLOCKED_RESOURCE: an external resource prevents the registered work;
- STALE: a dependency changed or a predecessor is no longer valid;
- SUPERSEDED: an older revision result retained for history;
- RETAINED: reviewed as independent and valid for reuse.

STALE and SUPERSEDED are not failures and never count as PASS. A phase can be
marked RETAINED only with a manifest entry naming unchanged inputs. A blocked
resource cannot be converted into PASS by relaxing the gate.

## Preventing partial or inconsistent rewrites

Before any new phase starts, the agent checks:

- current revision manifest and closure queue;
- exact hashes/revisions of all direct inputs;
- no predecessor is STALE, SUPERSEDED, or missing;
- paper equations, Lean statements, implementation contract, and evaluator
  agree on names, shapes, domains, and statuses;
- reports do not mix result directories from different revisions without labels.

After any change, run a consistency scan for retired symbols, equations, phase
names, metric fields, and claims. A stale reference is a phase failure. The
agent must stop downstream work at the first unresolved dependency and repair
the contract or record a blocker.

## Completion definition

The project is properly completed only for a declared revision and deployment
envelope. The final phase must show that every required descendant is PASS,
FAILED_HYPOTHESIS, or BLOCKED_RESOURCE with status honestly reported, and that
no required phase remains STALE or NOT_STARTED. This does not turn a failed
hypothesis into a successful architecture or prove safety outside the tested
envelope.
