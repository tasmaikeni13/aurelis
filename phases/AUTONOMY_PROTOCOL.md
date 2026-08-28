# AURELIS autonomous phase protocol

Every numbered phase must obey this protocol. A phase prompt plus this file is
the complete operating instruction; later phases inherit all earlier gates.

## Non-negotiable objective

Build and falsify the AURELIS architecture defined in `aurelis.md`. Do not
replace it with an easier generic local-attention/recurrent hybrid. The defining
invariants are delayed occurrence-level handoff, disjoint recent and remote
stores, the read `Mq + g(vbar-Mkbar)`, a cross-covariance-aware Bayes gate, and
an explicit episodic override.

## Evidence hierarchy

Use current files, raw outputs, checkpoints, and hardware measurements as
authoritative. Treat prose, expected behavior, and prior success reports as
hypotheses until reproduced. Prefer primary sources: papers, official
documentation, source code, and standards. Record URLs, versions, access date,
assumptions, and the exact design decision each source supports.

## Mandatory failure-repair loop

Run this loop whenever any assertion, theorem, numerical tolerance, training
criterion, performance threshold, or inherited gate fails:

1. **Freeze the evidence.** Save the failing config, seed, command, environment,
   raw trace, metrics, and smallest reproducible case. Never overwrite or hide
   a failure row.
2. **Classify the failure.** Choose and justify one or more of: implementation,
   numerical conditioning, hardware/kernel, optimizer/data, representation,
   statistical-model misspecification, theorem/assumption, evaluator, or
   external resource.
3. **Research before patching.** Search primary literature and official docs
   for the failure mechanism and viable repairs. Add the sources and extracted
   mathematics to the phase research log. Do not tune blindly.
4. **Derive a repair.** State equations, invariants, predicted effect, valid
   domain, and a counterexample outside that domain. If the original claim is
   false, correct the paper and claim registry; never silently weaken a theorem
   or alter a metric.
5. **Formalize the repair.** Add or update a faithful Lean statement for every
   new deterministic algebraic claim that is realistically formalizable. Run
   `lake build`. A proof-script/API failure is not a theorem refutation. A
   faithful counterexample or missing premise is.
6. **Implement twice where feasible.** First update an independent fp64/CPU
   oracle, then the production or optimized path. Do not make two paths call the
   same helper in a way that creates false agreement.
7. **Test the mechanism.** Add a regression test that fails before the repair,
   property/pathology tests, and the phase experiment. Compare predicted and
   observed effects, not just the final aggregate score.
8. **Run all inherited gates.** A repair that passes the current test but breaks
   an earlier phase is rejected.
9. **Iterate.** Repeat from step 1 until every phase gate passes. Do not cap
   repair attempts merely for convenience.

A gate may change only when preserved evidence proves that its scientific
claim is false or its evaluator invalid. Version the gate, retain the failed
claim/result, update `aurelis.md`, `CLAIMS.md`, and Lean coverage, and replace it
with a stricter faithful test of the corrected claim. Never lower a threshold
because a run is slow, expensive, or disappointing. If an actual external
blocker prevents further in-scope work, report it and leave the phase failed;
do not manufacture PASS.

## Reproducibility contract

- Pin seeds, dependencies, data revisions/checksums, and configs.
- Record commit, dirty state, command, UTC time, wall time, device, dtype, peak
  memory, and exclusions with every run.
- Use fp64 as the numerical oracle. Reduced precision is always measured
  against it.
- Synchronize GPU timing and separate compile/warm-up/tuning time from steady
  state.
- Report all configured seeds, including nonfinite and failed jobs.
- Preserve equal-budget comparisons: parameters, tokens, optimizer, schedule,
  batch tokens, context, evaluation, and tuning opportunity.
- Never use a test set to redesign an architecture within the same experimental
  generation. Increment the generation and rerun all models.

## Formal-method contract

`lake build` must pass with pinned Lean/mathlib and no `sorry`, `admit`, or
unreviewed project axioms. `lean/PROOF_COVERAGE.md` must say exactly what each
theorem proves and what remains analytic or empirical. Do not encode a weaker
statement and cite it as proof of stronger prose.

## PASS record

Each phase ends with a generated `results/phaseN/PASS.md` containing:

- every gate and direct evidence path;
- exact reproduction commands;
- all failed iterations and their disposition;
- research and mathematical repairs made;
- Lean theorem/coverage changes;
- raw/aggregate metric paths and plots;
- tested commit and environment fingerprint; and
- remaining limitations that are not part of the phase claim.

Only then may the next phase start.
