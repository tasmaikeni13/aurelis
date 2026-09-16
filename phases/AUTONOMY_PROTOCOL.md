# AURELIS-R v2 research and implementation protocol

This protocol supersedes v1's requirement to preserve its ridge solver and
Bayes router. The user's authorized redesign explicitly changes those choices.
This file instructs future phase implementation; it does not authorize running
large training jobs during a theory/documentation revision.

## Authority and source of truth

Use aurelis.md v2, IMPLEMENTATION_CONTRACT.md, CHANGE_IMPACT_PROTOCOL.md, and
the numbered phase together.
Resolve equation inconsistencies before implementation. Preserve historical
code/results. New executable work implements the new theory; it cannot inherit
old PASS files, trained quality claims, or TPU claims.

The project is a hypothesis, not a requirement that experiments must succeed.
Status is one of NOT_STARTED, RUNNING, PASS, FAILED_HYPOTHESIS, or BLOCKED_RESOURCE.
A Lean build supports only its theorem statements. It is not a model benchmark.

## Adaptive dependency rule

Every phase result is valid only for its recorded theory revision. Before
starting work, read [CHANGE_IMPACT_PROTOCOL.md](CHANGE_IMPACT_PROTOCOL.md) and
the current `results/v2/REVISION_MANIFEST.yaml`. If any input, equation,
theorem premise, state transition, certificate, evaluator, metric, baseline,
data revision, dtype, device, or service assumption changed, compute the
invalidation closure before doing new work. Mark the earliest affected phase
and every dependent descendant `STALE`; mark old results `SUPERSEDED` for
comparison; preserve their raw artifacts; increment the revision; and rerun
from the earliest affected phase. A repaired phase passing does not make stale
descendants valid. Earlier phases may be `RETAINED` only after their direct
inputs and claims are checked to be independent.

## Required workflow

1. Record the tested theory revision, source commit/dirty patch, config, seeds,
   dataset revisions, environment, actual devices, and raw commands.
2. Register the hypothesis, primary metric, baselines, tolerances, acceptance
   margins, and compute limits before evaluation. Preserve the registration.
3. Implement an independent CPU reference and production path when the phase
   requires them. Two wrappers around one helper are not independent oracles.
4. Preserve failures with minimal reproductions. Distinguish implementation,
   numerical, specification, hardware, evaluation, and scientific failures.
5. Repair bugs and rerun affected checks. For a false claim, correct the theory,
   formal scope, registry, phase prompts, and revision manifest, then version
   the experimental generation and invalidate the full dependent closure.
6. Never lower a threshold or replace a baseline to make a disappointing run pass.
   A hypothesis failure ends that scaling branch. Resource failure remains blocked.
7. Publish all seeds, exclusions, nonfinite jobs, confidence intervals, and raw
   measurements. Stop at the declared budget; more compute needs existing user
   authorization or a concrete request, not an endless retry rule.

When a change is discovered after later phases have run, do not edit their
reports in place. Preserve them under the parent revision, create the new
revision manifest, and regenerate affected prompts and reports. The required
endpoint is PASS, FAILED_HYPOTHESIS, or BLOCKED_RESOURCE for every descendant;
STALE or NOT_STARTED is incomplete.

## Evidence rules

No hand-assigned accuracy/MSE, randomized proxy scores, invented timing, or
hard-coded hardware inventory. Estimates are labeled estimates and never
reported as measurements. Record outputs/targets/checkpoint IDs for task scores.
Measure actual populated caches and synchronize the backend used for timing.
Report warmup/compile separately, and count index construction, transfer,
fallback, padding, and host storage in end-to-end costs.

Use strong optimized baselines and comparable parameter, token, optimizer,
context, tuning, KV head, precision, and total-memory budgets. Report both
equal-token and equal-training-cost comparisons where extra teacher work exists.
Test data cannot tune the architecture within the same generation.

## Formal and numerical rules

Run the pinned Lean/mathlib build; no admitted goals or new project axioms.
Standard Lean foundations are allowed and must not be advertised as “no
assumptions.” PROOF_COVERAGE.md must identify analytic, formal, implementation,
floating-point, and empirical boundaries.

Certification requires a valid disjoint cover, conservative envelopes, and
accounted arithmetic error. Probabilistic confidence, a learned sigmoid, or
empirical fp64 agreement is not a deterministic certificate. Budget exhaustion
and invalid data return explicit statuses. Do not silently downgrade strict mode.

## Required phase record

Write results/v2/phaseN/report.md with equation-to-code mapping, raw evidence
paths, hypotheses and outcomes, reproduction commands, failures/dispositions,
resource usage, and next decision. Generate PASS.md only when every registered
gate passes. Otherwise generate FAILED_HYPOTHESIS.md or BLOCKED_RESOURCE.md.
Include artifact hashes and tested revision. Earlier v2 correctness gates
remain dependencies, but old v1 empirical gates are retired.
