# Phase 8 — Scale and make a deployment decision

Depends on phases 0–7 and sufficient existing compute authorization. Scale is
conditional on evidence, not an obligation to spend until a positive result.

## Adaptive start gate

Scale only the current revision whose Phase 0–7 dependencies are PASS or
explicitly RETAINED. A scale run that changes dtype, hardware, data, serving
assumptions, or acceptance SLO creates a new infrastructure/evaluation
revision and invalidates its dependent closure. Do not merge results across
revisions without a manifest entry.

## Deliver

Choose a larger model/context regime from the phase 0 budget (350M is a
possible intermediate, not proof of industry scale). Repeat matched baselines,
paired seeds, independent held-out evaluations, and all serving contracts.
Select hardware and datasets from actually available resources. Record total
training and teacher FLOPs, wall time, energy/cost only when measured,
checkpointing overhead, archive storage and network limits.

Evaluate both long-prefill and long-generation workloads, many concurrent
sessions, multi-turn prefix reuse, speculative branching, and archive contention.
Disclose the useful context/concurrency region and the crossover where recurrence,
metadata scans, transfer, or fallback erase gains. Pilot figures cannot be
extrapolated to a larger model without measurements.

## Gates and outcome

Use the preregistered quality, SLO, cost, and total-memory criteria against the
strongest comparator. Require a reproducible practical Pareto gain and no
certificate/status violation. Evaluate sensitivity to page sizes, precision,
tolerance and workload shift on a separate validation split.

Produce a decision: qualified for the explicitly tested deployment envelope;
research-only with surviving but unscaled hypothesis; or FAILED_HYPOTHESIS.
Do not claim general industry readiness from one accelerator, one needle test,
a memory formula, or a single seed. A narrower qualified envelope is acceptable
only when explicitly reported and supported by its original gates.
