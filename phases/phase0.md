# Phase 0 — Reset evidence and register the experiment

Read the v2 paper, AUTONOMY_PROTOCOL.md, IMPLEMENTATION_CONTRACT.md, and
research/V1_AUDIT.md. This is a new experimental generation. Old PASS files
do not authorize skipping a phase. Implement new evaluators/scripts as needed;
the theory revision deliberately left those files untouched.

## Adaptive start gate

Phase 0 establishes the baseline revision and can be invalidated by any later
discovery that changes the research question, evidence classification, or
global resource/metric contract. Before starting, create or update the
revision manifest and mark inherited v1 evidence as historical. If a later
phase reports a changed global assumption, return here, increment the revision,
regenerate the registration, and rerun the affected closure.

## Deliver

Create results/v2/phase0 with an actual environment/device inventory,
implementation gap map, resource budget, and preregistration. Record installed
Lean/mathlib versions without changing their pins. Run the existing formal
build, mapping theorems to their exact statements.

Audit old diagnostic, hardware, memory, and decode claims against their
generating code. Preserve the old artifacts and classify measured, analytical,
hard-coded, unsupported, or not audited. Do not extrapolate this audit to
uninspected experiments.

Register these distinct hypotheses:

- H1: solve-free bounded execution removes v1's solver/all-prefix bottleneck.
- H2: recurrent completion lowers measured cost at fixed certified error
  relative to the best recurrence-free completion.
- H3: archive mode preserves held-out task quality with acceptable memory,
  p99 latency, fallback rate, and throughput.
- H4: trained bounded mode has useful quality at its fixed state budget.

Fix workload/context/batch grids, held-out splits, paired seeds, total compute,
baseline versions, reference encoding, epsilon grid, and SLO. Set numerical
and quality noninferiority margins and practical speed/memory improvement
margins before viewing new results. Suggested pilot defaults: 3 paired seeds,
quality margin 1% relative validation NLL and 2 percentage points on registered
recall tasks, and ≥10% measured cost improvement for H2 with a paired 95%
interval excluding zero. These are research decisions, not observed results;
replace them before execution if the target application needs different limits.

## Gates

Every claimed device is returned by runtime enumeration. Every metric type has
a raw-data provenance plan. No legacy claim is inherited by v2. Compute limits
and stop conditions are explicit. PASS means the audit and registration are
complete, not that any architecture hypothesis passed.
