# Phase 7 — Matched multi-seed scaling study

Start only after Phase 6 PASS. Read all prior artifacts and
`phases/AUTONOMY_PROTOCOL.md`. Execute the failure-repair loop until PASS.

This phase tests whether the tiny-model mechanism persists under a meaningful
but one-MI300X-feasible scaling budget. It may not redefine success around
constant state alone.

## Frozen design

Preregister one generation with:

- strongest Transformer, published-style hybrid, Gated DeltaNet or Kimi-style
  feasible comparator, cumulative least-squares comparator, and strongest
  AURELIS variant selected without Phase 7 test results;
- `25M–75M` parameters, at least three paired seeds, and at least `100M`
  training tokens per model/seed unless a larger equal budget is feasible;
- identical corpus, tokenizer, optimizer, schedule, batch tokens, context,
  precision policy, checkpoint cadence, and evaluator;
- parameter/FLOP/state reconciliation; and
- fixed primary and secondary claims with confidence intervals and multiple
  comparison treatment.

Do not omit a comparator because it was strong in Phase 6. Any post-test
architecture change creates a new generation and reruns every model/seed.

## Evaluation

Use the Phase 6 diagnostic suite plus natural validation at several contexts,
long-document subsets, downstream zero/few-shot probes appropriate to scale,
and generation-time prefill/decode sweeps. Evaluate the same checkpoint at
trained and extrapolated contexts. Report per-seed and aggregate results,
throughput, peak VRAM, wall time, energy/power if reliable, checkpoint size,
live state, and latency distributions.

## Failure repair

Diagnose divergence, quality loss, or systems regressions from preserved
traces. Research scaling, optimization, recurrence, attention, and numerical
solver literature before changing the design. Mathematical repairs must update
the paper, fp64 oracle, Lean coverage, and all prior experiments. Hyperparameter
tuning must be symmetric across models or separately budgeted and reported.

## PASS gates

- All preregistered models and all paired seeds complete the equal token budget
  or the phase remains failed.
- AURELIS validation loss is non-inferior within the preregistered margin to
  the strongest efficient hybrid, with confidence intervals across seeds.
- AURELIS retains a statistically supported targeted-memory advantage and a
  context-independent remote-plus-window decode-state advantage.
- At least one end-to-end quality-qualified throughput/latency Pareto advantage
  survives at long context; theoretical bytes alone are insufficient.
- No primary claim depends on one seed, one context, test-driven exclusions,
  or different data/optimizer budgets.
- Negative natural/downstream results and all resource costs are retained.
- All inherited gates and Lean build pass, and
  `results/phase7/PASS.md` satisfies the shared PASS record.
