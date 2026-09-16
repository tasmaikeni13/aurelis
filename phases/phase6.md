# Phase 6 — Trained language-model pilot

Depends on phases 0–5. Implement new training/evaluation scripts; none of the
legacy Phase 6 diagnostic constants are valid input evidence.

## Adaptive start gate

A training or evaluation failure can be an optimizer/data issue, an evaluator
defect, or a scientific hypothesis failure. Classify it before changing the
model. If the repair changes the objective, architecture, tokenizer, data
contract, metric, or certificate semantics, create a new revision and rerun
from its earliest affected phase; do not reuse old checkpoints as if they were
trained under the repaired theory.

## Training design

Use a pilot size appropriate to the registered compute limit (approximately
125M parameters is a planning option). Train modern dense GQA, optimized
recurrent/hybrid, strongest recurrence-free archive completion, and AURELIS-R.
Include bounded and archive variants as separately trained/reported configs.
Use paired seeds, tokenizer, corpus/revision, data order, optimizer, context,
and tuning budgets. Record actual parameter counts rather than target labels.

Implement (14) with sampled dense teacher queries. Report teacher fraction,
extra compute, loss scales, route gradients/stop-gradients, and both equal-token
and equal-cost comparisons. Detect value/projection collapse rather than
mistaking small certificates for preserved model quality.

## Evaluations and gates

Score held-out next-token likelihood and actual generated recall outputs from
saved checkpoints. Include multiple keys, old exceptions, duplicates,
multi-hop and late-disambiguated queries, context extrapolation, and ordinary
short-context tasks. Save predictions, target answers, tokenization, and
checkpoint hashes. Untrained toy formulas cannot stand in for these results.

Report quality against retrieved bytes, certificate/tolerance, wall-clock cost,
memory by tier, and fallback frequency. Apply registered paired noninferiority
and practical-improvement gates; report every seed and confidence interval.
Check bounded-mode loss independently: archive fallback cannot conceal amnesia.

PASS supports only pilot-scale learned viability and a surviving mechanism.
It does not establish broad deployment or large-model scaling. Failed quality,
collapse, or recurrence-free dominance ends the scaling branch.
