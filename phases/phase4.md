# Phase 4 — Falsification and novelty-critical comparisons

Depends on phases 0–3. This phase decides whether the recurrent-completion
idea merits accelerator/training investment.

## Adaptive start gate

This phase tests whether the current mechanism earns its cost. If a comparator,
metric, certificate, or evaluator is found invalid, mark the earliest affected
phase and this phase STALE; preserve old comparisons under their parent
revision. Re-preregister changed gates and rerun the complete comparison
matrix after the repair.

## Required comparators

Use identical Q/K/V, archive pages, encoding, epsilon, and accounting for:
local-value completion; zero completion; summary-based global completion;
per-page center completion (13); recurrence with simple additive/gated fusion;
sparse attention with no completion; value-blind mass selection; and AURELIS-R.
Retain full softmax and historical ridge as references. Compare at equal total
state/index budgets as well as equal fetched-KV budgets.

## Experiments

Sweep known structured linear relations mixed with rare exceptions; random
incompressible associations; diffuse attention; large value outliers; repeated
keys with different values; near-collisions; delayed disambiguation; abrupt
drift; multi-hop queries; and retrieval at window/page boundaries.
Include contexts where certificate metadata costs more than dense reads.

Ablate delayed versus immediate writes, transport versus local values,
recurrence dimension, mass correction, residual-sensitive selection, and
summary quality. Charge index build/write, predictor compute, summary scans,
sorting, and transfers. Error≤bound alone is not a useful research outcome.

## Decision

Evaluate H2 at the preregistered tolerance/quality budgets. If recurrence does
not beat the strongest cheap completion, record FAILED_HYPOTHESIS and retire
the claim that it makes certified retrieval cheaper. Do not scale merely because
AURELIS beats a weak zero-predictor baseline. A bounded-mode result may be
reported separately, with its own H4 outcome.

PASS requires a distinct surviving mechanism with paired uncertainty estimates,
reproducible counterexamples outside its useful regime, and an updated novelty
comparison against ResKV, Quest, certified quantized attention, and hybrid memory.
Synthetic results justify further experiments, not industry readiness.
