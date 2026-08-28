# AURELIS gated research plan

The architecture and present evidence are frozen in `aurelis.md`. Every phase
uses `phases/AUTONOMY_PROTOCOL.md`: on failure it preserves the counterexample,
researches primary sources, derives a mathematical repair, updates Lean where
faithful, implements an independent oracle and production fix, tests all prior
gates, and iterates. A threshold cannot be weakened to manufacture PASS.

## Phase graph

```text
theory + numerical analysis + Lean (current revision)
                         |
                         v
P0 identity migration, reference implementation, ROCm substrate
                         |
                         v
P1 mathematical oracle, calibration, pathologies
                         |
                         v
P2 matched mechanism separation and baselines
                         |
                         v
P3 learned features and episodic routing
                         |
                         v
P4 nonstationarity, multihop, capacity limits
                         |
                         v
P5 MI300X/ROCm optimization and systems gate
                         |
                         v
P6 tiny language-model viability
                         |
                         v
P7 matched multi-seed scaling study
                         |
                         v
P8 independent reproduction, paper, and release audit
```

There are exactly nine numbered phases (`0`–`8`). NLP is forbidden until the
synthetic, learned, formal, and MI300X systems prerequisites through Phase 5
are passing.

## Current completed foundation

- Literature search through 2026-08-28 with explicit closest-work boundary.
- Standalone architecture, deterministic/probabilistic analysis, complexity,
  limitations, algorithms, and falsifiable predictions.
- Reproducible fp64 numerical program with raw artifacts and alternate-seed
  checks.
- Lean 4 proof project for core deterministic algebra and router optimality.
- Autonomous failure-repair protocol and nine phase prompts.

These are foundation deliverables, not Phase 0 PASS. Implementation and all
trained-model claims remain pending.

## Release invariants

- The defining read remains `Mq+g(vbar-Mkbar)` with delayed disjoint handoff.
- Bayes denoising and episodic copy remain separately evaluated.
- Fixed-state arbitrary-recall failures are preserved.
- Every number traces to raw data/config/commit/environment.
- Every formal claim traces to an exact Lean statement and coverage boundary.
- Equal-budget comparisons report parameters, tokens, optimizer, batch tokens,
  context, state bytes, FLOPs, VRAM, throughput, and latency separately.
- The final paper remains standalone and never presents migration history as a
  scientific contribution.
