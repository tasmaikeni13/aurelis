# AURELIS Research Plan

The central question is whether a solve-free recurrent predictor can reduce the cost of retrieving exact history under an explicit local attention-error budget.

The research execution follows [the implementation contract](phases/IMPLEMENTATION_CONTRACT.md), [the autonomy protocol](phases/AUTONOMY_PROTOCOL.md), and [the phase index](phases/README.md).

## Execution Sequence

1. **Phase 0:** Environment inventory, machine-checked formal build, architecture specification mapping, preregistration, and resource budget.
2. **Phase 1:** Independent float64 mathematical reference oracles and formal correspondence.
3. **Phase 2:** Controlled baselines and mechanism isolation.
4. **Phase 3:** Learned feature projections, window kernels, and chunk scans.
5. **Phase 4:** Nonstationarity, multi-hop pointer chasing, and capacity limits.
6. **Phase 5:** Accelerated kernel implementation and hardware verification.
7. **Phase 6:** Language model viability, benchmarks, and scaling.
8. **Phase 7:** Long-context stress tests and serving integration.
9. **Phase 8:** Deployment qualification and ablation audits.
10. **Phase 9:** Publication synthesis and external reproducibility.

## Rejection and Stop Criteria

The primary rejection criterion is recurrence-free dominance at matched quality, error, and total service cost. Additional stop conditions include loose certificates forcing frequent full reads, index/transfer overhead erasing retrieval gains, degenerate trained quality, invalid numerical certification, and unacceptable tail latency.

A bounded-state approximate model and an archive-backed locally certified model represent distinct operational contracts.
