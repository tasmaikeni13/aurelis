# AURELIS Research Phases

These documents specify the implementation and empirical evaluation of the [AURELIS specification](../aurelis.md).

Read [AUTONOMY_PROTOCOL.md](AUTONOMY_PROTOCOL.md) and [IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md) with each phase. Before starting or resuming any phase, also consult [CHANGE_IMPACT_PROTOCOL.md](CHANGE_IMPACT_PROTOCOL.md).

| Phase | Deliverable | Status |
|---|---|---|
| [0](phase0.md) | Runtime inventory, formal verification, module mapping, and preregistration | **PASS** |
| [1](phase1.md) | Independent math oracles and formal correspondence | **PASS** |
| [2](phase2.md) | Streaming state, bounded read, exact archive reference | **PASS** |
| [3](phase3.md) | Validated certificate, refinement, failure semantics | **PASS** |
| [4](phase4.md) | Falsification and novelty-critical ablations | Ready for execution |
| [5](phase5.md) | Chunk training and accelerator kernels | Pending |
| [6](phase6.md) | Trained LM pilot with actual held-out measurements | Pending |
| [7](phase7.md) | Serving integration and failure injection | Pending |
| [8](phase8.md) | Matched-budget scale and deployment decision | Pending |
| [9](phase9.md) | Independent reproduction and publication audit | Pending |

Phases execute in order. A disproved research hypothesis is a valid negative result and a stop condition for scaling. Deliverables are recorded in `results/phaseN` and `plots/phaseN`.
