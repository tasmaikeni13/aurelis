# AURELIS-R v2 phase index

These prompts specify future implementation of [the v2 paper](../aurelis.md).
The current revision changes theory, Lean, and instructions only. **You must
write the Python/JAX/kernel, evaluator, and serving implementations accordingly.**
Existing scripts/configs/results are v1 and do not implement these contracts.

Read [AUTONOMY_PROTOCOL.md](AUTONOMY_PROTOCOL.md) and
[IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md) with each phase.
Before starting or resuming any phase, also read the
[adaptive change-impact protocol](CHANGE_IMPACT_PROTOCOL.md). It is the rule
for invalidating and regenerating dependent phases when any premise changes.

| Phase | Deliverable | Status |
|---|---|---|
| [0](phase0.md) | Evidence reset, provenance, resource and hypothesis registration | Pending |
| [1](phase1.md) | Independent math oracles and formal correspondence | Pending; core Lean derivations supplied |
| [2](phase2.md) | Streaming state, bounded read, exact archive reference | Pending |
| [3](phase3.md) | Validated certificate, refinement, failure semantics | Pending |
| [4](phase4.md) | Falsification and novelty-critical ablations | Pending |
| [5](phase5.md) | Chunk training and accelerator kernels | Pending |
| [6](phase6.md) | Trained LM pilot with actual held-out measurements | Pending |
| [7](phase7.md) | Serving integration and failure injection | Pending |
| [8](phase8.md) | Matched-budget scale and deployment decision | Pending |
| [9](phase9.md) | Independent reproduction and publication audit | Pending |

Phases execute in order. A disproved research hypothesis is a valid negative
result and a stop condition for scaling. Do not manufacture a PASS by changing
the question. Use results/v2/phaseN and plots/v2/phaseN; do not overwrite v1
artifacts or reuse their pass status. Legacy executable names are not evidence
that the corresponding new phase is implemented.

Phase status is revision-scoped. If a later discovery changes an earlier
equation, theorem, contract, evaluator, metric, data/device assumption, or
semantic invariant, mark the earliest consumer and its transitive descendants
STALE, preserve their old artifacts as SUPERSEDED, update the prompts, and
rerun the closure. A phase passing after a repair is never enough by itself.
