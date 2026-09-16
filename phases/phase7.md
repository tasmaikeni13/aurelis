# Phase 7 — Serving integration and failure injection

Depends on phases 0–6. Use a trained surviving checkpoint. Implement the
serving path and scripts explicitly; “kernel parity” is insufficient.

## Adaptive start gate

Serving discoveries can reach backward: a rollback, prefix-cache, archive, or
state-layout defect can invalidate Phase 2 semantics and every later result.
Compute the earliest affected phase before fixing serving code. Preserve old
load tests as superseded, regenerate the affected correctness contract, and
rerun all descendants before making a deployment decision.

## Deliver

Integrate per-request S/ring/archive state with continuous batching, true cached
decode, prefix sharing, chunked prefill, pause/resume, and cancellation.
Snapshots include ring gates, positions, archive length/index version, and model
identity. Implement copy-on-write forks and speculative rejection rollback.

Define bounded, strict archive, and permissive archive service policies.
Strict archive does not emit an uncertified estimate on deadline; it returns
an explicit failure/escalation. A full read certifies same-Q/K/V attention
only, not a dense model trajectory. Document exact-checkpoint replay requirements
if a trajectory-level guarantee is offered.

## Failure and security boundaries

Inject lost/stale/corrupt pages, invalid summary intervals, offload failures,
device reset, cancellation mid-read, memory pressure, and repeated retry.
Verify no cross-request data disclosure, use-after-free pages, stale certificate,
duplicate writes, or mutated shared prefix. Model-output factual safety is
outside the attention proof and needs application evaluation.

## Gates

Under the registered arrival/load distribution, report TTFT, inter-token
p50/p95/p99 latency, throughput within SLO, failure/fallback rates, and peak
memory in every tier. Include worst-case diffuse attention and high archive
miss rates. Compare fixed SLO and available resources with modern dense/hybrid
serving. Include correctness replay and state/checkpoint accounting.

PASS requires all invariants, fault outcomes, and registered SLO/quality gates.
A faster average with unacceptable p99, budget failures, or unsupported state
operations is not a deployable result.
