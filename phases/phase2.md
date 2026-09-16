# Phase 2 — Streaming semantics and exact archive reference

Depends on phases 0–1. Implement state/types/model interfaces in the migration
map, plus new scripts/tests/configs. This is correctness-first implementation.

## Adaptive start gate

This phase consumes the Phase 1 equations and formal correspondence. If any
later discovery changes causality, state layout, write order, snapshot
contents, or the meaning of an archive observation, invalidate Phase 2 and all
descendants. Regenerate this semantic reference first and rerun the closure.

## Deliver

Implement S, recent ring with source-position gates, exactly-once eviction
writes, bounded read (3), and append-only raw archive. Archive data retains
causal occurrence IDs and original encoding. Build exact selected sums and
full-history softmax reference; implement midpoint completion (8).
Reference archive mode may read everything. Do not claim retrieval speed yet.

Expose reset/snapshot/restore/fork/cancel and read statuses from the contract.
Implement actual populated-cache autoregressive decode. Make token-by-token
execution, multi-token prefill, and continuation agree for both modes under
the same math and precision. Use per-query causal masks in partial pages;
index construction may not leak later observations.

## Gates

Exercise t below/equal/above w, empty remote state, multiple page boundaries,
duplicate content with distinct IDs, packed examples, variable lengths, and
interleaved requests. Verify no cross-request state contamination.
Reading pages never writes S; retrying a read never duplicates an observation.
Prefix snapshots restore the complete state tuple. Rejected speculative tokens
can be rolled back by a correctness reference before optimizing it.

Measure live tensor state as context grows; bounded state must plateau after
the window fills. Archive bytes must grow as predicted and be labeled by tier.
A wrong archive length/index version must fail rather than return a certificate.
PASS requires streaming/history agreement and true cached decode, not isolated
one-token forward timings.
