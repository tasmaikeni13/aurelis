# Phase 3 — Certificate implementation and retrieval policy

Depends on phases 0–2. Implement the actual envelope, refinement, and numerical
status logic. A learned gate cannot replace this phase.

## Adaptive start gate

This phase consumes the exact selected/unread semantics from Phase 2 and the
certificate equations from the paper. An unsound bound, interval, rounding
assumption, or status rule invalidates this phase and every consumer, including
passing retrieval, kernel, training, and serving results. Update the equations,
Lean scope, implementation contract, and prompts, increment the revision, and
restart at the earliest affected phase.

## Deliver

Build key boxes, value centers/radii, partial-page handling, and a flat unread
page scan first. Implement (9)–(12), common shifted exponentials, stable exact
selected sums, and the priority heuristic in paper §6.2. Prove or independently
validate every summary contains its actual page and covers all unread IDs.

Implement a conservative arithmetic-error policy for the actual dtype:
outward intervals or documented validated error bounds covering dot products,
exp, norms, reductions, division, and output rounding. If the accelerator lacks
such support, use a slower validated path and label ordinary kernels uncertified.
No arbitrary tolerance inflation is accepted as a proof.

Return certified only when E+delta_num≤epsilon. Full reads, exhausted budgets,
invalid intervals, and unavailable archives have distinct statuses. Keep the
best valid previous candidate; intermediate error/bound need not be monotone.
Optional hierarchical/ANN acceleration cannot omit unknown pages from the
bound. Count summary visits and transfer/selection cost.

## Gates

Dense fp64 reference outputs must lie within valid returned enclosures on
registered stress cases; interval/kernel correspondence receives a separate
review. Test underflow, overflow, nearly cancelled numerators, zero unread mass,
partial pages, negative query coordinates, corrupt bounds, missing pages,
NaNs, I/O timeout, and budget exhaustion. Deliberately inject unsound bounds and
verify validation detects them; no invalid result receives certified status.

Report numerical allowance separately from approximation bound, bound/actual
error ratio, pages/bytes, full-read rate, and failure rate by context/dtype.
PASS is numerical/semantic correctness, not an assertion that the bound is tight
or the implementation is fast.
