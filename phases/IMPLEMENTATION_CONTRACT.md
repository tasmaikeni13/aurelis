# Implementation handoff: v1 to AURELIS-R

The paper and Lean changed; the implementation did not. **Rewrite the scripts,
model code, tests, configs, and kernels in subsequent phase work to implement
these equations. Do not run the current verifiers and call their output v2.**

## Revision and invalidation

This contract is revision-scoped. Every implementation, checkpoint, benchmark,
and serving artifact records the theory revision and hashes of this contract,
the paper, Lean imports, and evaluator. If implementation reveals that an
equation, invariant, theorem premise, numerical bound, evaluator, or state
semantic is wrong, stop downstream work and follow
[CHANGE_IMPACT_PROTOCOL.md](CHANGE_IMPACT_PROTOCOL.md). Mark the earliest
affected phase and its transitive descendants `STALE`, preserve their old
artifacts as `SUPERSEDED`, increment the revision, rewrite the affected phase
prompts, and regenerate from the earliest affected phase. A local code fix may
be limited to one phase only when an independent equivalence check proves the
published contract unchanged; record that proof in the revision manifest.

## State and operations

Per layer, batch item, and recurrent head:

- S: [batch, heads, d_value, d_key], initially zero; FP32 accumulation baseline.
- ring keys/values: [batch, heads, window, d_key/d_value]; stored alpha/beta,
  causal occurrence IDs, head pointer and valid count.
- alpha, beta in [0,1], computed at the observation's source position; key
  normalization ensures norm≤1 with a finite nonzero floor.
- Optional archive: original encoded remote KV, counts, page IDs, key coordinate
  minima/maxima, value center/radius, and complete index/frontier metadata.
- Decode checkpoint: S, ring including pending gates, position, archive length,
  index version, RNG state if stochastic operations are enabled, and model ID.

Use paper equation (2) for evicted writes and (3) for bounded output. The
archive predictor r is fixed for a read's refinement rounds. Archive selected
sums and midpoint completion implement (7)–(8); envelopes and bound implement
(9)–(12). All recent tokens are selected. Every unread remote occurrence
belongs to exactly one frontier node. Repeated content is not a duplicate ID.

Public read result must include output, mode, certificate bound or null,
reference encoding, numerical allowance, status, counts/bytes, and reason.
Statuses: approximate (bounded), certified (archive), full_read (archive),
budget_exhausted, invalid_state, archive_error. An exact-real certificate cannot
be attached to an unchecked floating-point bound. Full_read means all reference
observations participated, not bitwise agreement between different reductions.

Specify prefill, append, read, reset, snapshot, restore, fork, seal_page, and
cancel semantics. Packed sequences never share state unless explicitly prefix
shared. No retrieval operation mutates S. No future token contributes to a
summary or output. Causal partial pages require masking/consistent summaries.

## Module migration map (future work)

| Existing area | New implementation requirement |
|---|---|
| functional.py / oracle.py | Independent equation (2)–(12) fp64 references; preserve v1 under explicit legacy names if retained |
| types.py / streaming.py | New state/result types, exactly-once writes, archive IDs, snapshots, modes and failure statuses |
| nn.py / nn_phase3.py / nn_phase4.py | Shared query/key coordinate convention, delayed gated delta recurrence, bounded/archive objectives |
| models/aurelis_lm.py / jax_aurelis.py | Actual cached decode, true local attention, structured chunk training; no dense masked L×L local path |
| models/tpu_kernels.py | Explicit actual backend; chunk delta, local attention, page gathers/reductions and interval support where valid |
| baselines.py / models/hybrid_ssm.py | Faithful optimized comparator or clearly labeled simplified reference |
| experiments/ and benchmarks/ | Real trained predictions, populated-cache timings, fetched bytes and metadata costs |
| scripts/verify_phase*.py and run_phase*.sh | V2 gates, real provenance, no score constants and no inherited v1 PASS |
| tests/ / configs/ | New state API, precision/error contracts, causality, rollout parity, budgets and failure injection |
| analysis/ | Independent evaluation, certificate violations and cost/quality Pareto analysis |

Do not carry P, C, Cholesky caches, g_B, posterior-covariance labels, or
AURELIS-B/E meanings into v2. Delta state is not CP⁻¹. V1 oracle/checkpoints
are not compatible with v2; migration requires explicit conversion research or
fresh training.

## Shapes, hardware, and accounting

Start with separate query/recurrent heads and equal head sharing across
comparators. If GQA shares archive KV, each query head still has its own
scores/certificate; union fetched pages but preserve per-head bounds.

First kernel candidates: d_key=d_value in {32,64,128}, window in {64,128,256},
page in {32,64,128}, training chunk in {64,128}. These are experiment grids,
not a selected winning configuration. Select production shapes using phase 5.
The host/device inventory is detected, never inferred from filenames.

Bounded mode has fixed persistent per-request mixing state. Archive mode has
linear total KV plus metadata; a device-resident global index is not bounded
HBM. Count checkpoint, page-table, branch, allocator and transfer buffers.
No promise of constant archive search time, monotonic bound improvement, or
universal exact recall is permitted.

## Initial acceptance conventions

Use explicit absolute tolerances with recorded value scales. Phase 1's fp64
algebra check starts at atol=1e-10, rtol=1e-9 on moderate, nonoverflowing
inputs; ill-conditioned/overflow cases have separately specified outcomes.
These tolerances validate arithmetic agreement, not runtime certification.
Reduced precision needs a separately derived envelope. Register all task and
latency margins before running held-out experiments, following phase 0.
