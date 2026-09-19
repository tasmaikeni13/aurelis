# Phase 5 — Chunk training, kernels, and cost breakdown

**Revision:** `1.1` (Following Phase 4 Falsification of H2; surviving branch: Bounded Mode H1/H4).  
Depends on phases 0–4 with a surviving hypothesis. Implement the new algorithms
on the actual available hardware (240-core AMD EPYC host platform); no hardware family is assumed.

## Adaptive start gate

Per `phases/CHANGE_IMPACT_PROTOCOL.md`, Phase 4 falsified Hypothesis H2 (archive retrieval).
Archive retrieval scaling is retired. Phase 5 proceeds strictly on the surviving
novelty branch: **Bounded Mode (Hypotheses H1 & H4)**.
Kernel or chunk changes may remain local only after independent semantic and
numerical equivalence checks. If profiling exposes a theory or state-semantic
defect, invalidate the earlier consumer and all descendants using the
change-impact closure. A faster kernel for a changed operation is not evidence
for the published operation.

## Deliver

Implement fused delta updates/reads (Eq. 2 & 3), true causal window attention,
structured delta chunk forward/backward with associative state updates and checkpointing.
Validate gradients against the reference implementation and causal streaming continuation.
Preserve source-position gate/delay semantics.

No per-token Cholesky/linear solve or all-prefix d×d state materialization.
No L×L score allocation for the local path. Inspect kernel profiles and peak
allocated memory; lower FLOP counts alone do not satisfy these requirements.
Chunkwise unit-triangular work, recomputation, and backward memory are disclosed.

Compare modern optimized GQA attention, a faithful optimized delta/SSM,
and the bounded AURELIS hybrid. Historical handwritten baselines are labeled
as references, not state of the art.

## Measurement gates

Use actual populated-cache continuation, synchronized timing, warmup exclusion,
multiple repetitions, registered shapes/batches/contexts, and actual dtype.
Report prefill, forward/backward, decode, launches, state traffic, compile time,
peak host memory, and storage separately.

PASS requires numerical/gradient contracts, solve-free complexity
verification ($O(d_v d_k + w(d_k + d_v))$ per decode step), and preregistered
speed/memory thresholds on the host hardware. If removal of solves does not translate
to useful end-to-end gain, preserve the profile and stop that design's scaling.

