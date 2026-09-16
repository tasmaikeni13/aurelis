# Phase 5 — Chunk training, kernels, and cost breakdown

Depends on phases 0–4 with a surviving hypothesis. Implement the new algorithms
on the actual available accelerator; no hardware family is assumed.

## Adaptive start gate

Kernel or chunk changes may remain local only after independent semantic and
numerical equivalence checks. If profiling exposes a theory or state-semantic
defect, invalidate the earlier consumer and all descendants using the
change-impact closure. A faster kernel for a changed operation is not evidence
for the published operation.

## Deliver

Implement fused delta updates/reads, true window attention, structured delta
chunk forward/backward with checkpointing, selected-page attention, and grouped
refinement rounds. Validate gradients against the small reference and causal
streaming continuation. Preserve source-position gate/delay semantics.

No per-token Cholesky/linear solve or all-prefix d×d state materialization.
No L×L score allocation for the local path. Inspect kernel profiles and peak
allocated memory; lower FLOP counts alone do not satisfy these requirements.
Chunkwise unit-triangular work, recomputation, and backward memory are disclosed.

Compare modern optimized GQA attention (appropriate FlashAttention/serving
backend), a faithful optimized delta/SSM, and a strong real hybrid. Historical
handwritten baselines are labeled as references, not state of the art.

## Measurement gates

Use actual populated-cache continuation, synchronized timing, warmup exclusion,
multiple repetitions, registered shapes/batches/contexts, and actual dtype.
Report prefill, forward/backward, decode, launches, state traffic, index traversal,
page transfer, compile time, peak HBM, host memory, and storage separately.
Archive mode gets end-to-end measurements including unfavorable access patterns.

PASS requires numerical/gradient contracts, structural elimination of v1's
bottlenecks, and preregistered speed/memory thresholds in a stated workload
region. If removal of solves does not translate to useful end-to-end gain,
preserve the profile and stop that design's scaling.
