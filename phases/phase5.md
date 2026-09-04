# Phase 5 — MI300X/ROCm optimization and systems gate

Start only after Phase 4 PASS. Read all prior artifacts, current official AMD
ROCm/MI300X and PyTorch HIP documentation, and
`phases/AUTONOMY_PROTOCOL.md`. Execute the failure-repair loop until PASS.

This phase optimizes the proven mechanism on the actual one-GPU AMD server. A
faster numerically different mechanism is a failure.

## Paths to evaluate

Compare, on identical quantized inputs:

- fp64 CPU and GPU reference;
- eager PyTorch ROCm;
- `torch.compile`/Inductor with documented modes;
- rocBLAS/hipBLASLt and rocSOLVER-backed primitives;
- sequential rank-one factor maintenance versus refactorization;
- batched dense exact training solves;
- fixed-step and residual-stopped conjugate gradient;
- chunkwise/Woodbury construction where derived; and
- Triton/ROCm or Composable Kernel fusion only when profiling identifies a
  fusible bottleneck.

Profile complete heads and components over:

- `d_k,d_v in {16,32,64,128,256}` where memory permits;
- windows `{8,16,32,64,128,256,512}`;
- 125M parameter architecture targets (`d_model=768`, 12 heads, `d_k=64, d_v=64`, 12 layers);
- batch, heads, context, and decode batch axes;
- bf16/fp16/fp32 plus fp64 oracle;
- cold compile, warm-up, tuning, and steady state; and
- conditioned and pathological key streams.

Measure wall time with synchronization, tokens/s, latency distribution, HBM
traffic, achieved bandwidth/FLOPs where profiler support is reliable, launch
count, occupancy, peak VRAM, live decode state, solve iterations/residuals,
forward error, gradient error, and nonfinite rate.

## Matched systems baselines

Include full attention using the strongest available ROCm backend, sliding
window attention, positive-feature linear attention, Gated DeltaNet, and the
best cumulative least-squares remote implementation. Compare same dimensions,
parameters, state bytes, and quality-qualified configurations separately.
Record library/backend availability rather than silently substituting one.

## Optimization discipline

Use a profiler trace before each material kernel rewrite. Research current
official guidance and primary kernel work for the measured bottleneck. Derive
traffic/work predictions, implement, verify against fp64, and keep the change
only if end-to-end results agree. Never report a health-check GEMM as AURELIS
throughput or exclude compile/tuning costs without also reporting them.

## PASS gates

- Every retained optimized path meets dtype- and condition-aware forward,
  backward, state, gate, and handoff tolerances versus fp64.
- No CUDA/NVIDIA package or device assumption is present; ROCm is detected via
  HIP even though PyTorch uses `torch.cuda` APIs.
- The best exact and approximate training paths have a declared accuracy/cost
  frontier; iterative stopping error is linked to output error.
- At least one quality-valid long-context regime demonstrates both lower live
  decode state and higher steady-state decode throughput than matched global
  attention; regimes without a win remain in the report.
- The selected production path is not more than 2x slower than the strongest
  comparable local/recurrent hybrid at the preregistered deployment point, or
  a researched algorithmic repair closes the gap before PASS.
- OOM, nonfinite, compile, and unsupported-backend rows are retained.
- One command reproduces correctness plus a bounded smoke benchmark; a separate
  command reproduces the full sweep.
- All inherited gates and Lean proofs pass, and
  `results/phase5/PASS.md` satisfies the shared PASS record.
