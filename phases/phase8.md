This is Phase 8.

Goal:
Determine whether the validated CSM mechanism can be implemented efficiently on the available AMD MI300X under ROCm.

Correctness is already established in previous phases.
Now optimize without changing the mathematical operation.

Start by benchmarking the current PyTorch implementation.

Measure separately:

1. outer-product state updates
2. construction of S and C
3. Cholesky factorization
4. triangular solves
5. sequential decode
6. training forward
7. backward
8. memory movement

Profile:
- GPU utilization
- HBM bandwidth
- achieved FLOPs
- kernel launch overhead
- peak VRAM
- tokens/sec
- microseconds/token for decode

Sweep:
d_k in {16,32,64,128}
d_v in {16,32,64,128}
batch size
sequence length
number of heads
dtype

Precision policy to test:
- bf16 activations/features
- fp32 S/C accumulation
- fp32 Cholesky and solves
and relevant alternatives.

Implement optimizations incrementally:

A. vectorized PyTorch
B. torch.compile where supported and useful
C. chunked processing
D. associative segment summaries
E. fused implementation using the best available ROCm-compatible mechanism if justified

Do not assume a CUDA-only implementation strategy.
Detect what ROCm stack actually supports.

Implement and validate associative chunk composition for recurrence summaries.

Every optimized path must be numerically compared against the Phase 1 oracle.

Benchmark against:
- attention at equivalent width/context
- simple recurrent/linear-memory baseline where useful

Report results TWO ways:

1. theoretical operation/state complexity
2. actual MI300X wall-clock and memory behavior

Critical metric:
memory quality per byte and per unit wall-clock cost.

The manuscript proposes many small heads as the economical d^2 regime. Test that claim rather than assuming it.

PASS GATE:
Obtain a stable implementation whose performance is sufficiently understood that later model comparisons will not merely measure an obviously inefficient prototype.

Do NOT require CSM to beat FlashAttention yet.
The goal is to determine the actual hardware tax.

Write:
results/phase8_mi300x_systems.md
