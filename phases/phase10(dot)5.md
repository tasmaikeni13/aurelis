This is Phase 10.5 of the CSM research program.

DO NOT scale model size.

The Phase 10 experiment established:
- stable natural-language training,
- approximately matched model quality,
- constant incremental CSM state,
- a large state-memory advantage at long context,

but also:
- a substantial training throughput deficit,
- larger training VRAM,
- slower incremental decode.

Before scaling, determine how much of that deficit comes from the current
implementation rather than the CSM algorithm itself.

CRITICAL OBSERVATION:

The current incremental CSMMixer.step() recomputes

    chol(S_t + epsilon I)

from scratch at every token.

The current training read path also performs Cholesky factorization over every
materialized prefix system.

The intended CSM systems algorithm maintains a Cholesky factor using rank-one
updates, reducing recurrent factor maintenance from repeated O(d_k^3)
factorization toward O(d_k^2) rank-one maintenance.

PHASE 10.5 GOAL:

Implement and validate the intended maintained-factor CSM path on AMD MI300X /
ROCm, then determine the actual long-context latency crossover against attention.

PART A — maintained Cholesky decode

Extend CSMDecodeState to maintain whatever state is required for efficient reads,
including a Cholesky factor R satisfying approximately:

    R R^T = S + epsilon I

For the current Phase 10 language model, lambda = 1, so each update is:

    S_new = S_old + beta k k^T

Implement a numerically stable rank-one Cholesky update:

    R_new = cholupdate(R_old, sqrt(beta) k)

Do NOT silently change the mathematical CSM recurrence.

Implement at least:

1. full-refactorization reference path
2. maintained-factor path

Every optimized step must be checked against the reference for:
- S
- C
- solved query
- final read
- model logits over sequential decoding

Sweep:
d_k = {8,16,32,64}
heads
batch sizes
sequence lengths
bf16 feature + fp32 state policy

Track cumulative numerical error over at least:
128
512
2048
8192
32768
sequential updates.

PART B — optimize ROCm implementation

Profile the rank-one update.

Investigate ROCm-appropriate implementations:
- vectorized PyTorch
- torch.compile / Inductor
- Triton where supported
- custom HIP extension only if necessary

Do not write a complex custom kernel merely to obtain a benchmark win.
Correctness remains mandatory.

PART C — training path

Profile the exact Phase 10 training path and attribute runtime among:

- feature projections
- construction of token S/C increments
- prefix state construction
- Cholesky factorization
- triangular solves
- FFN
- backward

The current path factorizes every prefix system independently.

Prototype a training evaluation strategy closer to the manuscript's intended
chunk algorithm:

1. associative/chunk boundary summaries
2. boundary state computation
3. intra-chunk maintained-factor updates
4. reads inside the chunk

The objective is not necessarily to produce the final fused kernel yet.

The objective is to determine whether the intended O(d_k^2)-per-token local
algorithm materially reduces the measured hardware tax.

Compare all paths against the Phase 1 mathematical oracle.

PART D — actual long-context crossover

Using the Phase 10 checkpoints, measure true incremental generation at prompt
lengths:

128
512
2048
4096
8192
16384
32768

and, if memory permits,

65536

Measure:

- microseconds/token
- tokens/sec
- live recurrent/cache state bytes
- peak VRAM
- HBM traffic where measurable

Compare:
1. Transformer KV decoding
2. old CSM full-factorization decoding
3. maintained-factor CSM decoding

Determine the empirical context length at which CSM:
- uses less state
- becomes latency competitive, if one exists

Do not extrapolate a crossover from asymptotics if it is not measured.

PART E — decision

Produce:

results/phase10_5_systems_alignment.md

The report must answer:

1. How much faster is maintained-factor decode than the old implementation?
2. Does numerical drift remain controlled?
3. What is CSM's actual long-context latency curve?
4. Is there a measured CSM/Transformer latency crossover?
5. What component dominates CSM training after optimization?
6. Is the current implementation sufficiently representative to justify
   scaling to 125M?
7. What work remains kernel engineering rather than architectural research?

PASS TO PHASE 10.6 if:
- the maintained-factor path is numerically faithful,
- long-sequence stability is demonstrated,
- and the hardware behavior is sufficiently understood that a larger model
  would measure CSM rather than an obviously avoidable implementation artifact.

Do not proceed automatically to 125M.
