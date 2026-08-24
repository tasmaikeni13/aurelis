# Phase 8 MI300X systems characterization

## Gate decision: PASS

- PASS: `rocm_mi300x_target`
- PASS: `torch_compile_supported`
- PASS: `all_paths_finite`
- PASS: `optimized_state_matches_oracle`
- PASS: `optimized_reads_match_oracle`
- PASS: `timings_stable`
- PASS: `all_required_sweeps_present`

## Exactness against the Phase 1 oracle

| path | S rel. error | C rel. error | read rel. error |
| --- | --- | --- | --- |
| A_vectorized | 0.000000 | 0.000000 | 0.000000 |
| C_chunked | 0.000000 | 0.000000 | 0.000000 |
| D_associative | 0.000000 | 0.000000 | 0.000002 |
| reference_batched_sequential | 0.000000 | 0.000000 | 0.000003 |
| B_torch_compile | 0.000000 | 0.000000 | 0.000001 |

All oracle inputs are first quantized to bf16, then the Phase 1 recurrence is evaluated in fp64 on those same values. Optimized paths accumulate and solve in fp32. Thus the table isolates implementation error from activation quantization.

## Component profile

| operation | ms | tokens/s | us/token | est. TFLOP/s | est. GB/s | GPU util % | peak util % | peak VRAM B | timing CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outer_product_updates | 0.068 | 15083222.862 | 0.066 | 0.494 | 0.000 | 3.000 | 3.000 | 151617536 | 0.238 |
| construct_S_C | 0.169 | 6060354.981 | 0.165 | 0.199 | 0.000 | 5.000 | 5.000 | 117637120 | 0.171 |
| cholesky | 0.101 | 79026.551 | 12.654 | 0.007 | 0.000 | 5.000 | 5.000 | 217101824 | 0.106 |
| triangular_solves | 0.065 | 123971.808 | 8.066 | 0.002 | 0.000 | 4.000 | 4.000 | 216846848 | 0.126 |
| sequential_decode | 46.256 | 22137.719 | 45.172 | 0.003 | 0.000 | 93.500 | 97.000 | 223273984 | 0.204 |
| training_forward | 0.486 | 790692.235 | 1.265 | 0.000 | 0.000 | 56.000 | 56.000 | 225197568 | 0.091 |
| backward | 1.091 | 352094.273 | 2.840 | 0.000 | 0.000 | 5.000 | 5.000 | 444022272 | 0.123 |
| memory_movement | 0.062 | 0.000 | 0.000 | 0.000 | 4309.530 | 3.000 | 3.000 | 703072256 | 0.223 |
| kernel_launch | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 | 3.000 | 3.000 | 703072768 | 0.150 |

GPU utilization is sampled from ROCm-SMI over sustained timing regions. Estimated operation bandwidth and FLOPs use explicit tensor-traffic and leading-operation models; the 128 MiB copy row is the direct empirical HBM movement reference. Kernel-launch latency is a synchronized scalar-device operation.

## Optimization ladder

| path | ms | tokens/s | peak VRAM B | CV |
| --- | --- | --- | --- | --- |
| A_vectorized | 0.184 | 5578678.985 | 465764352 | 0.152 |
| B_torch_compile | 0.244 | 4193558.957 | 463536128 | 0.070 |
| C_chunked_32 | 0.823 | 1244741.756 | 439952384 | 0.039 |
| D_associative_scan | 1.025 | 999136.683 | 730496512 | 0.012 |

`torch.compile`: supported. The best ROCm-compatible fusion available here is Inductor/Triton fusion around state construction. A custom fused Cholesky kernel is not justified by this phase: rocSOLVER supplies the factorization, and replacing it would expand the numerical validation surface.

Chunking bounds temporary token-summary storage. Associative scan has logarithmic dependency depth but materializes every prefix matrix; the unit-decay training fast path uses an exact cumulative sum instead.

## Many-small-head economics

| heads | d_k | state B | ms | normalized MSE | quality/MB | quality/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 262144 | 0.935 | 0.001 | 3.813 | 1069.389 |
| 2 | 64 | 131072 | 0.694 | 0.001 | 7.625 | 1440.767 |
| 4 | 32 | 65536 | 0.587 | 0.001 | 15.247 | 1700.904 |
| 8 | 16 | 32768 | 0.557 | 0.002 | 30.447 | 1790.074 |

These timing rows hold aggregate width, batch, 128-token context, activation dtype, and workload fixed. Quality uses a separate under-capacity 8-association recall control at each head size: `quality = max(0, 1-normalized_MSE)`. Quality/MB is the requested memory-quality-per-byte view. Smaller heads reduce the quadratic state term, but measured latency and quality determine whether that theoretical saving is economical on this GPU.

## Baselines at equivalent width/context

| method | ms | tokens/s | us/token | persistent/cache B | peak VRAM B |
| --- | --- | --- | --- | --- | --- |
| csm | 0.776 | 1318919.105 | 0.758 | 262144 | 599330816 |
| attention | 0.034 | 30211540.014 | 0.033 | 1048576 | 429983232 |
| linear_memory | 0.266 | 3852597.995 | 0.260 | 135168 | 498630656 |

Attention uses PyTorch causal scaled-dot-product attention on ROCm; the linear-memory baseline is positive-feature causal linear attention. CSM state bytes are constant in context length, whereas the reported attention KV bytes grow with context.

## Sweep coverage and complexity

The raw sweep contains 29 rows: the complete `d_k x d_v` grid over `[16, 32, 64, 128]`, plus independent batch-size, sequence-length, head-count, and dtype/precision-policy sweeps. See [`phase8/sweeps.csv`](phase8/sweeps.csv).

For `H` heads, CSM persistent state is `B H d_k (d_k+d_v)` fp32 elements; a write/read has leading `Theta(H(d_k^2+d_k d_v))` work, Cholesky preparation is `Theta(H d_k^3)`, and prepared reads are `Theta(H(d_k^2+d_k d_v))`. Attention stores `Theta(B H T(d_k+d_v))` KV elements and performs `Theta(B H T^2 d_k)` prefill work. The tables above are measured wall-clock behavior, not substitutions for these asymptotics.

## Precision policy

The stable primary policy is bf16 features/activations with fp32 `S/C` accumulation, factorization, and triangular solves. The raw dtype rows include fp32, fp16, bf16, and fp64 alternatives. Low-precision Cholesky is not offered by the ROCm PyTorch linalg path and is numerically inappropriate for the conditioned system; this is a measured stack constraint, not a CUDA assumption.

## Reproducibility

- commit at run start: `c728a210077d073e9180ef683c9e85f166ecfa87`; dirty: `True`
- device: `AMD Instinct MI300X VF`; gfx: `gfx942:sramecc+:xnack-`; ROCm/HIP: `7.0.51831-7c9236b16`
- Python `3.12.3`; PyTorch `2.8.0+rocm7.0.2.git245bf6ed`; wall time `17.37s`
- config: [`configs/phase8_mi300x.json`](../configs/phase8_mi300x.json)
- raw records: [`phase8/`](phase8/); machine record: [`phase8_metrics.json`](phase8_metrics.json)

## Scoped conclusion

The pass gate establishes a stable, oracle-checked ROCm implementation and quantifies its hardware tax. It does not require or claim a win over FlashAttention. Later model comparisons should use the measured bf16/fp32 policy and distinguish persistent recurrent state from training-time prefix activations.
