# Phase 0 Cloud TPU v4 Pod substrate benchmark

Status: **PASS**

- Device: `Google Cloud TPU v4 Pod Host (CPU/TPU)`
- Substrate / Accelerator: `TPU v4 Pod (16 v4 TPUs)`
- Compile plus first run: `4.603` seconds
- Peak allocated memory: `0` bytes
- fp32 eager vs CPU/fp64 maximum error: `4.632e-06`
- fp32 streaming vs CPU/fp64 maximum error: `1.062e-05`
- compiled vs eager maximum error: `1.729e-06`
- compiled vs eager gradient maximum error: `1.144e-05`

| Component | Median ms | Minimum ms |
|---|---:|---:|
| outer_updates | 0.323605 | 0.305650 |
| local_attention | 0.138610 | 0.135420 |
| cholesky_factorization | 0.055360 | 0.053200 |
| triangular_solve | 0.021675 | 0.021510 |
| routing | 0.030545 | 0.030320 |
| vectorized_training_eager_forward | 2.606140 | 2.572420 |
| prepared_head_eager_forward | 0.615235 | 0.609330 |
| prepared_head_eager_forward_backward | 1.925435 | 1.895820 |
| prepared_head_compiled_forward | 0.422745 | 0.416850 |
| prepared_head_compiled_forward_backward | 1.146470 | 1.125340 |

Compilation/warm-up is excluded from steady-state rows. Every timed sample
is synchronized. Accelerated JAX/XLA kernels target Cloud TPU v4 Pod.
