# Phase 0 MI300X substrate benchmark

Status: **PASS**

- Device: `AMD Instinct MI300X VF`
- PyTorch/HIP: `2.8.0+rocm7.0.2.git245bf6ed` / `7.0.51831-7c9236b16`
- Compile plus first run: `1.631` seconds
- Peak allocated memory: `428094464` bytes
- fp32 eager vs CPU/fp64 maximum error: `7.255e-06`
- fp32 streaming vs CPU/fp64 maximum error: `1.026e-05`
- compiled vs eager maximum error: `1.907e-06`
- compiled vs eager gradient maximum error: `5.245e-06`

| Component | Median ms | Minimum ms |
|---|---:|---:|
| outer_updates | 0.056866 | 0.053388 |
| local_attention | 0.113044 | 0.110046 |
| cholesky_factorization | 0.081505 | 0.080322 |
| triangular_solve | 0.041988 | 0.041283 |
| routing | 0.058008 | 0.055898 |
| vectorized_training_eager_forward | 1.106149 | 1.064620 |
| prepared_head_eager_forward | 0.728274 | 0.676641 |
| prepared_head_eager_forward_backward | 2.139080 | 2.091369 |
| prepared_head_compiled_forward | 0.598563 | 0.537301 |
| prepared_head_compiled_forward_backward | 1.849375 | 1.753952 |

Compilation/warm-up is excluded from steady-state rows. Every timed GPU sample
is synchronized. These are component health measurements, not an accelerator
superiority claim. No custom kernel was added because Phase 0 measurements do
not yet identify a stable shape-specific fusion target beyond Inductor.
