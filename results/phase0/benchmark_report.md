# Phase 0 MI300X substrate benchmark

Status: **PASS**

- Device: `AMD Instinct MI300X VF`
- PyTorch/HIP: `2.8.0+rocm7.0.2.git245bf6ed` / `7.0.51831-7c9236b16`
- Compile plus first run: `1.729` seconds
- Peak allocated memory: `428094464` bytes
- fp32 eager vs CPU/fp64 maximum error: `7.255e-06`
- fp32 streaming vs CPU/fp64 maximum error: `1.026e-05`
- compiled vs eager maximum error: `1.907e-06`
- compiled vs eager gradient maximum error: `3.457e-06`

| Component | Median ms | Minimum ms |
|---|---:|---:|
| outer_updates | 0.054839 | 0.053424 |
| local_attention | 0.113631 | 0.108562 |
| cholesky_factorization | 0.081965 | 0.079581 |
| triangular_solve | 0.042359 | 0.041435 |
| routing | 0.050703 | 0.049274 |
| vectorized_training_eager_forward | 1.133153 | 1.083397 |
| prepared_head_eager_forward | 0.692621 | 0.666695 |
| prepared_head_eager_forward_backward | 2.158422 | 2.057068 |
| prepared_head_compiled_forward | 0.584419 | 0.539309 |
| prepared_head_compiled_forward_backward | 1.792926 | 1.660431 |

Compilation/warm-up is excluded from steady-state rows. Every timed GPU sample
is synchronized. These are component health measurements, not an accelerator
superiority claim. No custom kernel was added because Phase 0 measurements do
not yet identify a stable shape-specific fusion target beyond Inductor.
