# Phase 0 MI300X substrate benchmark

Status: **PASS**

- Device: `AMD Instinct MI300X VF`
- PyTorch/HIP: `2.8.0+rocm7.0.2.git245bf6ed` / `7.0.51831-7c9236b16`
- Compile plus first run: `2.291` seconds
- Peak allocated memory: `428094464` bytes
- fp32 eager vs CPU/fp64 maximum error: `7.255e-06`
- fp32 streaming vs CPU/fp64 maximum error: `1.026e-05`
- compiled vs eager maximum error: `1.907e-06`
- compiled vs eager gradient maximum error: `3.457e-06`

| Component | Median ms | Minimum ms |
|---|---:|---:|
| outer_updates | 0.065493 | 0.061902 |
| local_attention | 0.128908 | 0.123709 |
| cholesky_factorization | 0.088057 | 0.086780 |
| triangular_solve | 0.046831 | 0.045621 |
| routing | 0.059887 | 0.057358 |
| vectorized_training_eager_forward | 1.257369 | 1.153133 |
| prepared_head_eager_forward | 0.807257 | 0.774591 |
| prepared_head_eager_forward_backward | 2.915648 | 2.524545 |
| prepared_head_compiled_forward | 0.603923 | 0.581926 |
| prepared_head_compiled_forward_backward | 2.038974 | 1.994292 |

Compilation/warm-up is excluded from steady-state rows. Every timed GPU sample
is synchronized. These are component health measurements, not an accelerator
superiority claim. No custom kernel was added because Phase 0 measurements do
not yet identify a stable shape-specific fusion target beyond Inductor.
