# Phase 0 MI300X substrate benchmark

Status: **PASS**

- Device: `AMD Instinct MI300X VF`
- PyTorch/HIP: `2.8.0+rocm7.0.2.git245bf6ed` / `7.0.51831-7c9236b16`
- Compile plus first run: `1.821` seconds
- Peak allocated memory: `428094464` bytes
- fp32 eager vs CPU/fp64 maximum error: `7.255e-06`
- fp32 streaming vs CPU/fp64 maximum error: `1.026e-05`
- compiled vs eager maximum error: `1.907e-06`
- compiled vs eager gradient maximum error: `5.245e-06`

| Component | Median ms | Minimum ms |
|---|---:|---:|
| outer_updates | 0.061288 | 0.058552 |
| local_attention | 0.127608 | 0.125000 |
| cholesky_factorization | 0.087659 | 0.085750 |
| triangular_solve | 0.045134 | 0.044255 |
| routing | 0.057563 | 0.056859 |
| vectorized_training_eager_forward | 1.248599 | 1.188836 |
| prepared_head_eager_forward | 0.740702 | 0.728746 |
| prepared_head_eager_forward_backward | 2.333702 | 2.302403 |
| prepared_head_compiled_forward | 0.592864 | 0.564788 |
| prepared_head_compiled_forward_backward | 2.007323 | 1.951566 |

Compilation/warm-up is excluded from steady-state rows. Every timed GPU sample
is synchronized. These are component health measurements, not an accelerator
superiority claim. No custom kernel was added because Phase 0 measurements do
not yet identify a stable shape-specific fusion target beyond Inductor.
