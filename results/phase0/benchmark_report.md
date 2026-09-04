# Phase 0 MI300X substrate benchmark

Status: **PASS**

- Device: `AMD Instinct MI300X VF`
- PyTorch/HIP: `2.8.0+rocm7.0.2.git245bf6ed` / `7.0.51831-7c9236b16`
- Compile plus first run: `1.767` seconds
- Peak allocated memory: `428094464` bytes
- fp32 eager vs CPU/fp64 maximum error: `7.255e-06`
- fp32 streaming vs CPU/fp64 maximum error: `1.026e-05`
- compiled vs eager maximum error: `1.907e-06`
- compiled vs eager gradient maximum error: `5.245e-06`

| Component | Median ms | Minimum ms |
|---|---:|---:|
| outer_updates | 0.063018 | 0.060763 |
| local_attention | 0.135888 | 0.131906 |
| cholesky_factorization | 0.087140 | 0.085824 |
| triangular_solve | 0.045979 | 0.045352 |
| routing | 0.058007 | 0.056468 |
| vectorized_training_eager_forward | 1.112108 | 1.066790 |
| prepared_head_eager_forward | 0.733484 | 0.700409 |
| prepared_head_eager_forward_backward | 2.151060 | 2.109534 |
| prepared_head_compiled_forward | 0.639686 | 0.586237 |
| prepared_head_compiled_forward_backward | 1.923812 | 1.893743 |

Compilation/warm-up is excluded from steady-state rows. Every timed GPU sample
is synchronized. These are component health measurements, not an accelerator
superiority claim. No custom kernel was added because Phase 0 measurements do
not yet identify a stable shape-specific fusion target beyond Inductor.
