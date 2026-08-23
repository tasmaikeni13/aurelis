# Phase 1 report

## Gate decision: PASS

The fp64 sequential recurrence and the independent historical recomputation were evaluated for `d_k in {2,4,8,16,32}`, five deterministic seeds per dimension, random beta, random lambda, and Cholesky-based reads. The direct inverse appears only as a tiny-matrix oracle.

- PASS: `state_agreement`
- PASS: `read_agreement`
- PASS: `positive_semidefinite`
- PASS: `gradcheck`
- PASS: `pathologies_finite`

| Gate metric | Observed | Required |
|---|---:|---:|
| maximum state absolute error | 8.881784e-16 | <= 1.0e-11 |
| maximum read absolute error | 2.273737e-12 | <= 1.0e-10 |
| minimum eigenvalue of S | -2.855521e-16 | >= -1.0e-10 |
| gradcheck | True | True |

## Equation-level findings

- Sequential and recomputed states agree across 25 randomized cases. Agreement covers both sufficient statistics and queries, not merely final loss values.
- Cholesky reads and the inverse oracle agree in the automated test suite. Production-style reads never construct an inverse.
- The interpolation sweep shows stored-key and signed linear-functional errors approaching zero with epsilon.
- Repeated noisy observations follow the predicted `sigma^2 d_v / n` curve; empirical/predicted ratios range from 0.905 to 1.104 over 256 trials per point.
- The fp32 conditioning sweep reaches a maximum measured condition number of 1.000e+06; the associated error curve is an empirical finite-precision diagnostic, not a Phase 1 pass criterion.
- Manuscript `c_t(q)` is mathematically a posterior variance: smaller values mean higher confidence. The implementation names the public method `confidence` for equation compatibility and documents this direction explicitly.

## Pathological cases

| Case | Finite state | Finite read | min eig(S) |
|---|---:|---:|---:|
| repeated_keys | True | True | 0.000e+00 |
| nearly_collinear_tiny_epsilon | True | True | 5.000e-15 |
| beta_zero | True | True | 0.000e+00 |
| lambda_below_one_large_beta | True | True | 1.000e+149 |
| zero_value_single_observation | True | True | 0.000e+00 |

The test suite additionally covers lambda=1, lambda<1, tiny epsilon with a well-conditioned basis, beta=0, very large beta, zero values, a single observation with a closed-form answer, repeated keys, and nearly collinear keys.

## Plots

- [`recurrence_consistency.png`](../plots/phase1/recurrence_consistency.png)
- [`interpolation_error.png`](../plots/phase1/interpolation_error.png)
- [`conditioning_fp32.png`](../plots/phase1/conditioning_fp32.png)
- [`noise_averaging.png`](../plots/phase1/noise_averaging.png)

## Reproducibility record

- git commit tested: `60d5922bac15526599352f805a6382e32ae0d331`
- working tree dirty at run time: `False`
- config: [`configs/phase1_reference.json`](../configs/phase1_reference.json)
- seeds: `[0, 1, 2, 3, 4]` plus fixed per-experiment seeds recorded in source
- hardware: `AMD Instinct MI300X VF`
- software: Python `3.12.3`, PyTorch `2.8.0+rocm7.0.2.git245bf6ed`, HIP `7.0.51831-7c9236b16`
- wall-clock time: `25.054` seconds
- peak allocated VRAM: `0.125433` GiB
- machine-readable metrics: [`phase1_metrics.json`](phase1_metrics.json)

## Interpretation

The reference implementation represents the stated recurrence and reads to fp64 numerical precision under the tested gate distribution.

This gate validates the Phase 1 implementation against Definition 5.1. It does not validate learned encoders, optimized scans, the dyadic cascade, language modeling, or the broader architecture claims. No work beyond Phase 1 is authorized by this result.

## Independent verification

After generation, the repository-wide suite passed `36/36` pytest cases and `lake build` accepted all declared Lean theorems. These checks are rerun during the final completion audit; the experiment's internal gradcheck and numerical metrics remain the machine-generated gate evidence.
