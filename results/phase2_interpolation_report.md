# Phase 2 interpolation report

## Gate decision: FAIL

This sweep retains all 12,600 aggregate rows from 1,800 deterministic key/value datasets. It covers five key dimensions, eight loads, seven logarithmic epsilon values, three seeds, five key regimes, and three value regimes. No seed or difficult regime was removed.

- PASS: `independent_limit`
- PASS: `finite_epsilon_direction`
- FAIL: `finite_epsilon_theorem_bound`
- PASS: `dependent_overcapacity_breakdown`

| Measurement | Observed | Gate threshold |
|---|---:|---:|
| independent under-capacity median relative error at minimum epsilon | 2.214321e-12 | <= 5.0e-04 |
| independent under-capacity p99 relative error at minimum epsilon | 6.195306e-08 | diagnostic |
| cases with error no worse at minimum than maximum epsilon | 100.000% | >= 95.0% |
| maximum actual / valid theorem bound | 1.003804 | <= 1.000100 |
| dependent/over-capacity to independent median error ratio | 3.034e+11 | >= 100.0 |

## Mathematical interpretation

For a row-key matrix `K` with full row rank and value matrix `V`, stored-key recall is `K(K^T K + epsilon I)^(-1)K^T V`. The exact push-through identity makes its error `-epsilon (K K^T + epsilon I)^(-1)V`, hence

`||error||_F <= epsilon / (lambda_min(K K^T) + epsilon) ||V||_F`.

The automated bound check covers 6,363 full-row-rank cases and is deliberately not applied to duplicate or over-capacity cases. The observed limit and finite-epsilon direction support the theorem in its stated domain. Conditioning controls how small epsilon must become before that limit is numerically visible.

Dependent keys and `K > d_key` cannot interpolate arbitrary values: the read is a regularized projection through a rank-limited key geometry. Their nonzero low-epsilon error is therefore a structural failure of arbitrary association recall, not evidence against the scoped full-row-rank theorem. Near-collinear keys are mathematically independent in most generated cases but expose the predicted finite-precision and regularization sensitivity.

`c(q) = q^T(S + epsilon I)^(-1)q` is plotted as a posterior-variance statistic; lower means surer. It is not asserted to be a calibrated per-item error probability.

## Plots

- [`error_vs_epsilon.png`](../plots/phase2/error_vs_epsilon.png)
- [`error_vs_load.png`](../plots/phase2/error_vs_load.png)
- [`error_vs_min_gram_eigenvalue.png`](../plots/phase2/error_vs_min_gram_eigenvalue.png)
- [`confidence_vs_error.png`](../plots/phase2/confidence_vs_error.png)
- [`load_conditioning_heatmap.png`](../plots/phase2/load_conditioning_heatmap.png)

## Reproducibility

- git source checkpoint: `d35e29768dbe689fa427b5abc107860e7ff37100`
- working tree dirty at experiment start: `False`
- config: [`configs/phase2_interpolation.json`](../configs/phase2_interpolation.json)
- complete seeds: `[0, 1, 2]`
- hardware: `AMD Instinct MI300X VF`
- software: Python `3.12.3`, PyTorch `2.8.0+rocm7.0.2.git245bf6ed`, HIP `7.0.51831-7c9236b16`, NumPy `2.3.2`
- wall-clock time: `151.392` seconds
- peak allocated VRAM: `0.126585` GiB
- all aggregate rows: [`phase2/interpolation_sweep.csv`](phase2/interpolation_sweep.csv)
- machine-readable summary and resolved config: [`phase2_metrics.json`](phase2_metrics.json)

## Scoped conclusion

At least one Phase 2 criterion failed. The raw record is retained; the affected mathematical or empirical claim must be narrowed or corrected before relying on it.

This phase tests exact synthetic association geometry only. It does not establish learned-memory, language-model, wall-clock, or broad architectural superiority claims.
