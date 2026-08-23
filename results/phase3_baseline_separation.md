# Phase 3 baseline separation

## Gate decision: PASS

- PASS: `compressed_correlated_advantage`
- PASS: `equal_memory_nonconvex_functional_fidelity`
- PASS: `equal_memory_softmax_separation`
- PASS: `softmax_convex_hull_constraint`

The same-dimension regime gives every method the same `d_key` and `d_value`; explicit methods may therefore use state growing with `K`. The equal-state-budget regime caps explicit key/value storage at the byte count of dense fp64 CSM `S` and `C`. Hebbian and linear attention use every write but consume less than that maximum. In the linear-functional experiment `K = d_key = d_value`, so CSM and explicit softmax each use exactly `16 * d_key^2` bytes.

| Gate measurement | Observed | Required |
|---|---:|---:|
| correlated compressed-baseline / CSM median error ratio | 2.427e+06 | >= 100.0 |
| equal-memory nonconvex CSM median absolute error | 1.399e-07 | <= 1.0e-05 |
| equal-memory softmax / CSM nonconvex median error ratio | 4.496e+06 | >= 1.0e+04 |
| minimum softmax error minus convex-hull distance | 0.000e+00 | >= -1.0e-10 |
| minimum softmax weight | 0.000e+00 | >= -1.0e-10 |
| maximum softmax weight-sum error | 2.220e-16 | <= 1.0e-10 |

## Associative recall

The table aggregates the minimum-epsilon, at-or-below-capacity cases under the equal byte budget. Softmax receives oracle selection over all committed temperatures for each dataset; this is intentionally favorable to it. Least squares is an explicit-pair oracle, not a compressed practical winner.

| key regime | method | median error | p90 error |
| --- | ---: | ---: | ---: |
| correlated | csm | 3.934e-07 | 6.737e-05 |
| correlated | hebbian | 3.714e+00 | 6.529e+00 |
| correlated | least_squares | 1.041e-14 | 4.599e-13 |
| correlated | linear_attention | 9.696e-01 | 9.922e-01 |
| correlated | softmax | 0.000e+00 | 0.000e+00 |
| near_collinear | csm | 1.343e-03 | 1.016e-01 |
| near_collinear | hebbian | 3.800e+00 | 7.729e+00 |
| near_collinear | least_squares | 2.317e-11 | 2.513e-09 |
| near_collinear | linear_attention | 9.704e-01 | 9.915e-01 |
| near_collinear | softmax | 0.000e+00 | 0.000e+00 |
| random_gaussian | csm | 4.438e-08 | 1.444e-05 |
| random_gaussian | hebbian | 7.928e-01 | 1.016e+00 |
| random_gaussian | least_squares | 7.800e-16 | 1.935e-14 |
| random_gaussian | linear_attention | 9.697e-01 | 9.917e-01 |
| random_gaussian | softmax | 0.000e+00 | 0.000e+00 |

The solver has a reproducible fidelity advantage over the compressed Hebbian and positive-feature linear-attention states for correlated keys. It does **not** dominate explicit retrieval: for random independent stored-key recall, median errors are CSM `4.438e-08`, oracle-tuned softmax `0.000e+00`, and least squares `7.800e-16`. At `K <= d_key`, explicit softmax fits the equal budget and is often the better stored-key mechanism. Above capacity, arbitrary-value recall fails for CSM and the least-squares projection, while budgeted explicit methods also omit pairs; no broad winner is claimed there.

### Value-dimension sweep

The following isolates correlated keys at or below capacity, minimum epsilon, and the equal state-byte budget. All three committed value dimensions are reported rather than averaged away.

| d_value | method | median error | p90 error |
| --- | ---: | ---: | ---: |
| 4 | csm | 4.591e-07 | 3.964e-05 |
| 4 | hebbian | 3.697e+00 | 6.172e+00 |
| 4 | least_squares | 1.010e-14 | 2.285e-13 |
| 4 | linear_attention | 9.739e-01 | 9.947e-01 |
| 4 | softmax | 0.000e+00 | 0.000e+00 |
| 16 | csm | 3.829e-07 | 8.707e-05 |
| 16 | hebbian | 3.805e+00 | 6.877e+00 |
| 16 | least_squares | 1.115e-14 | 6.803e-13 |
| 16 | linear_attention | 9.667e-01 | 9.895e-01 |
| 16 | softmax | 0.000e+00 | 0.000e+00 |
| 64 | csm | 4.018e-07 | 7.666e-05 |
| 64 | hebbian | 3.519e+00 | 6.101e+00 |
| 64 | least_squares | 1.087e-14 | 5.834e-13 |
| 64 | linear_attention | 9.714e-01 | 9.904e-01 |
| 64 | softmax | 0.000e+00 | 0.000e+00 |

## Linear-functional separation

Values are the standard basis, so the target is exactly `alpha`. Every normalized softmax output is therefore a simplex point. Negative coefficients, coefficients above one, and sums other than one put the target outside that convex hull; their Euclidean simplex-projection distance is a method-independent lower bound on softmax error. CSM and least squares can instead produce signed linear-span coefficients.

| coefficients | method | median absolute error | hull distance |
| --- | ---: | ---: | ---: |
| coefficient_gt_one | csm | 9.314e-08 | 5.590e-01 |
| coefficient_gt_one | hebbian | 3.904e+00 | 5.590e-01 |
| coefficient_gt_one | linear_attention | 1.495e+00 | 5.590e-01 |
| coefficient_gt_one | softmax | 5.590e-01 | 5.590e-01 |
| mixed_nonunit | csm | 6.873e-07 | 1.969e+00 |
| mixed_nonunit | hebbian | 4.376e+00 | 1.969e+00 |
| mixed_nonunit | linear_attention | 2.609e+00 | 1.969e+00 |
| mixed_nonunit | softmax | 1.969e+00 | 1.969e+00 |
| negative_coefficients | csm | 2.966e-07 | 6.481e-01 |
| negative_coefficients | hebbian | 1.507e+00 | 6.481e-01 |
| negative_coefficients | linear_attention | 1.097e+00 | 6.481e-01 |
| negative_coefficients | softmax | 7.631e-01 | 6.481e-01 |
| positive_simplex | csm | 1.245e-08 | 2.776e-17 |
| positive_simplex | hebbian | 1.836e+00 | 2.776e-17 |
| positive_simplex | linear_attention | 9.884e-02 | 2.776e-17 |
| positive_simplex | softmax | 9.586e-02 | 2.776e-17 |
| positive_sum_two | csm | 2.491e-08 | 1.844e-01 |
| positive_sum_two | hebbian | 3.671e+00 | 1.844e-01 |
| positive_sum_two | linear_attention | 2.652e-01 | 1.844e-01 |
| positive_sum_two | softmax | 2.610e-01 | 1.844e-01 |

The positive-simplex case is included as a no-structural-separation control: its hull distance is zero, so the convexity argument alone predicts no CSM advantage. The equal-memory claim is limited to the characterized nonconvex coefficient classes.

## State, operations, and latency

The following is the `d_key=64`, `d_value=16`, `K=64` prepared-read measurement. CSM timing uses a precomputed Cholesky factor, just as least squares uses a precomputed pseudoinverse; neither derived factor is counted as recurrent state. FLOPs are leading-operation estimates, not profiler counts. GPU timings use batches of 256 queries and are latency diagnostics rather than optimized-kernel claims.

| method | state bytes | estimated FLOPs/query | measured us/query |
| --- | ---: | ---: | ---: |
| csm | 40960 | 10240 | 3.399 |
| hebbian | 8192 | 2048 | 0.088 |
| softmax | 40960 | 10560 | 0.167 |
| linear_attention | 8704 | 2256 | 0.196 |
| least_squares | 40960 | 10240 | 0.117 |

## Plots

- [`recall_vs_load_same_dimension.png`](../plots/phase3/recall_vs_load_same_dimension.png)
- [`recall_vs_load_equal_state_budget.png`](../plots/phase3/recall_vs_load_equal_state_budget.png)
- [`csm_epsilon_sweep.png`](../plots/phase3/csm_epsilon_sweep.png)
- [`linear_functional_separation.png`](../plots/phase3/linear_functional_separation.png)
- [`prepared_read_latency.png`](../plots/phase3/prepared_read_latency.png)

## Reproducibility

- git source checkpoint: `e1c592feb09dc769477b3e8fd566ac07a55e9e0e`
- working tree dirty at experiment start: `False`
- config: [`configs/phase3_baselines.json`](../configs/phase3_baselines.json)
- seeds: `[0, 1, 2]`; deterministic regime-specific mixing is in source
- hardware: `AMD Instinct MI300X VF`
- software: Python `3.12.3`, PyTorch `2.8.0+rocm7.0.2.git245bf6ed`, HIP `7.0.51831-7c9236b16`, NumPy `2.3.2`
- wall-clock time: `23.320` seconds
- peak allocated VRAM: `0.126283` GiB
- complete recall rows: [`phase3/associative_recall.csv`](phase3/associative_recall.csv)
- complete linear-functional rows: [`phase3/linear_functional.csv`](phase3/linear_functional.csv)
- latency rows: [`phase3/latency.csv`](phase3/latency.csv)
- machine-readable summary: [`phase3_metrics.json`](phase3_metrics.json)

## Scoped conclusion

A solver fidelity advantage survives the equal-state-byte comparison in the stated correlated compressed-memory and nonconvex linear-functional regimes. Explicit softmax and least squares match or beat CSM in important stored-key regimes, so the result is a separation, not universal superiority.

This is a synthetic, unlearned comparison. It does not establish an NLP, throughput, learned-representation, or universal memory advantage.
