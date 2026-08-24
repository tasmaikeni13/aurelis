# Phase 4 uncertainty and noisy evidence

## Gate decision: PASS

- PASS: `csm_matches_oracle_ridge`
- PASS: `gaussian_duplicates_follow_inverse_n_risk`
- PASS: `known_precision_reduces_risk`
- PASS: `gaussian_uncertainty_tracks_error`
- PASS: `gaussian_uncertainty_detects_high_error`
- PASS: `gaussian_uncertainty_is_calibrated`
- PASS: `gaussian_selective_prediction_reduces_risk`
- PASS: `unseen_directions_increase_uncertainty`

The pass decision applies only to the declared linear-Gaussian model. Laplace, Student-like scale-mixture, and nonlinear rows are stress tests whose degradation is reported, not evidence for Bayesian optimality.

| Gate measurement | Observed |
|---|---:|
| maximum CSM/oracle-ridge relative difference | 5.535e-16 |
| repeated-evidence log-risk slope (median across sigma) | -1.018 |
| uniform / precision-weighted Gaussian risk | 23.206x |
| Gaussian uncertainty/error Spearman | 0.704 |
| Gaussian high-error AUROC | 0.875 |
| Gaussian actual/predicted MSE | 0.957 |
| roughly half-coverage / full-coverage risk | 0.404 |
| minimum uncertainty increment toward unseen span | 5.556e-02 |

## A. Noisy duplicates and conflicting observations

Every dataset stores 8 orthogonal associations repeatedly, varies observation noise, and evaluates the latent (noise-free) value. Softmax receives oracle temperature selection from the committed grid, which favors it. Oracle ridge is independently assembled from the weighted normal equations. At `sigma=0.1`:

| model | method | n=1 MSE | n=16 MSE | risk ratio |
| --- | ---: | ---: | ---: | ---: |
| linear_gaussian | csm | 4.001e-02 | 2.313e-03 | 17.30x |
| linear_gaussian | simple_average | 4.001e-02 | 2.313e-03 | 17.30x |
| linear_gaussian | hebbian | 4.001e-02 | 5.385e+01 | 0.00x |
| linear_gaussian | softmax | 3.998e-02 | 2.311e-03 | 17.30x |
| linear_gaussian | oracle_ridge | 4.001e-02 | 2.313e-03 | 17.30x |
| laplace | csm | 3.740e-02 | 2.310e-03 | 16.19x |
| laplace | simple_average | 3.740e-02 | 2.310e-03 | 16.19x |
| laplace | hebbian | 3.740e-02 | 5.336e+01 | 0.00x |
| laplace | softmax | 3.737e-02 | 2.307e-03 | 16.20x |
| laplace | oracle_ridge | 3.740e-02 | 2.310e-03 | 16.19x |
| student_like | csm | 3.677e-02 | 2.701e-03 | 13.61x |
| student_like | simple_average | 3.677e-02 | 2.701e-03 | 13.61x |
| student_like | hebbian | 3.677e-02 | 5.462e+01 | 0.00x |
| student_like | softmax | 3.501e-02 | 2.698e-03 | 12.97x |
| student_like | oracle_ridge | 3.677e-02 | 2.701e-03 | 13.61x |
| nonlinear | csm | 3.792e-02 | 2.620e-03 | 14.47x |
| nonlinear | simple_average | 3.792e-02 | 2.620e-03 | 14.47x |
| nonlinear | hebbian | 3.792e-02 | 1.640e+02 | 0.00x |
| nonlinear | softmax | 3.789e-02 | 2.616e-03 | 14.48x |
| nonlinear | oracle_ridge | 3.792e-02 | 2.620e-03 | 14.47x |

In the Gaussian rows, CSM's empirical risk follows the predicted `1/n` law and matches ridge to fp64 resolution. Simple averaging is nearly identical in this orthogonal repeated-key special case. Hebbian memory sums duplicates instead of averaging them and therefore becomes worse as repeats grow. The explicit conflicting pair has precisions 25 and 1; its complete estimator-to-consensus distances are retained in `conflicting_observations.csv`.

## B. Known beta precision

Known inverse variances are passed as `beta`. Across all Gaussian precision patterns, uniform weighting has 23.206 times the aggregate risk of precision-weighted CSM. The raw record also includes Laplace and Student-like noise. This establishes the expected weighted least-squares behavior; it does not show that a learned gate will infer correct precisions.

## C. Confidence, calibration, high-error detection, and abstention

Training keys span 8 of 16 key dimensions. Queries move continuously from that observed subspace into its orthogonal complement. `c(q)` is treated as uncertainty (larger means less confident), and predicted latent squared error is `d_value * c(q)`.

| data model | Spearman | high-error AUROC | actual/predicted MSE | 95% coordinate coverage | p95 normalized error |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear_gaussian | 0.704 | 0.875 | 0.957 | 0.952 | 2.302 |
| laplace | 0.710 | 0.875 | 0.921 | 0.956 | 2.278 |
| student_like | 0.717 | 0.873 | 1.034 | 0.944 | 2.559 |
| nonlinear | 0.745 | 0.888 | 1.164 | 0.934 | 2.630 |

The Gaussian calibration ratio and 95% coordinate coverage are in-model checks. The non-Gaussian rows preserve equal variance but expose tail sensitivity through coverage and p95 normalized error. The nonlinear row violates the latent linear-map assumption; its calibration/ranking values characterize that misspecification and are not relabeled as Bayesian guarantees.

Selective prediction retains queries with the smallest `c(q)`. The committed curves report risk at retained fractions `[1.0, 0.8, 0.6, 0.5, 0.4, 0.2]`; this is regression risk rather than a classification "accuracy" surrogate.

## D. Missing and out-of-distribution directions

| outside-span component | mean c(q) | Gaussian mean squared error |
| --- | ---: | ---: |
| 0.00 | 0.1111 | 0.4401 |
| 0.25 | 0.1667 | 0.6634 |
| 0.50 | 0.3333 | 1.3026 |
| 0.75 | 0.6111 | 2.3417 |
| 1.00 | 1.0000 | 3.7547 |

The uncertainty increase is algebraic and remains present for every value/noise model because it depends only on the key statistic `S`. Only the Gaussian correspondence between its magnitude and prediction risk is an optimal/calibrated claim.

## Plots

- [`noisy_duplicates.png`](../plots/phase4/noisy_duplicates.png)
- [`calibration.png`](../plots/phase4/calibration.png)
- [`unseen_directions.png`](../plots/phase4/unseen_directions.png)
- [`selective_prediction.png`](../plots/phase4/selective_prediction.png)
- [`uncertainty_ranking.png`](../plots/phase4/uncertainty_ranking.png)

## Reproducibility

- source checkpoint: `a8ad6e59db43ee61adadf4888d69edcb9ac705c5`; dirty at experiment start: `True`
- config: [`configs/phase4_uncertainty.json`](../configs/phase4_uncertainty.json)
- device: `AMD Instinct MI300X VF`
- software: Python `3.12.3`, PyTorch `2.8.0+rocm7.0.2.git245bf6ed`, HIP `7.0.51831-7c9236b16`, NumPy `2.3.2`
- wall time: 19.907 seconds; peak allocated VRAM: 0.125083 GiB
- raw data: [`phase4/noisy_duplicates.csv`](phase4/noisy_duplicates.csv), [`phase4/precision_weighting.csv`](phase4/precision_weighting.csv), [`phase4/conflicting_observations.csv`](phase4/conflicting_observations.csv), [`phase4/confidence_queries.csv`](phase4/confidence_queries.csv)
- derived data: [`phase4/calibration.csv`](phase4/calibration.csv), [`phase4/selective_prediction.csv`](phase4/selective_prediction.csv), [`phase4/unseen_directions.csv`](phase4/unseen_directions.csv)
- machine-readable record: [`phase4_metrics.json`](phase4_metrics.json)

## Scoped conclusion

Within the exactly linear-Gaussian model, CSM matches weighted ridge, follows inverse-repeat risk, and its posterior variance is calibrated and useful for ranking and selective prediction. Under heavy-tailed and nonlinear misspecification the same score remains an algebraic coverage diagnostic, but tail and calibration degradation in the tables must replace any Bayesian-optimality claim.
