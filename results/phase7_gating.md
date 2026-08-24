# Phase 7 learned Bayesian gates

## Gate decision: PASS

- PASS: `learned_beta_improves_fixed_every_seed`
- PASS: `learned_beta_orders_reliability`
- PASS: `learned_lambda_beats_fixed_tradeoffs_every_seed`
- PASS: `learned_lambda_tracks_drift`
- PASS: `joint_beta_remains_ordered`
- PASS: `joint_gates_improve_fixed_every_seed`
- PASS: `joint_lambda_remains_drift_sensitive`

All models use the successful Phase 6 shared key/query feature chart. The evidence and drift experiments are completed and gated separately before joint training begins.

## Evidence precision beta

| method | risk | clean beta | noisy beta | corrupt beta | distractor beta |
| --- | ---: | ---: | ---: | ---: | ---: |
| beta_fixed_1 | 0.687 | 1.000 | 1.000 | 1.000 | 1.000 |
| beta_generic_value | 0.157 | 3.247 | 2.242 | 0.095 | 0.596 |
| beta_learned_precision | 0.031 | 2.941 | 0.692 | 0.047 | 0.158 |
| beta_oracle | 0.078 | 4.000 | 0.040 | 0.001 | 0.001 |

`beta_learned_precision` weights both sufficient statistics, as required by weighted least squares. `beta_generic_value` is an intentionally unconstrained control that weights only the cross-statistic. The gate sees noisy observable sensor-quality, consistency, and relevance cues—not the oracle category or target. Absolute beta has a global scale interaction with epsilon and the decoder, so ordering, oracle correlation, and risk are interpreted more strongly than raw magnitude.

## Drift lambda

| method | risk | post-change risk | stationary lambda | change lambda | drift corr |
| --- | ---: | ---: | ---: | ---: | ---: |
| lambda_fixed_1 | 0.609 | 1.253 | 1.000 | 1.000 | 0.000 |
| lambda_fixed_0.95 | 0.522 | 1.257 | 0.950 | 0.950 | 0.000 |
| lambda_fixed_0.8 | 0.354 | 1.163 | 0.800 | 0.800 | 0.000 |
| lambda_learned_cue | 0.167 | 0.646 | 0.901 | 0.018 | 0.870 |
| lambda_innovation_ablation | 0.247 | 0.875 | 0.765 | 0.358 | 0.513 |
| lambda_oracle_change | 0.158 | 0.653 | 1.000 | 0.010 | 1.000 |

The primary learned gate sees a noisy, locally observable drift cue, so it remains a token-emitted quantity compatible with the affine scan. `lambda_innovation_ablation` instead sees the detached pre-write innovation and posterior uncertainty. Its failure is retained because a single innovation is not a changepoint posterior. Exact causal inference of hidden changes requires an additional run-length belief state; lambda is the action conditional on such evidence, not the detector itself. The oracle receives the true change point. Fixed values expose the stationary/adaptation tradeoff. This distinction follows Bayesian online changepoint detection, which explicitly maintains a posterior over run length, and variable-forgetting-factor RLS, which treats forgetting adaptation as an additional mechanism rather than a consequence of naming a scalar lambda.

## Joint identifiability

| method | risk | clean beta | bad beta | stationary lambda | change lambda |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint_fixed | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 |
| joint_learned | 0.672 | 2.372 | 0.006 | 0.886 | 0.202 |
| joint_oracle | 0.634 | 4.000 | 0.001 | 1.000 | 0.010 |

In the joint experiment the lambda gate additionally sees evidence-quality cues, allowing it to distinguish a low-quality outlier from persistent operator drift. The tables report whether beta ordering and lambda drift sensitivity survive joint optimization; names alone are not treated as semantics.

## Failure study and mathematical basis

- Adams and MacKay, [Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742)
- Leung and So, [Gradient-based variable forgetting factor RLS](https://www.sciencedirect.com/science/article/pii/S0165168403000379)

The initial innovation-only run failed the lambda gate and correctly prevented joint training. A 120-step observable-cue probe recovered change response but missed the stationary-lambda threshold; the final run gives every drift and joint method the same 360-step budget without changing any gate threshold. These failed assumptions and probes are recorded in `EXPERIMENT_LOG.md`; their invalid partial artifacts are not gate inputs.

## Plots

- [`evidence_precision.png`](../plots/phase7/evidence_precision.png)
- [`drift_lambda.png`](../plots/phase7/drift_lambda.png)
- [`joint_gating.png`](../plots/phase7/joint_gating.png)

## Reproducibility

- commit at run start: `7e6d923a254c2e9c23485d97d389d83558d8110d`; dirty: `True`
- device: `AMD Instinct MI300X VF`; Python `3.12.3`; PyTorch `2.8.0+rocm7.0.2.git245bf6ed`
- wall time: 681.61s; peak VRAM: 0.402751 GiB
- config: [`configs/phase7_gating.json`](../configs/phase7_gating.json)
- raw seed rows: [`phase7/`](phase7/)
- machine-readable record: [`phase7_metrics.json`](phase7_metrics.json)

## Scoped conclusion

Learned precision and forgetting gates reproducibly improve risk over fixed gates while preserving the predicted reliability ordering and drift response; joint training retains both signals without claiming exact parameter identifiability.
