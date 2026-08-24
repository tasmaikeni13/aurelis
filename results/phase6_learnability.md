# Phase 6 learned feature maps

## Gate decision: PASS

- PASS: `learned_csm_learns_every_discrete_task_across_seeds`
- PASS: `learned_csm_learns_regression_tasks_across_seeds`
- PASS: `learned_csm_outperforms_random_features_every_seed`
- PASS: `natural_geometry_uses_nontrivial_capacity`

No general language model is trained. Every row is a small synthetic episodic model with `beta=lambda=1`. The primary sweep has no geometry regularizer; the regularizer ablation was executed only after all natural runs completed.

| Gate measurement | Observed |
|---|---:|
| minimum learned-CSM discrete success across tasks/seeds | 0.959 |
| maximum learned-CSM regression normalized MSE | 0.312 |
| learned/random aggregate risk ratios by seed | 0.208, 0.155, 0.136 |
| minimum natural effective-capacity fraction | 0.304 |

## Task and baseline results

| task | method | normalized MSE | success | effective rank | condition |
| --- | ---: | ---: | ---: | ---: | ---: |
| associative_recall | fixed_random_csm | 0.649 | 0.634 | 1.80 | 7.13e+01 |
| associative_recall | learned_attention | 0.055 | 0.983 | 5.60 | 2.67e+01 |
| associative_recall | learned_csm | 0.098 | 0.983 | 5.88 | 2.53e+01 |
| associative_recall | learned_hebbian | 0.516 | 0.734 | 5.81 | 2.57e+01 |
| contextual_associative_recall | fixed_random_csm | 0.988 | 0.133 | 3.34 | 5.17e+01 |
| contextual_associative_recall | learned_attention | 0.028 | 0.992 | 4.88 | 2.83e+01 |
| contextual_associative_recall | learned_csm | 0.040 | 0.992 | 7.17 | 2.41e+01 |
| contextual_associative_recall | learned_hebbian | 0.719 | 0.476 | 3.98 | 2.90e+01 |
| correlated_key_lookup | fixed_random_csm | 0.792 | 0.385 | 1.19 | 7.89e+01 |
| correlated_key_lookup | learned_attention | 0.028 | 0.990 | 2.45 | 6.27e+01 |
| correlated_key_lookup | learned_csm | 0.162 | 0.967 | 2.45 | 6.28e+01 |
| correlated_key_lookup | learned_hebbian | 0.833 | 0.201 | 2.35 | 6.24e+01 |
| in_context_linear_regression | fixed_random_csm | 0.752 | — | 2.88 | 5.26e+01 |
| in_context_linear_regression | learned_attention | 0.687 | — | 4.20 | 3.91e+01 |
| in_context_linear_regression | learned_csm | 0.213 | — | 4.27 | 3.68e+01 |
| in_context_linear_regression | learned_hebbian | 0.567 | — | 4.35 | 3.22e+01 |
| key_value_lookup | fixed_random_csm | 0.634 | 0.637 | 1.69 | 7.34e+01 |
| key_value_lookup | learned_attention | 0.041 | 0.979 | 5.77 | 2.37e+01 |
| key_value_lookup | learned_csm | 0.041 | 0.979 | 8.00 | 1.15e+01 |
| key_value_lookup | learned_hebbian | 0.223 | 0.967 | 7.98 | 1.23e+01 |
| noisy_in_context_regression | fixed_random_csm | 0.745 | — | 3.17 | 4.89e+01 |
| noisy_in_context_regression | learned_attention | 0.673 | — | 4.27 | 3.83e+01 |
| noisy_in_context_regression | learned_csm | 0.297 | — | 4.48 | 3.19e+01 |
| noisy_in_context_regression | learned_hebbian | 0.535 | — | 4.34 | 3.23e+01 |
| selective_copy | fixed_random_csm | 0.552 | 0.835 | 1.60 | 7.31e+01 |
| selective_copy | learned_attention | 0.009 | 1.000 | 5.94 | 2.17e+01 |
| selective_copy | learned_csm | 0.001 | 1.000 | 7.99 | 1.16e+01 |
| selective_copy | learned_hebbian | 0.518 | 0.863 | 6.79 | 1.48e+01 |

`fixed_random_csm` freezes tied random key/query features and a random value map while training only its output decoder, making it a favorable exact-query random-feature control. Hebbian and attention models receive learned encoders of the same size. Attention is not expected to lose on every task; the scientific gate is whether learned CSM features beat their own untrained/random representation and solve all tasks reproducibly.

### Matched-coordinate correction

The natural architecture uses one shared feature map for keys and queries plus one learned positive scalar query calibration. This is not an orthogonality constraint: the loss remains retrieval error alone, and the Gram spectrum is free to emerge. It enforces the mathematical requirement that a query evaluate the ridge operator in the coordinate chart in which it was fitted. The post-natural ablation restores independent key/query maps while holding every other setting fixed.

| task | shared success | independent success | shared rank | independent rank |
| --- | ---: | ---: | ---: | ---: |
| contextual_associative_recall | 0.992 | 0.376 | 7.17 | 2.88 |

The initial diagnostic implementation normalized two independent encoders and therefore violated this compatibility condition. Its selective-copy scorer additionally confused symbol identity with support-slot identity. Those diagnostic outputs are not used by this report or its gate.

## Natural representation geometry

Every seed-level row records the full mean Gram eigenvalue spectrum, pairwise cosine statistics, effective rank, minimum singular value, `cond(S+epsilon I)`, nominal and effective capacity fractions, gradient norms, epsilon, query scale, and retrieval error. Effective capacity is `effective_rank / min(K, d_key)`, the reachable rank of a `K`-association episode. The primary learned CSM results above are entirely unassisted by orthogonality loss.

### Explicit post-hoc regularizer ablation

| task | natural MSE | regularized MSE | natural cosine | regularized cosine |
| --- | ---: | ---: | ---: | ---: |
| correlated_key_lookup | 0.162 | 0.165 | 0.739 | 0.738 |
| contextual_associative_recall | 0.040 | 0.278 | 0.166 | 0.140 |

This ablation is diagnostic, not part of the pass gate. If it is required to rescue a task, that fact is a limitation rather than evidence that gradient descent naturally found good geometry.

## Plots

- [`task_performance.png`](../plots/phase6/task_performance.png)
- [`learning_curves.png`](../plots/phase6/learning_curves.png)
- [`natural_geometry.png`](../plots/phase6/natural_geometry.png)
- [`regularizer_ablation.png`](../plots/phase6/regularizer_ablation.png)

## Reproducibility

- source checkpoint: `7e6d923a254c2e9c23485d97d389d83558d8110d`; dirty at experiment start: `True`
- config: [`configs/phase6_learnability.json`](../configs/phase6_learnability.json)
- device: `AMD Instinct MI300X VF`
- software: Python `3.12.3`, PyTorch `2.8.0+rocm7.0.2.git245bf6ed`, HIP `7.0.51831-7c9236b16`, NumPy `2.3.2`
- wall time: 89.854 seconds; peak allocated VRAM: 0.401106 GiB
- seed metrics: [`phase6/seed_metrics.csv`](phase6/seed_metrics.csv)
- learning curves: [`phase6/learning_curves.csv`](phase6/learning_curves.csv)
- machine-readable record: [`phase6_metrics.json`](phase6_metrics.json)

## Failure study and mathematical basis

The correction follows differentiable ridge meta-learning, which applies one feature extractor to both support and held-out examples and backpropagates through the closed-form solver. It is also the construction required by deep-kernel learning: one learned feature map defines both arguments of a kernel. Representation-collapse work motivates the separately labeled regularizer ablation, but no covariance or orthogonality penalty enters the natural gate.

- Bertinetto et al., [Meta-learning with differentiable closed-form solvers](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/bertinetto19meta-learning.pdf)
- Lee et al., [Meta-Learning With Differentiable Convex Optimization](https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Meta-Learning_With_Differentiable_Convex_Optimization_CVPR_2019_paper.html)
- Wilson et al., [Deep Kernel Learning](https://proceedings.mlr.press/v51/wilson16.html)
- Bardes et al., [VICReg](https://arxiv.org/abs/2105.04906)

## Scoped conclusion

Across every seed, the unregularized learned CSM solves all seven synthetic tasks and improves aggregate risk over its frozen random-feature control. The geometry tables state whether this happened through naturally separated keys; regularized rows are post-hoc ablations only.
