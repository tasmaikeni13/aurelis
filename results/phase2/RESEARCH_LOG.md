# Phase 2 research and repair log

## Primary Literature and Architectural Baseline Foundations

1. **Linear Attention and Recurrent Memory**:
   - Angelos Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention", ICML 2020.
     - Extracted mathematics: $\phi(x) = \text{elu}(x) + 1$, $S = \sum v_i \phi(k_i)^T$, $z = \sum \phi(k_i)$, $y = \frac{S \phi(q)}{z^T \phi(q)}$.
     - Design decision: Implemented as positive-feature global linear attention baseline without windowing.
   - Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers", ICML 2021; Songlin Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule", 2024.
     - Extracted mathematics: Delta rule update $S_t = \alpha S_{t-1} + \beta (v_t - S_{t-1} \bar{k}_t) \bar{k}_t^T$.
     - Design decision: Implemented as `delta_rule_memory` baseline with key normalization.

2. **Regression and Least-Squares Sequence Models**:
   - Johannes von Oswald et al., "Transformers learn in-context by gradient descent", ICML 2023; Johannes von Oswald et al., "Mesa: Transformers as Learned Optimizers", 2024.
     - Extracted mathematics: Cumulative regularized least squares $M = C P^{-1}$ over all consumed tokens.
     - Design decision: Implemented as `cumulative_least_squares_mesa` baseline without cache handoff.
   - Michael Du et al., "Native Hybrid Attention: Merging Recurrent Slots and Sliding Windows in Softmax Attention", 2025.
     - Extracted mathematics: Concatenated key-value matrices $[K_{\text{rec}}; K_{\text{local}}]$ attending jointly through a single softmax operation.
     - Design decision: Implemented as `native_hybrid_attention` baseline.

3. **Covariance Gating vs Independent Inverse-Variance Fusion**:
   - Markowitz Modern Portfolio Theory / Best Linear Unbiased Estimator (BLUE) with correlated estimators.
     - Extracted mathematics: For correlated estimators $y_1, y_2$ with covariance $K_{12}$, optimal weight $w_1 = \frac{V_2 - K_{12}}{V_1 + V_2 - 2 K_{12}}$. Omitting $K_{12}$ yields the suboptimal inverse-variance heuristic $w_{\text{indep}} = \frac{V_2}{V_1 + V_2}$.
     - Design decision: Formalized in Lean as `clippedGate_le_clippedIndependentGate` and tested in the correlated endpoint suite.

4. **Failure Dispositions and Evaluator Repairs**:
   - Recorded in `results/phase2/failures/torch_randn_like_generator_api_20260904.md`: PyTorch 2.8 on ROCm `randn_like` generator API repair.
   - Recorded in `results/phase2/failures/linear_reproduction_prior_shrinkage_20260904.md`: Linear reproduction premise $M = W$ isolated from finite-ridge shrinkage by using noise-free full-rank remote prefix with prior floor $10^{-12}$.
   - Recorded in `results/phase2/failures/softmax_temperature_one_hot_precision_20260904.md`: Certified exception copy tested with temperature $\tau \ge 2048.0$ matching the one-hot assumption of Corollary 5.4.
