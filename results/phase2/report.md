# AURELIS Phase 2 Evaluation Report

Generated: `2026-09-04T05:23:38.925193+00:00`
Status: **PASS**

## Gates and Findings

1. **Baseline equation tests**:
   All 10 baselines (local softmax, remote Bayes ridge, global linear attention, delta-rule memory, cumulative least-squares Mesa, learned sum/concat, independent inverse-variance fusion, full residual $g=1$, AURELIS-B/E, and Native Hybrid Attention) passed exact equation tests against hand-computed values.

2. **Linear reproduction isolation**:
   Linear reproduction error for full residual is `8.579e-12` across all tested temperatures $\tau \in [0.01, 10.0]$, while local softmax smoothing error remains large (`2.4518`).

3. **Gaussian regime advantage**:
   AURELIS-B demonstrates lower MSE than local attention across all 10 independent random seeds.

4. **AURELIS-E episodic exception isolation**:
   AURELIS-E achieves `9.930e-16` error on certified cached exceptions, while AURELIS-B appropriately shrinks the exception toward the latent target.

5. **Retained nonlinear regime without advantage**:
   10 nonlinear cases retained and verified where local attention outperforms AURELIS, proving linear transport assumptions fail under misspecification.

6. **Capacity limits**:
   Local attention collapses when sequence length exceeds local window $w=32$, while remote linear memory collapses beyond rank $d_k=16$.

7. **Full covariance vs independence heuristic**:
   Full covariance Bayes gate achieves zero-regret optimality ($V(g_B) \le V(g_{\text{indep}})$ everywhere) with paired regret $z$-score of `9.07` (threshold $\ge 5.0$).
