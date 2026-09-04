# Phase 3 research and repair log

## Primary Literature and Architectural Baseline Foundations

1. **Effective Rank and Representation Dimensionality**:
   - Olivier Roy and Martin Vetterli, "The Effective Rank: A Measure of Effective Dimensionality", EUSIPCO 2007.
     - Extracted mathematics: For singular values $\sigma_1 \ge \dots \ge \sigma_r > 0$, define $p_i = \sigma_i / \sum_{j} \sigma_j$. Shannon entropy $H(p) = -\sum_i p_i \ln p_i$. Effective rank is $\text{erank}(A) = \exp(H(p)) \in [1, \min(m, n)]$.
     - Design decision: Used to measure representation preservation and identify rank collapse in shared vs independent chart ablations without ad-hoc orthogonality penalties.

2. **Gram Metric Symmetry in Shared Feature Charts**:
   - In AURELIS, keys and queries are projected through a shared chart $W_{kq} \in \mathbb{R}^{d_{\text{model}} \to heads \times d_k}$.
     - Extracted mathematics: Inner product $q^T k = x_q^T (W_{kq}^T W_{kq}) x_k$ induces a symmetric positive semidefinite metric $M = W_{kq}^T W_{kq} \succeq 0$. Remote precision $P = \lambda I + \sum k_t k_t^T = \lambda I + W_{kq} (\sum x_t x_t^T) W_{kq}^T$ shares the exact column space of $q = W_{kq} x_q$.
     - Ablation mechanism: Independent charts $W_k \ne W_q$ produce asymmetric, indefinite transport $W_q^T W_k$ and project queries into nullspaces of $P$, resulting in representation collapse and elevated risk.

3. **Subgradient Flat Plateau and Straight-Through Estimator for Episodic Gate**:
   - Stephen Boyd and Lieven Vandenberghe, "Convex Optimization", Cambridge University Press, 2004; Yoshua Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation", 2013.
     - Extracted mathematics: Hard maximum $g_E = \max(g_B, e_t)$ yields subgradient $\partial_{e_t} \max(g_B, e_t) = 0$ whenever $e_t < g_B$. With standard initialization ($e_t \approx 0.1 < g_B$), gradient descent encounters a dead zone where $e_t$ receives zero gradient.
     - Mathematical repair: Straight-Through Estimator (STE) combining exact hard forward execution $g_E^{\text{hard}} = \max(g_B, e_t)$ with smooth gradient propagation $g_E^{\text{soft}} = g_B + (1 - g_B) e_t$. The forward pass is 100% faithful to Eq. 4.4 and Lean `episodicGate`, while the backward pass supplies strictly positive gradient $\partial_{e_t} g_E = 1 - g_B > 0$.

4. **Task Identifiability and Observable Cue Signaling**:
   - The sequence presents an explicit observable cue bit in input channel $X[t, -1]$ (1.0 for episodic exception queries, 0.0 for latent/trend queries).
     - Extracted mathematics: Episodic responsibility $e_t = \sigma(W_e h_t)$ is trained with auxiliary BCE loss against the observable channel $X[t, -1]$. This directly fulfills Gate 5: the override is governed by explicit observable features rather than hidden evaluator labels.

5. **Cache Overlap Double-Counting Redundancy**:
   - Formalized in Lean 4 as `cache_overlap_redundancy`: counting tokens in both recent window $w$ and remote store strictly inflates representation history length, proving that delayed handoff ($Take\ w$ and $Drop\ w$) is the unique partition preserving token conservation.
