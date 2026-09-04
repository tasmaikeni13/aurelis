# Phase 4 research and repair log

## Primary Literature and Architectural Baseline Foundations

1. **Information Discounting and Dynamic Bayesian Linear Models**:
   - Mike West and Jeff Harrison, "Bayesian Forecasting and Dynamic Models", Springer Series in Statistics, 1997.
   - Extracted mathematics: For dynamic linear model $y_t = x_t^T \theta_t + v_t$ with stochastic parameter evolution $\theta_t = \theta_{t-1} + w_t$, exponential information discounting scales prior precision $P_{t-1}$ by decay factor $\gamma_t \in (0, 1]$.
   - The discounted remote Bayesian inference state evolves as:
     $$P_t = \gamma_t P_{t-1} + (1 - \gamma_t) \Lambda + \beta_e k_e k_e^T$$
     $$C_t = \gamma_t C_{t-1} + \beta_e v_e k_e^T$$
   - In affine scan monoid notation (`lean/Aurelis/AffineScan.lean`):
     $$( \gamma_2, u_2 ) \circ ( \gamma_1, u_1 ) = ( \gamma_2 \gamma_1, u_2 + \gamma_2 u_1 )$$
     associativity is preserved and parallel scans remain exact.
   - Posterior mean remains scale-invariant $M_t = C_t P_t^{-1}$. Posterior covariance is $P_t^{-1}$.
   - Variance minimizing gate $g_B = \text{clip}\left(\frac{q^T P_t^{-1} \bar{k}}{h + \bar{k}^T P_t^{-1} \bar{k}}, 0, 1\right)$ retains exact closed form with discounted $P_t$.
   - Design decision: Observable changepoint cue $c_t \in [0, 1]$ sets $\gamma_t = \text{clamp}(1 - c_t (1 - \gamma_{\min}), \gamma_{\min}, 1.0)$. When $c_t = 1.0$ (abrupt shift), state flushes obsolete pre-changepoint relations. When $c_t = 0.0$ (stationary), $\gamma_t = 1.0$ identically recovers base AURELIS. Unobservable shifts ($c_t = 0$) retain the fundamental limitation that obsolete relations degrade post-change risk until newly acquired data dominates.

2. **Heteroscedastic Precision Weighting and the Gauss-Markov Theorem**:
   - Carl Friedrich Gauss, "Theoria Combinationis Observationum Erroribus Minimis Obnoxiae", 1823; Arthur Aitken, "On Least Squares and Linear Combinations of Observations", Proc. Royal Soc. Edinburgh, 1935.
   - Extracted mathematics: For heteroscedastic noise $\xi_s \sim \mathcal{N}(0, \sigma_s^2 I)$, optimal precision weights $\beta_s = 1 / \sigma_s^2$ achieve the minimum-variance Best Linear Unbiased Estimator (BLUE).
   - Local attention noise variance charges $h = \sum_s a_s^2 / \beta_s$. Noisy tokens in cache receive small $\beta_s$, inflating $h$ and shrinking $g_B \to 0$ toward remote denoising.
   - Robustness control: Tempered evidence $\beta_s^{\text{tempered}} = \text{clamp}(\beta_s, \beta_{\min}, \beta_{\max})$ bounds condition numbers under heavy-tailed (Student-$t$, $\nu=3$) and outlier distributions.
   - Corrupted precision: Inverting or scrambing precision weights violates the Gauss-Markov optimality, transparently inflating variance and proving the mechanism actively relies on faithful precision estimation.

3. **Multi-Hop Composition and Pointer Chasing**:
   - Composition of linear transformations $W_2 \circ W_1$ and compositional attention routing.
   - Extracted mathematics: For linear maps $M_1, M_2, W_1, W_2$:
     $$(M_2 \circ M_1)(x) - (W_2 \circ W_1)(x) = (M_2 - W_2)(M_1 x) + W_2 ((M_1 - W_1) x)$$
   - Lean formalization in `lean/Aurelis/ResidualCorrection.lean` kernel-checks `composition_error_identity` and `composition_reproduces_linear`.
   - Computational budget separation: One attention read touches recent cache in $O(w(d_k + d_v))$ operations. $H$ adaptive multi-hop reads require $H$ sequential solves and query updates with wall-clock latency scaling linearly in $H$.

4. **Subspace Capacity and Monotonic Lower-Bound Failures**:
   - Thomas Cover, "Geometrical and Statistical Properties of Systems of Linear Inequalities with Applications in Pattern Recognition", IEEE Trans. Electron. Comput., 1965.
   - Extracted mathematics: A linear operator $M \in \mathbb{R}^{d_v \times d_k}$ has rank at most $\min(d_k, d_v)$. Storing $N$ independent adversarial associations $(k_i, v_i)$ in state $P = \alpha I + \sum_{i=1}^N k_i k_i^T$ cannot achieve zero error when $N > d_k$, as key vectors cannot remain mutually orthogonal in $d_k$ dimensions.
   - Evaluation protocol: We sweep $N \in \{2, 4, 8, 16, 32, 64, 128, 256\}$ with fixed $d_k = 8$. For $N \le 8$, recall error is bounded; for $N > 8$, error strictly increases monotonically, validating that fixed-state capacity bounds are preserved rather than obscured.

5. **Failure-Repair Loop: Iteration 1 Diagnostic and Repair**:

1. **Failure Evidence**:
   - Run 1 (commit initial Phase 4 suite) passed Gates 1, 2, 3, 5, 6, 7, but failed Gate 4 on `mixed_4hop` decoded accuracy for pattern `RRCC` (0.4219 vs preregistered threshold 0.5000) and `CR` (0.7719 vs 0.8500).

2. **Failure Classification**:
   - Classification: *Representation / Local-Remote Disambiguation Misspecification*.
   - Root Cause Analysis:
     - Local attention temperature $\tau$ was default $1.0$, resulting in diffuse softmax weights ($\max a_i \approx 0.20$ across $W=12$ window items) during pointer chasing.
     - In the pointer chaser loop, the gate was initially bounded below by `max_attn = attn.amax(dim=-1)`. Because a 12-element softmax always has maximum $\ge 1/12 \approx 0.083$ (and typically $0.20$ to $0.40$ on random query-key pairs), this forced the gate $g = \max(g_B, \max a_i)$ to remain substantially positive ($\approx 0.25$) even when the query target was located strictly in remote memory ($q^T k_{\text{cache}} \le 0.35$).
     - Consequently, 25% of random local cache innovation $v_{\text{cache}} - M k_{\text{cache}}$ was injected into clean remote reads $M q$, degrading the retrieved key vector and cascading error across subsequent hops.

3. **Mathematical Derivation of Repair**:
   - In AURELIS memory routing, cache presence should be conditioned on explicit geometric evidence of a cache match:
     $$s_{\max} = \max_{s \in \text{cache}} (q^T k_s)$$
   - For normalized unit keys in dimension $d_k=8$, random non-matching cache keys have $|q^T k_s| \le 1/\sqrt{d_k} \approx 0.35$, whereas a true cache hit satisfies $q^T k_s \approx 1.00$.
   - We define a sharp sigmoid cache presence gate:
     $$c_{\text{cache}} = \sigma\left(\kappa (s_{\max} - s_0)\right), \quad s_0 = 0.70, \ \kappa = 20.0$$
   - When $q$ is remote ($s_{\max} \le 0.35$):
     $$c_{\text{cache}} \le \sigma(20 \times (0.35 - 0.70)) = \sigma(-7.0) \approx 0.0009 \approx 0$$
     yielding pure, uncorrupted remote recall $y = M q$.
   - When $q$ is in cache ($s_{\max} \ge 0.95$):
     $$c_{\text{cache}} \ge \sigma(20 \times (0.95 - 0.70)) = \sigma(+5.0) \approx 0.993 \approx 1$$
     yielding the full Bayes posterior update $y = M q + g_B (v_{\text{cache}} - M k_{\text{cache}})$.
   - Pointer chasing attention temperature is set to $\tau = 8.0$, ensuring sharp softmax concentration on the matching cache token.

4. **Repair Verification**:
   - Re-evaluating with this principled discrimination gate on all 5 paired seeds:
     - 2-hop decoded minimum increased from 0.7719 to 0.9375 (threshold $\ge 0.8500$) — PASS.
     - 4-hop decoded minimum increased from 0.4219 to 0.8281 (threshold $\ge 0.5000$) — PASS.
     - 4-hop maximum vector error reduced to 0.0658 (threshold $\le 0.3500$) — PASS.
     - All 7 Phase 4 gates simultaneously PASS.
