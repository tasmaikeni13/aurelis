# AURELIS Experiment Log

This log chronicles every phase of development, experimental findings, failure-repair iterations, and hardware measurements on our AMD Instinct MI300X system.

---

## 2026-08-28 — Theoretical Foundations & Numerical Groundwork

### Literature Boundary & Novelty
We reviewed primary literature across hybrid sequence architectures: local/recurrent mixtures (Based, Griffin, Samba, Jamba), delta-rule memories (DeltaNet, Gated DeltaNet), and in-context regression (Mesa). The closest existing designs either alternate layers or concatenate local and remote outputs. Nobody was formulating same-head uncertainty-routed residuals over an exact partitioned Bayesian state. Our formal boundary analysis is detailed in [`research/LITERATURE_REVIEW.md`](research/LITERATURE_REVIEW.md).

### Core Architectural Decisions
1. **Disjoint Partition**: Older tokens are dropped from the local window and handed off exactly once to the remote Bayesian state. No double-counting.
2. **Residual Read**: $y(q) = Mq + g(q)[\bar{v} - M\bar{k}]$. When the remote slope $M$ is accurate, the residual completely removes local smoothing bias.
3. **Correlated Gating & Episodic Separation**: We derived the true cross-covariance $K_{RH}$ rather than relying on an independence heuristic. We also explicitly separated the objective for denoising a noisy latent trend from the objective for verbatim exception recall using $g_E = \max(g_B, e_t)$.

### Initial Float64 Numerical Checks
Running `analysis/aurelis_numerical.py` under seed `20260828`:
- Residual identity max absolute error: $9.992 \times 10^{-16}$
- Linear reproduction error: $2.285 \times 10^{-16}$
- Hard one-hot exception error: exactly `0.0`
- 50,000-trial Monte Carlo variance relative error: $0.293\%$
- All assertions passed cleanly across alternate seeds `17`, `29`, and `41`.

### Lean 4 Machine Verification
Set up the `Aurelis` Lean 4 formal project under `mathlib` 4.19.0. Initial build passed with zero `sorry`, `admit`, or undeclared axioms.

---

## 2026-08-29 — Phase 0: Identity Migration & MI300X Hardware Substrate

### Clean Repository Migration
Purged all legacy project names, outdated imports, and obsolete test files. Rebuilt `src/aurelis/` from scratch around the verified equations.

### Hardware Audit & ROCm Findings
Inspected our AMD Instinct MI300X VF accelerator under ROCm 7.0.2 with PyTorch 2.8.0.
- Detected 191.69 GiB total VRAM and 304 compute units.
- Confirmed PyTorch HIP namespace compatibility (`torch.cuda.is_available()` returns True, `torch.version.hip` active).
- Benchmarked peak GEMMs: 257.9 TFLOPS in bfloat16, 245.8 TFLOPS in float16.
- Ran into an initial hurdle with the bundled Triton compiler rejecting exact prefix cumulative-sum scans; we resolved this by keeping prefix state construction in PyTorch eager mode and compiling the outer head.

---

## 2026-08-29 — Phase 1: Oracles & Numerical Pathology Suite

### Verification of Core Invariants
Ran `experiments/phase1_oracle.py` across 90 streaming cases and 1,467 prefixes:
- Max streaming tolerance ratio: $2.74 \times 10^{-4}$ against double precision.
- Proved partition handoff exactness and zero history omission.
- Documented finite precision behavior under near-singular and collinear key matrices.
- Formalized `gated_error_identity`, `gatedRead_one`, and scalar ridge bounds in Lean. All 13 Phase 1 gates passed.

---

## 2026-09-04 — Phase 2: Controlled Baselines & Falsification Matrix

### Head-to-Head Comparison Suites
Implemented 10 baselines in `src/aurelis/baselines.py` (Local Attention, Bayesian Ridge, Global Linear Attention, DeltaNet, Mesa Least Squares, Concat/Sum hybrids, and Independent Inverse-Variance Fusion).
- Ran evaluations across matched parameters, matched dimensions, matched state bytes, and matched FLOPs.
- Tested across 9 falsification suites plus a correlated error suite over 10 random seeds.
- **Key Takeaway**: When local and remote errors correlate, our closed-form cross-covariance gate strictly outperforms the independence heuristic ($z \ge 5.0$). In misspecified nonlinear regimes, AURELIS behaves gracefully, matching theoretical predictions without claiming false universal dominance.

---

## 2026-09-04 — Phase 3: Learned Features & Straight-Through Episodic Routing

### Overcoming the Subgradient Dead Zone
When training neural feature projections, the hard maximum in $g_E = \max(g_B, e_t)$ creates zero subgradients for the episodic branch whenever $e_t < g_B$. We solved this by using a Straight-Through Estimator (STE): evaluating the hard maximum during the forward pass while backpropagating through a smooth surrogate $g_B + (1 - g_B)e_t$.

### 7-Task Curriculum Results (5 Paired Seeds)
- Shared key/query feature projections ($W_{kq}$) maintained an effective rank of $13.35 \ge 2.0$ and achieved an aggregate risk of $0.4625$ (vs $0.6029$ for independent projections and $2.3868$ for random frozen features).
- The episodic router achieved an AUROC of $1.0000$ and cue correlation $R^2 = 0.9478$ driven entirely by observable input features.
- AURELIS-E cut exception error by $1.77\times$ over AURELIS-B without degrading latent anti-copy performance.

---

## 2026-09-04 — Phase 4: Nonstationarity, Multi-Hop Composition & Capacity Limits

### Information Discounting on Observable Changes
Integrated dynamic linear model information discounting: $\gamma_t = \text{clamp}(1 - c_t(1 - \gamma_{\min}), \gamma_{\min}, 1.0)$.
- On observable changepoints, drift-aware AURELIS cut post-change MSE by $10.8\times$ compared to the stationary model ($0.0807$ vs $0.8767$).
- Under unobservable changepoints, performance bounded gracefully ($0.8593$ vs $0.8036$).

### Gauss-Markov Weighting & Multi-Hop Pointer Chasing
- Heteroscedastic inverse-variance weighting ($\beta_t = 1/\sigma_t^2$) reduced MSE by $12\times$ over uniform weighting ($0.0029$ vs $0.0354$). Corrupting precision weights blew up risk by $55\times$, proving active reliance on evidence quality.
- Solved error propagation across mixed cache/remote chains, achieving $83\%–100\%$ decoded success across 2-hop and 4-hop mixed pointer chains.
- Monotonically enforced rank capacity limits: error grew strictly as association count exceeded subspace rank $d_k=8$.

---

## 2026-09-05 — Phase 6: Language-Model Viability & Publication Gate (125M & 350M)

### The Three Publication Candidates
To produce an airtight paper for publication, we implemented and calibrated three matched architectures:
1. **AURELIS** (AURELIS-E and AURELIS-B): Dual-store local sliding window + delayed Bayesian state with constant-size decoding cache.
2. **Modern Causal Transformer**: RoPE rotary embeddings, Pre-RMSNorm, causal multi-head self-attention, SwiGLU MLP.
3. **Strong SSM + Attention Hybrid**: Alternating selective state-space scan (Mamba-2 style) + causal multi-head attention layers with Pre-RMSNorm and SwiGLU MLP.

### Parameter Accounting on Target Scales
- **125M Scale**: Transformer (123.5M), SSM Hybrid (120.3M), AURELIS-E (116.7M) — max deviation $2.89\% \le 8\%$.
- **350M Scale**: Transformer (353.5M), SSM Hybrid (341.6M), AURELIS-E (329.1M) — max deviation $3.60\% \le 8\%$.

### Custom HIP Kernel Optimization on AMD MI300X (`gfx942`)
Wrote and compiled inline HIP kernels for the accelerator:
- `recurrent_scan_f32_kernel`: Fused recurrent sequence scan ($h_t = a_t h_{t-1} + x_t$), max absolute error $9.54 \times 10^{-7}$ vs reference.
- `fused_residual_gate_f32_kernel`: Fused GPU gate evaluation ($y = \text{remote} + g \cdot (\bar{v} - M\bar{k})$), max absolute error $4.77 \times 10^{-7}$.

### Systems & Diagnostic Findings
- **Constant Decode State Memory**: While Transformer KV cache grew from 4.5 MB at 512 tokens to 36.0 MB at 4096 tokens, AURELIS remained rock-solid at **4.50 MB** ($8.0\times$ memory reduction at 4k).
- **Episodic Exception Recall**: AURELIS-E cut memorized exception MSE by **$4.48\times$** over AURELIS-B ($0.042$ vs $0.188$) while preserving identical latent denoising MSE ($0.015$ vs $0.014$).
- **Long-Context Passkey Retrieval**: AURELIS achieved **$100\%$** accuracy at 2048 tokens and $98\%$ at 4096 tokens.
- All Phase 6 gates passed cleanly; results logged in `results/phase6/PASS.md`.
