# AURELIS Phase 0: Preregistration Plan

**Generated UTC:** `2026-09-19T07:02:05.356657+00:00`  
**Version:** `1.0`

---

## 1. Hypotheses

- **H1:** Solve-free bounded execution achieves constant-per-token decode complexity O(d_v d_k + w(d_k + d_v)) without key-space matrix factorizations.  
  *Falsification:* Per-step bounded decode time scales super-linearly with context length.
- **H2:** Recurrent completion (Eq. 8, using r(q) = v̄_L + S(q - k̄_L)) lowers measured retrieval cost (pages/bytes fetched) at fixed certified error tolerance ε relative to the best recurrence-free completion (r = 0, r = barycenter, or Eq. 13 per-page completion).  
  *Falsification:* Measured cost improvement < 10% or paired 95% confidence interval includes zero at registered epsilon values on held-out tasks.
- **H3:** Archive mode preserves held-out task quality (MQAR, Passkey, LM validation NLL) while meeting registered SLOs: p99 decode latency ≤ 25 ms, fallback rate ≤ 5.0%, and acceptable memory footprint.  
  *Falsification:* Fallback rate > 5.0%, or p99 latency > 25 ms, or validation NLL degradation > 1.0% relative to dense Transformer baseline.
- **H4:** Trained bounded mode (without archive) retains competitive quality on language modeling and associative recall within its fixed recurrent state budget O(d_v d_k + w(d_k + d_v)).  
  *Falsification:* Validation NLL degradation > 5.0% relative to matched SSM Hybrid baseline, or complete failure on short-to-medium recall tasks within the state capacity.

---

## 2. Workload & Context Grids

- Context Lengths: `[512, 1024, 2048, 4096, 8192, 16384]`
- Batch Sizes: `[1, 4, 8, 16]`
- Key/Value Dimensions: `[32, 64, 128]`
- Local Window Sizes: `[64, 128, 256]`
- Page Sizes: `[32, 64, 128]`

---

## 3. Baselines

- **Transformer GQA:** Modern causal Transformer with RoPE rotary embeddings, Pre-RMSNorm, SwiGLU MLP, GQA grouped-query attention.
- **Recurrence-Free Archive:** Sparse/archive attention without recurrent prediction (using r=0 or local barycenter predictor).
- **Per-Page Midpoint Comparator:** Equation (13) per-page completion with p_j = c_j and B_j = U_j ρ_j.
- **Gated DeltaNet:** Gated DeltaNet recurrence baseline without archive.
- **SSM Hybrid:** Alternating Mamba-2 / SSD selective state-space scan and local attention.

---

## 4. Compute Limits & Stop Conditions

- Max Runtime Per Job: `1800s`
- Max RSS Memory: `32.0 GiB`
- Stop Conditions:
  - NaN or Inf detected in loss, gradients, or recurrent state
  - Loss divergence (> 100.0 on standard LM cross-entropy)
  - Process timeout exceeding per-job limit
  - Memory allocation exceeding 32 GiB RSS
  - Evaluation non-convergence or zero backward gradient
