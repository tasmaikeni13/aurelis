# AURELIS-R v2 Phase 0: Experiment Preregistration

**Registration ID:** `v2.0-phase0`  
**Timestamp UTC:** `2026-09-18T06:45:55.318757+00:00`  
**Governing Protocols:** `phases/AUTONOMY_PROTOCOL.md`, `phases/IMPLEMENTATION_CONTRACT.md`

---

## 1. Registered Hypotheses & Falsification Criteria

| Hypothesis ID | Name | Formal Statement | Falsification / Rejection Criterion | Target Phase |
|---|---|---|---|---|
| **H1** | Solve-free bounded execution removes v1 solver/all-prefix bottleneck | Solve-free bounded execution removes v1's solver and all-prefix intermediate bottleneck, achieving constant-per-token decode complexity O(d_v d_k + w(d_k + d_v)) without Cholesky factorizations. | Per-step bounded decode time scales super-linearly with context length, or exceeds v1 ridge decode time at matched dimensions. | Phase 2 & Phase 5 |
| **H2** | Recurrent completion lowers measured cost at fixed certified error | Recurrent completion (Eq. 8, using r(q) = v̄_L + S(q - k̄_L)) lowers measured retrieval cost (pages/bytes fetched) at fixed certified error tolerance ε relative to the best recurrence-free completion (r = 0, r = barycenter, or Eq. 13 per-page completion). | Measured cost improvement < 10% or paired 95% confidence interval includes zero at registered epsilon values on held-out tasks. | Phase 3 & Phase 4 |
| **H3** | Archive mode preserves held-out task quality with acceptable systems metrics | Archive mode preserves held-out task quality (MQAR, Passkey, LM validation NLL) while meeting registered SLOs: p99 decode latency ≤ 25 ms, fallback rate ≤ 5.0%, and acceptable memory footprint. | Fallback rate > 5.0%, or p99 latency > 25 ms, or validation NLL degradation > 1.0% relative to dense Transformer baseline. | Phase 6 & Phase 7 |
| **H4** | Trained bounded mode has useful quality at its fixed state budget | Trained bounded mode (without archive) retains competitive quality on language modeling and associative recall within its fixed recurrent state budget O(d_v d_k + w(d_k + d_v)). | Validation NLL degradation > 5.0% relative to matched SSM Hybrid baseline, or complete failure on short-to-medium recall tasks within the state capacity. | Phase 6 & Phase 8 |

---

## 2. Workload & Experimental Grids

- **Context Length Grid:** `[512, 1024, 2048, 4096, 8192, 16384]`
- **Batch Size Grid:** `[1, 4, 8, 16]`
- **Key/Value Dimensions ($d_k = d_v$):** `[32, 64, 128]`
- **Local Window Sizes ($w$):** `[64, 128, 256]`
- **Page Sizes ($B$):** `[32, 64, 128]`
- **Training Chunk Lengths ($C$):** `[64, 128]`
- **Tolerance Epsilon Grid ($\epsilon$):** `[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]`

---

## 3. Held-Out Datasets & Evaluation Suites

1. **MQAR (Multi-Query Associative Recall):** 10,000 train, 1,000 validation, 2,000 test sequences; vocabulary size 4,096; evaluated on exact string key-value retrieval.
2. **Passkey Needle Retrieval:** Insertion depths in $[0.1, 0.25, 0.5, 0.75, 0.9]$; context lengths from 512 to 16,384; 50 trials per depth.
3. **Multi-Hop Pointer Chasing:** 1,000 test chains across 2-hop and 4-hop mixed cache/remote chains.
4. **Language Modeling:** WikiText-103 standard train, validation, and test splits; token budget fixed at 10M training tokens for pilot models.

---

## 4. Paired Seeds & Compute Limits

- **Paired Seeds:** `[42, 137, 2026]` (applied identically across all architectures)
- **Per-Job Wall-Clock Limit:** `1800 seconds (30 minutes)`
- **Max Memory Budget:** `32.0 GiB RSS`
- **Total Compute Pilot Limit:** `100 CPU core-hours`
- **Explicit Stop Conditions:**
  - Nonfinite outputs (NaN or Inf) detected in activations, gradients, or states.
  - Loss divergence exceeding 100.0 on standard language modeling cross-entropy.
  - Job execution time exceeding per-job timeout.
  - Resident memory exceeding 32 GiB RSS.

---

## 5. Baselines & Comparators

1. **Dense GQA Transformer:** Modern causal Transformer with RoPE rotary embeddings, Pre-RMSNorm, SwiGLU MLP, and grouped-query attention.
2. **Recurrence-Free Archive Completion:** Sparse/archive attention with zero or local value barycenter predictor.
3. **Per-Page Midpoint Completion (Eq. 13):** Cheap baseline using page centers $p_j = c_j$ with error bound $B_j = U_j \rho_j$.
4. **Gated DeltaNet:** Bounded recurrence without archive.
5. **SSM Hybrid:** Alternating Mamba-2 / SSD selective state-space scan and local attention.
6. **Legacy v1 Bayesian Ridge:** Ridge solver mechanism ablation.

---

## 6. Acceptance & Noninferiority Margins

- **Float64 Numerical Agreement:** atol $= 10^{-10}$, rtol $= 10^{-9}$ on non-overflowing inputs.
- **Quality Noninferiority:** $\le 1.0\%$ relative validation NLL and $\le 2.0$ percentage points on registered recall tasks.
- **Cost Improvement (H2):** $\ge 10.0\%$ measured cost reduction with paired 95% confidence interval excluding zero.
- **Serving SLO:** Target p99 decode latency $\le 25.0\text{ ms}$, maximum fallback rate $\le 5.0\%$.
- **Bounded Mode State Memory:** $\ge 2.0\times$ reduction vs Transformer KV cache at $\ge 4096$ context.

---

## 7. Raw-Data Provenance Plan

| Metric Type | Primary Data Source | Serialization Format | Verification Mechanism |
|---|---|---|---|
| Accuracy & Recall | Raw model generated tokens | `.jsonl` with prediction records | Exact string match vs ground truth; checkpoint hash attached |
| Validation NLL | Cross-entropy per token | `.jsonl` per-batch loss arrays | Recomputed against raw validation token streams |
| Latency | High-resolution `time.perf_counter()` | `.jsonl` timing samples | Backend synchronized; warmups logged separately; populated cache |
| Memory Footprint | `psutil` RSS & tensor `element_size() * numel()` | JSON memory trace logs | Resident set size verified at sequence checkpoints |
| Certificate Error | $\|y_* - \widehat{y}_A\|_2$ vs fp64 full softmax | `.jsonl` bound records | 100% mathematical enclosure verified against fp64 oracle |

---

## 8. Preregistration Gate Verdict

- Hypotheses H1-H4 formally registered: **PASS**
- Experimental grids, baselines, and margins fixed before execution: **PASS**
- Explicit compute limits and stop conditions declared: **PASS**
- Provenance plan established for every metric type: **PASS**
