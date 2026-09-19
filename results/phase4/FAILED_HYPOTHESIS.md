# Phase 4 Hypothesis Falsification: FAILED_HYPOTHESIS

**Phase:** Phase 4 — Falsification and Novelty-Critical Comparisons  
**Evaluated At UTC:** `2026-09-19T13:08:43.394615+00:00`  
**Git Commit:** `046b499a90bdac34716abbeb04ab7e8facc1c1c6`  
**Overall Verdict:** **FAILED_HYPOTHESIS**

### Falsification Record: Hypothesis H2 (H-RECURRENCE)

- **Preregistered Claim:** Recurrent completion (Eq. 8, using $r(q) = \bar{v}_L + S(q - \bar{k}_L)$) lowers measured retrieval cost (pages/bytes fetched) at fixed certified error tolerance $\epsilon$ by at least 10% relative to the best recurrence-free completion (Eq. 13 per-page center completion, $r=0$, or local barycenter).
- **Empirical Result:** Across 60 paired evaluations over 20 random seeds, mean cost reduction was `-0.41%` (< 10.0%) and the 95% confidence interval was `[-52.80, -52.80]`.
- **Mathematical Cause:** Equation (13) per-page completion uses local page centers $p_j = c_j$, giving residual bounds $B_j = U_j \rho_j$. In contrast, AURELIS uses a single global predictor $r(q)$, requiring page bounds $b_j(r) = U_j(\|c_j - r\| + \rho_j) \ge U_j \rho_j$. The static per-page comparator is mathematically tighter and incurs zero recurrent compute/memory overhead.

### Research Decision & Next Steps

1. **Retirement of Claim:** We retire the scientific claim that solve-free recurrent predictors make certified exact retrieval cheaper than static per-page completion.
2. **Scaling Branch Terminated:** The archive-backed certified retrieval branch will NOT receive further accelerator/serving scaling investment in Phases 5–8.
3. **Bounded Mode Preserved:** Bounded mode (H1 / H4), operating without an archive in $O(d_v d_k + w(d_k + d_v))$ constant per-step decode time, survives as an independent fast approximate hybrid mechanism.
