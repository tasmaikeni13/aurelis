# Phase 4 Hypothesis Falsification: FAILED_HYPOTHESIS

**Phase:** Phase 4 — Falsification and Novelty-Critical Comparisons  
**Revision:** `1.1` (Supersedes `1.0` per `results/CHANGE_MANIFEST.yaml`)  
**Evaluated At UTC:** `2026-09-19T13:26:22.800384+00:00`  
**Git Commit:** `c48dac340334a1428195f46c432eefb39d6b7698`  
**Overall Verdict:** **FAILED_HYPOTHESIS**

### Falsification Record: Hypothesis H2 (H-RECURRENCE)

- **Preregistered Claim:** Recurrent completion (using $r(q) = \bar{v}_L + S(q - \bar{k}_L)$) lowers measured retrieval cost (pages/bytes fetched) at fixed certified error tolerance $\epsilon$ by at least 10% relative to the best recurrence-free completion (Eq. 13 per-page center completion, $r=0$, or local barycenter) with 95% confidence interval strictly greater than 0.
- **Empirical Result:** Across 60 paired evaluations over 20 random seeds:
  - Mean cost reduction was `-0.66%` (< 10.0%).
  - Paired 95% confidence interval was `[-86.40, -86.40]`, strictly negative.
  - Across all 20 seeds and all tolerance levels $\epsilon \in \{0.2, 0.1, 0.05\}$, AURELIS and `per_page_center` required the exact same number of page reads, while AURELIS incurred an uncompensated $O(d_v d_k)$ FLOP and matrix memory overhead for computing and maintaining recurrent state $S$.

### Mathematical Mechanism & Proof of Falsification:

1. **Generation 1.0 Defect (Global Prior Inflation):**
   In Generation 1.0, Eq. (12) bounded unread pages by $b_j(r) = U_j(\|c_j - r\| + \rho_j)$. Because $\|c_j - r\| \ge 0$, $b_j(r) \ge U_j \rho_j = B_j(\text{per\_page\_center})$ everywhere.
2. **Revision 1.1 Iteration & Minimality of Chebyshev Page Centers Theorem:**
   In Revision 1.1, we repaired the score intervals using Euclidean key-ball bounds intersected with coordinate boxes, and unified archive completion under Eq. (13). We then proved:
   $$\forall p \in \mathbb{R}^{d_v}, \quad U_j(\|c_j - p\| + \rho_j) + \eta_j \|p - \widehat y\| \ge U_j \rho_j + \eta_j \|c_j - \widehat y\|$$
   identically because $U_j \ge \eta_j = (U_j - L_j)/2$. Under triangle inequality over Chebyshev centers, static per-page completion $p_j = c_j$ with bound $B_j = U_j \rho_j$ mathematically minimizes the worst-case certificate bound over all possible predictors.
3. **Priority Invariance:**
   In empirical evaluation, page picking priority is dominated by the $U_j \rho_j$ mass-radius product ($U_j \rho_j \gg \eta_j \|c_j - r\|$), resulting in identical page fetch sequences.

### Research Decision & Next Steps per Protocol:

1. **Retirement of Claim:** In accordance with `phases/phase4.md` line 39 and `phases/CHANGE_IMPACT_PROTOCOL.md`, we formally retire the scientific claim that solve-free recurrent predictors make certified exact retrieval cheaper than static per-page completion.
2. **Scaling Branch Terminated:** The archive-backed certified retrieval branch will NOT receive further accelerator/serving scaling investment in Phases 5–8.
3. **Bounded Mode Preserved:** Bounded mode (H1 / H4), operating without an archive in $O(d_v d_k + w(d_k + d_v))$ constant per-step decode time, survives as an independent fast approximate hybrid mechanism.

