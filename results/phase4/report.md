# AURELIS Phase 4 Report: Falsification & Novelty-Critical Comparisons

**Date UTC:** `2026-09-19T13:26:22.800384+00:00`  
**Git Commit:** `c48dac340334a1428195f46c432eefb39d6b7698`  
**Revision:** `1.1` (Supersedes `1.0` per `results/CHANGE_MANIFEST.yaml`)  
**Phase Status:** **FAILED_HYPOTHESIS**

---

## 1. Executive Summary & Scientific Outcome

Phase 4 evaluated the core research hypothesis **H2 (H-RECURRENCE)**: whether coupling a solve-free recurrent predictor $r(q) = \bar{v}_L + S_t(q - \bar{k}_L)$ to an exact archive lowers the retrieval cost at fixed certified error tolerance $\epsilon$ relative to the strongest cheap recurrence-free completion.

### The Key Scientific Discovery & Protocol Iteration:
The hypothesis **H2 is conclusively FALSIFIED** under both Generation 1.0 and Revision 1.1.

1. **Theorem (Minimality of Chebyshev Page Centers):** Under triangle inequality splitting at page Chebyshev center $c_j$, for any predictor $p$, $U_j(\|c_j - p\| + \rho_j) + \eta_j \|p - \widehat y\| \ge U_j \rho_j + \eta_j \|c_j - \widehat y\|$ identically because $U_j \ge \eta_j = (U_j - L_j)/2$. Static per-page Chebyshev center completion (Eq. 13) mathematically minimizes the worst-case certificate bound over all possible predictors.
2. **Empirical Verification:** Across 60 paired evaluations over 20 random seeds, `per_page_center` achieved equal page reads and strictly lower composite service cost than AURELIS. Recurrence incurs an uncompensated $O(d_v d_k)$ FLOP and matrix memory overhead without saving page fetches. Mean cost difference was negative, conclusively failing the preregistered $\ge 10\%$ margin.
3. **Protocol Action:** In accordance with `phases/phase4.md`, `aurelis.md` §6.3, and `phases/CHANGE_IMPACT_PROTOCOL.md`, we record **FAILED_HYPOTHESIS** for H2 and retire the claim that solve-free recurrence makes certified archive retrieval cheaper. As mandated by protocol, we do not scale this archive retrieval branch to Phases 5–8.
4. **Bounded Mode Preserved:** Bounded mode (H1 / H4), which operates strictly without an archive in $O(d_v d_k + w(d_k + d_v))$ constant per-step decode time, remains fully validated as an independent fast approximate architecture.

---

## 2. Deliverables Summary

| Deliverable Artifact | Description | Primary Status |
|---|---|---|
| `comparator_benchmark.json` / `.md` | Full evaluation across all 8 required comparators | Completed |
| `workload_sweep_results.json` / `.md` | 11 synthetic and structural workload sweeps | Completed |
| `ablation_results.json` / `.md` | 6 required ablations with full cost accounting | Completed |
| `counterexamples.json` / `.md` | 4 reproducible counterexamples outside useful regime | Completed |
| `novelty_comparison.json` / `.md` | Novelty comparison vs ResKV, Quest, CQA, Hybrids | Completed |
| `gate_records.json` | Complete audit of all 7 Phase 4 gate criteria | Completed |
| `raw/lean_build.log` | Lean 4 formal compiler trace (0 sorry, 0 custom axioms) | PASS |
| `raw/pytest_run.log` | Full PyTest suite execution trace (149 tests passed) | PASS |

---

## 3. Hypothesis H2 Evaluation & Paired Uncertainty Estimates

- **Sample Size:** 60 paired evaluations across 20 seeds
- **Mean Difference (Best Cheap Baseline - AURELIS):** -86.4000
- **Standard Error:** 0.0000
- **Paired 95% Confidence Interval:** `[-86.4000, -86.4000]`
- **Mean Cost Reduction:** `-0.66%` (Threshold: $\ge 10.0\%$)
- **Formal Scientific Verdict:** **FAILED_HYPOTHESIS**

**Rationale:** Under Revision 1.1, AURELIS adopts Eq. (13) grouped completion with Euclidean key-ball score bounds. Per the Minimality of Chebyshev Page Centers Theorem, for any predictor p in R^{d_v}, U_j(||c_j - p|| + rho_j) + eta_j ||p - y_hat|| >= U_j * rho_j + eta_j ||c_j - y_hat|| identically. Because static Chebyshev page center completion p_j = c_j minimizes the worst-case certificate bound and requires zero recurrent state memory/compute, recurrence incurs an uncompensated FLOP/memory overhead. Across 60 paired evaluations over 20 random seeds, mean cost difference was -86.40 (95% CI [-86.40, -86.40]), with mean reduction -0.66% < 10%. H2 is conclusively falsified under both Revision 1.0 and Revision 1.1.

---

## 4. Gate Verification & Outcomes

| Gate | Criterion | Evidence | Status |
|---|---|---|---|
| Gate 1 | Identical Q/K/V, archive pages, encoding, epsilon, and accounting across all 8 comparators | Verified in test_phase4_comparators.py & comparator_benchmark.json | **PASS** |
| Gate 2 | Equal total state/index budgets and equal fetched-KV budgets comparison | Verified in test_phase4_comparators.py & comparator_benchmark.json | **PASS** |
| Gate 3 | Workload sweep across all 11 registered regimes (including metadata-dominant) | Verified in test_phase4_workloads.py & workload_sweep_results.json | **PASS** |
| Gate 4 | All 6 required ablations with comprehensive arithmetic and systems cost accounting | Verified in test_phase4_ablations.py & ablation_results.json | **PASS** |
| Gate 5 | Statistical evaluation of H2 with paired uncertainty estimates and confidence intervals | H2 falsified: mean cost improvement < 10% (-0.66%) and CI [-86.40, -86.40] does not demonstrate required superiority over per_page_center (Eq. 13). | **FAILED_HYPOTHESIS** |
| Gate 6 | Reproducible counterexamples documented outside useful regime | Verified in test_phase4_counterexamples.py & counterexamples.json | **PASS** |
| Gate 7 | Updated novelty comparison against ResKV, Quest, certified quantized attention, and hybrid memory | Documented in novelty_comparison.json & novelty_comparison.md | **PASS** |

---

## 5. Artifact Hashes

| File | SHA-256 Checksum |
|---|---|
| `comparator_benchmark.json` | `848db6f52d2a4f4814e5984fb37f50e6d1083dfa35899eaf832718e337bf1b55` |
| `comparator_benchmark.md` | `f7c1d89a66ec9a55594c3f2866a1e4460a3698637039ba9c586d0e698e270e4e` |
| `workload_sweep_results.json` | `ab4f350153fe78c191e1df081b015f9488b6b7f66c342f4f71d120111a6c599e` |
| `workload_sweep_results.md` | `ecf01ffb43032074190a2cf314549604e1292dee905e36a6f1409fd6375913b8` |
| `ablation_results.json` | `51507306a266bbf93008191d752776061f8a571c7a5a49217268c6cc5165b832` |
| `ablation_results.md` | `3471a5c4c5e2316fb7c2cd4621cbb21b31d1813bceaee7d124fc8db6af93b7d9` |
| `counterexamples.json` | `272b111258135570308649ec5930e7c72c86edeb8887e3499e268b7d8475e185` |
| `counterexamples.md` | `64ec7c5bb0f0c54f0cedfc1e09b1f7fbbf80b21a97d04fab7ad360581102ed94` |
| `novelty_comparison.json` | `c9c509d1aebfbc9d25ad92f8654ad927b69846399ef0a7d516ffb5c590371d84` |
| `novelty_comparison.md` | `95f7c4ebad3fa3d0fdbfa70d165ef23527abf063eb3ac23c95d6063d1782adf7` |
| `gate_records.json` | `e227982a511271eab224a5a6cf4f83c1c0de1dd796cbce27546f3b06fd7ea826` |
| `FAILED_HYPOTHESIS.md` | `7f5b078cc60b29025b3dd08df0468ab6def069d988212208e6959284877ea651` |

