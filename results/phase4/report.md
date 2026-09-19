# AURELIS Phase 4 Report: Falsification & Novelty-Critical Comparisons

**Date UTC:** `2026-09-19T13:08:43.394615+00:00`  
**Git Commit:** `046b499a90bdac34716abbeb04ab7e8facc1c1c6`  
**Phase Status:** **FAILED_HYPOTHESIS**

---

## 1. Executive Summary & Scientific Outcome

Phase 4 evaluated the core research hypothesis **H2 (H-RECURRENCE)**: whether coupling a solve-free recurrent predictor $r(q) = \bar{v}_L + S_t(q - \bar{k}_L)$ to an exact archive lowers the retrieval cost at fixed certified error tolerance $\epsilon$ relative to the strongest cheap recurrence-free completion.

### The Key Scientific Discovery:
The hypothesis **H2 is conclusively FALSIFIED**.

1. **Mathematical Dominance of Equation (13):** The per-page center completion comparator (Eq. 13 in `aurelis.md` §6.3), which uses static page centers $p_j = c_j$ and bounds $B_j = U_j \rho_j$, mathematically dominates AURELIS's residual bound $b_j(r) = U_j(\|c_j - r\| + \rho_j)$ because $\|c_j - r\| \ge 0$ everywhere.
2. **Empirical Verification:** Across 60 paired evaluations over 20 random seeds, `per_page_center` achieved equal or fewer page reads and lower composite service cost than AURELIS. The mean cost improvement of AURELIS was negative, failing the preregistered $\ge 10\%$ margin.
3. **Protocol Action:** In accordance with `phases/phase4.md`, `aurelis.md` §6.3, and `phases/AUTONOMY_PROTOCOL.md`, we record **FAILED_HYPOTHESIS** for H2 and retire the claim that solve-free recurrence makes certified archive retrieval cheaper. We do not scale this archive mechanism.
4. **Bounded Mode Preserved:** Bounded mode (H1 / H4), which operates strictly without an archive in $O(d_v d_k + w(d_k + d_v))$ constant per-step decode time, remains valid as an independent fast approximate model.

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
| `raw/pytest_run.log` | Full PyTest suite execution trace (147 tests passed) | PASS |

---

## 3. Hypothesis H2 Evaluation & Paired Uncertainty Estimates

- **Sample Size:** 60 paired evaluations across 20 seeds
- **Mean Difference (Best Cheap Baseline - AURELIS):** -52.8000
- **Standard Error:** 0.0000
- **Paired 95% Confidence Interval:** `[-52.8000, -52.8000]`
- **Mean Cost Reduction:** `-0.41%` (Threshold: $\ge 10.0\%$)
- **Formal Scientific Verdict:** **FAILED_HYPOTHESIS**

**Rationale:** The recurrence-free comparator per_page_center (Eq. 13) mathematically and empirically dominates recurrent completion. Because b_j(r) = U_j(||c_j - r|| + rho_j) >= U_j * rho_j = B_j(Eq. 13), per-page center completion achieves strictly tighter certificate bounds, equal or lower error, and zero recurrent state memory/compute overhead. Mean cost difference was -52.80 (95% CI [-52.80, -52.80]), with mean reduction -0.41% < 10%. H2 is conclusively falsified.

---

## 4. Gate Verification & Outcomes

| Gate | Criterion | Evidence | Status |
|---|---|---|---|
| Gate 1 | Identical Q/K/V, archive pages, encoding, epsilon, and accounting across all 8 comparators | Verified in test_phase4_comparators.py & comparator_benchmark.json | **PASS** |
| Gate 2 | Equal total state/index budgets and equal fetched-KV budgets comparison | Verified in test_phase4_comparators.py & comparator_benchmark.json | **PASS** |
| Gate 3 | Workload sweep across all 11 registered regimes (including metadata-dominant) | Verified in test_phase4_workloads.py & workload_sweep_results.json | **PASS** |
| Gate 4 | All 6 required ablations with comprehensive arithmetic and systems cost accounting | Verified in test_phase4_ablations.py & ablation_results.json | **PASS** |
| Gate 5 | Statistical evaluation of H2 with paired uncertainty estimates and confidence intervals | H2 falsified: mean cost improvement < 10% (-0.41%) and CI [-52.80, -52.80] does not demonstrate required superiority over per_page_center (Eq. 13). | **FAILED_HYPOTHESIS** |
| Gate 6 | Reproducible counterexamples documented outside useful regime | Verified in test_phase4_counterexamples.py & counterexamples.json | **PASS** |
| Gate 7 | Updated novelty comparison against ResKV, Quest, certified quantized attention, and hybrid memory | Documented in novelty_comparison.json & novelty_comparison.md | **PASS** |

---

## 5. Artifact Hashes

| File | SHA-256 Checksum |
|---|---|
| `comparator_benchmark.json` | `e401f6880cb54fb6bce97c1613b371c436a58a283b7d8bba56a3744d03038172` |
| `comparator_benchmark.md` | `93a0ca1dcbde29dc11434548f840c81727e84681d41e620f0042fff6bcb5bd16` |
| `workload_sweep_results.json` | `40fce607c1a401595abc05ad8945da2e0c31ee70a5c89260ea811d4d5bced1ea` |
| `workload_sweep_results.md` | `da33d933d6b5e067bffc53d8b516f98bda13e1fb45958ae25fe434a531777f80` |
| `ablation_results.json` | `f6e4db9104d33ba74432f2e9d4215aa70aa253dab6ab605fd39f1463a2f3bb16` |
| `ablation_results.md` | `f043fb0b0884995977f57a328e054a0413695503f25720578df69070dd69c936` |
| `counterexamples.json` | `272b111258135570308649ec5930e7c72c86edeb8887e3499e268b7d8475e185` |
| `counterexamples.md` | `64ec7c5bb0f0c54f0cedfc1e09b1f7fbbf80b21a97d04fab7ad360581102ed94` |
| `novelty_comparison.json` | `c9c509d1aebfbc9d25ad92f8654ad927b69846399ef0a7d516ffb5c590371d84` |
| `novelty_comparison.md` | `95f7c4ebad3fa3d0fdbfa70d165ef23527abf063eb3ac23c95d6063d1782adf7` |
| `gate_records.json` | `e5bd3bd68867305182effa8607a7f082d77ae4abe45f5c6791cf28456847df23` |
| `FAILED_HYPOTHESIS.md` | `1d4f4f270924885dd4dbbd99375a64fbb52a25c95e4be5fc8e0566bd692599ea` |

