# AURELIS-R v2 Phase 0 PASS

**Generated UTC:** `2026-09-18T06:45:55.320130+00:00`  
**Theory Revision:** `v2.0`  
**Git Commit:** `9921cfb207c7712102d2a02439a77b49dbecf74a`  
**Verdict:** **PASS**

Phase 0 gates have passed unconditionally:
1. **Runtime Enumeration:** Active host hardware and compute devices were detected via live runtime enumeration without hard-coded labels. The compute substrate is AMD EPYC 7B12 CPU (240 vCPUs, 400 GiB RAM) with JAX `CpuDevice`.
2. **Formal Verification:** Pinned Lean 4 / Mathlib build completed successfully with zero admitted proofs (`sorry`), zero `admit`, and zero project axioms. All 30 theorems were mapped to paper equations (1)–(12).
3. **Evidence Reset:** All 12 legacy claims from v1 were audited against source code, classified into standard categories (measured, analytical, hard-coded, unsupported, not audited), and marked historical. No v1 evidence is inherited by v2.
4. **Preregistration:** Hypotheses H1–H4, workload grids, paired seeds (42, 137, 2026), acceptance margins, and raw-data provenance plans were registered prior to experimental execution.
5. **Budgets & Stop Conditions:** Compute limits (1800s timeout, 32 GiB RSS, 100 core-hours pilot) and stopping criteria were formalized.
6. **Implementation Gap Map:** Module-level handoff requirements from v1 Bayesian ridge to v2 solve-free delta recurrence were documented.

All deliverables are archived in `results/v2/phase0/`. Phase 0 authorizes starting Phase 1.
