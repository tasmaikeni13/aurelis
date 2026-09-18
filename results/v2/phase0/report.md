# AURELIS-R v2 Phase 0 Report: Evidence Reset & Registration

**Date UTC:** `2026-09-18T06:45:55.320130+00:00`  
**Theory Revision:** `v2.0`  
**Git Commit:** `9921cfb207c7712102d2a02439a77b49dbecf74a`  
**Phase Status:** **PASS**

---

## 1. Executive Summary

Phase 0 establishes the baseline revision and evidence foundation for the AURELIS-R v2 experimental generation. In compliance with `AUTONOMY_PROTOCOL.md`, `IMPLEMENTATION_CONTRACT.md`, and `CHANGE_IMPACT_PROTOCOL.md`:
1. Inherited v1 empirical evidence was audited, classified, and marked historical; no legacy claim or PASS status is inherited by v2.
2. Host and runtime environments were inventoried using runtime device enumeration; active execution device is confirmed as CPU (AMD EPYC 7B12, 240 vCPUs, 400 GiB RAM).
3. The pinned Lean 4 formal build (`lake build`) was executed, verifying zero errors, zero warnings, zero admitted proofs (`sorry`), and zero project axioms across all 30 formal theorems.
4. An implementation gap map was constructed comparing existing v1 code against v2 equations (2)–(12).
5. Four core hypotheses (H1–H4), workload grids, paired seeds, compute budgets, baseline versions, and raw-data provenance plans were preregistered before viewing downstream results.

---

## 2. Equation-to-Code Mapping & Deliverables

| Deliverable Artifact | Description | Primary Equations / Direct Contracts |
|---|---|---|
| `environment.json` / `environment_inventory.md` | Genuine runtime device and software enumeration | AUTONOMY_PROTOCOL.md § Evidence rules |
| `formal_build_report.md` | Lean 4 build report and theorem mapping | Eqs. (1), (2), (5), (6), (8), (9), (10), (11), (12) |
| `legacy_claims_audit.json` / `legacy_claims_audit.md` | Source audit of old Phase 6 diagnostics and hardware claims | research/V1_AUDIT.md |
| `implementation_gap_map.json` / `implementation_gap_map.md` | Module migration map from v1 to v2 | IMPLEMENTATION_CONTRACT.md |
| `preregistration.json` / `preregistration.md` | Formal registration of hypotheses, grids, margins, and provenance | aurelis.md §9, AUTONOMY_PROTOCOL.md |
| `resource_budget.json` / `resource_budget.md` | Compute, wall-clock, and memory budgets | AUTONOMY_PROTOCOL.md § Required workflow |

---

## 3. Hypotheses & Status

- **H1 (Solve-free bounded execution):** Registered; primary evaluation in Phase 2 & Phase 5.
- **H2 (Recurrent completion lowers cost):** Registered; primary evaluation in Phase 3 & Phase 4.
- **H3 (Archive mode quality and SLO):** Registered; primary evaluation in Phase 6 & Phase 7.
- **H4 (Trained bounded mode quality):** Registered; primary evaluation in Phase 6 & Phase 8.

---

## 4. Gate Verification & Outcomes

| Gate | Criterion | Evidence | Status |
|---|---|---|---|
| Gate 1 | Every claimed device is returned by runtime enumeration | `environment.json`; AMD EPYC 7B12 CPU, JAX CpuDevice; no TPU/GPU claimed | **PASS** |
| Gate 2 | Every metric type has a raw-data provenance plan | `preregistration.md` § 7; explicit logging plans for accuracy, loss, latency, memory, certificates | **PASS** |
| Gate 3 | No legacy claim is inherited by v2 | `legacy_claims_audit.md`; all 12 v1 claims audited, classified, and marked historical | **PASS** |
| Gate 4 | Compute limits and stop conditions are explicit | `preregistration.md` § 4, `resource_budget.md`; 1800s timeout, 32 GiB RSS cap, NaN/Inf stops | **PASS** |
| Gate 5 | Formal Lean build passes with zero axioms/sorry | `formal_build_report.md`, `raw/lean_build.log`; 30 theorems mapped | **PASS** |
| Gate 6 | Implementation gap map and preregistration complete | `implementation_gap_map.md`, `preregistration.md` complete | **PASS** |

---

## 5. Artifact Hashes

| File | SHA-256 Checksum |
|---|---|
| `environment.json` | `2164fe75f11a71771c0b8bcdb5042a30e999cc81fecaeaab49d3fad1f5207549` |
| `environment_inventory.md` | `1b830b1b4a1b10d07480ed67966ee28590233debf3b90ec1ed87e6fdce559de9` |
| `formal_build_report.md` | `e8aefcec4a5385e092149b2556a9c9454880fce4076c8d573c4afd3237240c76` |
| `implementation_gap_map.json` | `23437167a7276fb0c0cc72707928916b788aa14f4cab366c93f35be5d7f0bca1` |
| `implementation_gap_map.md` | `10b8084a6a29df9f0a9db12d541590d24c2c590fa0042ab4cbb62a088cc440db` |
| `legacy_claims_audit.json` | `a4244c19c142110a8501d4e90f40cf1c4711862544c3bf6647ca689cf138b258` |
| `legacy_claims_audit.md` | `ff2dbd06d45de68168a8f161dba124799b47d67604316f5f0ff63ce63635d65f` |
| `preregistration.json` | `fcde5456c728730d5c4d73b296e9bc48f2b648a2965e4a4f088a2f643dac1e6d` |
| `preregistration.md` | `8be11cc2e9f42059ccb7e8fd02f8a24d620165e26148ce0c8bb04ce2a56bb530` |
| `resource_budget.json` | `237c539cc56c6b048e2fb204e42caf5bd7414e6982051413027a2fd0ced232b9` |
| `resource_budget.md` | `4179d7fde3e77002be50b88475c38cf14e10d5be6ea9661a5b5b31f8f6093bd5` |

---

## 6. Next Decision

Phase 0 is complete with status **PASS**. Proceed to **Phase 1: Independent Math Oracles & Formal Correspondence**, implementing clean float64 reference oracles for equations (2)–(12) and validating numerical agreement with Lean formal statements.
