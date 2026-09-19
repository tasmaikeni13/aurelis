# AURELIS Phase 0 Report: Environment Inventory, Formal Verification, & Registration

**Date UTC:** `2026-09-19T07:02:05.358144+00:00`  
**Git Commit:** `684c0fe24dffb2aa1c550a7ab051f36d9ed1a389`  
**Phase Status:** **PASS**

---

## 1. Executive Summary

Phase 0 establishes the experimental baseline and formal foundation for AURELIS:
1. Host and runtime environments were inventoried using runtime device introspection; active execution substrate is confirmed as CPU (AMD EPYC 7B12, 240 vCPUs, 400 GiB RAM).
2. The pinned Lean 4 formal build (`lake build`) was executed, verifying zero errors, zero warnings, zero admitted proofs (`sorry`), and zero project axioms across all formal theorems in namespace `Aurelis`.
3. An architecture module specification map was constructed linking equations (2)–(12) to planned modules.
4. Four core hypotheses (H1–H4), workload grids, paired seeds, compute budgets, baseline versions, and raw-data provenance plans were preregistered before viewing downstream results.

---

## 2. Equation-to-Code Mapping & Deliverables

| Deliverable Artifact | Description | Primary Equations / Contracts |
|---|---|---|
| `environment.json` / `environment_inventory.md` | Genuine runtime device and software enumeration | AUTONOMY_PROTOCOL.md § Evidence rules |
| `formal_build_report.md` | Lean 4 build report and theorem mapping | Eqs. (1), (2), (5), (6), (8), (9), (10), (11), (12) |
| `module_specification_map.json` / `module_specification_map.md` | Module specification and responsibilities | IMPLEMENTATION_CONTRACT.md |
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
| Gate 1 | Every claimed device is returned by runtime enumeration | `environment.json`; AMD EPYC 7B12 CPU, JAX CpuDevice | **PASS** |
| Gate 2 | Every metric type has a raw-data provenance plan | `preregistration.md` § 7; explicit logging plans for accuracy, loss, latency, memory, certificates | **PASS** |
| Gate 3 | Compute limits and stop conditions are explicit | `preregistration.md` § 4, `resource_budget.md`; 1800s timeout, 32 GiB RSS cap, NaN/Inf stops | **PASS** |
| Gate 4 | Formal Lean build passes with zero axioms/sorry | `formal_build_report.md`, `raw/lean_build.log`; formal theorems mapped | **PASS** |
| Gate 5 | Architecture module specification and preregistration complete | `module_specification_map.md`, `preregistration.md` complete | **PASS** |

---

## 5. Artifact Hashes

| File | SHA-256 Checksum |
|---|---|
| `environment.json` | `eb5875b11e01e4879e040f7cc1a247a2405bb9c8d3db19543c25d9e201f87616` |
| `environment_inventory.md` | `6a54b5b1ff6ce3664cffde9cedc48b8ff90943fee7d989b139cb5bd7cd878a4a` |
| `formal_build_report.md` | `2788015a97031f452463616b4c878e77a6d28248082da9b137d547b8adb6c295` |
| `module_specification_map.json` | `e27597c2c52c79d14564c7fb1336ab94dbc1ea4ef8c23bb1c09c383af8c792c2` |
| `module_specification_map.md` | `c6e852e9934f96b2b08b00899a45365306cec7484cc2c37ff7d25d592f5f14b5` |
| `preregistration.json` | `fbbcc8a8cebc4faebef12a98de1b1f4b074b8c0c9874be87a293a8b544ab725e` |
| `preregistration.md` | `7b4aea4bfeb57e5ac8b0f6830326491052033d671e833467651083f7112bdece` |
| `resource_budget.json` | `9f576e50aac69def63300dd4cc6d5eb813ff27b6aed2b88d88d87c7e837702ff` |
| `resource_budget.md` | `4ec3a8cdb632f1e21b4ef43977b1a99aff1716eaf1d42faa0370f2d6eff5daa9` |

---

## 6. Next Decision

Phase 0 is complete with status **PASS**. Proceed to **Phase 1: Independent Math Oracles & Formal Correspondence**, implementing clean float64 reference oracles for equations (2)–(12) and validating numerical agreement with Lean formal statements.
