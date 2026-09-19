# AURELIS Experiment Log

This log chronicles experimental progress, verification gates, and findings for the AURELIS architecture.

---

## 2026-09-18 — Phase 0: Environment Inventory, Formal Verification, & Preregistration

### 1. Architectural Foundations
AURELIS formulates solve-free recurrent memory with residual-certified retrieval:
- **Solve-Free Gated Delta Recurrence:** Persistent state update follows $S^+ = \alpha S + \beta(v - \alpha S k)k^\top$, completely eliminating per-token key-space matrix inversions ($O(d_k^3)$ Cholesky factorizations).
- **Disjoint Causal Partition:** Cache observations within the recent local window $w$ and evicted observations are strictly partitioned, ensuring newly evicted items enter the recurrent state exactly once.
- **Mass-Consistent Residual Certification:** Completed archive retrieval combines selected exact observations with predicted unread mass under deterministic norm bounds (Eq. 10).

### 2. Machine-Checked Formal Verification (Lean 4)
Built Lean 4 (mathlib 4.19.0) formal core with zero errors, zero warnings, zero admitted proofs (`sorry`), and zero project axioms across all formal theorems in namespace `Aurelis`:
- `exact_recall_injective` & `exact_recall_capacity`: Finite-state recall lower bounds ($|\mathcal{S}| \ge m^n$).
- `deltaRead_add` & `deltaRead_exact_write`: Query linearity and unit-key write reproduction.
- `deltaTransition_energy` & `deltaTransition_nonexpansive`: Rank-one row perturbation energy identity and contraction condition ($\beta \ge 0, \beta \|k\|^2 \le 2$).
- `completedRead_balance`, `completedRead_full`, `completion_error_identity`, & `completedRead_certificate`: Mass normalization, empty-unread endpoint, and deterministic residual norm certificates.
- `exp_score_interval`, `weighted_residual_bound`, `coordinate_product_interval`, & `page_residual_ball`: Coordinate key boxes and page exponential score intervals.
- `handoff_partition` & `recent_length_le_window`: Causal cache handoff partitioning.

### 3. Environment & Hardware Inventory
Enumerated runtime environment using direct device introspection:
- Host: AMD EPYC 7B12 64-Core Processor (240 vCPUs, 400 GiB RAM, Linux 6.6).
- Runtimes: Python 3.10.12, PyTorch 2.14.0 (CPU), JAX 0.4.30 (CPU device).

### 4. Preregistration & Resource Budget
- Formally preregistered core hypotheses H1–H4, workload configurations, paired random seed schedules, baseline versions, and explicit compute caps (1800s timeout, 32 GiB RSS cap, immediate stop on nonfinite outputs).
- Generated complete deliverables in `results/phase0/`: `environment.json`, `environment_inventory.md`, `formal_build_report.md`, `module_specification_map.json`, `module_specification_map.md`, `preregistration.json`, `preregistration.md`, `resource_budget.json`, `resource_budget.md`, `report.md`, and `PASS.md`.
- Phase 0 verification status: **PASS**.
