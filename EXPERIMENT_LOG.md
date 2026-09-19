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

---

## 2026-09-19 — Phase 1: Independent Mathematics & Formal Correspondence

### 1. Dual Independent CPU Oracles (fp64)
- Built scalar and tensor mathematical oracles (`ScalarOracle` and `TensorOracle`) without shared helpers to eliminate circular verification.
- Verified agreement at `atol=1e-10, rtol=1e-9` across all core equations (2)–(13): gated delta recurrence, unit-key write reproduction, bounded read transport, linear reproduction, perturbation energy identity, normalized archive completion, two-term error identity, residual certificates, page coordinate score intervals, and grouped completion comparator.

### 2. Machine-Checked Lean 4 Extension
- Compiled Lean 4 formal core with zero errors, zero warnings, zero admitted proofs (`sorry`), and zero custom axioms.
- Extended machine-checked proofs to include grouped completion balance and certificate bounds (Eq. 13).

### 3. Pathologies and Counterexamples
- Evaluated 12/12 registered numerical pathologies including zero keys, beta endpoints, negative query coordinates, uniform/concentrated scores, extreme value outliers ($10^6$), and stale-state residuals.
- Verified 4 formal counterexamples: normalizer omission error underestimation, capacity impossibility, finite softmax non-lookup, and non-monotonic error reduction on page fetch.
- Phase 1 verification status: **PASS**.

---

## 2026-09-19 — Phase 2: Streaming Semantics & Exact Archive Reference

### 1. State Layout & Causal Handoff
- Implemented fixed-capacity recurrent state $S \in \mathbb{R}^{d_v \times d_k}$ and recent ring buffer with source-position write gates $\alpha, \beta \in [0, 1]$.
- Validated disjoint occurrence partitioning ($L_t \cap R_t = \emptyset, L_t \cup R_t = \{1 \dots t\}$) across $t < w$, $t = w$, and $t > w$.
- Verified exactly-once eviction writes into $S$ and preserved duplicate content with distinct occurrence IDs.

### 2. Exact Append-Only Raw Archive
- Built paged raw storage retaining causal occurrence IDs and original encodings.
- Implemented sealed page coordinate bounding boxes ($k^-, k^+$) and value balls ($c_j, \rho_j$) with strict per-query causal masking in partial pages.
- Verified that full archive read recovers exact full history softmax ($y_*$) to machine precision ($< 10^{-15}$).

### 3. Autoregressive Decode Equivalence
- Verified exact numerical equivalence across token-by-token execution, multi-token prefill, and continuation for both Bounded and Archive operating modes ($0.00$ error in bounded mode, $< 10^{-14}$ in archive mode).
- Verified true cached decode $O(1)$ complexity in bounded mode.

### 4. Session Lifecycle & Speculative Rollback
- Implemented stateful `AurelisSession` exposing `reset`, `snapshot`, `restore`, `fork`, and `cancel`.
- Proved that speculative draft tokens can be rolled back via `cancel` to produce bit-exact continuation matching clean runs.
- Verified complete request isolation without cross-session state contamination across interleaved streams.

### 5. Memory Profiling & Live Tensor Accounting
- Measured live tensor state across context lengths $t \in [1, 128]$, empirically confirming that bounded working state strictly plateaus once $t \ge w$.
- Verified linear archive byte growth labeled across storage tiers (`tier1_working_ram`, `tier2_host_archive`, `tier3_cold_index`).

### 6. Operational Invariants & Integrity Checks
- Proved that reading archive pages never writes to $S$ (bit-identical before and after read).
- Proved read idempotence: retrying reads never duplicates observations or inflates mass.
- Verified that corrupted archive length or index version triggers explicit `invalid_state` failure rather than returning a certificate.
- Phase 2 verification status: **PASS**.

