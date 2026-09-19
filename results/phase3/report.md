# AURELIS Phase 3 Report: Certificate Implementation & Retrieval Policy

**Date UTC:** `2026-09-19T13:01:50.699428+00:00`  
**Git Commit:** `0d7144962b4a1d85c59495b2f4a783b68ae5c057`  
**Phase Status:** **PASS**

---

## 1. Executive Summary

Phase 3 implements and verifies the complete deterministic certificate, conservative arithmetic error policy, and retrieval policy for AURELIS:
1. **Computable Page Envelopes (Eqs. 11, 12):** Built coordinate key boxes $[k^-, k^+]$, outward value centers $c_j$ and radii $\rho_j$, partial-page handling, and flat unread scanning with independent summary containment validation.
2. **Conservative Arithmetic Error Policy ($\delta_{\text{num}}$):** Implemented documented Higham forward error bounds covering dot products, shifted exponentials, Euclidean norms, vector/scalar reductions, division perturbation, and output vector rounding across float32 and float64.
3. **Certified Stopping Rule:** Returned certified strictly when $\mathcal{E}_A + \delta_{\text{num}} \le \epsilon$. Proved that dense fp64 reference outputs lie within returned enclosures everywhere (ratio $\ge 1.0$).
4. **Retrieval Policy & Non-Monotone Candidate Tracking (Paper §6.2):** Implemented priority heuristic $[b_j + \eta_j \|r - \widehat{y}_A\|] / \text{cost}_j$, candidate snapshotting, and strict retention of the best valid candidate across intermediate steps.
5. **Distinct Failure & Lifecycle Statuses:** Enforced distinct statuses: `certified`, `full_read`, `budget_exhausted`, `invalid_interval`, `invalid_state`, `archive_unavailable`, `archive_error`.
6. **Comprehensive Stress Suite & Pathology Verification:** Audited all 11 registered stress cases including underflow, overflow, numerator cancellation, zero unread mass, partial pages, negative query coordinates, corrupt bounds injection, missing pages, NaNs, and I/O timeouts. Confirmed that no invalid result receives certified status.

---

## 2. Equation-to-Code Mapping & Deliverables

| Deliverable Artifact | Description | Primary Equations / Contracts |
|---|---|---|
| `src/aurelis/certificate.py` | Outward score & page envelopes, arithmetic error bounds, summary validator | Eqs. (10), (11), (12) |
| `src/aurelis/policy.py` | Retrieval policy, priority heuristic, candidate tracking, budget management | Eqs. (8), (10), Paper §6.2 |
| `src/aurelis/archive.py` | Integrated reference read with validated certificate & policy | Operating contract archive mode |
| `src/aurelis/types.py` | `CertificateBound` (approximation bound, allowance), distinct `ReadStatus` | Types and status contracts |
| `certificate_validation.json` / `.md` | Summary containment and disjoint unread cover verification | §6.1, §6.2 |
| `arithmetic_error_model.json` / `.md` | Documented forward error bounds for float32/float64 | §6.4 |
| `retrieval_policy_benchmark.json` / `.md` | Context, page size, tolerance benchmark & error ratios | §6.2, §8 |
| `stress_pathology_results.json` / `.md` | 11 registered stress cases, failure injection, timeout | Gates §Phase 3 |
| `gate_records.json` | Complete audit of all 11 Phase 3 gate criteria | phase3.md |
| `raw/lean_build.log` | Lean 4 formal compiler build trace | Formal integrity |
| `raw/pytest_run.log` | Full Pytest suite execution trace (116 passed tests) | Test verification |

---

## 3. Gate Verification & Outcomes

| Gate | Criterion | Evidence | Status |
|---|---|---|---|
| Gate 1 | Computable page envelopes (key boxes, value centers, radii, partial pages) | Verified in test_phase3_envelopes.py & certificate_validation.json | **PASS** |
| Gate 2 | Disjoint exhaustive unread partition cover validation | Verified in test_phase3_envelopes.py & certificate_validation.json | **PASS** |
| Gate 3 | Documented validated forward arithmetic error bounds (delta_num) | Verified in test_phase3_arithmetic_error.py & arithmetic_error_model.json | **PASS** |
| Gate 4 | Stopping certified only when E_A + delta_num <= epsilon | Verified in test_phase3_enclosure.py & policy.py | **PASS** |
| Gate 5 | Dense fp64 reference outputs strictly enclosed in returned bounds (ratio >= 1.0) | Verified in test_phase3_enclosure.py & retrieval_policy_benchmark.json | **PASS** |
| Gate 6 | Non-monotone candidate tracking (best valid candidate preserved) | Verified in test_phase3_retrieval_policy.py & policy.py | **PASS** |
| Gate 7 | Paper §6.2 priority heuristic and cost accounting | Verified in test_phase3_retrieval_policy.py & policy.py | **PASS** |
| Gate 8 | Distinct statuses: certified, full_read, budget_exhausted, invalid_interval, invalid_state, archive_unavailable, archive_error | Verified in types.py, policy.py & stress_pathology_results.json | **PASS** |
| Gate 9 | Stress suite: underflow, overflow, cancellation, zero unread, negative coords, partial pages | Verified in test_phase3_stress_and_pathologies.py & stress_pathology_results.json | **PASS** |
| Gate 10 | Deliberate corrupt bounds injection detected and rejected from certification | Verified in test_phase3_stress_and_pathologies.py & stress_pathology_results.json | **PASS** |
| Gate 11 | Missing pages, NaNs, and I/O timeouts return explicit non-certified failure statuses | Verified in test_phase3_stress_and_pathologies.py & stress_pathology_results.json | **PASS** |

---

## 4. Next Decision

Phase 3 is complete with status **PASS**. The deterministic residual certificate, conservative arithmetic error policy, summary containment validator, and retrieval policy are fully verified and stress-tested. Proceed to **Phase 4: Falsification and Novelty-Critical Ablations**.

---

## 5. Artifact Hashes

| File | SHA-256 Checksum |
|---|---|
| `certificate_validation.json` | `3511ec8e111a651ddba39862e6e4664987890c2cf5d6fd815f475735bb2ae0e7` |
| `certificate_validation.md` | `68330cf065e8ecd7eae5104a272f643d1beb51c03bc34ee4ce8d4b9cb6d51e96` |
| `arithmetic_error_model.json` | `e8a2496d0702e319f160a4edc44b2a9067ffc16a3ce39436488160b808ce732b` |
| `arithmetic_error_model.md` | `097b74f0babb9cefd152a7b68d7760442e5c2e5c51fb8ca945000fc246c1c73e` |
| `retrieval_policy_benchmark.json` | `2ff682eb15f1f9aad163aee47518b92f53181433ef01d7436ccdb37d101b68b7` |
| `retrieval_policy_benchmark.md` | `f62ea416ef0528c7bed7babb1e108a5e9552d17df862e3e4e22d9cf4b4a1738f` |
| `stress_pathology_results.json` | `091d4faf71a871d4d0d4d5322550659a966aab684e137190f8ec3ae5493dabeb` |
| `stress_pathology_results.md` | `b3312ff18318726c912c574f9528d4ae82e2554320e6cc6216f2d2533cac8f5e` |
| `gate_records.json` | `8d9c6204925f0d1b10b1e81afc1bbc456037140e06eea767bbc68a2418c8db95` |
| `PASS.md` | `d8f8c075f3edf7949b3f0a71598d6f1bc95e6acc9e8d23f7a871558da98fa230` |

