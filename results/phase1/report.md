# AURELIS Phase 1 Report: Independent Mathematics & Formal Correspondence

**Date UTC:** `2026-09-19T07:46:25.105833+00:00`  
**Git Commit:** `ee14a250a7d99e035e134cd97e07decafc4d7048`  
**Phase Status:** **PASS**

---

## 1. Executive Summary

Phase 1 establishes the mathematical foundation, dual independent reference oracles, formal theorem correspondence, and pathology test suite for AURELIS:
1. **Dual Independent CPU Oracles (fp64):** Clean scalar loop oracle (`ScalarOracle`) and tensor oracle (`TensorOracle`) implemented without shared math helpers to avoid circular agreement.
2. **Full Equation Coverage:** All equations (2)–(12) and Eq. (13) verified across multiple vector dimensions ($d_k, d_v \in \{2, 4, 8, 16, 32\}$) at fp64 tolerances `atol=1e-10, rtol=1e-9`.
3. **Lean Formal Correspondence & Extension:** Built `lake build` with zero errors, zero warnings, zero admitted proofs (`sorry`), zero custom axioms. Extended formal theorems to include grouped completion balance and certificate (Eq. 13).
4. **Pathology Verification:** Tested all 12 registered numerical pathologies including zero keys, beta endpoints, negative query coordinates, uniform/concentrated scores, huge value outliers ($10^6$), and stale-state residuals.
5. **Counterexamples and Impossibility Demonstrations:** Formally demonstrated that dropping the normalizer term underestimates error, bounded state cannot meet arbitrary exact recall past capacity, finite softmax is an interpolator rather than hard lookup, and error reduction is non-monotonic on fetch.

---

## 2. Equation-to-Code Mapping & Deliverables

| Deliverable Artifact | Description | Primary Equations / Contracts |
|---|---|---|
| `oracles_verification.json` / `.md` | Dual fp64 oracle agreement results | Eqs. (2)–(13) |
| `pathology_results.json` / `.md` | Error and scale records across 12 pathology families | AUTONOMY_PROTOCOL.md § Evidence rules |
| `counterexamples.json` / `.md` | 4 counterexamples & impossibility proofs | phase1.md § Pathologies and gates |
| `theorem_ledger.md` | Formal correspondence between Lean 4 and paper equations | aurelis.md §1-§6 |
| `raw/lean_build.log` | Lean compiler log | Formal build integrity |
| `raw/pytest_run.log` | Pytest test execution trace | Diagnostic test suite |

---

## 3. Gate Verification & Outcomes

| Gate | Criterion | Evidence | Status |
|---|---|---|---|
| Gate 1 | Dual independent CPU oracles agree at fp64 atol=1e-10, rtol=1e-9 | `oracles_verification.md`; all 11 operations pass | **PASS** |
| Gate 2 | All 12 pathology families evaluated and recorded | `pathology_results.md`; 12/12 pass with scales recorded | **PASS** |
| Gate 3 | Normalizer omission counterexample demonstrated | `counterexamples.md` § 1; naive bound=0.0 < actual error=0.35 | **PASS** |
| Gate 4 | Capacity impossibility and finite softmax non-lookup demonstrated | `counterexamples.md` §§ 2, 3 | **PASS** |
| Gate 5 | Non-monotonic error reduction property demonstrated | `counterexamples.md` § 4; bound increases on fetch | **PASS** |
| Gate 6 | Lean 4 formal build passes with zero sorry and zero project axioms | `raw/lean_build.log`, `theorem_ledger.md` | **PASS** |
| Gate 7 | Streaming and history fp64 calculations agree exactly | `test_phase1_oracles.py::test_streaming_vs_history_equivalence` | **PASS** |

---

## 4. Next Decision

Phase 1 is complete with status **PASS**. All deterministic mathematical contracts and formal correspondences have been verified. Proceed to **Phase 2: Streaming State, Bounded Read, Exact Archive Reference**.

---

## 5. Artifact Hashes

| File | SHA-256 Checksum |
|---|---|
| `oracles_verification.json` | `842c0ca7723b0e0536f66b7bb23b460251a6d8bb9671d056fd8215af2c557f39` |
| `oracles_verification.md` | `321aafddb21c24d81b7359725dbcac27a2e6f8c4306716a0bcff0aab8e36b391` |
| `pathology_results.json` | `341102c65b76f5e74ccaa949171fee15fb4340c5e7a0c09cddf5d294a4bad2e0` |
| `pathology_results.md` | `fc0a2e890789797480c4b2838f38f3e59a55cb16ab4d0485e82ddf183a008b8b` |
| `counterexamples.json` | `b67d84206af7214393a5b0a4a256279c042c97ec62b31561c12c5bc85c2023b3` |
| `counterexamples.md` | `030d1d60175d4d434a2d4adfe4c34b6b4b7577dfd0890cf253c1ce2c0ee9b684` |
| `theorem_ledger.md` | `8ece167bf69248160b46e57f009b4028341fbbc7c3cebfc91afc157a81ad15a6` |
| `PASS.md` | `860d610e331f4c9bb51c38a8280594793efb6f854047ab77fefdbed3d9d82c5a` |
