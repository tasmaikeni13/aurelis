# AURELIS Phase 2 Report: Streaming Semantics & Exact Archive Reference

**Date UTC:** `2026-09-19T10:37:19.644301+00:00`  
**Git Commit:** `3915e076592a52f31805f2153b54bf9b5b055aa4`  
**Phase Status:** **PASS**

---

## 1. Executive Summary

Phase 2 implements and verifies the complete streaming semantics, state lifecycle, and exact archive reference retrieval for AURELIS:
1. **Recurrent State & Ring Buffer:** Implemented fixed-capacity recurrent state $S \in \mathbb{R}^{d_v \times d_k}$, causal ring buffer with source-position write gates ($\alpha, \beta$), and disjoint occurrence partitioning.
2. **Exact Append-Only Raw Archive:** Implemented paged storage retaining causal occurrence IDs, original encodings, and coordinate score and residual envelopes ($k^-, k^+, c_j, \rho_j$) with strict per-query causal masking in partial pages.
3. **Decode Equivalence:** Proved bit-for-bit / numerical equivalence across token-by-token execution, multi-token prefill, and continuation across both Bounded and Archive operating contracts.
4. **Session Lifecycle & Speculative Rollback:** Exposed `reset`, `snapshot`, `restore`, `fork`, `cancel` operations. Demonstrated complete tuple checkpointing and exact speculative draft token rollback.
5. **Memory Profile & Live Tensor Accounting:** Measured live tensor bytes across context lengths up to $t=128$, empirically verifying that bounded working state strictly plateaus once $t \ge w$, with archive bytes labeled by tier (`tier1_working_ram`, `tier2_host_archive`, `tier3_cold_index`).
6. **Critical Operational Invariants:** Verified that reading archive pages never mutates $S$, retrying reads is strictly idempotent, and corrupted archive length or index version triggers `invalid_state` failure rather than returning a certificate.

---

## 2. Equation-to-Code Mapping & Deliverables

| Deliverable Artifact | Description | Primary Equations / Contracts |
|---|---|---|
| `src/aurelis/archive.py` | Append-only raw archive, paged envelopes, exact reference read | Eqs. (7), (8), (10), (11), (12) |
| `src/aurelis/streaming.py` | Streaming processor, ring buffer handoff, exact partitioning | Eqs. (2), (3) |
| `src/aurelis/session.py` | Stateful `AurelisSession` with snapshot, fork, and rollback | §3, §8 |
| `src/aurelis/types.py` | `StateSnapshot`, `ArchiveEntry`, `MemoryProfile`, `ReadResult` | IMPLEMENTATION_CONTRACT.md |
| `streaming_verification.json` / `.md` | Handoff and partition across steps, request isolation | §3 |
| `archive_reference.json` / `.md` | Full softmax recovery, idempotence, integrity failure injection | Eqs. (7), (8), (10) |
| `memory_profile.json` / `.md` | Live tensor memory measurements and bounded plateau proof | §1.1, §8 |
| `decode_equivalence.json` / `.md` | Prefill vs token-by-token vs continuation equivalence | §8 |
| `gate_records.json` | Complete audit of all 15 Phase 2 gate criteria | phase2.md |
| `raw/lean_build.log` | Lean 4 compiler build trace | Formal integrity |
| `raw/pytest_run.log` | Full Pytest suite execution trace (83 passed tests) | Test verification |

---

## 3. Gate Verification & Outcomes

| Gate | Criterion | Evidence | Status |
|---|---|---|---|
| Gate 1 | Exercise t < w, t = w, and t > w | Verified by test suite and empirical logs | **PASS** |
| Gate 2 | Empty remote state behavior | Verified by test suite and empirical logs | **PASS** |
| Gate 3 | Multiple page boundaries crossed cleanly | Verified by test suite and empirical logs | **PASS** |
| Gate 4 | Duplicate content with distinct IDs preserved | Verified by test suite and empirical logs | **PASS** |
| Gate 5 | Packed examples with document boundary resets | Verified by test suite and empirical logs | **PASS** |
| Gate 6 | Variable lengths and interleaved requests without contamination | Verified by test suite and empirical logs | **PASS** |
| Gate 7 | Reading pages never writes recurrent state S | Verified by test suite and empirical logs | **PASS** |
| Gate 8 | Retrying read never duplicates an observation | Verified by test suite and empirical logs | **PASS** |
| Gate 9 | Prefix snapshots restore complete state tuple | Verified by test suite and empirical logs | **PASS** |
| Gate 10 | Rejected speculative tokens rolled back cleanly | Verified by test suite and empirical logs | **PASS** |
| Gate 11 | Bounded state live tensor memory strictly plateaus after window fills | Verified by test suite and empirical logs | **PASS** |
| Gate 12 | Archive bytes grow as predicted and labeled by tier | Verified by test suite and empirical logs | **PASS** |
| Gate 13 | Wrong archive length or index version fails with invalid_state | Verified by test suite and empirical logs | **PASS** |
| Gate 14 | Token-by-token, prefill, and continuation agree for bounded and archive modes | Verified by test suite and empirical logs | **PASS** |
| Gate 15 | Full history softmax recovered when all pages read | Verified by test suite and empirical logs | **PASS** |

---

## 4. Next Decision

Phase 2 is complete with status **PASS**. Streaming state semantics, exact archive reference retrieval, and session lifecycle are fully verified. Proceed to **Phase 3: Validated Certificate, Refinement, and Failure Semantics**.

---

## 5. Artifact Hashes

| File | SHA-256 Checksum |
|---|---|
| `streaming_verification.json` | `d86e3b7f220198d3309fc91c4acd38c76cb6591b21ee36b49fe7a971fdea40a3` |
| `streaming_verification.md` | `d57e59f931a8fa6c509d5c863e3d38144f7b919a1965db59c9e4da0e8cb6b5cd` |
| `archive_reference.json` | `c1aad6ed08c57bf0669ef012b757b0c2b513341059880b9cd883e85a9cf647a5` |
| `archive_reference.md` | `dc1ab5dd09d9634841624c2b55f8bc26e0759c66d1fcb3dd4f0f7707f27d756e` |
| `memory_profile.json` | `fea734fc8112fe61c9882b1ce02831530c82be42cbfc08fdf95b09457274c085` |
| `memory_profile.md` | `6677119446d38cb7ebac5ebd17608839ba502e623f266d55b262a6cb603edad6` |
| `decode_equivalence.json` | `11d678d88be99fd15bd4625b0f10858339790587c479f1c0682833662886a534` |
| `decode_equivalence.md` | `4cddc52711861e402726fea3f692ce1b37d77d35c8b3b7f24b0837841bbc76d9` |
| `gate_records.json` | `ae485f6e69697cb82bd5e9675976a53e954b737c2e56751c6622ed8708c2d912` |
| `PASS.md` | `06275609054582656712e8021bdf488881879608e33babb196ce85c9b485f603` |
