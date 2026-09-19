# Phase 2: Exact Archive Reference & Invariant Verification

| Invariant / Check | Criterion | Max Abs Error / Status | Status |
|---|---|---|---|
| Full softmax recovery | All pages read recovers exact $y_*$ | 4.88e-16 | **PASS** |
| Reading never writes S | Recurrent state $S$ bit-identical | 0.00e+00 | **PASS** |
| Read idempotence | Retrying read never duplicates mass | 0.00e+00 | **PASS** |
| Integrity failure injection | Corrupted length/version returns `invalid_state` | invalid_state | **PASS** |
