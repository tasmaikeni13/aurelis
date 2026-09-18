# AURELIS-R v2 Phase 0: Resource Budget & Stop Conditions

**Generated UTC:** `2026-09-18T06:45:55.319086+00:00`  
**Hardware Environment:** `AMD EPYC 7B12 (240 vCPUs)` (400.0 GiB RAM)  
**Compute Execution Mode:** Strictly runtime-enumerated CPU execution

---

## 1. Phase-by-Phase Compute Allocation

| Phase | Compute Allocation | Max Wall-Clock Timeout | Status |
|---|---|---|---|
| `phase0` | 2 core-hrs | 30 mins | **COMPLETING** |
| `phase1` | 5 core-hrs | 45 mins | **NOT_STARTED** |
| `phase2` | 8 core-hrs | 60 mins | **NOT_STARTED** |
| `phase3` | 12 core-hrs | 90 mins | **NOT_STARTED** |
| `phase4` | 20 core-hrs | 120 mins | **NOT_STARTED** |
| `phase5` | 25 core-hrs | 180 mins | **NOT_STARTED** |
| `phase6` | 50 core-hrs | 360 mins | **NOT_STARTED** |
| `phase7` | 20 core-hrs | 120 mins | **NOT_STARTED** |
| `phase8` | 60 core-hrs | 480 mins | **NOT_STARTED** |
| `phase9` | 10 core-hrs | 60 mins | **NOT_STARTED** |

---

## 2. Resource Policies & Stopping Conditions

1. **Strict Resource Caps:** No phase may consume more than its allotted compute budget without explicit prior authorization.
2. **Negative Results as Valid Endpoints:** If an architecture hypothesis is rejected (e.g. recurrence does not beat recurrence-free completion in Phase 4), that scaling branch terminates cleanly as `FAILED_HYPOTHESIS`. Compute is not wasted on unpromising models.
3. **Graceful Resource Failure:** If external resource constraints prevent evaluation (e.g. host memory exhaustion), record `BLOCKED_RESOURCE` rather than relaxing scientific gates.
