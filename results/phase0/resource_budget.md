# AURELIS Phase 0: Resource Budget

**Generated UTC:** `2026-09-19T07:02:05.357014+00:00`

| Phase | Target Device | Max Wallclock | RSS Memory Cap |
|---|---|---|---|
| `phase0` | CPU | 600s | 16.0 GiB |
| `phase1` | CPU | 1200s | 16.0 GiB |
| `phase2` | CPU | 1800s | 16.0 GiB |
| `phase3` | CPU | 2400s | 24.0 GiB |
| `phase4` | CPU | 3600s | 32.0 GiB |
| `phase5` | CPU/Accelerator | 7200s | 32.0 GiB |
| `phase6` | CPU/Accelerator | 14400s | 64.0 GiB |
| `phase7` | CPU/Accelerator | 7200s | 32.0 GiB |
| `phase8` | CPU/Accelerator | 14400s | 64.0 GiB |
| `phase9` | CPU | 3600s | 16.0 GiB |


### Global Guardrails

- **Process Timeout:** Terminate job and log `BLOCKED_RESOURCE`.
- **Memory Cap:** Enforce 32 GiB RSS cap.
- **Numerical Sanity:** Abort on nonfinite values (NaN/Inf).
