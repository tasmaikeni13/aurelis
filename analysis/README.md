# AURELIS theory analysis

This directory contains deterministic, CPU/fp64 numerical checks for the
AURELIS head.  It is intentionally separate from the legacy implementation:
the implementation migration and MI300X/ROCm kernels are the job of Phase 0.

Run:

```bash
.venv/bin/python analysis/aurelis_numerical.py
```

The command regenerates `results/` and `plots/`, fails on violated analytic
gates, and records conditioning pathologies rather than discarding them.
