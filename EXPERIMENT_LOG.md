# AURELIS research log

## 2026-08-28 — Theory foundation

### Literature research

Reviewed primary work on local/recurrent hybrids, linear and delta attention,
test-time regression, cumulative least-squares sequence layers, same-layer
hybrids, and 2026 hybrid mechanism studies. The frozen comparison and novelty
boundary is `research/LITERATURE_REVIEW.md`.

### Architecture decision

Named the architecture **AURELIS** and fixed delayed handoff plus the residual
read `Mq+g(vbar-Mkbar)`. Derived the correlated endpoint covariance and
projected analytic Bayes gate. Separated latent-denoising and episodic-copy
targets with a distinct override rather than claiming one universal gate.

### Numerical analysis

Command:

```bash
.venv/bin/python analysis/aurelis_numerical.py
```

Seed `20260828` passed all assertions. Highlights:

- residual identity maximum absolute error: `9.992e-16`;
- gate-form equivalence: `3.331e-16`;
- linear reproduction error: `2.285e-16`;
- hard one-hot exception error: `0`;
- conditional routed predicted/empirical MSE: `0.143750 / 0.143329`;
- routed relative calibration error: `0.293%`; and
- gate regret and endpoint-noninferiority slack: `0` at fp64 resolution.

Alternate seeds `17`, `29`, and `41` passed the same assertion suite using
separate temporary output directories. Raw committed evidence is under
`analysis/results/` and `analysis/plots/`.

### Formal analysis

Rebuilt the proof project under namespace `Aurelis`, pinned to Lean/mathlib
4.19.0. `lake build` passed. Search found no `sorry`, `admit`, or declared
project axiom. Coverage is recorded in `lean/PROOF_COVERAGE.md`.

### Paper and automation

Deleted the prior manuscript and wrote `aurelis.md` as a standalone paper.
Replaced the previous phase set with exactly nine numbered prompts and a shared
self-correction protocol. These prompts are not execution reports; Phase 0 and
all trained/system phases remain pending.
