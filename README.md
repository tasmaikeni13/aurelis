# AURELIS

**Attention with Uncertainty-Routed Residuals over an Episodic–Long-range
Inference State**

AURELIS is a proposed same-head hybrid of exact local softmax attention and a
bounded remote Bayesian least-squares state. Recent associations remain in a
window; evicted associations enter the remote state exactly once. A query is
read as

```text
y(q) = Mq + g(q) [vbar(q) - M kbar(q)]
```

where the analytic Bayes gate includes the cross-covariance between the remote
and residual estimators. An explicit episodic responsibility can override that
gate when the target is an observed cached exception rather than a denoised
latent relation.

## Current scope

This revision is the requested theory foundation:

- [standalone paper](aurelis.md);
- [literature review and novelty boundary](research/LITERATURE_REVIEW.md);
- [reproducible fp64 numerical analysis](analysis/README.md);
- [Lean 4 formalization](lean/README.md); and
- exactly nine autonomous research phases, [Phase 0](phases/phase0.md) through
  [Phase 8](phases/phase8.md), governed by the
  [self-correction protocol](phases/AUTONOMY_PROTOCOL.md).

The existing Python experiment directories are pre-Phase-0 substrate and are
not evidence for AURELIS. Phase 0 is deliberately responsible for deleting or
re-deriving those artifacts, migrating the package identity, and implementing
the hybrid on the AMD MI300X/ROCm server. Old results must never be relabeled as
hybrid results.

## Reproduce the current evidence

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r analysis/requirements.txt
.venv/bin/python analysis/aurelis_numerical.py
(cd lean && lake update && lake build)
```

The numerical run regenerates raw CSV/JSON, a report, and all paper figures.
The Lean project is pinned to Lean/mathlib 4.19.0 and contains no proof
placeholders or project axioms.

## Evidence status

| Claim class | Current evidence |
|---|---|
| Handoff, residual, routing, matrix algebra | Analytic derivations plus Lean kernel checks |
| Conditional Gaussian uncertainty | Derivation plus 50,000-trial Monte Carlo calibration |
| Finite-precision mechanism | Deterministic fp64 checks and conditioning sweep |
| Learned features and episodic detection | Pending Phases 3–4 |
| MI300X/ROCm correctness and speed | Pending Phase 5 after Phase 0 implementation |
| Language-model quality | Pending Phases 6–7; no present claim |

See [CLAIMS.md](CLAIMS.md) for the claim-by-claim boundary and
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the gated program.
