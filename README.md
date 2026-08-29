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

## Phase 0 implementation

The repository now contains an AURELIS-native reference substrate:

- immutable batched/multi-head streaming state with a fixed ring cache and
  exact delayed occurrence handoff;
- an independently assembled fp64 historical oracle and a dimension-capped
  explicit-inverse test oracle;
- Cholesky, dense-solve, remote, full-residual, AURELIS-B, and AURELIS-E paths
  with all router and solve diagnostics;
- an exact vectorized all-boundary training reference and learned projection
  wrapper with autograd/gradcheck coverage;
- eager and TorchInductor forward/backward measurements on one AMD Instinct
  MI300X under ROCm; and
- structured configs, raw records, reports, plots, tests, benchmarks, scripts,
  formal proofs, and research sources.

Phase 0 makes no language-model quality or accelerator-superiority claim.

## Reproduce Phase 0

```bash
./scripts/bootstrap.sh
./scripts/run_phase0.sh
```

The second command is fail-fast: it runs the live environment audit, Python
unit/property/gradcheck suite, pinned Lean build, fp64 reference experiment,
MI300X eager/compiled benchmark, and completion audit. Direct gate evidence is
in [`results/phase0/PASS.md`](results/phase0/PASS.md).

The standalone manuscript and its earlier deterministic numerical evidence
remain separately reproducible with `./scripts/run_aurelis_theory.sh`.

## Repository map

| Path | Purpose |
|---|---|
| `src/aurelis/` | Functional read, immutable streaming state, independent history oracle, vectorized training path, learned projections |
| `tests/` | Partition, oracle agreement, solver pathology, router, autograd, and Inductor regression gates |
| `experiments/` | Small fp64 scientific reference experiment |
| `benchmarks/` | Synchronized ROCm component and full-head measurements |
| `configs/` | Versioned Phase 0 experiment/benchmark inputs |
| `results/phase0/` | Raw logs, machine-readable metrics, reports, failure records, and PASS record |
| `plots/phase0/` | Generated Phase 0 figure |
| `scripts/` | Isolated bootstrap, environment audit, fail-fast runner, and verifier |
| `lean/` | Pinned Lean 4 formalization and exact proof-coverage boundary |
| `analysis/` | Manuscript-level deterministic fp64 analysis |
| `phases/` | Governing protocol and Phases 0–8 |

## Evidence status

| Claim class | Current evidence |
|---|---|
| Handoff, residual, routing, matrix algebra | Lean checks, unit/property tests, independent fp64 paths |
| Conditional Gaussian uncertainty | Derivation, Monte Carlo calibration, dense gate minimization test |
| Finite-precision mechanism | CPU/fp64 oracle plus MI300X fp32 eager/compiled comparison |
| Learned features and episodic detection | Pending Phases 3–4 |
| MI300X/ROCm substrate correctness | Phase 0 measured; optimization/comparative speed remains pending |
| Language-model quality | Pending Phases 6–7; no present claim |

See [CLAIMS.md](CLAIMS.md) for the claim-by-claim boundary and
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the gated program.
