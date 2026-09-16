# AURELIS Phase 0 PASS

Generated: `2026-09-16T08:46:56.005880+00:00`

Phase 0 status: **PASS**. This record covers migration and the reference/TPU
substrate only. It makes no language-model quality or accelerator-superiority
claim.

## Gate evidence

| Gate | Direct evidence |
|---|---|
| Obsolete identity absent | `results/phase0/migration_audit.txt` (zero matches outside that generated audit) |
| Streaming/history fp64 agreement | `results/phase0/reference_metrics.json`; maximum error `1.066e-13` |
| Disjoint exhaustive occurrence partition | `tests/test_partition.py`, `results/phase0/raw/reference_cases.jsonl`; zero failures |
| Cholesky/dense/capped-inverse agreement and conditioned failure domains | `tests/test_solvers.py`, `results/phase0/reference_metrics.json`; maximum error `3.997e-15` |
| Autograd and gradcheck for inputs/projections | `tests/test_autograd.py`, `results/phase0/raw/pytest.log` |
| Analytic Bayes route and exact episodic hit | `tests/test_routing.py`, `results/phase0/raw/pytest.log` |
| Eager/Inductor/fp64 agreement | `results/phase0/benchmark_metrics.json`; fp32/fp64 `4.632e-06`, compiled/eager `1.729e-06` |
| Cloud TPU v4 Pod measured; forbidden accelerator dependencies absent | `environment.txt`, `results/phase0/environment.json` |
| Lean build; no proof placeholders/project axioms | `results/phase0/raw/lean_build.log`, `lean/PROOF_COVERAGE.md` |
| Full documented command | `scripts/run_phase0.sh` and the five raw command logs |

## Exact reproduction

```bash
./scripts/bootstrap.sh
./scripts/run_phase0.sh
```

The fail-fast command runs the environment audit, Python unit/property and
gradcheck suite, full Lean build, small fp64 reference experiment, Cloud TPU v4
component benchmark, and this completion audit.

## Failed iterations and disposition

- `results/phase0/failures/bootstrap_ensurepip_20260829.md`: the first venv
  bootstrap lacked Ubuntu's matching venv package; it was installed without
  changing the driver, accelerator runtime stack, or Python version.
- `results/phase0/failures/reference_dtype_20260829.md`: the standalone
  experiment inherited fp32 inputs against an fp64 state; all oracle tensors
  now declare fp64 explicitly.
- `results/phase0/failures/environment_missing_python_headers_20260829.md`:
  Inductor compilation helper lacked Python development headers; the matching Ubuntu
  compiler-header package repaired the environment without changing PyTorch or
  accelerator libraries.
- `results/phase0/failures/vectorized_inductor_cumsum_20260829.md`: the bundled
  Triton compiler rejected its generated all-prefix cumulative-sum kernel. The
  exact prefix constructor remains eager, while the complete prepared head is
  compiled as one graph and checked forward/backward.
- `results/phase0/failures/lean_toolchain_missing_20260829.md`: the first full
  runner lacked `lake`; user-scoped elan selected the already pinned Lean
  4.19.0 toolchain, after which the unchanged proof project built.

No scientific gate was weakened. Expected non-positive-definite and explicit
inverse dimension failures remain regression tests.

## Research and mathematical repairs

- Cloud TPU v4 architecture, JAX TPU runtime, OpenXLA HLO compilation, and
  Cholesky solve sources are embedded in
  `results/phase0/environment.json` with the design decision each supports.
- The TPU v4-32 pod slice topology (2x2x4 3D torus, 16 chips / 32 TensorCores)
  is verified directly via local and cluster environment checks.
- No theorem or manuscript equation required correction in this phase.

## Lean coverage

The existing faithful theorems were retained unchanged. Exact coverage and
unformalized probability/numerical/system boundaries are listed in
`lean/PROOF_COVERAGE.md`. The full pinned build output is retained.

## Raw and aggregate artifacts

- `results/phase0/raw/reference_cases.jsonl`
- `results/phase0/raw/component_timings.jsonl`
- `results/phase0/reference_metrics.json`
- `results/phase0/benchmark_metrics.json`
- `results/phase0/reference_report.md`
- `results/phase0/benchmark_report.md`
- `plots/phase0/reference_agreement.png`

## Tested revision and environment fingerprint

- Base commit: `a178c5d74493f98347b3235d48785a5c96b29795`
- Working tree was intentionally dirty with `64` migration paths;
  each JSON record stores the dirty flag, path count, and status SHA-256.
- Environment SHA-256: `a661029915d0bb8085e1411682742e8288755e6107beb9dd1b50a0c90d935c9f`
- Reference metrics SHA-256: `f7f36cbd581fdf49f9247af1b4388386116140edafeaace273fe227c4bb08a8f`
- Benchmark metrics SHA-256: `bfb24397ec1c554b40de35feb78d6ba4d46124b9ddcf90fce3a1599833dbdeb5`

## Remaining limitations outside the Phase 0 claim

- The stable streaming factor is freshly refactorized after handoff; optimized
  rank-one factor updates and periodic refactor policy remain systems work.
- The exact vectorized training path materializes all prefix precision
  matrices, remains eager after the retained Triton cumulative-sum failure,
  and does not claim favorable large-sequence memory use.
- No custom Triton kernel was added because the measured Phase 0 shapes did
  not yet establish a stable fusion target beyond Inductor.
- Learned feature quality, drift, large-scale language modeling, and matched
  throughput comparisons belong to later phases and remain pending.
