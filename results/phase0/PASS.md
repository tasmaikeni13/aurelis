# AURELIS Phase 0 PASS

Generated: `2026-08-29T17:21:47.536795+00:00`

Phase 0 status: **PASS**. This record covers migration and the reference/ROCm
substrate only. It makes no language-model quality or accelerator-superiority
claim.

## Gate evidence

| Gate | Direct evidence |
|---|---|
| Obsolete identity absent | `results/phase0/migration_audit.txt` (zero matches outside that generated audit) |
| Streaming/history fp64 agreement | `results/phase0/reference_metrics.json`; maximum error `7.105e-14` |
| Disjoint exhaustive occurrence partition | `tests/test_partition.py`, `results/phase0/raw/reference_cases.jsonl`; zero failures |
| Cholesky/dense/capped-inverse agreement and conditioned failure domains | `tests/test_solvers.py`, `results/phase0/reference_metrics.json`; maximum error `1.776e-15` |
| Autograd and gradcheck for inputs/projections | `tests/test_autograd.py`, `results/phase0/raw/pytest.log` |
| Analytic Bayes route and exact episodic hit | `tests/test_routing.py`, `results/phase0/raw/pytest.log` |
| Eager/Inductor/fp64 agreement | `results/phase0/benchmark_metrics.json`; fp32/fp64 `7.255e-06`, compiled/eager `1.907e-06` |
| MI300X/ROCm measured; forbidden accelerator dependencies absent | `environment.txt`, `results/phase0/environment.json` |
| Lean build; no proof placeholders/project axioms | `results/phase0/raw/lean_build.log`, `lean/PROOF_COVERAGE.md` |
| Full documented command | `scripts/run_phase0.sh` and the five raw command logs |

## Exact reproduction

```bash
./scripts/bootstrap.sh
./scripts/run_phase0.sh
```

The fail-fast command runs the environment audit, Python unit/property and
gradcheck suite, full Lean build, small fp64 reference experiment, MI300X
eager/compiled component benchmark, and this completion audit.

## Failed iterations and disposition

- `results/phase0/failures/bootstrap_ensurepip_20260829.md`: the first venv
  bootstrap lacked Ubuntu's matching venv package; it was installed without
  changing the driver, ROCm stack, or Python version.
- `results/phase0/failures/reference_dtype_20260829.md`: the standalone
  experiment inherited fp32 inputs against an fp64 state; all oracle tensors
  now declare fp64 explicitly.
- `results/phase0/failures/environment_missing_python_headers_20260829.md`:
  Inductor's HIP helper lacked Python development headers; the matching Ubuntu
  compiler-header package repaired the environment without changing PyTorch or
  ROCm.
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

- Current AMD compatibility, MI300X optimization, rocSOLVER Cholesky, and
  PyTorch HIP-semantics sources (accessed 2026-08-29) are embedded in
  `results/phase0/environment.json` with the design decision each supports.
- The host/wheel version difference is reported as measured behavior, not an
  unsupported compatibility assumption.
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

- Base commit: `db7ef177b56cbda36bcebeaa4f31bc2a542d134c`
- Working tree was intentionally dirty with `20` migration paths;
  each JSON record stores the dirty flag, path count, and status SHA-256.
- Environment SHA-256: `368d3c1bbd5b3518fad6280b66353313c8dab4361866c34059dc241d01baa342`
- Reference metrics SHA-256: `a2e92723df23cf6a500f5d17cc47af8c9f2354fcbbfcd0dbd831423a38e5ae51`
- Benchmark metrics SHA-256: `e6d25427fc357f7ce91456ef39c69a89dcca3fd92366b5d9283bbb5b48ac6a5f`

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
