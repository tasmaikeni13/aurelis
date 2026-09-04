# AURELIS Phase 1 PASS

Generated: `2026-09-04T17:55:49.041156+00:00`

Phase 1 status: **PASS**. This record covers the mathematical oracle,
calibration, dtype/conditioning pathologies, handoff boundary, and analytic
cost gate. It makes no learned-feature, language-model-quality, or competitive
throughput claim.

## Gate evidence

| Gate | Direct evidence |
|---|---|
| Pinned dimension/window/rank/prior/temperature/evidence/pathology sweep | `configs/phase1_oracle.json`, `results/phase1/raw/deterministic_rows.jsonl`, `streaming_rows.jsonl`; 90 cases and 1467 prefixes |
| Streaming state equals full-history oracle at every prefix; no occurrence loss/duplication | `metrics.json`, `raw/streaming_rows.jsonl`; maximum tolerance ratio `2.742e-04`, zero partition failures |
| Residual, general-gate, reproduction, finite-ridge, exact-hit, and two raw-gate identities | `metrics.json`, `tests/test_phase1_identities.py`; maximum valid identity ratio `3.354e-06` |
| Below/interior/above clipping, dense-grid optimum, endpoint non-inferiority | `metrics.json` routing/deterministic rows; all three clipping regions exercised |
| Exact conditional `V_R,V_H,K_RH,V(g)` Monte Carlo with preregistered 99% intervals | `metrics.json`; 5 matched regimes × 50000 trials |
| Covariance omission control is measurably wrong | `metrics.json`; paired regret z-score `128.630` |
| Latent denoising / episodic copy conflict and misspecification counterexample | `metrics.json` target-conflict record |
| Handoff output and gradients immediately before/at/after | `metrics.json` handoff boundaries/discontinuities; tracked occurrence count always one |
| Condition, solve residual, forward error, Cholesky/nonfinite, fp64/fp32/bf16-storage | `metrics.json`, `raw/conditioning_rows.jsonl`, `plots/phase1/conditioning_precision.png`; identical-quantized-input policy recorded |
| Analytic/observed state bytes and operation counts | `metrics.json` cost record; every implementation byte prediction exact, profiler calls retained |
| Invalid domains remain visible | `metrics.json`; 12 cancellation-unresolved and 4 explicitly invalid rows |
| Complete Python suite | `results/phase1/raw/pytest.log`; 46 tests |
| Lean 4.19/mathlib build and statement audit | `results/phase1/raw/lean_build.log`, `FORMAL_AUDIT.md`, `lean/PROOF_COVERAGE.md`; no `sorry`, `admit`, or project `axiom` |
| All inherited Phase 0 gates | `results/phase1/raw/inherited_phase0.log`, `results/phase0/PASS.md` |

## Exact reproduction

```bash
./scripts/bootstrap.sh
./scripts/run_phase1.sh
```

The fail-fast runner executes the complete inherited Phase 0 environment,
Python, Lean, reference, and ROCm gates; reruns all Phase 1 Python tests and the
full Lean build; runs the pinned Phase 1 experiment; then performs this
requirement-level audit.

## Failed iterations and disposition

- `results/phase1/failures/cancellation_conditioning_evaluator_20260829.md` — retained failure and disposition.
- `results/phase1/failures/completion_audit_cuda_device_label_20260829.md` — retained failure and disposition.
- `results/phase1/failures/environment_missing_python_headers_20260829.md` — retained failure and disposition.
- `results/phase1/failures/lean_gated_endpoint_normalization_20260829.md` — retained failure and disposition.
- `results/phase1/failures/lean_toolchain_absent_20260829.md` — retained failure and disposition.
- `results/phase1/failures/missing_pinned_venv_20260829.md` — retained failure and disposition.
- `results/phase1/failures/relative_config_path_20260829.md` — retained failure and disposition.
- `results/phase1/failures/torch_geomspace_api_20260829.md` — retained failure and disposition.

The scientific equations and preregistered constants were not weakened. The
only evaluator repair composes solve conditioning with scalar subtractive-
cancellation conditioning and retains unresolvable rows as invalid-domain
evidence.

## Research and mathematical repairs

- `results/phase1/RESEARCH_LOG.md` records the LAPACK/Goldberg basis for the
  cancellation-aware evaluator repair and its exact design consequence.
- Lean gained `gated_error_identity`, `gatedRead_one`,
  `scalar_ridge_slope_error`, and `scalar_ridge_residual_bound`; statement
  fidelity and remaining analytic boundaries are audited in
  `results/phase1/FORMAL_AUDIT.md` and `lean/PROOF_COVERAGE.md`.

## Raw, aggregate, and plot artifacts

- `results/phase1/metrics.json`
- `results/phase1/report.md`
- `results/phase1/raw/streaming_rows.jsonl`
- `results/phase1/raw/deterministic_rows.jsonl`
- `results/phase1/raw/conditioning_rows.jsonl`
- `results/phase1/raw/experiment.log`
- `plots/phase1/conditional_calibration.png`
- `plots/phase1/conditioning_precision.png`

## Tested revision and environment fingerprint

- Base commit: `e4ac4e66de08384096f1cd3f06bfa288d6a3eb3f`
- Working tree intentionally dirty with `42` Phase 0/1 research paths.
- Phase 1 config SHA-256: `50e309cddf5476ed8808687cc0e18e6782eb057e2349a84844c3d029ae6a9f84`
- Phase 1 metrics SHA-256: `c23b919b419425972f6528aefe489315aa62f2f86aae32dbf3e1322a52563499`
- Phase 0 environment SHA-256: `9d64fa4773bdca6cd522dc7d70b267129ea9878aba1b1928f5f0b507f9ad9ae8`

## Remaining limitations outside the Phase 1 claim

- Native bfloat16 Cholesky on the active `cuda` device is unsupported on the pinned Torch build;
  bfloat16 storage is therefore promoted to fp32 compute and labeled as such.
- The profiler does not report FLOPs for Cholesky/triangular solves; analytic
  arithmetic and observed supported FLOPs/operator invocations are separate.
- Conditional probability calibration assumes the declared disjoint
  linear-Gaussian model; the retained misspecification row demonstrates the
  boundary rather than extending the theorem.
- Full matrix spectral-norm ridge bounds remain analytic; Lean covers a
  faithful scalar specialization. Learned features, drift, LM quality, and
  competitive throughput belong to later phases.
