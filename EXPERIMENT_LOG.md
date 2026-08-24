# Experiment log

Results are append-only records. A rerun creates or replaces generated machine-readable files only when its complete record is preserved in Git history.

## Required record format

```yaml
experiment_id: unique descriptive identifier
timestamp_utc: ISO-8601 timestamp
git_commit: exact tested source commit
working_tree_dirty: true_or_false
config: committed config path plus resolved values
seed: scalar or complete seed list
hardware: GPU model/architecture/VRAM and relevant host facts
software_versions: Python, PyTorch, HIP, ROCm, Triton and relevant libraries
wall_clock_time: seconds
peak_vram: bytes and GiB
metrics: names, values, units, and aggregation rule
plots: committed plot paths
interpretation: scoped conclusion, failures, and non-conclusions
```

## Runs

### P0-ENV-20260823

- timestamp UTC: `2026-08-23T15:45:30.482905+00:00`
- git commit: `223afde` (audit script source was introduced at `80235c0`)
- working tree dirty: `true` because generated audit outputs were created after the source checkpoint
- config: `scripts/audit_environment.py`, GEMM size 8192, 7 measured repetitions after 3 warmups
- seed: `0`
- hardware: AMD Instinct MI300X VF, `gfx942`, 191.688 GiB visible VRAM; 20 host cores; 235.948 GiB RAM
- software versions: Python 3.12.3; ROCm 7.0.2; PyTorch 2.8.0+rocm7.0.2; HIP 7.0.51831; Triton 3.4.0+rocm7.0.2
- wall-clock time: 17.452 seconds (machine-readable record is authoritative)
- peak VRAM: 2.387 GiB allocated during audit
- metrics: `torch.compile` pass; direct Triton pass; all four dtype GEMMs finite; median sanity TFLOP/s in `results/environment.json`
- plots: none (environment audit)
- interpretation: host is suitable for the Phase 1 fp64 reference. Throughput numbers are untuned health checks, not benchmark claims.

### P1-REF-20260823

- timestamp UTC: `2026-08-23T15:52:13.672343+00:00`
- git commit: `60d5922bac15526599352f805a6382e32ae0d331`
- working tree dirty: `false` at experiment start
- config: `configs/phase1_reference.json` with resolved values embedded in `results/phase1_metrics.json`
- seeds: `[0,1,2,3,4]` for the 25 recurrence cases; specialized fixed seeds are visible in experiment source
- hardware: AMD Instinct MI300X VF
- software versions: Python 3.12.3; PyTorch 2.8.0+rocm7.0.2; HIP 7.0.51831; NumPy 2.3.2
- wall-clock time: 25.054 seconds
- peak VRAM: 134,683,136 bytes (0.125433 GiB allocated)
- metrics: state max error `8.881784e-16`; read max error `2.273737e-12`; minimum S eigenvalue `-2.855521e-16`; gradcheck pass; all pathology outputs finite; noisy-risk ratio range `[0.905,1.104]`
- plots: `plots/phase1/recurrence_consistency.png`, `interpolation_error.png`, `conditioning_fp32.png`, `noise_averaging.png`
- interpretation: Phase 1 reference equations pass their configured fp64 gate. This does not validate learned memory, kernel performance, the cascade, NLP, or claims beyond the measured/formalized subset.

Machine-readable record: [`results/phase1_metrics.json`](results/phase1_metrics.json). Human report: [`results/phase1_report.md`](results/phase1_report.md).

### P2-INTERPOLATION-STRICT-20260823 — FAIL

- timestamp UTC: `2026-08-23T16:08:25.972587+00:00`
- git commit: `d35e29768dbe689fa427b5abc107860e7ff37100`; complete failed outputs preserved at commit `2f7dba6`
- working tree dirty: `false` at experiment start
- config: the initial committed `configs/phase2_interpolation.json`; same full grid as the corrected run
- seeds: `[0,1,2]`, deterministically mixed with dimensions and regime indices
- hardware: AMD Instinct MI300X VF
- software versions: Python 3.12.3; PyTorch 2.8.0+rocm7.0.2; HIP 7.0.51831; NumPy 2.3.2
- wall-clock time: 151.392 seconds
- peak VRAM: 135,920,128 bytes (0.126585 GiB allocated)
- metrics: 12,600 rows / 1,800 datasets; three of four gates passed; maximum measured-error / exact-real-bound ratio `1.003804` exceeded the strict `1.000100` fp64 threshold
- plots: the five Phase 2 plot paths at commit `2f7dba6`
- interpretation: **FAIL retained.** The strict gate compared a floating-point Cholesky result with an exact-real theorem bound. Its worst excess was about `1.5e-14` absolute on an error of order `1e-12`; this diagnosed a missing numerical-roundoff qualification, not a counterexample to the inequality.

### P2-INTERPOLATION-20260823 — PASS

- timestamp UTC: `2026-08-23T16:13:29.098769+00:00`
- git commit: `3862d607f306830db1f558fcc4d5738ace0253f7`
- working tree dirty: `false` at experiment start
- config: `configs/phase2_interpolation.json`, with all resolved values embedded in `results/phase2_metrics.json`
- seeds: `[0,1,2]`, deterministically mixed with dimensions and regime indices
- hardware: AMD Instinct MI300X VF
- software versions: Python 3.12.3; PyTorch 2.8.0+rocm7.0.2; HIP 7.0.51831; NumPy 2.3.2
- wall-clock time: 150.704 seconds
- peak VRAM: 135,920,128 bytes (0.126585 GiB allocated)
- metrics: 12,600 rows / 1,800 datasets; under-capacity independent median low-epsilon error `2.214321e-12`; p99 `6.195306e-8`; epsilon direction `100%`; 672/6,363 exact-bound comparisons exceeded only at fp64 scale; maximum ratio after the committed conditioning-scaled allowance `0.9999999999998866`; dependent/over-capacity median error separation `3.034e11`
- plots: `plots/phase2/error_vs_epsilon.png`, `error_vs_load.png`, `error_vs_min_gram_eigenvalue.png`, `confidence_vs_error.png`, `load_conditioning_heatmap.png`
- interpretation: Phase 2 supports full-row-rank interpolation and the finite-epsilon direction in its scoped synthetic domain, and clearly exposes rank-limited failure for dependent/over-capacity associations. It does not establish baseline superiority or learned/NLP performance.

Machine-readable record: [`results/phase2_metrics.json`](results/phase2_metrics.json). Human report: [`results/phase2_interpolation_report.md`](results/phase2_interpolation_report.md).

### P3-BASELINE-SEPARATION-20260823 — PASS

- timestamp UTC: `2026-08-23T16:33:24.974687+00:00`
- git commit: `e1c592feb09dc769477b3e8fd566ac07a55e9e0e`; the preceding complete passing output set is also preserved at that checkpoint
- working tree dirty: `false` at experiment start
- config: `configs/phase3_baselines.json`, with all resolved values embedded in `results/phase3_metrics.json`
- seeds: `[0,1,2]`, deterministically mixed with dimensions and regime indices
- hardware: AMD Instinct MI300X VF
- software versions: Python 3.12.3; PyTorch 2.8.0+rocm7.0.2; HIP 7.0.51831; NumPy 2.3.2
- wall-clock time: 23.320 seconds
- peak VRAM: 135,595,008 bytes (0.126283 GiB allocated)
- metrics: 17,010 recall rows, 900 linear-functional rows, and 15 latency rows; correlated compressed-baseline/CSM median error ratio `2.427e6`; equal-byte nonconvex CSM median absolute error `1.399e-7`; softmax/CSM median separation `4.496e6`; convex-hull checks exact within `2.220e-16`
- plots: `plots/phase3/recall_vs_load_same_dimension.png`, `recall_vs_load_equal_state_budget.png`, `csm_epsilon_sweep.png`, `linear_functional_separation.png`, `prepared_read_latency.png`
- interpretation: **PASS with important no-win regimes.** CSM separates from compressed Hebbian/linear memories on correlated keys and from normalized softmax on equal-byte nonconvex linear functionals. Oracle-tuned explicit softmax and least squares match or beat CSM on stored-key recall; the reference CSM read is also much slower in the recorded prepared-read latency diagnostic. No universal superiority is claimed.

Machine-readable record: [`results/phase3_metrics.json`](results/phase3_metrics.json). Human report: [`results/phase3_baseline_separation.md`](results/phase3_baseline_separation.md).

### P4-UNCERTAINTY-STRICT-20260824 — FAIL

- timestamp UTC: initial complete run immediately preceding `2026-08-24T13:18:33.004071+00:00`
- git commit: `a8ad6e59db43ee61adadf4888d69edcb9ac705c5`; working tree dirty because Phase 4 source and generated outputs were not checkpointed separately
- config: initial `configs/phase4_uncertainty.json`; selective coverages omitted exact `0.5` and required a risk ratio no larger than `0.4`
- seeds: `[0..15]` for duplicate/precision sweeps; 48 deterministic specialized confidence trials per model
- hardware/software: AMD Instinct MI300X VF; Python 3.12.3; PyTorch 2.8.0+rocm7.0.2; HIP 7.0.51831; NumPy 2.3.2
- metrics: seven of eight checks passed; the closest retained fraction to one half was `0.6`, with Gaussian selective/full risk ratio `0.4716538778`, above the initial `0.4` threshold
- interpretation: **FAIL retained.** The mismatch was between the named half-coverage check and a grid that omitted one half. `results/phase4_initial_strict_gate_failure.md` preserves all initial gate measurements and the correction rationale.

### P4-UNCERTAINTY-20260824 — PASS

- timestamp UTC: `2026-08-24T13:25:02.626168+00:00` (final rerun with the ridge oracle assembled independently from raw observations)
- git commit: `a8ad6e59db43ee61adadf4888d69edcb9ac705c5`; working tree dirty at experiment start
- config: `configs/phase4_uncertainty.json`, including exact `0.5` selective coverage and all resolved values in `results/phase4_metrics.json`
- seeds: `[0..15]` plus 48 deterministic confidence trials per data model
- hardware/software: AMD Instinct MI300X VF; Python 3.12.3; PyTorch 2.8.0+rocm7.0.2; HIP 7.0.51831; NumPy 2.3.2
- wall-clock time: 19.907 seconds; peak VRAM: 134,306,816 bytes (0.125083 GiB)
- metrics: 4,800 duplicate rows, 432 precision rows, 3 conflict rows, 23,040 confidence-query rows; CSM/oracle difference `5.535e-16`; repeat-risk slope `-1.018`; uniform/precision risk `23.206x`; Gaussian Spearman `0.704`; high-error AUROC `0.875`; actual/predicted MSE `0.957`; half/full selective risk `0.404`
- plots: five Phase 4 plots under `plots/phase4/`, all visually inspected
- interpretation: **PASS only in-model.** Linear-Gaussian ridge and calibration behavior pass. Laplace, Student-like, and nonlinear results characterize misspecification; no Bayesian optimality is claimed for them.

Machine-readable record: [`results/phase4_metrics.json`](results/phase4_metrics.json). Human report: [`results/phase4_uncertainty_and_noise.md`](results/phase4_uncertainty_and_noise.md).

### P5-MULTIHOP-20260824 — PASS

- timestamp UTC: `2026-08-24T13:20:17.591147+00:00`
- git commit: `a8ad6e59db43ee61adadf4888d69edcb9ac705c5`; working tree dirty at experiment start
- config: `configs/phase5_multihop.json`, with all resolved values embedded in `results/phase5_metrics.json`
- seeds: `[0,1,2]`, deterministically mixed with dimension, edge count, representation, and graph regime
- hardware/software: AMD Instinct MI300X VF; Python 3.12.3; PyTorch 2.8.0+rocm7.0.2; HIP 7.0.51831; NumPy 2.3.2
- wall-clock time: 92.646 seconds; peak VRAM: 135,149,056 bytes (0.125867 GiB)
- metrics: 12,960 pointer-chasing rows and 90 latency rows; controlled minimum decoded success `1.0`; maximum controlled H=16 vector error `1.600e-7`; maximum propagation-bound relative excess `7.973e-9`; many-to-one operator norm up to `3.0`
- systems diagnostic: at `d_key=K=64`, H=16 prepared fp64 reference latency was 12,260.259 us for CSM and 522.226 us for repeated explicit softmax at comparable leading FLOPs; no practical-efficiency win is claimed
- plots: four Phase 5 plots under `plots/phase5/`, all visually inspected
- interpretation: **PASS architectural gate.** One maintained CSM state supports 16 chained adaptive reads on controlled codes. Random geometry, capacity, epsilon, amplification, and reference runtime failures are retained, and the systems claim remains open.

Machine-readable record: [`results/phase5_metrics.json`](results/phase5_metrics.json). Human report: [`results/phase5_multihop.md`](results/phase5_multihop.md).
