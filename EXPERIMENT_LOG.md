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
