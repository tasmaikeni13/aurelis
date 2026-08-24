# Conjugate State Machines: falsification repository

This repository turns the claims in [`conjugate-state-machines.md`](conjugate-state-machines.md) into reproducible tests. The current completed scope is **Phase 0 through Phase 5**: environment characterization, a mathematically transparent fp64 Gauss–Markov memory, finite-epsilon/capacity sweeps, controlled baseline separation, noisy-evidence calibration, and multi-hop functional graphs. It does not contain a language model, learned encoders, the dyadic cascade, or optimized kernels.

Current status: **Phase 0 PASS; Phase 1 PASS; Phase 2 PASS; Phase 3 PASS; Phase 4 PASS; Phase 5 PASS** on the recorded MI300X environment. See the phase reports under [`results/`](results/). The initial strict Phase 4 selective-risk failure is retained alongside the corrected passing run.

> **Hard gate:** NO NLP SCALE EXPERIMENT is allowed until both the synthetic-memory gate and the learned-memory gate pass. Phases 1–5 remain synthetic and unlearned, so these results do not unlock NLP experiments.

## Reproduce on the ROCm host

The commands below create an isolated environment; they do not modify the system Python.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e '.[dev]'

.venv/bin/python scripts/audit_environment.py --output environment.txt \
  --json-output results/environment.json
.venv/bin/python -m pytest
.venv/bin/python experiments/phase1_numerics.py \
  --config configs/phase1_reference.json
.venv/bin/python experiments/phase2_interpolation.py \
  --config configs/phase2_interpolation.json
.venv/bin/python experiments/phase3_baseline_separation.py \
  --config configs/phase3_baselines.json
.venv/bin/python experiments/phase4_uncertainty_and_noise.py \
  --config configs/phase4_uncertainty.json
.venv/bin/python experiments/phase5_multihop.py \
  --config configs/phase5_multihop.json
```

PyTorch uses the `torch.cuda` API namespace on ROCm. No CUDA toolkit, NVIDIA device, or CUDA-specific package is assumed.

## Repository map

- [`CLAIMS.md`](CLAIMS.md): falsifiable claims and failure interpretations.
- [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md): dependency graph, gates, and risk controls.
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md): immutable experiment-record schema and run log.
- [`phases/`](phases/): authoritative Phase 0–7 specifications.
- [`src/csm/memory.py`](src/csm/memory.py): fp64 reference recurrence, Cholesky/solve reads, recomputation, and tiny inverse oracle.
- [`tests/`](tests): equation-level and pathological tests.
- [`experiments/phase1_numerics.py`](experiments/phase1_numerics.py): quantitative Phase 1 experiment and plot generator.
- [`experiments/phase2_interpolation.py`](experiments/phase2_interpolation.py): complete interpolation, conditioning, and capacity sweep.
- [`experiments/phase3_baseline_separation.py`](experiments/phase3_baseline_separation.py): same-dimension/equal-byte baseline and linear-functional comparison.
- [`experiments/phase4_uncertainty_and_noise.py`](experiments/phase4_uncertainty_and_noise.py): noisy duplicates, precision weighting, calibration, OOD queries, and misspecification.
- [`experiments/phase5_multihop.py`](experiments/phase5_multihop.py): controlled/random functional graphs, adaptive-read baselines, propagation bounds, and systems diagnostics.
- [`results/phase0_report.md`](results/phase0_report.md): Phase 0 decision report.
- [`results/phase1_report.md`](results/phase1_report.md): Phase 1 gate evidence.
- [`results/phase2_interpolation_report.md`](results/phase2_interpolation_report.md): Phase 2 gate evidence, including the retained initial numerical-bound failure.
- [`results/phase3_baseline_separation.md`](results/phase3_baseline_separation.md): Phase 3 separation, resource, latency, and no-win evidence.
- [`results/phase4_uncertainty_and_noise.md`](results/phase4_uncertainty_and_noise.md): Phase 4 in-model calibration and outside-model degradation evidence.
- [`results/phase5_multihop.md`](results/phase5_multihop.md): Phase 5 chained-read accuracy, error attribution, FLOPs, and latency evidence.
- [`lean/`](lean): Lean 4 proofs of scan/recurrence identities, matrix invariants, ridge-factor properties, and normalized-softmax separation.

## Interpretation discipline

A passing numerical test validates this implementation against the stated equations; it does not establish the manuscript's architectural or empirical value. A failed theoretical prediction is recorded as evidence against the claim. Results are never silently discarded, and changes to mathematical claims must be paired with a manuscript correction and a new experiment record.
