# Conjugate State Machines: falsification repository

This repository turns the claims in [`conjugate-state-machines.md`](conjugate-state-machines.md) into reproducible tests. The current completed scope is **Phase 0 through Phase 3**: environment characterization, a mathematically transparent fp64 Gauss–Markov memory, exhaustive finite-epsilon/capacity sweeps, and controlled baseline separation. It does not contain a language model, learned encoders, the dyadic cascade, or optimized kernels.

Current status: **Phase 0 PASS; Phase 1 PASS; Phase 2 PASS; Phase 3 PASS** on the recorded MI300X environment. See [`results/phase0_report.md`](results/phase0_report.md), [`results/phase1_report.md`](results/phase1_report.md), [`results/phase2_interpolation_report.md`](results/phase2_interpolation_report.md), and [`results/phase3_baseline_separation.md`](results/phase3_baseline_separation.md). Later gates remain closed.

> **Hard gate:** NO NLP SCALE EXPERIMENT is allowed until both the synthetic-memory gate and the learned-memory gate pass. Phases 1–3 remain synthetic and unlearned, so they do not unlock NLP experiments.

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
```

PyTorch uses the `torch.cuda` API namespace on ROCm. No CUDA toolkit, NVIDIA device, or CUDA-specific package is assumed.

## Repository map

- [`CLAIMS.md`](CLAIMS.md): falsifiable claims and failure interpretations.
- [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md): dependency graph, gates, and risk controls.
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md): immutable experiment-record schema and run log.
- [`src/csm/memory.py`](src/csm/memory.py): fp64 reference recurrence, Cholesky/solve reads, recomputation, and tiny inverse oracle.
- [`tests/`](tests): equation-level and pathological tests.
- [`experiments/phase1_numerics.py`](experiments/phase1_numerics.py): quantitative Phase 1 experiment and plot generator.
- [`experiments/phase2_interpolation.py`](experiments/phase2_interpolation.py): complete interpolation, conditioning, and capacity sweep.
- [`experiments/phase3_baseline_separation.py`](experiments/phase3_baseline_separation.py): same-dimension/equal-byte baseline and linear-functional comparison.
- [`results/phase0_report.md`](results/phase0_report.md): Phase 0 decision report.
- [`results/phase1_report.md`](results/phase1_report.md): Phase 1 gate evidence.
- [`results/phase2_interpolation_report.md`](results/phase2_interpolation_report.md): Phase 2 gate evidence, including the retained initial numerical-bound failure.
- [`results/phase3_baseline_separation.md`](results/phase3_baseline_separation.md): Phase 3 separation, resource, latency, and no-win evidence.
- [`lean/`](lean): Lean 4 proofs of scan/recurrence identities, matrix invariants, ridge-factor properties, and normalized-softmax separation.

## Interpretation discipline

A passing numerical test validates this implementation against the stated equations; it does not establish the manuscript's architectural or empirical value. A failed theoretical prediction is recorded as evidence against the claim. Results are never silently discarded, and changes to mathematical claims must be paired with a manuscript correction and a new experiment record.
