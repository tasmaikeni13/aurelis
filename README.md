# Conjugate State Machines: falsification repository

This repository turns the claims in [`conjugate-state-machines.md`](conjugate-state-machines.md) into reproducible tests. The completed scope is **Phase 0 through Phase 10**: environment characterization, a mathematically transparent fp64 Gauss–Markov memory, finite-epsilon/capacity sweeps, controlled baseline separation, noisy-evidence calibration, multi-hop functional graphs, learned feature maps and gates, ROCm systems optimization, a tiny language-model gate, and a seeded 100M-token natural-language comparison. It does not contain the dyadic cascade or an approximate inverse.

Current status: **Phase 0 PASS; Phase 1 PASS; Phase 2 PASS; Phase 3 PASS; Phase 4 PASS; Phase 5 PASS; Phase 6 PASS; Phase 7 PASS; Phase 8 PASS; Phase 9 PASS; Phase 10 scaling gate PASS** on the recorded MI300X environment. See the phase reports under [`results/`](results/). Failed strict gates and failed assumptions are retained in reports, ablations, the experiment log, or Git history rather than silently discarded.

> **Hard gate:** NO NLP SCALE EXPERIMENT is allowed until both the synthetic-memory gate and the learned-memory gate pass. Those prerequisites passed through Phase 7 before the explicitly authorized Phase 8–10 work began. Phase 10's qualifying advantage is constant incremental state, not a blanket quality or speed win: CSM remained slower and used more training VRAM in the recorded comparison.

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
.venv/bin/python experiments/phase6_learnability.py \
  --config configs/phase6_learnability.json
.venv/bin/python experiments/phase7_gating.py \
  --config configs/phase7_gating.json
.venv/bin/python experiments/phase8_mi300x_systems.py \
  --config configs/phase8_mi300x.json
.venv/bin/python experiments/phase9_tiny_lm.py \
  --config configs/phase9_tiny_lm.json
.venv/bin/python experiments/phase10_small_nlp.py \
  --config configs/phase10_small_nlp.json
```

PyTorch uses the `torch.cuda` API namespace on ROCm. No CUDA toolkit, NVIDIA device, or CUDA-specific package is assumed.

## Repository map

- [`CLAIMS.md`](CLAIMS.md): falsifiable claims and failure interpretations.
- [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md): dependency graph, gates, and risk controls.
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md): immutable experiment-record schema and run log.
- [`phases/`](phases/): authoritative Phase 0–10 specifications.
- [`src/csm/memory.py`](src/csm/memory.py): fp64 reference recurrence, Cholesky/solve reads, recomputation, and tiny inverse oracle.
- [`src/csm/learning.py`](src/csm/learning.py): differentiable batched CSM, shared key/query feature chart, baselines, and bounded scalar gates.
- [`src/csm/systems.py`](src/csm/systems.py): batched affine summaries, chunk composition, associative prefix scan, and fp32 CSM reads.
- [`src/csm/language.py`](src/csm/language.py): matched Transformer, CSM, hybrid, and recurrent decoder implementations with incremental state.
- [`src/csm/textdata.py`](src/csm/textdata.py): checksum-pinned WikiText loading, byte tokenizer, and six diagnostic sequence families.
- [`tests/`](tests): equation-level and pathological tests.
- [`experiments/phase1_numerics.py`](experiments/phase1_numerics.py): quantitative Phase 1 experiment and plot generator.
- [`experiments/phase2_interpolation.py`](experiments/phase2_interpolation.py): complete interpolation, conditioning, and capacity sweep.
- [`experiments/phase3_baseline_separation.py`](experiments/phase3_baseline_separation.py): same-dimension/equal-byte baseline and linear-functional comparison.
- [`experiments/phase4_uncertainty_and_noise.py`](experiments/phase4_uncertainty_and_noise.py): noisy duplicates, precision weighting, calibration, OOD queries, and misspecification.
- [`experiments/phase5_multihop.py`](experiments/phase5_multihop.py): controlled/random functional graphs, adaptive-read baselines, propagation bounds, and systems diagnostics.
- [`experiments/phase6_learnability.py`](experiments/phase6_learnability.py): learned-feature tasks, random/Hebbian/attention controls, geometry measurements, and post-hoc ablations.
- [`experiments/phase7_gating.py`](experiments/phase7_gating.py): learned evidence precision, observable-cue forgetting, innovation-only failure ablation, and joint gates.
- [`experiments/phase8_mi300x_systems.py`](experiments/phase8_mi300x_systems.py): MI300X component profile, precision/path sweeps, oracle checks, and baseline measurements.
- [`experiments/phase9_tiny_lm.py`](experiments/phase9_tiny_lm.py): matched 5M–7M-parameter, 10M-token decoder gate.
- [`experiments/phase10_small_nlp.py`](experiments/phase10_small_nlp.py): three-seed, 27M–30M-parameter, 100M-token controlled comparison.
- [`results/phase0_report.md`](results/phase0_report.md): Phase 0 decision report.
- [`results/phase1_report.md`](results/phase1_report.md): Phase 1 gate evidence.
- [`results/phase2_interpolation_report.md`](results/phase2_interpolation_report.md): Phase 2 gate evidence, including the retained initial numerical-bound failure.
- [`results/phase3_baseline_separation.md`](results/phase3_baseline_separation.md): Phase 3 separation, resource, latency, and no-win evidence.
- [`results/phase4_uncertainty_and_noise.md`](results/phase4_uncertainty_and_noise.md): Phase 4 in-model calibration and outside-model degradation evidence.
- [`results/phase5_multihop.md`](results/phase5_multihop.md): Phase 5 chained-read accuracy, error attribution, FLOPs, and latency evidence.
- [`results/phase6_learnability.md`](results/phase6_learnability.md): Phase 6 gate evidence and the matched-coordinate failure study.
- [`results/phase7_gating.md`](results/phase7_gating.md): Phase 7 evidence/drift gates and the changepoint-detection scope correction.
- [`results/phase8_mi300x_systems.md`](results/phase8_mi300x_systems.md): Phase 8 hardware tax and optimized-path evidence.
- [`results/phase9_tiny_lm.md`](results/phase9_tiny_lm.md): Phase 9 stability, perplexity, diagnostics, and systems evidence.
- [`results/phase10_small_nlp.md`](results/phase10_small_nlp.md): Phase 10 seeded natural-language, long-context, and decode-state comparison.
- [`lean/`](lean): Lean 4 proofs of scan/recurrence identities, matrix invariants, ridge-factor properties, and normalized-softmax separation.

## Interpretation discipline

A passing numerical test validates this implementation against the stated equations; it does not establish the manuscript's architectural or empirical value. A failed theoretical prediction is recorded as evidence against the claim. Results are never silently discarded, and changes to mathematical claims must be paired with a manuscript correction and a new experiment record.
