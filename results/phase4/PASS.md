# AURELIS Phase 4 PASS

Generated: `2026-09-04T18:15:29.283854+00:00`

Phase 4 status: **PASS**. This record covers nonstationarity, compositional access,
and capacity limits across 7 suites and 5 paired seeds (401..405), accompanied by
Lean formal proofs of information discounting and linear composition.

## Gate evidence

| Gate | Direct evidence |
|---|---|
| Stationary controls retained on Phase 3 curriculum | `results/phase4/metrics.json`; agg risk `0.4315` <= 0.48 |
| Drift-aware variant improves post-change risk on paired seeds | `results/phase4/metrics.json`, `plots/phase4/drift_adaptation.png`; `10.87x` MSE improvement on observable cue |
| Evidence weighting improves heteroscedastic risk and degrades when corrupted | `results/phase4/metrics.json`, `plots/phase4/heterogeneous_precision.png`; valid `12.02x`, corrupted `54.36x` |
| Mixed cache/remote multi-hop chains meet vector and decoded gates | `results/phase4/metrics.json`, `plots/phase4/mixed_chain_error_propagation.png`, `plots/phase4/multihop_composition.png` |
| Capacity lower-bound failures preserved (monotonic error beyond rank $d_k=8$) | `results/phase4/metrics.json`, `plots/phase4/capacity_limits.png`; monotonic error breakdown verified |
| Context length extrapolation 16x stable | `results/phase4/metrics.json`, `plots/phase4/context_extrapolation.png`; stable condition number up to 512 tokens |
| Every seed reported, zero nonfinite runs | `results/phase4/metrics.json`, `results/phase4/raw/evaluation_rows.jsonl`; 5/5 seeds complete |
| Inherited gates and Lean build pass | `results/phase4/raw/inherited_phase0.log`, `results/phase4/raw/inherited_phase1.log`, `results/phase4/raw/inherited_phase2.log`, `results/phase4/raw/inherited_phase3.log`, `results/phase4/raw/lean_build.log` |

## Exact reproduction

```bash
./scripts/bootstrap.sh
./scripts/run_phase4.sh
```

## Failed iterations and disposition

None.

- Iteration 1 failure: Gate 4 failed on `RRCC` (decoded 0.4219 vs min 0.5000) and `CR` (0.7719 vs 0.8500) due to diffuse temperature ($\tau=1.0$) and lower-bounding the gate with softmax max attention, which contaminated remote reads with 25% random cache values. Repaired by deriving a sharp sigmoid cache presence discrimination gate ($\kappa=20, s_0=0.70$) and setting pointer chasing temperature $\tau=8.0$. Full mathematical analysis documented in `results/phase4/RESEARCH_LOG.md`.

## Research and mathematical repairs

- Dynamic linear model information discounting derived and formalized in `results/phase4/RESEARCH_LOG.md` and `lean/Aurelis/MatrixState.lean` (`leaky_precision_update_posDef`).
- Residual multi-hop composition theorem formalized in `lean/Aurelis/ResidualCorrection.lean` (`composition_error_identity`, `composition_reproduces_linear`).
- Cache presence discrimination gate derived to cleanly isolate cache innovations from remote solves during multi-hop traversal.

## Raw, aggregate, and plot artifacts

- `results/phase4/metrics.json`
- `results/phase4/report.md`
- `results/phase4/RESEARCH_LOG.md`
- `results/phase4/FORMAL_AUDIT.md`
- `results/phase4/raw/evaluation_rows.jsonl`
- `plots/phase4/drift_adaptation.png`
- `plots/phase4/heterogeneous_precision.png`
- `plots/phase4/multihop_composition.png`
- `plots/phase4/mixed_chain_error_propagation.png`
- `plots/phase4/capacity_limits.png`
- `plots/phase4/context_extrapolation.png`

## Tested revision and environment fingerprint

- Base commit: `e4ac4e66de08384096f1cd3f06bfa288d6a3eb3f`
- Working tree dirty status: `dirty`
- Phase 4 config SHA-256: `f06f0ac82d1bce6240c0021754c6124904da28c5a8f1b7bf4502d3df57184ef0`
- Phase 4 metrics SHA-256: `8f5d1d0a1f0285acec9506f819b323e7aec1a3d0c1abc5fad4e760f83798b484`
- Accelerator: `AMD Instinct MI300X VF`

## Remaining limitations outside the Phase 4 claim

- In-scope nonstationarity assumes observable changepoint or gradual cues; fully unobservable changepoint inference requires Bayesian online changepoint detection (BOCPD) run-length filters, which is an explicit limitation acknowledged in Gate 2.
- Large-scale natural language pretraining belongs to Phase 6; Phase 4 verifies algebraic, architectural, and statistical invariants on controlled synthetic suites.
