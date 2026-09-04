# AURELIS Phase 2 PASS

Generated: `2026-09-04T17:56:11.461194+00:00`

Phase 2 status: **PASS**. This record covers the hybrid mechanism separation,
four fair budget comparison views, nine falsification suites, multi-seed Gaussian
regime validation, and formal Lean proofs of covariance gate optimality.

## Gate evidence

| Gate | Direct evidence |
|---|---|
| 10 baseline models and ablations with fair budgets | `configs/phase2_baselines.json`, `tests/test_phase2_baselines.py`, `results/phase2/raw/baselines_rows.jsonl` |
| Linear reproduction isolated from temperature/state confounds | `metrics.json`, `results/phase2/raw/falsification_rows.jsonl`; error < 1e-10 across all tau |
| AURELIS-B advantage in Gaussian regimes across every seed | `metrics.json`; 10/10 seeds positive advantage |
| AURELIS-E episodic exception benefit isolated from Bayes | `metrics.json`; certified exception error < 1e-12 |
| Retained nonlinear/misspecified regime with no AURELIS advantage | `metrics.json`; 10 cases documented where local attention outperforms AURELIS |
| Capacity failures agree with rank/window limits | `metrics.json`, `plots/phase2/capacity_limits.png`; collapse at w=32 and d_k=16 |
| Full covariance gate outperforms or equals independence heuristic | `metrics.json`, `plots/phase2/covariance_advantage.png`; paired z-score `9.07` >= 5.0 |
| Lean proofs of independence heuristic suboptimality | `lean/Aurelis/Router.lean`, `lean/PROOF_COVERAGE.md`; `clippedGate_le_clippedIndependentGate` |
| All inherited Phase 0 and Phase 1 gates | `results/phase2/raw/inherited_phase0.log`, `results/phase2/raw/inherited_phase1.log` |

## Exact reproduction

```bash
./scripts/bootstrap.sh
./scripts/run_phase2.sh
```

## Failed iterations and disposition

- `results/phase2/failures/linear_reproduction_prior_shrinkage_20260904.md`
- `results/phase2/failures/report_md_generation_order_20260904.md`
- `results/phase2/failures/softmax_temperature_one_hot_precision_20260904.md`
- `results/phase2/failures/torch_randn_like_generator_api_20260904.md`

## Research and mathematical repairs

- `results/phase2/RESEARCH_LOG.md` documents baseline derivations and covariance mechanism.
- Lean gained `independentGate`, `clippedIndependentGate`, and `clippedGate_le_clippedIndependentGate` in `lean/Aurelis/Router.lean`.

## Raw, aggregate, and plot artifacts

- `results/phase2/metrics.json`
- `results/phase2/report.md`
- `results/phase2/raw/baselines_rows.jsonl`
- `results/phase2/raw/falsification_rows.jsonl`
- `results/phase2/raw/correlated_rows.jsonl`
- `results/phase2/raw/context_sweep_rows.jsonl`
- `plots/phase2/mechanism_separation.png`
- `plots/phase2/covariance_advantage.png`
- `plots/phase2/capacity_limits.png`
- `plots/phase2/state_latency_pareto.png`

## Tested revision and environment fingerprint

- Base commit: `e4ac4e66de08384096f1cd3f06bfa288d6a3eb3f`
- Working tree dirty status: `dirty`
- Phase 2 config SHA-256: `4655c62c9f7d46e3e69adb61429fc467ceadb268a918559b0da999c7a2698033`
- Phase 2 metrics SHA-256: `cc16bb3c40d51e256310417d265122aece712d12b3863fd96b476e352b6b0c7f`

## Remaining limitations outside the Phase 2 claim

- Evaluated in simulated memory regimes and synthetic sequence tasks; learned representation training belongs to Phase 3.
- Native Hybrid recurrent summary slots used fixed key/value summaries; end-to-end backprop through recurrent slot updates is left to later phases.
