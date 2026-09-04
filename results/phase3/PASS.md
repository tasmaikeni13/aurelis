# AURELIS Phase 3 PASS

Generated: `2026-09-04T18:12:54.955531+00:00`

Phase 3 status: **PASS**. This record covers the learned feature projections,
episodic routing mechanism, 7 curriculum task families across 5 paired seeds,
ablation falsifications, and formal Lean proofs of episodic gate properties.

## Gate evidence

| Gate | Direct evidence |
|---|---|
| Learned AURELIS solves every task family above preregistered threshold on every seed | `results/phase3/metrics.json`, `results/phase3/raw/evaluation_rows.jsonl` |
| Improves over frozen random features on aggregate risk for every seed | `results/phase3/metrics.json`; risk `0.4625` vs `2.3868` |
| Shared-chart AURELIS beats independent-chart failure ablation and retains usable effective rank | `results/phase3/metrics.json`, `plots/phase3/shared_vs_independent_spectra.png`; erank `13.35` >= 2.0 |
| AURELIS-B calibrated; AURELIS-E materially improves exact exception copy | `results/phase3/metrics.json`, `plots/phase3/episodic_cue_calibration.png`; `1.77x` improvement |
| Observable episodic cue explains override | `results/phase3/metrics.json`; AUROC `1.0000` >= 0.90, R2 `0.9478` >= 0.80 |
| Handoff-boundary degradation within declared tolerance | `results/phase3/metrics.json`, `plots/phase3/cache_boundary_continuity.png`; degradation `-0.0193` <= 0.15 |
| Every seed reported, zero nonfinite runs | `results/phase3/metrics.json`, `results/phase3/raw/models_training.jsonl`; 5/5 seeds complete |
| Inherited gates and Lean build pass | `results/phase3/raw/inherited_phase0.log`, `results/phase3/raw/inherited_phase1.log`, `results/phase3/raw/inherited_phase2.log`, `results/phase3/raw/lean_build.log` |

## Exact reproduction

```bash
./scripts/bootstrap.sh
./scripts/run_phase3.sh
```

## Failed iterations and disposition

None.

## Research and mathematical repairs

- `results/phase3/RESEARCH_LOG.md` documents straight-through estimator (STE) for $g_E = \max(g_B, e_t)$ overcoming the flat subgradient plateau $\partial_b \max(a, b) = 0$ when $b < a$.
- Lean formalization in `lean/Aurelis/Router.lean` gained `episodicGate`, `episodicGate_ge_bayes`, `episodicGate_ge_episodic`, and `episodicGate_bounds`.
- Lean formalization in `lean/Aurelis/Handoff.lean` gained `cache_overlap_redundancy` formalizing why double counting inflates representation history length.

## Raw, aggregate, and plot artifacts

- `results/phase3/metrics.json`
- `results/phase3/report.md`
- `results/phase3/raw/models_training.jsonl`
- `results/phase3/raw/evaluation_rows.jsonl`
- `plots/phase3/task_family_performance.png`
- `plots/phase3/episodic_cue_calibration.png`
- `plots/phase3/shared_vs_independent_spectra.png`
- `plots/phase3/cache_boundary_continuity.png`

## Tested revision and environment fingerprint

- Base commit: `e4ac4e66de08384096f1cd3f06bfa288d6a3eb3f`
- Working tree dirty status: `dirty`
- Phase 3 config SHA-256: `09cee4dd498c97497320bedbc618e3b77c02b62fb66ddaadb800c2bb82704fd8`
- Phase 3 metrics SHA-256: `211e6217f2b22da71eeada273a01bef3571542cd8cfd712dd3779fcdc47f218a`

## Remaining limitations outside the Phase 3 claim

- Small synthetic task curriculum; natural language pretraining and scaling belongs to later phases (Phases 6–7).
- Recurrent sequence state updates evaluated with exact causal prefix Cholesky solves and standard PyTorch autograd.
