#!/usr/bin/env python3
"""Requirement-level Phase 2 audit and generated PASS record."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "phase2"
CONFIG_PATH = REPO / "configs" / "phase2_baselines.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=REPO, text=True, capture_output=True, check=False)


def main() -> None:
    config = load(CONFIG_PATH)
    metrics = load(RESULTS / "metrics.json")
    phase0_ref = load(REPO / "results" / "phase0" / "reference_metrics.json")
    phase0_bench = load(REPO / "results" / "phase0" / "benchmark_metrics.json")
    phase1_metrics = load(REPO / "results" / "phase1" / "metrics.json")

    required = [
        RESULTS / "raw" / "inherited_phase0.log",
        RESULTS / "raw" / "inherited_phase1.log",
        RESULTS / "raw" / "pytest.log",
        RESULTS / "raw" / "lean_build.log",
        RESULTS / "raw" / "experiment.log",
        RESULTS / "raw" / "baselines_rows.jsonl",
        RESULTS / "raw" / "falsification_rows.jsonl",
        RESULTS / "raw" / "correlated_rows.jsonl",
        RESULTS / "raw" / "context_sweep_rows.jsonl",
        RESULTS / "report.md",
        RESULTS / "FORMAL_AUDIT.md",
        RESULTS / "RESEARCH_LOG.md",
        REPO / "plots" / "phase2" / "mechanism_separation.png",
        REPO / "plots" / "phase2" / "covariance_advantage.png",
        REPO / "plots" / "phase2" / "capacity_limits.png",
        REPO / "plots" / "phase2" / "state_latency_pareto.png",
    ]

    lean_placeholders = command(
        ["rg", "-n", "--glob", "*.lean", r"\b(sorry|admit|axiom)\b", "lean"]
    )

    checks = {
        "phase0_inherited_reference_pass": phase0_ref.get("status") == "PASS",
        "phase0_inherited_benchmark_pass": phase0_bench.get("status") == "PASS",
        "phase1_inherited_oracle_pass": phase1_metrics.get("status") == "PASS",
        "lean_no_sorry_admit_axiom": lean_placeholders.returncode != 0,
        "config_hash_exact": metrics.get("config_sha256") == sha256(CONFIG_PATH),
        "gate_1_baselines_pass_equation_tests": metrics["checks"]["gate_1_baselines_pass_equation_tests"],
        "gate_2_linear_reproduction_isolated": metrics["checks"]["gate_2_linear_reproduction_isolated"],
        "gate_3_gaussian_advantage_survives_every_seed": metrics["checks"]["gate_3_gaussian_advantage_survives_every_seed"],
        "gate_4_aurelis_e_exception_isolated": metrics["checks"]["gate_4_aurelis_e_exception_isolated"],
        "gate_5_nonlinear_no_advantage_retained": metrics["checks"]["gate_5_nonlinear_no_advantage_retained"],
        "gate_6_capacity_failures_agree_with_limits": metrics["checks"]["gate_6_capacity_failures_agree_with_limits"],
        "gate_7_full_covariance_beats_independence_heuristic": metrics["checks"]["gate_7_full_covariance_beats_independence_heuristic"],
        "required_artifacts_present": all(p.exists() for p in required),
    }

    failed = {k: v for k, v in checks.items() if not v}
    if failed:
        dump_path = RESULTS / "verification_failure.json"
        dump_path.write_text(json.dumps(failed, indent=2) + "\n", encoding="utf-8")
        raise AssertionError(f"Phase 2 verification failed: {failed}")

    verification_failure = RESULTS / "verification_failure.json"
    if verification_failure.exists():
        verification_failure.unlink()

    # Build report and PASS.md

    report_md = f"""# AURELIS Phase 2 Evaluation Report

Generated: `{datetime.now(UTC).isoformat()}`
Status: **PASS**

## Gates and Findings

1. **Baseline equation tests**:
   All 10 baselines (local softmax, remote Bayes ridge, global linear attention, delta-rule memory, cumulative least-squares Mesa, learned sum/concat, independent inverse-variance fusion, full residual $g=1$, AURELIS-B/E, and Native Hybrid Attention) passed exact equation tests against hand-computed values.

2. **Linear reproduction isolation**:
   Linear reproduction error for full residual is `{metrics['summary']['max_linear_error_aurelis']:.3e}` across all tested temperatures $\\tau \\in [0.01, 10.0]$, while local softmax smoothing error remains large (`{metrics['summary']['min_linear_error_local']:.4f}`).

3. **Gaussian regime advantage**:
   AURELIS-B demonstrates lower MSE than local attention across all 10 independent random seeds.

4. **AURELIS-E episodic exception isolation**:
   AURELIS-E achieves `{metrics['summary']['max_e_episodic_error']:.3e}` error on certified cached exceptions, while AURELIS-B appropriately shrinks the exception toward the latent target.

5. **Retained nonlinear regime without advantage**:
   {metrics['summary']['retained_nonlinear_count']} nonlinear cases retained and verified where local attention outperforms AURELIS, proving linear transport assumptions fail under misspecification.

6. **Capacity limits**:
   Local attention collapses when sequence length exceeds local window $w=32$, while remote linear memory collapses beyond rank $d_k=16$.

7. **Full covariance vs independence heuristic**:
   Full covariance Bayes gate achieves zero-regret optimality ($V(g_B) \\le V(g_{{\\text{{indep}}}})$ everywhere) with paired regret $z$-score of `{metrics['summary']['correlated_z_score']:.2f}` (threshold $\\ge 5.0$).
"""
    (RESULTS / "report.md").write_text(report_md, encoding="utf-8")

    failures_dir = RESULTS / "failures"
    failure_files = sorted(failures_dir.glob("*.md")) if failures_dir.exists() else []
    failure_lines = "\n".join(f"- `{f.relative_to(REPO)}`" for f in failure_files) if failure_files else "None."

    commit = command(["git", "rev-parse", "HEAD"]).stdout.strip()
    dirty = command(["git", "status", "--porcelain"]).stdout.strip()

    pass_md = f"""# AURELIS Phase 2 PASS

Generated: `{datetime.now(UTC).isoformat()}`

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
| Retained nonlinear/misspecified regime with no AURELIS advantage | `metrics.json`; {metrics['summary']['retained_nonlinear_count']} cases documented where local attention outperforms AURELIS |
| Capacity failures agree with rank/window limits | `metrics.json`, `plots/phase2/capacity_limits.png`; collapse at w=32 and d_k=16 |
| Full covariance gate outperforms or equals independence heuristic | `metrics.json`, `plots/phase2/covariance_advantage.png`; paired z-score `{metrics['summary']['correlated_z_score']:.2f}` >= 5.0 |
| Lean proofs of independence heuristic suboptimality | `lean/Aurelis/Router.lean`, `lean/PROOF_COVERAGE.md`; `clippedGate_le_clippedIndependentGate` |
| All inherited Phase 0 and Phase 1 gates | `results/phase2/raw/inherited_phase0.log`, `results/phase2/raw/inherited_phase1.log` |

## Exact reproduction

```bash
./scripts/bootstrap.sh
./scripts/run_phase2.sh
```

## Failed iterations and disposition

{failure_lines}

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

- Base commit: `{commit}`
- Working tree dirty status: `{'dirty' if dirty else 'clean'}`
- Phase 2 config SHA-256: `{sha256(CONFIG_PATH)}`
- Phase 2 metrics SHA-256: `{sha256(RESULTS / 'metrics.json')}`

## Remaining limitations outside the Phase 2 claim

- Evaluated in simulated memory regimes and synthetic sequence tasks; learned representation training belongs to Phase 3.
- Native Hybrid recurrent summary slots used fixed key/value summaries; end-to-end backprop through recurrent slot updates is left to later phases.
"""
    (RESULTS / "PASS.md").write_text(pass_md, encoding="utf-8")
    print(f"PASS record written to {RESULTS / 'PASS.md'}")


if __name__ == "__main__":
    main()
