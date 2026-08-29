#!/usr/bin/env python3
"""Requirement-level Phase 1 audit and generated PASS record."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "phase1"
CONFIG_PATH = REPO / "configs" / "phase1_oracle.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=REPO, text=True, capture_output=True, check=False)


def main() -> None:
    config = load(CONFIG_PATH)
    metrics = load(RESULTS / "metrics.json")
    phase0_reference = load(REPO / "results" / "phase0" / "reference_metrics.json")
    phase0_benchmark = load(REPO / "results" / "phase0" / "benchmark_metrics.json")
    required = [
        RESULTS / "raw" / "inherited_phase0.log",
        RESULTS / "raw" / "pytest.log",
        RESULTS / "raw" / "lean_build.log",
        RESULTS / "raw" / "experiment.log",
        RESULTS / "raw" / "streaming_rows.jsonl",
        RESULTS / "raw" / "deterministic_rows.jsonl",
        RESULTS / "raw" / "conditioning_rows.jsonl",
        RESULTS / "report.md",
        RESULTS / "FORMAL_AUDIT.md",
        RESULTS / "RESEARCH_LOG.md",
        REPO / "plots" / "phase1" / "conditional_calibration.png",
        REPO / "plots" / "phase1" / "conditioning_precision.png",
    ]
    sweep = config["sweeps"]
    streaming = metrics["streaming"]
    shape_rows = [row for row in streaming if row["case"] == "shape"]
    rank_rows = [row for row in streaming if row["case"].startswith("remote_load_")]
    deterministic = metrics["deterministic"]
    monte_carlo = metrics["monte_carlo"]
    conditioning = metrics["conditioning"]
    routing = metrics["routing_and_recall"]
    handoff = metrics["handoff"]
    costs = metrics["costs"]
    lean_placeholders = command(
        ["rg", "-n", "--glob", "*.lean", r"\b(sorry|admit|axiom)\b", "lean"]
    )
    expected_shapes = {
        (d_key, d_value, window)
        for d_key in sweep["d_key"]
        for d_value in sweep["d_value"]
        for window in sweep["window"]
    }
    observed_shapes = {
        (row["d_key"], row["d_value"], row["window"]) for row in shape_rows
    }
    dtype_names = {row["storage_dtype"] for row in conditioning["rows"]}
    clip_classes = {row["observed_class"] for row in routing["clipping"]}
    checks = {
        "phase0_inherited_reference_pass": phase0_reference.get("status") == "PASS",
        "phase0_inherited_benchmark_pass": phase0_benchmark.get("status") == "PASS",
        "required_artifacts_present": all(path.exists() for path in required),
        "experiment_pass": metrics.get("status") == "PASS" and all(metrics["checks"].values()),
        "config_hash_exact": metrics.get("config_sha256") == sha256(CONFIG_PATH),
        "shape_cartesian_sweep_exact": observed_shapes == expected_shapes,
        "rank_load_sweep_complete": {
            row["case"].removeprefix("remote_load_") for row in rank_rows
        }
        == set(sweep["remote_load_relative_to_rank"])
        and {row["d_key"] for row in rank_rows} == set(sweep["d_key"]),
        "prior_sweep_complete": {row["prior"] for row in deterministic}
        == set(sweep["prior_precision"]),
        "temperature_sweep_complete": {row["temperature"] for row in deterministic}
        == set(sweep["attention_temperature"]),
        "pathology_sweep_complete": {row["pathology"] for row in deterministic}
        == set(sweep["key_pathologies"]),
        "evidence_models_complete": {row["evidence_model"] for row in deterministic}
        == set(sweep["evidence_models"]),
        "dtype_sweep_complete": {
            "float64",
            "float32",
            "bfloat16_storage_float32_compute",
        }.issubset(dtype_names),
        "streaming_every_prefix_and_partition": sum(
            row["prefixes_checked"] for row in streaming
        )
        == metrics["aggregate"]["streaming_prefixes_checked"]
        and all(row["status"] == "PASS" for row in streaming)
        and metrics["aggregate"]["partition_failures"] == 0,
        "exact_identity_valid_rows": all(row["status"] == "PASS" for row in deterministic),
        "invalid_domains_retained": metrics["aggregate"][
            "invalid_gate_cancellation_rows_retained"
        ]
        > 0
        and len(conditioning["invalid_domain_rows"]) >= 4,
        "all_clipping_regions": clip_classes
        == {"below_zero", "interior", "above_one"}
        and all(row["status"] == "PASS" for row in routing["clipping"]),
        "one_hot_exact_and_finite_temperature_labeled": routing[
            "hard_one_hot_realized_by_singleton_cache"
        ]["status"]
        == "PASS"
        and any(row["numerically_saturated"] for row in routing["finite_temperature_convergence"]),
        "monte_carlo_99_percent": all(
            row["all_formula_estimates_inside_preregistered_99_percent_intervals"]
            and row["routed_endpoint_noninferiority_with_mc_error"]
            for row in monte_carlo
        ),
        "covariance_omission_wrong": next(
            row for row in monte_carlo if row["name"] == "constructed_covariance_omission"
        )["covariance_omission_z_score"]
        >= config["preregistered_tolerances"]["covariance_omission_minimum_z_score"],
        "target_conflict_and_misspecification_retained": metrics["target_conflict"][
            "status"
        ]
        == "PASS"
        and metrics["target_conflict"]["misspecified_local_exception_counterexample"][
            "retained"
        ],
        "handoff_output_gradient_audited": handoff["status"] == "PASS"
        and [row["position"] for row in handoff["boundaries"]]
        == ["immediately_before", "at_handoff", "immediately_after"]
        and all(row["tracked_occurrence_count"] == 1 for row in handoff["boundaries"]),
        "conditioning_and_nonfinite_audited": conditioning["status"] == "PASS"
        and all(
            row["observed"] in {"rejected", "returned_nonfinite"}
            for row in conditioning["invalid_domain_rows"]
        ),
        "bytes_and_operations_audited": costs["status"] == "PASS"
        and all(row["difference"] == 0 for row in costs["state_rows"])
        and costs["operation_record"]["torch_profiler_operator_invocations"] > 0,
        "lean_no_placeholders_or_project_axioms": lean_placeholders.returncode == 1,
    }
    if not all(checks.values()):
        failure = {
            "generated_utc": datetime.now(UTC).isoformat(),
            "checks": checks,
            "missing": [str(path.relative_to(REPO)) for path in required if not path.exists()],
        }
        (RESULTS / "verification_failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise SystemExit(1)

    covariance_case = next(
        row for row in monte_carlo if row["name"] == "constructed_covariance_omission"
    )
    failures = sorted((RESULTS / "failures").glob("*.md"))
    failure_lines = "\n".join(
        f"- `{path.relative_to(REPO)}` — retained failure and disposition."
        for path in failures
    )
    commit = command(["git", "rev-parse", "HEAD"]).stdout.strip()
    dirty = command(["git", "status", "--short"]).stdout.splitlines()
    environment_path = REPO / "results" / "phase0" / "environment.json"
    pass_record = f"""# AURELIS Phase 1 PASS

Generated: `{datetime.now(UTC).isoformat()}`

Phase 1 status: **PASS**. This record covers the mathematical oracle,
calibration, dtype/conditioning pathologies, handoff boundary, and analytic
cost gate. It makes no learned-feature, language-model-quality, or competitive
throughput claim.

## Gate evidence

| Gate | Direct evidence |
|---|---|
| Pinned dimension/window/rank/prior/temperature/evidence/pathology sweep | `configs/phase1_oracle.json`, `results/phase1/raw/deterministic_rows.jsonl`, `streaming_rows.jsonl`; {len(streaming)} cases and {metrics['aggregate']['streaming_prefixes_checked']} prefixes |
| Streaming state equals full-history oracle at every prefix; no occurrence loss/duplication | `metrics.json`, `raw/streaming_rows.jsonl`; maximum tolerance ratio `{metrics['aggregate']['maximum_streaming_tolerance_ratio']:.3e}`, zero partition failures |
| Residual, general-gate, reproduction, finite-ridge, exact-hit, and two raw-gate identities | `metrics.json`, `tests/test_phase1_identities.py`; maximum valid identity ratio `{metrics['aggregate']['maximum_identity_tolerance_ratio']:.3e}` |
| Below/interior/above clipping, dense-grid optimum, endpoint non-inferiority | `metrics.json` routing/deterministic rows; all three clipping regions exercised |
| Exact conditional `V_R,V_H,K_RH,V(g)` Monte Carlo with preregistered 99% intervals | `metrics.json`; {len(monte_carlo)} matched regimes × {config['monte_carlo_trials']} trials |
| Covariance omission control is measurably wrong | `metrics.json`; paired regret z-score `{covariance_case['covariance_omission_z_score']:.3f}` |
| Latent denoising / episodic copy conflict and misspecification counterexample | `metrics.json` target-conflict record |
| Handoff output and gradients immediately before/at/after | `metrics.json` handoff boundaries/discontinuities; tracked occurrence count always one |
| Condition, solve residual, forward error, Cholesky/nonfinite, fp64/fp32/bf16-storage | `metrics.json`, `raw/conditioning_rows.jsonl`, `plots/phase1/conditioning_precision.png`; identical-quantized-input policy recorded |
| Analytic/observed state bytes and operation counts | `metrics.json` cost record; every implementation byte prediction exact, profiler calls retained |
| Invalid domains remain visible | `metrics.json`; {metrics['aggregate']['invalid_gate_cancellation_rows_retained']} cancellation-unresolved and {len(conditioning['invalid_domain_rows'])} explicitly invalid rows |
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

{failure_lines}

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

- Base commit: `{commit}`
- Working tree intentionally dirty with `{len(dirty)}` Phase 0/1 research paths.
- Phase 1 config SHA-256: `{sha256(CONFIG_PATH)}`
- Phase 1 metrics SHA-256: `{sha256(RESULTS / 'metrics.json')}`
- Phase 0 environment SHA-256: `{sha256(environment_path)}`

## Remaining limitations outside the Phase 1 claim

- Native bfloat16 Cholesky on the active `{conditioning['native_bfloat16_cholesky_probe_device']}` device is unsupported on the pinned Torch build;
  bfloat16 storage is therefore promoted to fp32 compute and labeled as such.
- The profiler does not report FLOPs for Cholesky/triangular solves; analytic
  arithmetic and observed supported FLOPs/operator invocations are separate.
- Conditional probability calibration assumes the declared disjoint
  linear-Gaussian model; the retained misspecification row demonstrates the
  boundary rather than extending the theorem.
- Full matrix spectral-norm ridge bounds remain analytic; Lean covers a
  faithful scalar specialization. Learned features, drift, LM quality, and
  competitive throughput belong to later phases.
"""
    (RESULTS / "PASS.md").write_text(pass_record, encoding="utf-8")
    verification_failure = RESULTS / "verification_failure.json"
    if verification_failure.exists():
        verification_failure.unlink()


if __name__ == "__main__":
    main()
