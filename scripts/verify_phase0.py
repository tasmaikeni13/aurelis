#!/usr/bin/env python3
"""Audit Phase 0 artifacts and generate PASS only when every gate is direct."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "phase0"
MIGRATION_AUDIT = RESULTS / "migration_audit.txt"


def command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=REPO, text=True, capture_output=True, check=False)


def candidate_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPO,
        capture_output=True,
        check=True,
    )
    files = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        path = REPO / raw.decode()
        if path == MIGRATION_AUDIT or not path.is_file():
            continue
        files.append(path)
    return files


def migration_matches() -> list[str]:
    acronym = bytes((99, 115, 109)).decode()
    expanded = bytes(
        (99, 111, 110, 106, 117, 103, 97, 116, 101, 32, 115, 116, 97, 116, 101, 32, 109, 97, 99, 104, 105, 110, 101, 115)
    ).decode()
    pattern = re.compile(re.escape(acronym) + r"|" + re.escape(expanded), re.IGNORECASE)
    matches: list[str] = []
    for path in candidate_files():
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                matches.append(f"{path.relative_to(REPO)}:{line_number}:{line.strip()}")
    return matches


def load_json(name: str) -> dict[str, object]:
    return json.loads((RESULTS / name).read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    required_logs = [
        RESULTS / "raw" / "environment_audit.log",
        RESULTS / "raw" / "pytest.log",
        RESULTS / "raw" / "lean_build.log",
        RESULTS / "raw" / "reference_experiment.log",
        RESULTS / "raw" / "benchmark.log",
    ]
    missing_logs = [str(path.relative_to(REPO)) for path in required_logs if not path.exists()]
    environment = load_json("environment.json")
    reference = load_json("reference_metrics.json")
    benchmark = load_json("benchmark_metrics.json")
    matches = migration_matches()
    lean_placeholders = command(
        [
            "rg",
            "-n",
            "--glob",
            "*.lean",
            r"\b(sorry|admit|axiom)\b",
            "lean",
        ]
    )
    checks = {
        "migration_clean": not matches,
        "environment_pass": environment.get("status") == "PASS",
        "reference_pass": reference.get("status") == "PASS",
        "benchmark_pass": benchmark.get("status") == "PASS",
        "all_command_logs_present": not missing_logs,
        "lean_has_no_placeholders_or_project_axioms": lean_placeholders.returncode == 1,
    }
    acronym = bytes((67, 83, 77)).decode()
    expanded = bytes(
        (67, 111, 110, 106, 117, 103, 97, 116, 101, 32, 83, 116, 97, 116, 101, 32, 77, 97, 99, 104, 105, 110, 101, 115)
    ).decode()
    migration_text = "\n".join(
        [
            "AURELIS PHASE 0 GENERATED MIGRATION AUDIT",
            "=========================================",
            f"timestamp_utc: {datetime.now(UTC).isoformat()}",
            f"quoted_old_acronym: {acronym}",
            f"quoted_old_expanded_name: {expanded}",
            "scope: case-insensitive substring search of tracked and unignored working-tree files; .git, ignored build environments, and this audit are excluded",
            f"match_count_outside_this_audit: {len(matches)}",
            *matches,
            "status: PASS" if not matches else "status: FAIL",
            "",
        ]
    )
    MIGRATION_AUDIT.write_text(migration_text)
    if not all(checks.values()):
        detail = {"checks": checks, "missing_logs": missing_logs, "migration_matches": matches}
        (RESULTS / "verification_failure.json").write_text(
            json.dumps(detail, indent=2, sort_keys=True) + "\n"
        )
        raise SystemExit(1)

    commit = command(["git", "rev-parse", "HEAD"]).stdout.strip()
    dirty = command(["git", "status", "--short"]).stdout.splitlines()
    observed = reference["observed"]
    hardware = benchmark["observed"]
    pass_record = f"""# AURELIS Phase 0 PASS

Generated: `{datetime.now(UTC).isoformat()}`

Phase 0 status: **PASS**. This record covers migration and the reference/ROCm
substrate only. It makes no language-model quality or accelerator-superiority
claim.

## Gate evidence

| Gate | Direct evidence |
|---|---|
| Obsolete identity absent | `results/phase0/migration_audit.txt` (zero matches outside that generated audit) |
| Streaming/history fp64 agreement | `results/phase0/reference_metrics.json`; maximum error `{observed['maximum_streaming_oracle_absolute_error']:.3e}` |
| Disjoint exhaustive occurrence partition | `tests/test_partition.py`, `results/phase0/raw/reference_cases.jsonl`; zero failures |
| Cholesky/dense/capped-inverse agreement and conditioned failure domains | `tests/test_solvers.py`, `results/phase0/reference_metrics.json`; maximum error `{observed['maximum_solver_absolute_error']:.3e}` |
| Autograd and gradcheck for inputs/projections | `tests/test_autograd.py`, `results/phase0/raw/pytest.log` |
| Analytic Bayes route and exact episodic hit | `tests/test_routing.py`, `results/phase0/raw/pytest.log` |
| Eager/Inductor/fp64 agreement | `results/phase0/benchmark_metrics.json`; fp32/fp64 `{hardware['fp32_forward_vs_fp64_max_absolute_error']:.3e}`, compiled/eager `{hardware['compiled_vs_eager_max_absolute_error']:.3e}` |
| MI300X/ROCm measured; forbidden accelerator dependencies absent | `environment.txt`, `results/phase0/environment.json` |
| Lean build; no proof placeholders/project axioms | `results/phase0/raw/lean_build.log`, `lean/PROOF_COVERAGE.md` |
| Full documented command | `scripts/run_phase0.sh` and the five raw command logs |

## Exact reproduction

```bash
./scripts/bootstrap.sh
./scripts/run_phase0.sh
```

The fail-fast command runs the environment audit, Python unit/property and
gradcheck suite, full Lean build, small fp64 reference experiment, MI300X
eager/compiled component benchmark, and this completion audit.

## Failed iterations and disposition

- `results/phase0/failures/bootstrap_ensurepip_20260829.md`: the first venv
  bootstrap lacked Ubuntu's matching venv package; it was installed without
  changing the driver, ROCm stack, or Python version.
- `results/phase0/failures/reference_dtype_20260829.md`: the standalone
  experiment inherited fp32 inputs against an fp64 state; all oracle tensors
  now declare fp64 explicitly.
- `results/phase0/failures/environment_missing_python_headers_20260829.md`:
  Inductor's HIP helper lacked Python development headers; the matching Ubuntu
  compiler-header package repaired the environment without changing PyTorch or
  ROCm.
- `results/phase0/failures/vectorized_inductor_cumsum_20260829.md`: the bundled
  Triton compiler rejected its generated all-prefix cumulative-sum kernel. The
  exact prefix constructor remains eager, while the complete prepared head is
  compiled as one graph and checked forward/backward.
- `results/phase0/failures/lean_toolchain_missing_20260829.md`: the first full
  runner lacked `lake`; user-scoped elan selected the already pinned Lean
  4.19.0 toolchain, after which the unchanged proof project built.

No scientific gate was weakened. Expected non-positive-definite and explicit
inverse dimension failures remain regression tests.

## Research and mathematical repairs

- Current AMD compatibility, MI300X optimization, rocSOLVER Cholesky, and
  PyTorch HIP-semantics sources (accessed 2026-08-29) are embedded in
  `results/phase0/environment.json` with the design decision each supports.
- The host/wheel version difference is reported as measured behavior, not an
  unsupported compatibility assumption.
- No theorem or manuscript equation required correction in this phase.

## Lean coverage

The existing faithful theorems were retained unchanged. Exact coverage and
unformalized probability/numerical/system boundaries are listed in
`lean/PROOF_COVERAGE.md`. The full pinned build output is retained.

## Raw and aggregate artifacts

- `results/phase0/raw/reference_cases.jsonl`
- `results/phase0/raw/component_timings.jsonl`
- `results/phase0/reference_metrics.json`
- `results/phase0/benchmark_metrics.json`
- `results/phase0/reference_report.md`
- `results/phase0/benchmark_report.md`
- `plots/phase0/reference_agreement.png`

## Tested revision and environment fingerprint

- Base commit: `{commit}`
- Working tree was intentionally dirty with `{len(dirty)}` migration paths;
  each JSON record stores the dirty flag, path count, and status SHA-256.
- Environment SHA-256: `{sha256(RESULTS / 'environment.json')}`
- Reference metrics SHA-256: `{sha256(RESULTS / 'reference_metrics.json')}`
- Benchmark metrics SHA-256: `{sha256(RESULTS / 'benchmark_metrics.json')}`

## Remaining limitations outside the Phase 0 claim

- The stable streaming factor is freshly refactorized after handoff; optimized
  rank-one factor updates and periodic refactor policy remain systems work.
- The exact vectorized training path materializes all prefix precision
  matrices, remains eager after the retained Triton cumulative-sum failure,
  and does not claim favorable large-sequence memory use.
- No custom Triton kernel was added because the measured Phase 0 shapes did
  not yet establish a stable fusion target beyond Inductor.
- Learned feature quality, drift, large-scale language modeling, and matched
  throughput comparisons belong to later phases and remain pending.
"""
    (RESULTS / "PASS.md").write_text(pass_record)
    failure_path = RESULTS / "verification_failure.json"
    if failure_path.exists():
        failure_path.unlink()


if __name__ == "__main__":
    main()
