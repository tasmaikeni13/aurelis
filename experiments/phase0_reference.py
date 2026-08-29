#!/usr/bin/env python3
"""Generate Phase 0 fp64 streaming/oracle agreement evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from aurelis import (
    aurelis_read,
    consume,
    explicit_inverse_read,
    historical_oracle,
    initial_state,
    occurrence_partition,
    read,
)


REPO = Path(__file__).resolve().parents[1]


def git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command], cwd=REPO, text=True, capture_output=True, check=False
    )
    return result.stdout.strip()


def dirty_state() -> dict[str, Any]:
    lines = git(["status", "--short"]).splitlines()
    payload = "\n".join(lines).encode()
    return {
        "dirty": bool(lines),
        "path_count": len(lines),
        "status_sha256": hashlib.sha256(payload).hexdigest(),
    }


def max_output_error(actual: Any, expected: Any) -> float:
    fields = ("remote", "full_residual", "bayes", "episodic")
    errors = [
        float((getattr(actual, field) - getattr(expected, field)).abs().max().item())
        for field in fields
    ]
    diagnostics = ("h", "V_R", "V_H", "K_RH", "g_raw", "g_B", "g_E")
    for field in diagnostics:
        difference = (
            getattr(actual.diagnostics, field) - getattr(expected.diagnostics, field)
        ).abs()
        finite = difference[torch.isfinite(difference)]
        errors.append(float(finite.max().item()) if finite.numel() else 0.0)
    return max(errors)


def run(config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    partition_failures = 0
    dtype = torch.float64
    started = time.perf_counter()
    for seed in config["seeds"]:
        generator = torch.Generator().manual_seed(seed)
        for d_key in config["dimensions"]:
            for window in config["windows"]:
                for length in config["lengths"]:
                    shape = (config["batch"], config["heads"])
                    keys = torch.randn(
                        *shape, length, d_key, generator=generator, dtype=dtype
                    )
                    values = torch.randn(
                        *shape,
                        length,
                        config["d_value"],
                        generator=generator,
                        dtype=dtype,
                    )
                    evidence = torch.rand(
                        *shape, length, generator=generator, dtype=dtype
                    )
                    evidence = (
                        config["minimum_evidence"]
                        + (config["maximum_evidence"] - config["minimum_evidence"])
                        * evidence
                    )
                    query = torch.randn(*shape, d_key, generator=generator, dtype=dtype)
                    temperature = (
                        torch.rand(config["heads"], generator=generator, dtype=dtype)
                        + 0.25
                    )
                    responsibility: torch.Tensor | float = (
                        torch.rand(*shape, generator=generator, dtype=dtype)
                        if length
                        else 0.0
                    )
                    state = initial_state(
                        *shape,
                        d_key,
                        config["d_value"],
                        window,
                        prior=config["prior"],
                    )
                    for step in range(length):
                        state = consume(
                            state,
                            keys[:, :, step],
                            values[:, :, step],
                            evidence[:, :, step],
                        )
                        remote_ids, recent_ids = occurrence_partition(state)
                        if (
                            set(remote_ids).intersection(recent_ids)
                            or remote_ids + recent_ids != tuple(range(step + 1))
                        ):
                            partition_failures += 1
                    streaming = read(
                        state,
                        query,
                        temperature=temperature,
                        episodic_responsibility=responsibility,
                    )
                    oracle = historical_oracle(
                        keys,
                        values,
                        evidence,
                        query,
                        window=window,
                        prior=config["prior"],
                        temperature=temperature,
                        episodic_responsibility=responsibility,
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "length": length,
                            "window": window,
                            "d_key": d_key,
                            "over_capacity": length > d_key,
                            "handoff_boundary": length in (window, window + 1),
                            "streaming_oracle_max_absolute_error": max_output_error(
                                streaming, oracle
                            ),
                            "maximum_solve_residual": float(
                                torch.maximum(
                                    streaming.diagnostics.solve_residual_q,
                                    streaming.diagnostics.solve_residual_kbar,
                                ).max()
                            ),
                        }
                    )

    generator = torch.Generator().manual_seed(config["seeds"][0] + 99)
    d_key = max(config["dimensions"])
    matrix = torch.randn(2, 2, d_key, d_key, generator=generator, dtype=dtype)
    precision = matrix @ matrix.mT + config["prior"] * torch.eye(d_key, dtype=dtype)
    cross = torch.randn(
        2, 2, config["d_value"], d_key, generator=generator, dtype=dtype
    )
    keys = torch.randn(2, 2, 4, d_key, generator=generator, dtype=dtype)
    values = torch.randn(
        2, 2, 4, config["d_value"], generator=generator, dtype=dtype
    )
    evidence = torch.rand(2, 2, 4, generator=generator, dtype=dtype) + 0.1
    query = torch.randn(2, 2, d_key, generator=generator, dtype=dtype)
    arguments = (precision, cross, keys, values, evidence, query)
    cholesky = aurelis_read(*arguments)
    dense = aurelis_read(*arguments, solve_method="dense")
    inverse = explicit_inverse_read(*arguments, inverse_cap=16)
    solver_error = max(
        float((cholesky.bayes - dense.bayes).abs().max()),
        float((cholesky.bayes - inverse.bayes).abs().max()),
    )
    maximum_agreement = max(row["streaming_oracle_max_absolute_error"] for row in rows)
    thresholds = config["tolerances"]
    gate = {
        "streaming_matches_historical_oracle": maximum_agreement
        <= thresholds["streaming_oracle_max_absolute_error"],
        "cholesky_dense_inverse_agree": solver_error
        <= thresholds["solver_max_absolute_error"],
        "occurrence_partition_exact": partition_failures
        <= thresholds["partition_failures"],
    }
    return {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": "PASS" if all(gate.values()) else "FAIL",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "wall_time_seconds": time.perf_counter() - started,
        "command": ".venv/bin/python experiments/phase0_reference.py",
        "commit": git(["rev-parse", "HEAD"]),
        "dirty_state": dirty_state(),
        "device": "cpu",
        "dtype": "torch.float64",
        "config": config,
        "observed": {
            "case_count": len(rows),
            "maximum_streaming_oracle_absolute_error": maximum_agreement,
            "maximum_solver_absolute_error": solver_error,
            "partition_failures": partition_failures,
        },
        "gate": gate,
        "rows": rows,
    }


def write_outputs(record: dict[str, Any], output: Path, plot: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plot.mkdir(parents=True, exist_ok=True)
    (output / "reference_metrics.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    raw_path = output / "raw" / "reference_cases.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in record["rows"])
    )

    by_length: dict[int, float] = {}
    for row in record["rows"]:
        by_length[row["length"]] = max(
            by_length.get(row["length"], 0.0),
            row["streaming_oracle_max_absolute_error"],
        )
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.semilogy(list(by_length), [max(value, 1e-18) for value in by_length.values()], "o-")
    axis.axhline(
        record["config"]["tolerances"]["streaming_oracle_max_absolute_error"],
        color="tab:red",
        linestyle="--",
        label="gate",
    )
    axis.set_xlabel("consumed occurrences")
    axis.set_ylabel("max |streaming - historical oracle|")
    axis.set_title("AURELIS Phase 0 fp64 agreement")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot / "reference_agreement.png", dpi=150)
    plt.close(figure)

    observed = record["observed"]
    report = f"""# Phase 0 reference experiment

Status: **{record['status']}**

| Measure | Observed |
|---|---:|
| Random cases | {observed['case_count']} |
| Maximum streaming/oracle absolute error | {observed['maximum_streaming_oracle_absolute_error']:.3e} |
| Maximum Cholesky/dense/inverse error | {observed['maximum_solver_absolute_error']:.3e} |
| Partition failures | {observed['partition_failures']} |

The experiment uses CPU/fp64 and independently reconstructs every remote
prefix from full history. It covers empty and warm-up caches, the handoff
boundary, random windows/dimensions, and lengths above feature capacity. Unit
tests add repeated-key and near-singular pathologies, autograd, the analytic
router, and explicit expected-failure domains.

Reproduce with `{record['command']}`.
"""
    (output / "reference_report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO / "configs/phase0_reference.json")
    parser.add_argument("--output", type=Path, default=REPO / "results/phase0")
    parser.add_argument("--plot", type=Path, default=REPO / "plots/phase0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    record = run(config)
    write_outputs(record, args.output, args.plot)
    if record["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
