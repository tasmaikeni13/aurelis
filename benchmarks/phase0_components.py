#!/usr/bin/env python3
"""Measure AURELIS components and eager/Inductor correctness on one MI300X."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch

from aurelis import (
    consume,
    historical_oracle,
    initial_state,
    prepared_aurelis_head,
    read,
    vectorized_reference,
)


REPO = Path(__file__).resolve().parents[1]


def synchronize() -> None:
    torch.cuda.synchronize()


def benchmark(
    name: str, operation: Callable[[], Any], warmup: int, repetitions: int
) -> dict[str, Any]:
    for _ in range(warmup):
        operation()
    synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        operation()
        synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "component": name,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, text=True, capture_output=True, check=False
    ).stdout.strip()


def dirty_state() -> dict[str, Any]:
    lines = git(["status", "--short"]).splitlines()
    return {
        "dirty": bool(lines),
        "path_count": len(lines),
        "status_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
    }


def forward(
    keys: torch.Tensor,
    values: torch.Tensor,
    evidence: torch.Tensor,
    queries: torch.Tensor,
    window: int,
    prior: float,
) -> torch.Tensor:
    return vectorized_reference(
        keys,
        values,
        evidence,
        queries,
        window=window,
        prior=prior,
    ).bayes


def run(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("Phase 0 MI300X benchmark requires a PyTorch HIP device")
    torch.manual_seed(config["seed"])
    device = torch.device("cuda")
    shape = (config["batch"], config["heads"], config["length"])
    keys_cpu = torch.randn(*shape, config["d_key"], dtype=torch.float64)
    values_cpu = torch.randn(*shape, config["d_value"], dtype=torch.float64)
    evidence_cpu = torch.rand(*shape, dtype=torch.float64) + 0.2
    queries_cpu = torch.randn(*shape, config["d_key"], dtype=torch.float64)
    keys = keys_cpu.float().to(device).requires_grad_()
    values = values_cpu.float().to(device).requires_grad_()
    evidence = evidence_cpu.float().to(device).requires_grad_()
    queries = queries_cpu.float().to(device).requires_grad_()
    window = config["window"]
    prior = config["prior"]
    warmup = config["warmup"]
    repetitions = config["repetitions"]
    torch.cuda.reset_peak_memory_stats()

    outer_key = keys[:, :, 0]
    outer_value = values[:, :, 0]
    outer_beta = evidence[:, :, 0]
    matrix = torch.randn(
        config["batch"], config["heads"], config["d_key"], config["d_key"], device=device
    )
    precision = matrix @ matrix.mT + prior * torch.eye(config["d_key"], device=device)
    factor = torch.linalg.cholesky(precision)
    rhs = torch.randn(
        config["batch"], config["heads"], config["d_key"], 2, device=device
    )
    local_keys = keys[:, :, :window]
    local_values = values[:, :, :window]
    local_query = queries[:, :, window - 1]
    local_beta = evidence[:, :, :window]
    scores = torch.einsum("bhwd,bhd->bhw", local_keys, local_query)
    weights = torch.softmax(scores, dim=-1)
    kbar = torch.einsum("bhw,bhwd->bhd", weights, local_keys)
    p_query = torch.cholesky_solve(local_query.unsqueeze(-1), factor).squeeze(-1)
    p_kbar = torch.cholesky_solve(kbar.unsqueeze(-1), factor).squeeze(-1)
    h = torch.sum(weights.square() / local_beta, dim=-1)

    components = [
        benchmark(
            "outer_updates",
            lambda: (
                torch.einsum("bh,bhi,bhj->bhij", outer_beta, outer_key, outer_key),
                torch.einsum("bh,bhv,bhd->bhvd", outer_beta, outer_value, outer_key),
            ),
            warmup,
            repetitions,
        ),
        benchmark(
            "local_attention",
            lambda: (
                torch.softmax(torch.einsum("bhwd,bhd->bhw", local_keys, local_query), -1),
                torch.einsum("bhw,bhwd->bhd", weights, local_keys),
                torch.einsum("bhw,bhwv->bhv", weights, local_values),
            ),
            warmup,
            repetitions,
        ),
        benchmark(
            "cholesky_factorization",
            lambda: torch.linalg.cholesky(precision),
            warmup,
            repetitions,
        ),
        benchmark(
            "triangular_solve",
            lambda: torch.cholesky_solve(rhs, factor),
            warmup,
            repetitions,
        ),
        benchmark(
            "routing",
            lambda: torch.clamp(
                torch.sum(local_query * p_kbar, -1)
                / (h + torch.sum(kbar * p_kbar, -1)),
                0.0,
                1.0,
            ),
            warmup,
            repetitions,
        ),
    ]

    components.append(
        benchmark(
            "vectorized_training_eager_forward",
            lambda: forward(keys, values, evidence, queries, window, prior),
            warmup,
            repetitions,
        )
    )

    remote_end = config["length"] - window
    prepared_precision = prior * torch.eye(config["d_key"], device=device)
    prepared_precision = prepared_precision.view(1, 1, config["d_key"], config["d_key"])
    prepared_precision = prepared_precision + torch.einsum(
        "bhn,bhni,bhnj->bhij",
        evidence[:, :, :remote_end],
        keys[:, :, :remote_end],
        keys[:, :, :remote_end],
    )
    prepared_cross = torch.einsum(
        "bhn,bhnv,bhnd->bhvd",
        evidence[:, :, :remote_end],
        values[:, :, :remote_end],
        keys[:, :, :remote_end],
    )
    prepared_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (
            prepared_precision,
            prepared_cross,
            keys[:, :, -window:],
            values[:, :, -window:],
            evidence[:, :, -window:],
            queries[:, :, -1],
            torch.ones(config["batch"], config["heads"], device=device),
            torch.ones(config["batch"], config["heads"], device=device),
        )
    )
    eager_tuple = prepared_aurelis_head(*prepared_inputs)
    eager_result = eager_tuple[2]
    synchronize()
    eager_grad = torch.autograd.grad(
        eager_tuple[2].sum() + eager_tuple[3].sum(), prepared_inputs
    )
    components.append(
        benchmark(
            "prepared_head_eager_forward",
            lambda: prepared_aurelis_head(*prepared_inputs),
            warmup,
            repetitions,
        )
    )

    def eager_forward_backward() -> tuple[torch.Tensor, ...]:
        result = prepared_aurelis_head(*prepared_inputs)
        return torch.autograd.grad(result[2].sum() + result[3].sum(), prepared_inputs)

    components.append(
        benchmark(
            "prepared_head_eager_forward_backward",
            eager_forward_backward,
            warmup,
            repetitions,
        )
    )

    compiled_function = torch.compile(
        prepared_aurelis_head, backend="inductor", fullgraph=True
    )
    compile_started = time.perf_counter()
    compiled_tuple = compiled_function(*prepared_inputs)
    compiled_result = compiled_tuple[2]
    synchronize()
    compile_and_first_run_seconds = time.perf_counter() - compile_started
    compiled_grad = torch.autograd.grad(
        compiled_tuple[2].sum() + compiled_tuple[3].sum(),
        prepared_inputs,
        retain_graph=False,
    )
    components.append(
        benchmark(
            "prepared_head_compiled_forward",
            lambda: compiled_function(*prepared_inputs),
            warmup,
            repetitions,
        )
    )

    def compiled_forward_backward() -> tuple[torch.Tensor, ...]:
        result = compiled_function(*prepared_inputs)
        return torch.autograd.grad(result[2].sum() + result[3].sum(), prepared_inputs)

    components.append(
        benchmark(
            "prepared_head_compiled_forward_backward",
            compiled_forward_backward,
            warmup,
            repetitions,
        )
    )

    historical_cpu = historical_oracle(
        keys_cpu,
        values_cpu,
        evidence_cpu,
        queries_cpu[:, :, -1],
        window=window,
        prior=prior,
    ).bayes
    fp32_vs_fp64 = float(
        (eager_result.detach().cpu().double() - historical_cpu).abs().max()
    )
    compiled_vs_eager = max(
        float((actual.detach() - expected.detach()).abs().max())
        for actual, expected in zip(compiled_tuple, eager_tuple, strict=True)
    )
    gradient_error = max(
        float((actual.detach() - expected.detach()).abs().max())
        for actual, expected in zip(compiled_grad, eager_grad, strict=True)
    )

    stream = initial_state(
        config["batch"],
        config["heads"],
        config["d_key"],
        config["d_value"],
        window,
        prior=prior,
        dtype=torch.float32,
        device=device,
    )
    for step in range(config["length"]):
        stream = consume(
            stream,
            keys.detach()[:, :, step],
            values.detach()[:, :, step],
            evidence.detach()[:, :, step],
        )
    streaming_gpu = read(stream, queries.detach()[:, :, -1]).bayes
    streaming_vs_fp64 = float((streaming_gpu.cpu().double() - historical_cpu).abs().max())
    peak_memory = torch.cuda.max_memory_allocated()
    tolerance = config["tolerances"]
    gates = {
        "fp32_eager_matches_fp64": fp32_vs_fp64
        <= tolerance["fp32_forward_vs_fp64_max_absolute_error"],
        "fp32_streaming_matches_fp64": streaming_vs_fp64
        <= tolerance["fp32_forward_vs_fp64_max_absolute_error"],
        "compiled_matches_eager": compiled_vs_eager
        <= tolerance["compiled_vs_eager_max_absolute_error"],
        "compiled_gradient_matches_eager": gradient_error
        <= tolerance["compiled_vs_eager_gradient_max_absolute_error"],
    }
    return {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": "PASS" if all(gates.values()) else "FAIL",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "command": ".venv/bin/python benchmarks/phase0_components.py",
        "commit": git(["rev-parse", "HEAD"]),
        "dirty_state": dirty_state(),
        "device": torch.cuda.get_device_name(0),
        "architecture": str(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "dtype": "torch.float32 (compared with CPU torch.float64)",
        "config": config,
        "compile_and_first_run_seconds": compile_and_first_run_seconds,
        "peak_memory_bytes": peak_memory,
        "observed": {
            "fp32_forward_vs_fp64_max_absolute_error": fp32_vs_fp64,
            "fp32_streaming_vs_fp64_max_absolute_error": streaming_vs_fp64,
            "compiled_vs_eager_max_absolute_error": compiled_vs_eager,
            "compiled_vs_eager_gradient_max_absolute_error": gradient_error,
        },
        "gates": gates,
        "components": components,
        "custom_kernel": {
            "implemented": False,
            "disposition": (
                "Not justified in Phase 0: the head crosses factorization and "
                "solver library boundaries, and the required eager/Inductor paths "
                "are measured directly. A Triton prototype remains optional after "
                "shape sweeps identify a fusion target."
            ),
        },
    }


def write(record: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark_metrics.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    raw = output / "raw" / "component_timings.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in record["components"])
    )
    rows = "\n".join(
        f"| {row['component']} | {row['median_ms']:.6f} | {row['minimum_ms']:.6f} |"
        for row in record["components"]
    )
    observed = record["observed"]
    report = f"""# Phase 0 MI300X substrate benchmark

Status: **{record['status']}**

- Device: `{record['device']}`
- PyTorch/HIP: `{record['torch']}` / `{record['hip']}`
- Compile plus first run: `{record['compile_and_first_run_seconds']:.3f}` seconds
- Peak allocated memory: `{record['peak_memory_bytes']}` bytes
- fp32 eager vs CPU/fp64 maximum error: `{observed['fp32_forward_vs_fp64_max_absolute_error']:.3e}`
- fp32 streaming vs CPU/fp64 maximum error: `{observed['fp32_streaming_vs_fp64_max_absolute_error']:.3e}`
- compiled vs eager maximum error: `{observed['compiled_vs_eager_max_absolute_error']:.3e}`
- compiled vs eager gradient maximum error: `{observed['compiled_vs_eager_gradient_max_absolute_error']:.3e}`

| Component | Median ms | Minimum ms |
|---|---:|---:|
{rows}

Compilation/warm-up is excluded from steady-state rows. Every timed GPU sample
is synchronized. These are component health measurements, not an accelerator
superiority claim. No custom kernel was added because Phase 0 measurements do
not yet identify a stable shape-specific fusion target beyond Inductor.
"""
    (output / "benchmark_report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO / "configs/phase0_benchmark.json")
    parser.add_argument("--output", type=Path, default=REPO / "results/phase0")
    args = parser.parse_args()
    record = run(json.loads(args.config.read_text()))
    write(record, args.output)
    if record["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
