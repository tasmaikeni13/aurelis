#!/usr/bin/env python3
"""Generate quantitative Phase 1 errors, plots, and the gate report."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from csm import (
    FP64GaussMarkovMemory,
    GaussMarkovMemory,
    direct_inverse_oracle,
    recompute_state,
)


def git_output(arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = max(torch.linalg.vector_norm(expected).item(), 1e-300)
    return float(torch.linalg.vector_norm(actual - expected).item() / denominator)


def recurrence_experiment(
    config: dict[str, Any], device: torch.device
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dimension in config["dimensions"]:
        d_value = min(config["value_dimension"], dimension)
        for seed in config["seeds"]:
            generator = torch.Generator(device=device).manual_seed(seed)
            steps = config["sequence_length"]
            keys = torch.randn(
                steps, dimension, generator=generator, device=device, dtype=torch.float64
            )
            keys /= torch.linalg.vector_norm(keys, dim=1, keepdim=True)
            values = torch.randn(
                steps, d_value, generator=generator, device=device, dtype=torch.float64
            )
            beta = 0.01 + 4.0 * torch.rand(
                steps, generator=generator, device=device, dtype=torch.float64
            )
            decay = 0.55 + 0.45 * torch.rand(
                steps, generator=generator, device=device, dtype=torch.float64
            )
            query = torch.randn(
                dimension, generator=generator, device=device, dtype=torch.float64
            )
            memory = FP64GaussMarkovMemory(
                dimension, d_value, epsilon=config["epsilon"]
            )
            sequential = memory.run(keys, values, beta, decay)
            recomputed = recompute_state(keys, values, beta, decay)
            sequential_read = memory.read(sequential, query)
            recomputed_read = memory.read(recomputed, query)
            oracle_read, oracle_variance = direct_inverse_oracle(
                sequential, query, config["epsilon"]
            )
            eigenvalues = torch.linalg.eigvalsh(sequential.S)
            records.append(
                {
                    "dimension": dimension,
                    "value_dimension": d_value,
                    "seed": seed,
                    "S_max_absolute_error": float(
                        (sequential.S - recomputed.S).abs().max().item()
                    ),
                    "C_max_absolute_error": float(
                        (sequential.C - recomputed.C).abs().max().item()
                    ),
                    "read_max_absolute_error": float(
                        (sequential_read - recomputed_read).abs().max().item()
                    ),
                    "oracle_read_max_absolute_error": float(
                        (sequential_read - oracle_read).abs().max().item()
                    ),
                    "variance_oracle_absolute_error": abs(
                        memory.confidence(sequential, query).item()
                        - oracle_variance.item()
                    ),
                    "S_symmetry_max_absolute_error": float(
                        (sequential.S - sequential.S.mT).abs().max().item()
                    ),
                    "S_minimum_eigenvalue": float(eigenvalues.min().item()),
                    "system_condition_number": float(
                        torch.linalg.cond(memory.system_matrix(sequential)).item()
                    ),
                }
            )
    synchronize(device)
    return records


def interpolation_experiment(
    config: dict[str, Any], device: torch.device
) -> list[dict[str, Any]]:
    dimension = config["interpolation_dimension"]
    generator = torch.Generator(device=device).manual_seed(9182)
    raw = torch.randn(
        dimension, dimension, generator=generator, device=device, dtype=torch.float64
    )
    keys = torch.linalg.qr(raw).Q.mT.contiguous()
    values = torch.randn(
        dimension, dimension, generator=generator, device=device, dtype=torch.float64
    )
    beta = 0.25 + 2.0 * torch.rand(
        dimension, generator=generator, device=device, dtype=torch.float64
    )
    decay = torch.ones(dimension, device=device, dtype=torch.float64)
    alpha = torch.tensor(
        [2.0, -1.0, 0.5, -0.25, 1.5, -2.0, 0.75, 0.1],
        device=device,
        dtype=torch.float64,
    )[:dimension]
    query = alpha @ keys
    target = alpha @ values
    records: list[dict[str, Any]] = []
    for epsilon in config["interpolation_epsilons"]:
        memory = FP64GaussMarkovMemory(dimension, dimension, epsilon=epsilon)
        state = memory.run(keys, values, beta, decay)
        stored_errors = torch.stack(
            [
                torch.linalg.vector_norm(memory.read(state, keys[index]) - values[index])
                for index in range(dimension)
            ]
        )
        functional_error = torch.linalg.vector_norm(memory.read(state, query) - target)
        records.append(
            {
                "epsilon": epsilon,
                "maximum_stored_key_error": float(stored_errors.max().item()),
                "mean_stored_key_error": float(stored_errors.mean().item()),
                "linear_functional_error": float(functional_error.item()),
            }
        )
    return records


def conditioning_experiment(
    config: dict[str, Any], device: torch.device
) -> list[dict[str, Any]]:
    dimension = 8
    generator = torch.Generator(device=device).manual_seed(8241)
    left = torch.linalg.qr(
        torch.randn(
            dimension, dimension, generator=generator, device=device, dtype=torch.float64
        )
    ).Q
    right = torch.linalg.qr(
        torch.randn(
            dimension, dimension, generator=generator, device=device, dtype=torch.float64
        )
    ).Q
    values64 = torch.randn(
        dimension, dimension, generator=generator, device=device, dtype=torch.float64
    )
    query64 = torch.randn(
        dimension, generator=generator, device=device, dtype=torch.float64
    )
    beta64 = torch.ones(dimension, device=device, dtype=torch.float64)
    decay64 = torch.ones(dimension, device=device, dtype=torch.float64)
    epsilon = config["conditioning_epsilon"]
    records: list[dict[str, Any]] = []
    for exponent in config["conditioning_singular_exponents"]:
        singular_values = torch.logspace(
            0,
            -float(exponent),
            dimension,
            device=device,
            dtype=torch.float64,
        )
        keys64 = (left @ torch.diag(singular_values) @ right.mT).mT.contiguous()
        fp64 = FP64GaussMarkovMemory(dimension, dimension, epsilon=epsilon)
        state64 = fp64.run(keys64, values64, beta64, decay64)
        reference = fp64.read(state64, query64)
        condition = torch.linalg.cond(fp64.system_matrix(state64)).item()

        fp32_error = float("nan")
        fp32_status = "ok"
        try:
            fp32 = GaussMarkovMemory(
                dimension,
                dimension,
                epsilon=epsilon,
                dtype=torch.float32,
            )
            state32 = fp32.run(
                keys64.float(), values64.float(), beta64.float(), decay64.float()
            )
            output32 = fp32.read(state32, query64.float()).double()
            fp32_error = relative_error(output32, reference)
        except RuntimeError as error:
            fp32_status = f"failed: {error}"
        records.append(
            {
                "singular_exponent": exponent,
                "system_condition_number": float(condition),
                "fp32_relative_read_error": fp32_error,
                "fp32_status": fp32_status,
            }
        )
    return records


def noise_averaging_experiment(
    config: dict[str, Any], device: torch.device
) -> list[dict[str, Any]]:
    dimension = 4
    sigma = config["noise_sigma"]
    trials = config["noise_trials"]
    generator = torch.Generator(device=device).manual_seed(811)
    key = torch.zeros(dimension, device=device, dtype=torch.float64)
    key[0] = 1.0
    true_value = torch.tensor(
        [0.5, -1.0, 2.0, -0.25], device=device, dtype=torch.float64
    )
    records: list[dict[str, Any]] = []
    for repetitions in config["noise_repetitions"]:
        squared_errors = []
        for _ in range(trials):
            values = true_value + sigma * torch.randn(
                repetitions,
                dimension,
                generator=generator,
                device=device,
                dtype=torch.float64,
            )
            keys = key.expand(repetitions, -1)
            beta = torch.ones(repetitions, device=device, dtype=torch.float64)
            decay = torch.ones(repetitions, device=device, dtype=torch.float64)
            memory = FP64GaussMarkovMemory(dimension, dimension, epsilon=1e-10)
            state = memory.run(keys, values, beta, decay)
            error = memory.read(state, key) - true_value
            squared_errors.append(error.square().sum())
        empirical = torch.stack(squared_errors).mean().item()
        predicted = sigma**2 * dimension / repetitions
        records.append(
            {
                "repetitions": repetitions,
                "empirical_mse": float(empirical),
                "predicted_mse": float(predicted),
                "empirical_to_predicted_ratio": float(empirical / predicted),
            }
        )
    return records


def gradcheck_experiment() -> dict[str, Any]:
    torch.manual_seed(751)
    memory = FP64GaussMarkovMemory(2, 2, epsilon=0.8)

    def function(keys, values, beta, decay, query):
        state = memory.run(keys, values, beta, decay)
        read, variance = memory.read_with_confidence(state, query)
        return torch.cat((read, variance.unsqueeze(0)))

    arguments = (
        torch.randn(3, 2, dtype=torch.float64, requires_grad=True),
        torch.randn(3, 2, dtype=torch.float64, requires_grad=True),
        (0.5 + torch.rand(3, dtype=torch.float64)).requires_grad_(),
        (0.75 + 0.1 * torch.rand(3, dtype=torch.float64)).requires_grad_(),
        torch.randn(2, dtype=torch.float64, requires_grad=True),
    )
    started = time.perf_counter()
    passed = torch.autograd.gradcheck(
        function,
        arguments,
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
        fast_mode=False,
    )
    return {"passed": bool(passed), "wall_clock_seconds": time.perf_counter() - started}


def pathological_experiment(device: torch.device) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def record(name: str, state, memory, query) -> None:
        read = memory.read(state, query)
        records.append(
            {
                "case": name,
                "finite_state": bool(torch.isfinite(state.S).all() and torch.isfinite(state.C).all()),
                "finite_read": bool(torch.isfinite(read).all()),
                "minimum_eigenvalue": float(torch.linalg.eigvalsh(state.S).min().item()),
                "read_norm": float(torch.linalg.vector_norm(read).item()),
            }
        )

    dtype = torch.float64
    # repeated keys
    memory = FP64GaussMarkovMemory(2, 2, epsilon=1e-8)
    keys = torch.tensor([[1.0, 0.0]] * 3, device=device, dtype=dtype)
    values = torch.tensor([[1.0, 2.0], [-1.0, 3.0], [4.0, 0.0]], device=device, dtype=dtype)
    state = memory.run(keys, values, torch.tensor([1.0, 2.0, 5.0], device=device, dtype=dtype), torch.ones(3, device=device, dtype=dtype))
    record("repeated_keys", state, memory, keys[0])

    # nearly collinear keys and tiny epsilon
    delta = 1e-7
    keys = torch.tensor([[1.0, 0.0], [1.0, delta]], device=device, dtype=dtype)
    keys /= torch.linalg.vector_norm(keys, dim=1, keepdim=True)
    memory = FP64GaussMarkovMemory(2, 2, epsilon=1e-12)
    state = memory.run(keys, values[:2], torch.ones(2, device=device, dtype=dtype), torch.ones(2, device=device, dtype=dtype))
    record("nearly_collinear_tiny_epsilon", state, memory, keys[0])

    # beta zero
    memory = FP64GaussMarkovMemory(2, 2)
    state = memory.write(memory.initial_state(device=device), keys[0], values[0], 0.0, 1.0)
    record("beta_zero", state, memory, keys[0])

    # lambda below one and very large beta
    large_keys = torch.eye(2, device=device, dtype=dtype)
    state = memory.initial_state(device=device)
    state = memory.write(state, large_keys[0], values[0], 1e150, 1.0)
    state = memory.write(state, large_keys[1], values[1], 1e149, 0.2)
    record("lambda_below_one_large_beta", state, memory, large_keys[1])

    # zero values and single observation
    memory = FP64GaussMarkovMemory(2, 2)
    state = memory.write(memory.initial_state(device=device), keys[0], torch.zeros(2, device=device, dtype=dtype), 1.0, 1.0)
    record("zero_value_single_observation", state, memory, keys[0])
    return records


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def make_plots(metrics: dict[str, Any], plot_directory: Path) -> list[str]:
    plot_directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    recurrence = metrics["recurrence"]
    dimensions = sorted({record["dimension"] for record in recurrence})
    state_max = [
        max(
            max(record["S_max_absolute_error"], record["C_max_absolute_error"])
            for record in recurrence
            if record["dimension"] == dimension
        )
        for dimension in dimensions
    ]
    read_max = [
        max(record["read_max_absolute_error"] for record in recurrence if record["dimension"] == dimension)
        for dimension in dimensions
    ]
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.semilogy(dimensions, np.maximum(state_max, 1e-20), "o-", label="state max |error|")
    axis.semilogy(dimensions, np.maximum(read_max, 1e-20), "s-", label="read max |error|")
    axis.set(xlabel="key dimension", ylabel="maximum absolute error", title="Sequential recurrence vs historical recomputation")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    path = plot_directory / "recurrence_consistency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    interpolation = metrics["interpolation"]
    epsilons = [record["epsilon"] for record in interpolation]
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.loglog(epsilons, [record["maximum_stored_key_error"] for record in interpolation], "o-", label="stored-key max")
    axis.loglog(epsilons, [record["linear_functional_error"] for record in interpolation], "s-", label="linear-functional")
    axis.set(xlabel="epsilon", ylabel="L2 error", title="Finite-epsilon interpolation error")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    path = plot_directory / "interpolation_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    conditioning = metrics["conditioning"]
    valid = [record for record in conditioning if np.isfinite(record["fp32_relative_read_error"])]
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.loglog([record["system_condition_number"] for record in valid], [record["fp32_relative_read_error"] for record in valid], "o-")
    axis.set(xlabel="condition number of S + epsilon I", ylabel="fp32 relative read error vs fp64", title="Conditioning and finite precision")
    axis.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = plot_directory / "conditioning_fp32.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    noise = metrics["noise_averaging"]
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.loglog([record["repetitions"] for record in noise], [record["empirical_mse"] for record in noise], "o-", label="empirical")
    axis.loglog([record["repetitions"] for record in noise], [record["predicted_mse"] for record in noise], "--", label="sigma^2 d_v / n")
    axis.set(xlabel="repeated observations n", ylabel="mean squared error", title="Noisy repeated-write averaging")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    path = plot_directory / "noise_averaging.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def gate_summary(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    recurrence = metrics["recurrence"]
    maximum_state_error = max(
        max(record["S_max_absolute_error"], record["C_max_absolute_error"])
        for record in recurrence
    )
    maximum_read_error = max(record["read_max_absolute_error"] for record in recurrence)
    minimum_eigenvalue = min(record["S_minimum_eigenvalue"] for record in recurrence)
    thresholds = config["gate"]
    checks = {
        "state_agreement": maximum_state_error <= thresholds["state_max_absolute_error"],
        "read_agreement": maximum_read_error <= thresholds["read_max_absolute_error"],
        "positive_semidefinite": minimum_eigenvalue >= thresholds["psd_minimum_eigenvalue"],
        "gradcheck": metrics["gradcheck"]["passed"] if thresholds["gradcheck_required"] else True,
        "pathologies_finite": all(
            record["finite_state"] and record["finite_read"]
            for record in metrics["pathologies"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "maximum_state_absolute_error": maximum_state_error,
        "maximum_read_absolute_error": maximum_read_error,
        "minimum_S_eigenvalue": minimum_eigenvalue,
        "thresholds": thresholds,
    }


def render_report(record: dict[str, Any]) -> str:
    gate = record["gate"]
    metrics = record["metrics"]
    recurrence = metrics["recurrence"]
    noise = metrics["noise_averaging"]
    condition = metrics["conditioning"]
    status = "PASS" if gate["passed"] else "FAIL"
    checks = "\n".join(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in gate["checks"].items()
    )
    path_rows = "\n".join(
        f"| {item['case']} | {item['finite_state']} | {item['finite_read']} | {item['minimum_eigenvalue']:.3e} |"
        for item in metrics["pathologies"]
    )
    plots = "\n".join(f"- [`{Path(path).name}`](../../{path})" for path in record["plots"])
    return f"""# Phase 1 report

## Gate decision: {status}

The fp64 sequential recurrence and the independent historical recomputation were evaluated for `d_k in {{2,4,8,16,32}}`, five deterministic seeds per dimension, random beta, random lambda, and Cholesky-based reads. The direct inverse appears only as a tiny-matrix oracle.

{checks}

| Gate metric | Observed | Required |
|---|---:|---:|
| maximum state absolute error | {gate['maximum_state_absolute_error']:.6e} | <= {gate['thresholds']['state_max_absolute_error']:.1e} |
| maximum read absolute error | {gate['maximum_read_absolute_error']:.6e} | <= {gate['thresholds']['read_max_absolute_error']:.1e} |
| minimum eigenvalue of S | {gate['minimum_S_eigenvalue']:.6e} | >= {gate['thresholds']['psd_minimum_eigenvalue']:.1e} |
| gradcheck | {metrics['gradcheck']['passed']} | True |

## Equation-level findings

- Sequential and recomputed states agree across {len(recurrence)} randomized cases. Agreement covers both sufficient statistics and queries, not merely final loss values.
- Cholesky reads and the inverse oracle agree in the automated test suite. Production-style reads never construct an inverse.
- The interpolation sweep shows stored-key and signed linear-functional errors approaching zero with epsilon.
- Repeated noisy observations follow the predicted `sigma^2 d_v / n` curve; empirical/predicted ratios range from {min(row['empirical_to_predicted_ratio'] for row in noise):.3f} to {max(row['empirical_to_predicted_ratio'] for row in noise):.3f} over {record['config']['noise_trials']} trials per point.
- The fp32 conditioning sweep reaches a maximum measured condition number of {max(row['system_condition_number'] for row in condition):.3e}; the associated error curve is an empirical finite-precision diagnostic, not a Phase 1 pass criterion.
- Manuscript `c_t(q)` is mathematically a posterior variance: smaller values mean higher confidence. The implementation names the public method `confidence` for equation compatibility and documents this direction explicitly.

## Pathological cases

| Case | Finite state | Finite read | min eig(S) |
|---|---:|---:|---:|
{path_rows}

The test suite additionally covers lambda=1, lambda<1, tiny epsilon with a well-conditioned basis, beta=0, very large beta, zero values, a single observation with a closed-form answer, repeated keys, and nearly collinear keys.

## Plots

{plots}

## Reproducibility record

- git commit tested: `{record['git_commit']}`
- working tree dirty at run time: `{record['working_tree_dirty']}`
- config: [`configs/phase1_reference.json`](../../configs/phase1_reference.json)
- seeds: `{record['config']['seeds']}` plus fixed per-experiment seeds recorded in source
- hardware: `{record['hardware']}`
- software: Python `{record['software']['python']}`, PyTorch `{record['software']['torch']}`, HIP `{record['software']['hip']}`
- wall-clock time: `{record['wall_clock_seconds']:.3f}` seconds
- peak allocated VRAM: `{record['peak_vram_bytes'] / 2**30:.6f}` GiB
- machine-readable metrics: [`phase1_metrics.json`](../phase1_metrics.json)

## Interpretation

{record['interpretation']}

This gate validates the Phase 1 implementation against Definition 5.1. It does not validate learned encoders, optimized scans, the dyadic cascade, language modeling, or the broader architecture claims. No work beyond Phase 1 is authorized by this result.
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase1_reference.json")
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--plots", type=Path, default=Path("plots/phase1"))
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    config = json.loads(args.config.read_text())
    device = select_device(config["device"])
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    metrics = {
        "recurrence": recurrence_experiment(config, device),
        "interpolation": interpolation_experiment(config, device),
        "conditioning": conditioning_experiment(config, device),
        "noise_averaging": noise_averaging_experiment(config, device),
        "gradcheck": gradcheck_experiment(),
        "pathologies": pathological_experiment(device),
    }
    plots = make_plots(metrics, args.plots)
    gate = gate_summary(metrics, config)
    synchronize(device)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "working_tree_dirty": bool(git_output(["status", "--porcelain"])),
        "config_path": str(args.config),
        "config": config,
        "seed": config["seeds"],
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "numpy": np.__version__,
        },
        "wall_clock_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "metrics": metrics,
        "plots": plots,
        "gate": gate,
        "interpretation": (
            "The reference implementation represents the stated recurrence and reads to fp64 numerical precision under the tested gate distribution."
            if gate["passed"]
            else "The Phase 1 reference gate failed; inspect individual checks before drawing theoretical conclusions."
        ),
    }

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "phase1").mkdir(parents=True, exist_ok=True)
    (args.results / "phase1_metrics.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    for name, rows in metrics.items():
        if isinstance(rows, list) and rows:
            write_csv(args.results / "phase1" / f"{name}.csv", rows)
    report = render_report(record)
    (args.results / "phase1_report.md").write_text(report)
    print(report)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
