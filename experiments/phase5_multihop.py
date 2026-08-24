#!/usr/bin/env python3
"""Phase 5: chained adaptive reads on controlled functional graphs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from csm import (
    FP64GaussMarkovMemory,
    chase_indices,
    explicit_pair_state,
    nearest_code,
    prepared_read_operator,
    softmax_read_many,
)
from csm.baselines import csm_state_bytes, explicit_pair_state_bytes


METHODS = ("csm_chained", "softmax_repeated", "softmax_one")


def git_output(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: Iterable[float], q: float) -> float:
    return float(np.quantile(np.asarray(list(values), dtype=np.float64), q))


def normalize_rows(values: Tensor) -> Tensor:
    return values / torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(
        torch.finfo(values.dtype).tiny
    )


def make_codes(
    representation: str,
    condition: str,
    count: int,
    dimension: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    if representation == "controlled":
        if count > dimension:
            raise ValueError("orthogonal controlled codes require K <= d_key")
        return torch.linalg.qr(
            torch.randn(
                dimension,
                count,
                generator=generator,
                device=device,
                dtype=torch.float64,
            ),
            mode="reduced",
        ).Q.mT
    independent = torch.randn(
        count,
        dimension,
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    if condition == "random":
        return normalize_rows(independent)
    if condition.startswith("correlated_"):
        correlation = float(condition.split("_", 1)[1])
        common = normalize_rows(
            torch.randn(
                1,
                dimension,
                generator=generator,
                device=device,
                dtype=torch.float64,
            )
        )
        return normalize_rows(
            math.sqrt(correlation) * common
            + math.sqrt(1.0 - correlation) * independent / math.sqrt(dimension)
        )
    raise ValueError(f"unknown nonorthogonal condition: {condition}")


def make_successors(
    graph_regime: str,
    count: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    if graph_regime == "permutation":
        return torch.randperm(count, generator=generator, device=device)
    if graph_regime == "many_to_one":
        successor_pool = max(1, count // 4)
        return torch.randint(
            successor_pool,
            (count,),
            generator=generator,
            device=device,
        )
    raise ValueError(f"unknown graph regime: {graph_regime}")


def best_softmax_temperature(
    keys: Tensor, values: Tensor, temperatures: list[float]
) -> float:
    state = explicit_pair_state(keys, values)
    best_temperature = math.nan
    best_error = math.inf
    for temperature in temperatures:
        reads, _ = softmax_read_many(state, keys, temperature)
        error = torch.mean(torch.sum((reads - values) ** 2, dim=1)).item()
        if error < best_error:
            best_error = error
            best_temperature = temperature
    return best_temperature


def geometric_sum(value: float, terms: int) -> float:
    if abs(value - 1.0) < 1e-10:
        return float(terms)
    return float((value**terms - 1.0) / (value - 1.0))


def method_flops(method: str, dimension: int, edges: int, hops: int) -> int:
    csm_read = 4 * dimension * dimension
    softmax_read = 4 * edges * dimension + 5 * edges
    if method == "csm_chained":
        return hops * csm_read
    if method == "softmax_repeated":
        return hops * softmax_read
    if method == "softmax_one":
        return softmax_read
    raise ValueError(f"unknown method: {method}")


def evaluate_graph(
    *,
    codes: Tensor,
    successors: Tensor,
    starts: Tensor,
    representation: str,
    condition: str,
    graph_regime: str,
    dimension: int,
    epsilon: float,
    seed: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    edges = codes.shape[0]
    values = codes[successors]
    memory = FP64GaussMarkovMemory(dimension, dimension, epsilon=epsilon)
    state = memory.undiscounted_state(codes, values)
    system = memory.system_matrix(state)
    factor = torch.linalg.cholesky(system)
    operator = prepared_read_operator(memory, state)
    operator_norm = torch.linalg.matrix_norm(operator, ord=2).item()
    system_condition = torch.linalg.cond(system).item()
    gram = codes @ codes.mT
    gram_condition = (
        torch.linalg.cond(gram).item() if edges <= dimension else math.inf
    )
    off_diagonal = gram - torch.eye(
        edges, device=codes.device, dtype=torch.float64
    )
    coherence = torch.max(torch.abs(off_diagonal)).item() if edges > 1 else 0.0
    maximum_hops = max(config["hops"])
    target_indices = chase_indices(successors, starts, maximum_hops)

    def prepared_csm_read(queries: Tensor) -> tuple[Tensor, Tensor]:
        solved = torch.cholesky_solve(queries.mT, factor).mT
        return solved @ state.C.mT, torch.einsum("bd,bd->b", queries, solved)

    all_one_step, _ = prepared_csm_read(codes)
    one_hop_error = torch.linalg.vector_norm(all_one_step - values, dim=1)
    epsilon_one = one_hop_error.max().item()
    csm_queries = [codes[starts]]
    csm_confidence = []
    for _ in range(maximum_hops):
        read, confidence = prepared_csm_read(csm_queries[-1])
        csm_queries.append(read)
        csm_confidence.append(confidence)

    explicit = explicit_pair_state(codes, values)
    temperature = best_softmax_temperature(
        codes, values, config["softmax_temperatures"]
    )
    softmax_queries = [codes[starts]]
    for _ in range(maximum_hops):
        read, _ = softmax_read_many(explicit, softmax_queries[-1], temperature)
        softmax_queries.append(read)
    one_softmax = softmax_queries[1]

    rows: list[dict[str, Any]] = []
    for hop in range(1, maximum_hops + 1):
        target = codes[target_indices[hop]]
        predictions = {
            "csm_chained": csm_queries[hop],
            "softmax_repeated": softmax_queries[hop],
            "softmax_one": one_softmax,
        }
        if hop not in config["hops"]:
            continue
        for method, prediction in predictions.items():
            endpoint_errors = torch.linalg.vector_norm(
                prediction - target, dim=1
            )
            mean_error = endpoint_errors.mean().item()
            # Include skipped intermediate hops in the cumulative total.
            # Recompute the exact prefix sum to make each row independently auditable.
            if method == "csm_chained":
                prefix_predictions = csm_queries
            elif method == "softmax_repeated":
                prefix_predictions = softmax_queries
            else:
                prefix_predictions = [codes[starts]] + [one_softmax] * maximum_hops
            accumulated_error = sum(
                torch.linalg.vector_norm(
                    prefix_predictions[index] - codes[target_indices[index]], dim=1
                ).mean().item()
                for index in range(1, hop + 1)
            )
            decoded = nearest_code(codes, prediction)
            success = (decoded == target_indices[hop]).double().mean().item()
            bound = (
                epsilon_one * geometric_sum(operator_norm, hop)
                if method == "csm_chained"
                else math.nan
            )
            maximum_error = endpoint_errors.max().item()
            rows.append(
                {
                    "representation": representation,
                    "key_condition": condition,
                    "graph_regime": graph_regime,
                    "d_key": dimension,
                    "edges": edges,
                    "load": edges / dimension,
                    "capacity_domain": (
                        "at_or_below_capacity"
                        if edges <= dimension
                        else "over_capacity"
                    ),
                    "epsilon": epsilon,
                    "seed": seed,
                    "starts": starts.numel(),
                    "method": method,
                    "hops": hop,
                    "adaptive_reads": hop if method != "softmax_one" else 1,
                    "minimum_layer_depth": (
                        1 if method in ("csm_chained", "softmax_one") else hop
                    ),
                    "success_rate": success,
                    "mean_endpoint_error": mean_error,
                    "p90_endpoint_error": torch.quantile(
                        endpoint_errors, 0.9
                    ).item(),
                    "maximum_endpoint_error": maximum_error,
                    "accumulated_mean_error": accumulated_error,
                    "mean_confidence": (
                        csm_confidence[hop - 1].mean().item()
                        if method == "csm_chained"
                        else math.nan
                    ),
                    "maximum_one_hop_csm_error": (
                        epsilon_one if method == "csm_chained" else math.nan
                    ),
                    "read_operator_norm": (
                        operator_norm if method == "csm_chained" else math.nan
                    ),
                    "theorem_error_bound": bound,
                    "bound_relative_excess": (
                        max(0.0, maximum_error - bound) / max(bound, 1e-300)
                        if method == "csm_chained"
                        else math.nan
                    ),
                    "system_condition_number": system_condition,
                    "key_gram_condition_number": gram_condition,
                    "maximum_code_coherence": coherence,
                    "selected_softmax_temperature": (
                        temperature if method.startswith("softmax") else math.nan
                    ),
                    "state_bytes": (
                        csm_state_bytes(dimension, dimension)
                        if method == "csm_chained"
                        else explicit_pair_state_bytes(
                            edges, dimension, dimension
                        )
                    ),
                    "estimated_total_flops": method_flops(
                        method, dimension, edges, hop
                    ),
                }
            )
    return rows


def graph_sweep(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in config["dimensions"]:
        regimes = [
            ("controlled", "orthogonal", config["orthogonal_loads"]),
            *(
                ("random_nonorthogonal", condition, config["nonorthogonal_loads"])
                for condition in config["nonorthogonal_conditions"]
            ),
        ]
        for representation_index, (representation, condition, loads) in enumerate(
            regimes
        ):
            for load in loads:
                edges = max(2, int(round(dimension * load)))
                for graph_index, graph_regime in enumerate(config["graph_regimes"]):
                    for seed in config["seeds"]:
                        generator = torch.Generator(device=device).manual_seed(
                            seed * 10_000_019
                            + dimension * 100_003
                            + edges * 1_009
                            + representation_index * 101
                            + graph_index
                        )
                        codes = make_codes(
                            representation,
                            condition,
                            edges,
                            dimension,
                            generator,
                            device,
                        )
                        successors = make_successors(
                            graph_regime, edges, generator, device
                        )
                        start_count = min(edges, config["maximum_starts"])
                        starts = torch.randperm(
                            edges, generator=generator, device=device
                        )[:start_count]
                        for epsilon in config["epsilons"]:
                            rows.extend(
                                evaluate_graph(
                                    codes=codes,
                                    successors=successors,
                                    starts=starts,
                                    representation=representation,
                                    condition=condition,
                                    graph_regime=graph_regime,
                                    dimension=dimension,
                                    epsilon=epsilon,
                                    seed=seed,
                                    config=config,
                                )
                            )
    return rows


def benchmark(
    operation: Callable[[], Tensor],
    device: torch.device,
    warmups: int,
    repetitions: int,
) -> float:
    for _ in range(warmups):
        operation()
    synchronize(device)
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        operation()
        synchronize(device)
        timings.append(time.perf_counter_ns() - started)
    return float(np.median(timings) / 1_000.0)


def latency_sweep(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in config["latency_dimensions"]:
        for representation, condition in (
            ("controlled", "orthogonal"),
            ("random_nonorthogonal", "correlated_0.8"),
        ):
            generator = torch.Generator(device=device).manual_seed(
                8_000_009 + dimension + (0 if representation == "controlled" else 1)
            )
            edges = dimension
            codes = make_codes(
                representation,
                condition,
                edges,
                dimension,
                generator,
                device,
            )
            successors = make_successors("permutation", edges, generator, device)
            values = codes[successors]
            memory = FP64GaussMarkovMemory(dimension, dimension, epsilon=1e-4)
            state = memory.undiscounted_state(codes, values)
            factor = torch.linalg.cholesky(memory.system_matrix(state))
            explicit = explicit_pair_state(codes, values)
            temperature = best_softmax_temperature(
                codes, values, config["softmax_temperatures"]
            )
            start = codes[:1]

            def csm_operation(hops: int) -> Callable[[], Tensor]:
                def operation() -> Tensor:
                    current = start
                    for _ in range(hops):
                        solved = torch.cholesky_solve(current.mT, factor).mT
                        current = solved @ state.C.mT
                    return current

                return operation

            def softmax_operation(hops: int) -> Callable[[], Tensor]:
                def operation() -> Tensor:
                    current = start
                    for _ in range(hops):
                        current, _ = softmax_read_many(
                            explicit, current, temperature
                        )
                    return current

                return operation

            for hops in config["latency_hops"]:
                operations = {
                    "csm_chained": csm_operation(hops),
                    "softmax_repeated": softmax_operation(hops),
                    "softmax_one": softmax_operation(1),
                }
                for method, operation in operations.items():
                    rows.append(
                        {
                            "representation": representation,
                            "key_condition": condition,
                            "d_key": dimension,
                            "edges": edges,
                            "hops_target": hops,
                            "method": method,
                            "adaptive_reads": (
                                hops if method != "softmax_one" else 1
                            ),
                            "minimum_layer_depth": (
                                1
                                if method in ("csm_chained", "softmax_one")
                                else hops
                            ),
                            "state_bytes": (
                                csm_state_bytes(dimension, dimension)
                                if method == "csm_chained"
                                else explicit_pair_state_bytes(
                                    edges, dimension, dimension
                                )
                            ),
                            "estimated_total_flops": method_flops(
                                method, dimension, edges, hops
                            ),
                            "median_microseconds_per_chain": benchmark(
                                operation,
                                device,
                                config["latency_warmups"],
                                config["latency_repetitions"],
                            ),
                        }
                    )
    return rows


def aggregate_performance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["representation"],
                row["key_condition"],
                row["capacity_domain"],
                row["graph_regime"],
                row["epsilon"],
                row["method"],
                row["hops"],
            )
        ].append(row)
    output = []
    for key, values in sorted(groups.items()):
        csm_values = [
            row for row in values if math.isfinite(row["read_operator_norm"])
        ]
        output.append(
            {
                "representation": key[0],
                "key_condition": key[1],
                "capacity_domain": key[2],
                "graph_regime": key[3],
                "epsilon": key[4],
                "method": key[5],
                "hops": key[6],
                "mean_success_rate": float(
                    np.mean([row["success_rate"] for row in values])
                ),
                "median_endpoint_error": percentile(
                    [row["mean_endpoint_error"] for row in values], 0.5
                ),
                "p90_endpoint_error": percentile(
                    [row["mean_endpoint_error"] for row in values], 0.9
                ),
                "median_accumulated_error": percentile(
                    [row["accumulated_mean_error"] for row in values], 0.5
                ),
                "median_confidence": (
                    percentile([row["mean_confidence"] for row in values], 0.5)
                    if csm_values
                    else math.nan
                ),
                "median_operator_norm": (
                    percentile(
                        [row["read_operator_norm"] for row in csm_values], 0.5
                    )
                    if csm_values
                    else math.nan
                ),
                "case_count": len(values),
            }
        )
    return output


def gate_summary(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    minimum_epsilon = min(config["epsilons"])
    controlled = [
        row
        for row in rows
        if row["representation"] == "controlled"
        and row["method"] == "csm_chained"
        and row["epsilon"] == minimum_epsilon
    ]
    controlled_h16 = [row for row in controlled if row["hops"] == 16]
    minimum_success = min(row["success_rate"] for row in controlled)
    maximum_h16_error = max(row["maximum_endpoint_error"] for row in controlled_h16)
    maximum_bound_excess = max(row["bound_relative_excess"] for row in controlled)
    amplification = [
        row["read_operator_norm"]
        for row in controlled
        if row["graph_regime"] == "many_to_one" and row["hops"] == 16
    ]
    observed = {
        "controlled_minimum_success_rate": minimum_success,
        "controlled_maximum_h16_error": maximum_h16_error,
        "maximum_theorem_bound_relative_excess": maximum_bound_excess,
        "many_to_one_maximum_operator_norm": max(amplification),
        "controlled_case_count": len(controlled),
    }
    thresholds = config["gate"]
    checks = {
        "controlled_codes_reproduce_multihop_behavior": minimum_success
        >= thresholds["controlled_minimum_success_rate"],
        "controlled_h16_vector_error_is_small": maximum_h16_error
        <= thresholds["controlled_maximum_h16_error"],
        "error_propagation_bound_holds": maximum_bound_excess
        <= thresholds["bound_relative_tolerance"],
        "operator_amplification_counterexample_is_visible": max(amplification) > 1.0,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
    }, observed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    summary: list[dict[str, Any]],
    latency: list[dict[str, Any]],
    config: dict[str, Any],
    directory: Path,
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    minimum_epsilon = min(config["epsilons"])
    colors = dict(zip(METHODS, plt.cm.tab10.colors))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for axis, graph in zip(axes, config["graph_regimes"]):
        for method in METHODS:
            medians = []
            for hops in config["hops"]:
                values = [
                    row["mean_success_rate"]
                    for row in summary
                    if row["representation"] == "controlled"
                    and row["graph_regime"] == graph
                    and row["epsilon"] == minimum_epsilon
                    and row["method"] == method
                    and row["hops"] == hops
                ]
                medians.append(float(np.mean(values)))
            axis.plot(
                config["hops"],
                medians,
                marker="o",
                label=method,
                color=colors[method],
            )
        axis.set_title(graph)
        axis.set_xlabel("target hops")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("nearest-code success rate")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Controlled-code pointer chasing")
    fig.tight_layout()
    path = directory / "controlled_success_by_hop.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for epsilon in config["epsilons"]:
        medians = []
        for hops in config["hops"]:
            values = [
                row["median_endpoint_error"]
                for row in summary
                if row["representation"] == "controlled"
                and row["graph_regime"] == "permutation"
                and row["epsilon"] == epsilon
                and row["method"] == "csm_chained"
                and row["hops"] == hops
            ]
            medians.append(float(np.median(values)))
        axis.semilogy(config["hops"], medians, marker="o", label=f"eps={epsilon:g}")
    axis.set(
        xlabel="hops",
        ylabel="median endpoint vector error",
        title="Finite-epsilon accumulation on controlled permutations",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "epsilon_error_accumulation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    conditions = ["orthogonal", *config["nonorthogonal_conditions"]]
    for condition in conditions:
        subset = [
            row
            for row in summary
            if row["key_condition"] == condition
            and row["capacity_domain"] == "at_or_below_capacity"
            and row["epsilon"] == minimum_epsilon
            and row["method"] == "csm_chained"
            and row["hops"] == 16
        ]
        if subset:
            axis.scatter(
                [row["median_operator_norm"] for row in subset],
                [row["median_endpoint_error"] for row in subset],
                label=condition,
                s=45,
            )
    axis.set_yscale("log")
    axis.set(
        xlabel="read operator norm",
        ylabel="H=16 endpoint error",
        title="Geometry and operator amplification",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "operator_amplification.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for method in METHODS:
        subset = [
            row
            for row in latency
            if row["representation"] == "controlled"
            and row["d_key"] == max(config["latency_dimensions"])
            and row["method"] == method
        ]
        axis.plot(
            [row["hops_target"] for row in subset],
            [row["median_microseconds_per_chain"] for row in subset],
            marker="o",
            label=method,
            color=colors[method],
        )
    axis.set(
        xlabel="target hops",
        ylabel="median microseconds / single-query chain",
        title=f"Prepared reference latency at d={max(config['latency_dimensions'])}",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "chain_latency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def render_report(record: dict[str, Any]) -> str:
    status = "PASS" if record["gate"]["passed"] else "FAIL"
    checks = "\n".join(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in record["gate"]["checks"].items()
    )
    minimum_epsilon = min(record["config"]["epsilons"])
    controlled_table = markdown_table(
        ["graph", "method", "H=1 success", "H=16 success", "H=16 error"],
        [
            [
                graph,
                method,
                f"{np.mean([row['mean_success_rate'] for row in record['performance_summary'] if row['representation'] == 'controlled' and row['graph_regime'] == graph and row['epsilon'] == minimum_epsilon and row['method'] == method and row['hops'] == 1]):.3f}",
                f"{np.mean([row['mean_success_rate'] for row in record['performance_summary'] if row['representation'] == 'controlled' and row['graph_regime'] == graph and row['epsilon'] == minimum_epsilon and row['method'] == method and row['hops'] == 16]):.3f}",
                f"{np.median([row['median_endpoint_error'] for row in record['performance_summary'] if row['representation'] == 'controlled' and row['graph_regime'] == graph and row['epsilon'] == minimum_epsilon and row['method'] == method and row['hops'] == 16]):.3e}",
            ]
            for graph in record["config"]["graph_regimes"]
            for method in METHODS
        ],
    )
    degradation_table = markdown_table(
        ["geometry", "capacity", "epsilon", "H=16 success", "H=16 error", "operator norm"],
        [
            [
                condition,
                capacity,
                f"{epsilon:g}",
                f"{np.mean([row['mean_success_rate'] for row in record['performance_summary'] if row['key_condition'] == condition and row['capacity_domain'] == capacity and row['epsilon'] == epsilon and row['method'] == 'csm_chained' and row['hops'] == 16]):.3f}",
                f"{np.median([row['median_endpoint_error'] for row in record['performance_summary'] if row['key_condition'] == condition and row['capacity_domain'] == capacity and row['epsilon'] == epsilon and row['method'] == 'csm_chained' and row['hops'] == 16]):.3e}",
                f"{np.median([row['median_operator_norm'] for row in record['performance_summary'] if row['key_condition'] == condition and row['capacity_domain'] == capacity and row['epsilon'] == epsilon and row['method'] == 'csm_chained' and row['hops'] == 16]):.3f}",
            ]
            for condition, capacity in (
                ("orthogonal", "at_or_below_capacity"),
                ("random", "at_or_below_capacity"),
                ("correlated_0.8", "at_or_below_capacity"),
                ("correlated_0.98", "at_or_below_capacity"),
                ("random", "over_capacity"),
                ("correlated_0.98", "over_capacity"),
            )
            for epsilon in (minimum_epsilon, max(record["config"]["epsilons"]))
        ],
    )
    representative_latency = [
        row
        for row in record["latency"]
        if row["representation"] == "controlled"
        and row["d_key"] == max(record["config"]["latency_dimensions"])
        and row["hops_target"] == 16
    ]
    systems_table = markdown_table(
        ["method", "adaptive reads", "layer depth", "state bytes", "FLOPs", "microseconds"],
        [
            [
                row["method"],
                str(row["adaptive_reads"]),
                str(row["minimum_layer_depth"]),
                str(row["state_bytes"]),
                str(row["estimated_total_flops"]),
                f"{row['median_microseconds_per_chain']:.3f}",
            ]
            for row in representative_latency
        ],
    )
    observed = record["observed"]
    plots = "\n".join(
        f"- [`{Path(path).name}`](../{path})" for path in record["plots"]
    )
    return f"""# Phase 5 multi-hop functional graphs

## Gate decision: {status}

{checks}

| Gate measurement | Observed |
|---|---:|
| controlled minimum success rate, all H | {observed['controlled_minimum_success_rate']:.6f} |
| controlled maximum H=16 vector error | {observed['controlled_maximum_h16_error']:.3e} |
| maximum relative excess over propagation bound | {observed['maximum_theorem_bound_relative_excess']:.3e} |
| maximum many-to-one read-operator norm | {observed['many_to_one_maximum_operator_norm']:.3f} |

## Controlled pointer chasing

Each state stores a complete functional graph `node -> successor(node)`. The start code is read adaptively and the output becomes the next query, while `S` and `C` remain unchanged. Nearest-code success and vector error are both reported because finite epsilon can shrink a vector without changing its decoded node.

{controlled_table}

Repeated softmax is a strong equal-access baseline and often succeeds on these easy controlled codes. Its distinction is structural: H adaptive accesses require H attention layers, whereas the tested CSM layer exposes H reads against one maintained state. One softmax access has one adaptive round and therefore generally cannot produce an H-hop target for `H>1`.

## Error accumulation and cause attribution

{degradation_table}

The sweep varies edge count, `d_key`, epsilon, key geometry, `H in {{1,2,4,8,16}}`, and load `K/d_key`. Controlled orthogonal codes are necessarily restricted to `K/d_key <= 1`; random nonorthogonal codes include the over-capacity `1.5` load. The raw rows include per-hop endpoint error, accumulated prefix error, system/Gram conditioning, coherence, confidence, and exact small-matrix operator norms.

Many-to-one graphs produce operator norms above one even though every successor code is unit norm. This retains the manuscript's corrected amplification counterexample: the simpler `H * epsilon_1` bound is not used unless the operator is contractive. The full geometric bound is checked on every controlled row.

Observed failures are attributable through the recorded axes: larger epsilon creates systematic per-hop shrinkage; correlated geometry increases system conditioning and amplification; `K>d_key` crosses the value-linear capacity limit; and `L>1` permits perturbation growth. These factors are shown rather than filtered.

## Architectural claim versus systems claim

The architectural result is that `csm_chained` performs all 16 adaptive reads with one fixed state and a declared one-layer read loop. The systems result is separate. The following is a prepared fp64 single-query diagnostic at `d_key=K=64`, where CSM reuses a Cholesky factor and explicit softmax retains all pairs:

{systems_table}

FLOPs are leading-operation estimates. Timings include Python/PyTorch dispatch and synchronization, use unoptimized reference kernels, and are not throughput claims. A successful architectural demonstration does not imply these reads are cheap enough to matter in a trained system.

## Plots

{plots}

## Reproducibility

- source checkpoint: `{record['git_commit']}`; dirty at experiment start: `{record['working_tree_dirty']}`
- config: [`configs/phase5_multihop.json`](../configs/phase5_multihop.json)
- device: `{record['hardware']}`
- software: Python `{record['software']['python']}`, PyTorch `{record['software']['torch']}`, HIP `{record['software']['hip']}`, NumPy `{record['software']['numpy']}`
- wall time: {record['wall_clock_seconds']:.3f} seconds; peak allocated VRAM: {record['peak_vram_bytes'] / 2**30:.6f} GiB
- raw hop rows: [`phase5/pointer_chasing.csv`](phase5/pointer_chasing.csv)
- latency rows: [`phase5/latency.csv`](phase5/latency.csv)
- machine-readable record: [`phase5_metrics.json`](phase5_metrics.json)

## Scoped conclusion

{record['interpretation']}

This phase uses controlled and random representations, not learned encoders. It validates the state/read mechanism and exposes random-geometry failures; it does not satisfy the separate learned-memory or NLP gate.
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase5_multihop.json")
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--plots", type=Path, default=Path("plots/phase5"))
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    source_dirty = bool(git_output(["status", "--porcelain"]))
    config = json.loads(args.config.read_text())
    device = select_device(config["device"])
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows = graph_sweep(config, device)
    latency = latency_sweep(config, device)
    summary = aggregate_performance(rows)
    gate, observed = gate_summary(rows, config)
    plots = make_plots(summary, latency, config, args.plots)
    synchronize(device)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "working_tree_dirty": source_dirty,
        "config_path": str(args.config),
        "config": config,
        "hardware": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "numpy": np.__version__,
        },
        "wall_clock_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        ),
        "row_counts": {"pointer_chasing": len(rows), "latency": len(latency)},
        "performance_summary": summary,
        "latency": latency,
        "observed": observed,
        "plots": plots,
        "gate": gate,
        "interpretation": (
            "Controlled codes reproduce accurate H-hop chasing through 16 adaptive reads against one unchanged CSM state, and the full operator-norm propagation bound holds. Random/correlated and over-capacity rows show where epsilon, geometry, capacity, and amplification degrade the chain. Reference latency remains a separate systems diagnostic, not evidence of practical efficiency."
            if gate["passed"]
            else "The controlled-code Phase 5 gate failed. The complete hop and systems records are retained, and the multi-hop claim is not accepted."
        ),
    }
    args.results.mkdir(parents=True, exist_ok=True)
    write_csv(args.results / "phase5" / "pointer_chasing.csv", rows)
    write_csv(args.results / "phase5" / "latency.csv", latency)
    (args.results / "phase5_metrics.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    (args.results / "phase5_multihop.md").write_text(render_report(record))
    print(json.dumps(record["gate"], indent=2, sort_keys=True))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
