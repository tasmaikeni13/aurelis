#!/usr/bin/env python3
"""Compare the CSM solve with simple memories under two fairness regimes."""

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
    estimated_flops_per_query,
    explicit_pair_state,
    explicit_pair_state_bytes,
    hebbian_read_many,
    hebbian_state,
    hebbian_state_bytes,
    least_squares_read_many,
    least_squares_state,
    linear_attention_read_many,
    linear_attention_state,
    linear_attention_state_bytes,
    make_keys,
    make_values,
    maximum_pairs_for_budget,
    softmax_read_many,
)
from csm.baselines import csm_state_bytes, evenly_spaced_subset, positive_feature
from csm.synthetic import dataset_seed


METHODS = ("csm", "hebbian", "softmax", "linear_attention", "least_squares")
FAIRNESS_REGIMES = ("same_dimension", "equal_state_budget")


def git_output(arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: Iterable[float], q: float) -> float:
    return float(np.quantile(np.asarray(list(values), dtype=np.float64), q))


def relative_frobenius(actual: Tensor, expected: Tensor) -> float:
    denominator = max(torch.linalg.vector_norm(expected).item(), 1e-300)
    return float(torch.linalg.vector_norm(actual - expected).item() / denominator)


def recall_metrics(
    actual: Tensor, expected: Tensor, exact_tolerance: float
) -> dict[str, float]:
    differences = actual - expected
    relative = torch.linalg.vector_norm(differences, dim=1) / torch.linalg.vector_norm(
        expected, dim=1
    ).clamp_min(1e-300)
    return {
        "relative_frobenius_error": relative_frobenius(actual, expected),
        "median_relative_error": float(relative.median().item()),
        "p90_relative_error": float(torch.quantile(relative, 0.9).item()),
        "maximum_relative_error": float(relative.max().item()),
        "exact_recall_rate": float((relative <= exact_tolerance).double().mean().item()),
    }


def best_softmax_read(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    targets: Tensor,
    temperatures: list[float],
) -> tuple[Tensor, Tensor, float, float]:
    """Return the globally best temperature for the provided evaluation set."""

    state = explicit_pair_state(keys, values)
    best: tuple[float, Tensor, Tensor, float] | None = None
    for temperature in temperatures:
        reads, weights = softmax_read_many(state, queries, temperature)
        error = relative_frobenius(reads, targets)
        if best is None or error < best[0]:
            best = (error, reads, weights, temperature)
    assert best is not None
    return best[1], best[2], best[3], best[0]


def append_recall_row(
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    method: str,
    reads: Tensor,
    values: Tensor,
    state_bytes: int,
    budget_bytes: int,
    retained: int,
    temperature: float,
    exact_tolerance: float,
) -> None:
    row = {
        **metadata,
        "method": method,
        "retained_associations": retained,
        "state_bytes": state_bytes,
        "budget_bytes": budget_bytes,
        "budget_fraction": state_bytes / budget_bytes,
        "estimated_flops_per_query": estimated_flops_per_query(
            method,
            metadata["d_key"],
            metadata["d_value"],
            retained,
        ),
        "selected_temperature": temperature,
    }
    row.update(recall_metrics(reads, values, exact_tolerance))
    rows.append(row)


def recall_sweep(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d_key in config["key_dimensions"]:
        for load in config["loads"]:
            count = max(1, int(round(d_key * load)))
            for d_value in config["value_dimensions"]:
                for seed in config["seeds"]:
                    for regime_index, key_regime in enumerate(config["key_regimes"]):
                        key_seed = dataset_seed(seed, d_key, count, regime_index)
                        value_seed = dataset_seed(
                            seed, d_key, count, regime_index, d_value
                        )
                        keys = make_keys(
                            key_regime,
                            count,
                            d_key,
                            generator=torch.Generator(device=device).manual_seed(
                                key_seed
                            ),
                            device=device,
                            correlation=config["correlation"],
                            near_collinear_noise=config["near_collinear_noise"],
                        )
                        values = make_values(
                            "gaussian",
                            count,
                            d_value,
                            generator=torch.Generator(device=device).manual_seed(
                                value_seed
                            ),
                            device=device,
                        )
                        budget = csm_state_bytes(d_key, d_value)

                        # Compressed baselines use every write and remain below budget.
                        hebbian = hebbian_read_many(hebbian_state(keys, values), keys)
                        linear = linear_attention_read_many(
                            linear_attention_state(keys, values), keys
                        )

                        fairness_outputs: dict[str, dict[str, Any]] = {}
                        for fairness in FAIRNESS_REGIMES:
                            if fairness == "same_dimension":
                                indices = torch.arange(count, device=device)
                            else:
                                capacity = maximum_pairs_for_budget(
                                    budget, d_key, d_value
                                )
                                indices = evenly_spaced_subset(
                                    count, min(count, capacity), device
                                )
                            retained_keys = keys[indices]
                            retained_values = values[indices]
                            softmax_reads, _, temperature, _ = best_softmax_read(
                                retained_keys,
                                retained_values,
                                keys,
                                values,
                                config["softmax_temperatures"],
                            )
                            oracle_reads = least_squares_read_many(
                                least_squares_state(retained_keys, retained_values),
                                keys,
                            )
                            fairness_outputs[fairness] = {
                                "indices": indices,
                                "softmax": softmax_reads,
                                "temperature": temperature,
                                "least_squares": oracle_reads,
                            }

                        for epsilon in config["epsilons"]:
                            memory = FP64GaussMarkovMemory(
                                d_key, d_value, epsilon=epsilon
                            )
                            state = memory.undiscounted_state(keys, values)
                            csm_reads = memory.read_many(state, keys)
                            for fairness in FAIRNESS_REGIMES:
                                common = {
                                    "experiment": "associative_recall",
                                    "fairness": fairness,
                                    "d_key": d_key,
                                    "d_value": d_value,
                                    "associations": count,
                                    "load": load,
                                    "seed": seed,
                                    "key_regime": key_regime,
                                    "epsilon": epsilon,
                                }
                                retained = fairness_outputs[fairness]["indices"].numel()
                                append_recall_row(
                                    rows,
                                    metadata=common,
                                    method="csm",
                                    reads=csm_reads,
                                    values=values,
                                    state_bytes=budget,
                                    budget_bytes=budget,
                                    retained=count,
                                    temperature=float("nan"),
                                    exact_tolerance=config["exact_relative_tolerance"],
                                )
                                append_recall_row(
                                    rows,
                                    metadata=common,
                                    method="hebbian",
                                    reads=hebbian,
                                    values=values,
                                    state_bytes=hebbian_state_bytes(d_key, d_value),
                                    budget_bytes=budget,
                                    retained=count,
                                    temperature=float("nan"),
                                    exact_tolerance=config["exact_relative_tolerance"],
                                )
                                append_recall_row(
                                    rows,
                                    metadata=common,
                                    method="softmax",
                                    reads=fairness_outputs[fairness]["softmax"],
                                    values=values,
                                    state_bytes=explicit_pair_state_bytes(
                                        retained, d_key, d_value
                                    ),
                                    budget_bytes=budget,
                                    retained=retained,
                                    temperature=fairness_outputs[fairness]["temperature"],
                                    exact_tolerance=config["exact_relative_tolerance"],
                                )
                                append_recall_row(
                                    rows,
                                    metadata=common,
                                    method="linear_attention",
                                    reads=linear,
                                    values=values,
                                    state_bytes=linear_attention_state_bytes(
                                        d_key, d_value
                                    ),
                                    budget_bytes=budget,
                                    retained=count,
                                    temperature=float("nan"),
                                    exact_tolerance=config["exact_relative_tolerance"],
                                )
                                append_recall_row(
                                    rows,
                                    metadata=common,
                                    method="least_squares",
                                    reads=fairness_outputs[fairness]["least_squares"],
                                    values=values,
                                    state_bytes=explicit_pair_state_bytes(
                                        retained, d_key, d_value
                                    ),
                                    budget_bytes=budget,
                                    retained=retained,
                                    temperature=float("nan"),
                                    exact_tolerance=config["exact_relative_tolerance"],
                                )
    return rows


def coefficient_patterns(dimension: int, device: torch.device) -> dict[str, Tensor]:
    base = torch.arange(1, dimension + 1, device=device, dtype=torch.float64)
    base /= base.sum()
    negative = torch.zeros(dimension, device=device, dtype=torch.float64)
    negative[:4] = torch.tensor([0.8, -0.6, 0.4, -0.2], device=device)
    greater_than_one = torch.zeros(dimension, device=device, dtype=torch.float64)
    greater_than_one[:2] = torch.tensor([1.5, 0.25], device=device)
    mixed = torch.zeros(dimension, device=device, dtype=torch.float64)
    mixed[:4] = torch.tensor([2.0, -1.5, 0.75, 0.25], device=device)
    return {
        "positive_simplex": base,
        "negative_coefficients": negative,
        "coefficient_gt_one": greater_than_one,
        "positive_sum_two": 2.0 * base,
        "mixed_nonunit": mixed,
    }


def simplex_projection(vector: Tensor) -> Tensor:
    """Euclidean projection onto ``{x >= 0, sum x = 1}``."""

    sorted_values = torch.sort(vector, descending=True).values
    cumulative = torch.cumsum(sorted_values, dim=0) - 1.0
    indices = torch.arange(
        1, vector.numel() + 1, device=vector.device, dtype=vector.dtype
    )
    active = sorted_values - cumulative / indices > 0
    rho = int(torch.nonzero(active, as_tuple=False)[-1].item())
    threshold = cumulative[rho] / (rho + 1)
    return torch.clamp(vector - threshold, min=0.0)


def functional_sweep(
    config: dict[str, Any], device: torch.device
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    regimes = config["functional_key_regimes"]
    for dimension in config["functional_dimensions"]:
        values = torch.eye(dimension, device=device, dtype=torch.float64)
        patterns = coefficient_patterns(dimension, device)
        budget = csm_state_bytes(dimension, dimension)
        explicit_bytes = explicit_pair_state_bytes(
            dimension, dimension, dimension
        )
        assert explicit_bytes == budget
        for seed in config["seeds"]:
            for regime_index, key_regime in enumerate(regimes):
                keys = make_keys(
                    key_regime,
                    dimension,
                    dimension,
                    generator=torch.Generator(device=device).manual_seed(
                        dataset_seed(seed, dimension, dimension, regime_index)
                    ),
                    device=device,
                    correlation=config["correlation"],
                    near_collinear_noise=config["near_collinear_noise"],
                )
                query_matrix = torch.stack(
                    [coefficients @ keys for coefficients in patterns.values()]
                )
                targets = torch.stack(list(patterns.values()))
                memory = FP64GaussMarkovMemory(
                    dimension,
                    dimension,
                    epsilon=config["functional_epsilon"],
                )
                csm_reads = memory.read_many(
                    memory.undiscounted_state(keys, values), query_matrix
                )
                hebbian_reads = hebbian_read_many(
                    hebbian_state(keys, values), query_matrix
                )
                linear_reads = linear_attention_read_many(
                    linear_attention_state(keys, values), query_matrix
                )
                oracle_reads = least_squares_read_many(
                    least_squares_state(keys, values), query_matrix
                )

                for pattern_index, (kind, coefficients) in enumerate(patterns.items()):
                    query = query_matrix[pattern_index : pattern_index + 1]
                    target = targets[pattern_index]
                    softmax_reads, weights, temperature, _ = best_softmax_read(
                        keys,
                        values,
                        query,
                        target.unsqueeze(0),
                        config["softmax_temperatures"],
                    )
                    projection = simplex_projection(target)
                    hull_distance = float(
                        torch.linalg.vector_norm(target - projection).item()
                    )
                    outputs = {
                        "csm": csm_reads[pattern_index],
                        "hebbian": hebbian_reads[pattern_index],
                        "softmax": softmax_reads[0],
                        "linear_attention": linear_reads[pattern_index],
                        "least_squares": oracle_reads[pattern_index],
                    }
                    bytes_by_method = {
                        "csm": budget,
                        "hebbian": hebbian_state_bytes(dimension, dimension),
                        "softmax": explicit_bytes,
                        "linear_attention": linear_attention_state_bytes(
                            dimension, dimension
                        ),
                        "least_squares": explicit_bytes,
                    }
                    for fairness in FAIRNESS_REGIMES:
                        for method, output in outputs.items():
                            absolute_error = float(
                                torch.linalg.vector_norm(output - target).item()
                            )
                            rows.append(
                                {
                                    "experiment": "linear_functional",
                                    "fairness": fairness,
                                    "d_key": dimension,
                                    "d_value": dimension,
                                    "associations": dimension,
                                    "seed": seed,
                                    "key_regime": key_regime,
                                    "coefficient_kind": kind,
                                    "coefficient_sum": float(coefficients.sum().item()),
                                    "coefficient_minimum": float(coefficients.min().item()),
                                    "coefficient_maximum": float(coefficients.max().item()),
                                    "method": method,
                                    "state_bytes": bytes_by_method[method],
                                    "budget_bytes": budget,
                                    "budget_fraction": bytes_by_method[method] / budget,
                                    "estimated_flops_per_query": estimated_flops_per_query(
                                        method, dimension, dimension, dimension
                                    ),
                                    "selected_temperature": (
                                        temperature if method == "softmax" else float("nan")
                                    ),
                                    "absolute_error": absolute_error,
                                    "relative_error": absolute_error
                                    / max(torch.linalg.vector_norm(target).item(), 1e-300),
                                    "convex_hull_distance": hull_distance,
                                    "error_minus_hull_distance": (
                                        absolute_error - hull_distance
                                        if method == "softmax"
                                        else float("nan")
                                    ),
                                    "minimum_softmax_weight": (
                                        float(weights.min().item())
                                        if method == "softmax"
                                        else float("nan")
                                    ),
                                    "softmax_weight_sum_error": (
                                        abs(float(weights.sum().item()) - 1.0)
                                        if method == "softmax"
                                        else float("nan")
                                    ),
                                }
                            )
    return rows


def benchmark(
    operation: Callable[[], Tensor],
    device: torch.device,
    warmups: int,
    repetitions: int,
    query_count: int,
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
    return float(np.median(timings) / query_count / 1_000.0)


def latency_sweep(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    d_value = config["latency_value_dimension"]
    query_count = config["latency_queries"]
    for dimension in config["latency_dimensions"]:
        generator = torch.Generator(device=device).manual_seed(73 + dimension)
        keys = torch.randn(
            dimension,
            dimension,
            generator=generator,
            device=device,
            dtype=torch.float64,
        )
        keys /= torch.linalg.vector_norm(keys, dim=1, keepdim=True)
        values = torch.randn(
            dimension,
            d_value,
            generator=generator,
            device=device,
            dtype=torch.float64,
        )
        queries = torch.randn(
            query_count,
            dimension,
            generator=generator,
            device=device,
            dtype=torch.float64,
        )
        queries /= torch.linalg.vector_norm(queries, dim=1, keepdim=True)
        memory = FP64GaussMarkovMemory(dimension, d_value, epsilon=1e-8)
        csm_state = memory.undiscounted_state(keys, values)
        factor = torch.linalg.cholesky(memory.system_matrix(csm_state))
        hebb = hebbian_state(keys, values)
        explicit = explicit_pair_state(keys, values)
        linear = linear_attention_state(keys, values)
        oracle = least_squares_state(keys, values)

        operations: dict[str, Callable[[], Tensor]] = {
            "csm": lambda: torch.cholesky_solve(queries.mT, factor).mT
            @ csm_state.C.mT,
            "hebbian": lambda: hebbian_read_many(hebb, queries),
            "softmax": lambda: softmax_read_many(explicit, queries, 1e-4)[0],
            "linear_attention": lambda: linear_attention_read_many(linear, queries),
            "least_squares": lambda: least_squares_read_many(oracle, queries),
        }
        state_bytes = {
            "csm": csm_state_bytes(dimension, d_value),
            "hebbian": hebbian_state_bytes(dimension, d_value),
            "softmax": explicit_pair_state_bytes(dimension, dimension, d_value),
            "linear_attention": linear_attention_state_bytes(dimension, d_value),
            "least_squares": explicit_pair_state_bytes(
                dimension, dimension, d_value
            ),
        }
        for method, operation in operations.items():
            rows.append(
                {
                    "d_key": dimension,
                    "d_value": d_value,
                    "associations": dimension,
                    "method": method,
                    "state_bytes": state_bytes[method],
                    "estimated_flops_per_query": estimated_flops_per_query(
                        method, dimension, d_value, dimension
                    ),
                    "median_microseconds_per_query": benchmark(
                        operation,
                        device,
                        config["latency_warmups"],
                        config["latency_repetitions"],
                        query_count,
                    ),
                }
            )
    return rows


def aggregate_recall(rows: list[dict[str, Any]], epsilon: float) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["epsilon"] != epsilon:
            continue
        domain = "under_or_at_capacity" if row["load"] <= 1.0 else "over_capacity"
        groups[
            (row["fairness"], row["key_regime"], domain, row["method"])
        ].append(row["relative_frobenius_error"])
    return [
        {
            "fairness": key[0],
            "key_regime": key[1],
            "capacity_domain": key[2],
            "method": key[3],
            "median_relative_error": percentile(values, 0.5),
            "p90_relative_error": percentile(values, 0.9),
            "case_count": len(values),
        }
        for key, values in sorted(groups.items())
    ]


def aggregate_functional(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["fairness"] == "equal_state_budget":
            groups[(row["coefficient_kind"], row["method"])].append(row)
    return [
        {
            "coefficient_kind": key[0],
            "method": key[1],
            "median_absolute_error": percentile(
                [row["absolute_error"] for row in values], 0.5
            ),
            "p90_absolute_error": percentile(
                [row["absolute_error"] for row in values], 0.9
            ),
            "median_convex_hull_distance": percentile(
                [row["convex_hull_distance"] for row in values], 0.5
            ),
            "case_count": len(values),
        }
        for key, values in sorted(groups.items())
    ]


def aggregate_value_dimension(
    rows: list[dict[str, Any]], epsilon: float
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["fairness"] == "equal_state_budget"
            and row["key_regime"] == "correlated"
            and row["load"] <= 1.0
            and row["epsilon"] == epsilon
        ):
            groups[(row["d_value"], row["method"])].append(
                row["relative_frobenius_error"]
            )
    return [
        {
            "d_value": key[0],
            "method": key[1],
            "median_relative_error": percentile(values, 0.5),
            "p90_relative_error": percentile(values, 0.9),
            "case_count": len(values),
        }
        for key, values in sorted(groups.items())
    ]


def gate_summary(
    recall: list[dict[str, Any]],
    functional: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    minimum_epsilon = min(config["epsilons"])
    correlated: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in recall:
        if (
            row["fairness"] == "equal_state_budget"
            and row["key_regime"] == "correlated"
            and row["load"] <= 1.0
            and row["epsilon"] == minimum_epsilon
            and row["method"] in ("csm", "hebbian", "linear_attention")
        ):
            case = (
                row["d_key"],
                row["d_value"],
                row["associations"],
                row["seed"],
            )
            correlated[case][row["method"]] = row["relative_frobenius_error"]
    compressed_ratios = [
        min(values["hebbian"], values["linear_attention"])
        / max(values["csm"], 1e-300)
        for values in correlated.values()
    ]

    nonconvex = [
        row
        for row in functional
        if row["fairness"] == "equal_state_budget"
        and row["coefficient_kind"] != "positive_simplex"
    ]
    by_case: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in nonconvex:
        case = (
            row["d_key"],
            row["seed"],
            row["key_regime"],
            row["coefficient_kind"],
        )
        by_case[case][row["method"]] = row["absolute_error"]
    functional_ratios = [
        values["softmax"] / max(values["csm"], 1e-300)
        for values in by_case.values()
    ]
    csm_functional = [
        row["absolute_error"] for row in nonconvex if row["method"] == "csm"
    ]
    softmax_rows = [
        row
        for row in functional
        if row["fairness"] == "equal_state_budget" and row["method"] == "softmax"
    ]
    hull_margin = min(row["error_minus_hull_distance"] for row in softmax_rows)
    minimum_weight = min(row["minimum_softmax_weight"] for row in softmax_rows)
    maximum_sum_error = max(row["softmax_weight_sum_error"] for row in softmax_rows)
    observed = {
        "compressed_correlated_to_csm_median_error_ratio": percentile(
            compressed_ratios, 0.5
        ),
        "nonconvex_functional_csm_median_absolute_error": percentile(
            csm_functional, 0.5
        ),
        "nonconvex_softmax_to_csm_median_error_ratio": percentile(
            functional_ratios, 0.5
        ),
        "minimum_softmax_error_minus_hull_distance": hull_margin,
        "minimum_softmax_weight": minimum_weight,
        "maximum_softmax_weight_sum_error": maximum_sum_error,
        "correlated_case_count": len(compressed_ratios),
        "nonconvex_functional_case_count": len(functional_ratios),
    }
    thresholds = config["gate"]
    checks = {
        "compressed_correlated_advantage": observed[
            "compressed_correlated_to_csm_median_error_ratio"
        ]
        >= thresholds["compressed_correlated_to_csm_error_ratio"],
        "equal_memory_nonconvex_functional_fidelity": observed[
            "nonconvex_functional_csm_median_absolute_error"
        ]
        <= thresholds["nonconvex_functional_csm_median_error"],
        "equal_memory_softmax_separation": observed[
            "nonconvex_softmax_to_csm_median_error_ratio"
        ]
        >= thresholds["nonconvex_softmax_to_csm_error_ratio"],
        "softmax_convex_hull_constraint": (
            hull_margin >= -thresholds["convex_hull_numerical_tolerance"]
            and minimum_weight >= -thresholds["convex_hull_numerical_tolerance"]
            and maximum_sum_error <= thresholds["convex_hull_numerical_tolerance"]
        ),
    }
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
    recall: list[dict[str, Any]],
    functional: list[dict[str, Any]],
    latency: list[dict[str, Any]],
    config: dict[str, Any],
    directory: Path,
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    minimum_epsilon = min(config["epsilons"])
    colors = dict(zip(METHODS, plt.cm.tab10.colors))

    for fairness in FAIRNESS_REGIMES:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
        for axis, regime in zip(axes, config["key_regimes"]):
            for method in METHODS:
                medians = []
                for load in config["loads"]:
                    values = [
                        row["relative_frobenius_error"]
                        for row in recall
                        if row["fairness"] == fairness
                        and row["key_regime"] == regime
                        and row["method"] == method
                        and row["epsilon"] == minimum_epsilon
                        and row["load"] == load
                    ]
                    medians.append(percentile(values, 0.5))
                axis.semilogy(
                    config["loads"],
                    np.maximum(medians, 1e-16),
                    marker="o",
                    label=method,
                    color=colors[method],
                )
            axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
            axis.set_title(regime)
            axis.set_xlabel("load K / d_key")
            axis.grid(True, which="both", alpha=0.25)
        axes[0].set_ylabel("median relative recall error")
        axes[-1].legend(fontsize=7)
        fig.suptitle(f"Associative recall: {fairness.replace('_', ' ')}")
        fig.tight_layout()
        path = directory / f"recall_vs_load_{fairness}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for regime in config["key_regimes"]:
        medians = []
        for epsilon in config["epsilons"]:
            values = [
                row["relative_frobenius_error"]
                for row in recall
                if row["fairness"] == "equal_state_budget"
                and row["method"] == "csm"
                and row["key_regime"] == regime
                and row["load"] <= 1.0
                and row["epsilon"] == epsilon
            ]
            medians.append(percentile(values, 0.5))
        axis.loglog(config["epsilons"], medians, marker="o", label=regime)
    axis.set(
        xlabel="epsilon",
        ylabel="median relative recall error",
        title="CSM epsilon sweep at or below capacity",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = directory / "csm_epsilon_sweep.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    kinds = list(coefficient_patterns(16, torch.device("cpu")))
    x = np.arange(len(kinds))
    width = 0.15
    for method_index, method in enumerate(METHODS):
        medians = []
        for kind in kinds:
            values = [
                row["absolute_error"]
                for row in functional
                if row["fairness"] == "equal_state_budget"
                and row["method"] == method
                and row["coefficient_kind"] == kind
            ]
            medians.append(max(percentile(values, 0.5), 1e-16))
        axis.bar(
            x + (method_index - 2) * width,
            medians,
            width,
            label=method,
            color=colors[method],
        )
    axis.set_yscale("log")
    axis.set_xticks(x)
    axis.set_xticklabels(kinds, rotation=25, ha="right")
    axis.set(
        ylabel="median absolute target error",
        title="Equal-memory linear-functional separation",
    )
    axis.grid(True, which="both", axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "linear_functional_separation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for method in METHODS:
        subset = [row for row in latency if row["method"] == method]
        axis.plot(
            [row["d_key"] for row in subset],
            [row["median_microseconds_per_query"] for row in subset],
            marker="o",
            label=method,
            color=colors[method],
        )
    axis.set(
        xlabel="key dimension",
        ylabel="median microseconds / query",
        title=f"Prepared batched reads ({config['latency_queries']} queries)",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "prepared_read_latency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    separator = ["---"] + ["---:" for _ in headers[1:]]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(separator) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def render_report(record: dict[str, Any]) -> str:
    status = "PASS" if record["gate"]["passed"] else "FAIL"
    checks = "\n".join(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in record["gate"]["checks"].items()
    )
    observed = record["observed"]
    recall_summary = record["recall_summary"]
    functional_summary = record["functional_summary"]
    value_dimension_summary = record["value_dimension_summary"]
    selected_recall = [
        row
        for row in recall_summary
        if row["fairness"] == "equal_state_budget"
        and row["capacity_domain"] == "under_or_at_capacity"
    ]
    recall_table = markdown_table(
        ["key regime", "method", "median error", "p90 error"],
        [
            [
                row["key_regime"],
                row["method"],
                f"{row['median_relative_error']:.3e}",
                f"{row['p90_relative_error']:.3e}",
            ]
            for row in selected_recall
        ],
    )
    functional_table = markdown_table(
        ["coefficients", "method", "median absolute error", "hull distance"],
        [
            [
                row["coefficient_kind"],
                row["method"],
                f"{row['median_absolute_error']:.3e}",
                f"{row['median_convex_hull_distance']:.3e}",
            ]
            for row in functional_summary
            if row["method"] in ("csm", "softmax", "hebbian", "linear_attention")
        ],
    )
    value_dimension_table = markdown_table(
        ["d_value", "method", "median error", "p90 error"],
        [
            [
                str(row["d_value"]),
                row["method"],
                f"{row['median_relative_error']:.3e}",
                f"{row['p90_relative_error']:.3e}",
            ]
            for row in value_dimension_summary
        ],
    )
    representative = [
        row
        for row in record["latency"]
        if row["d_key"] == max(record["config"]["latency_dimensions"])
    ]
    resource_table = markdown_table(
        ["method", "state bytes", "estimated FLOPs/query", "measured us/query"],
        [
            [
                row["method"],
                str(row["state_bytes"]),
                str(row["estimated_flops_per_query"]),
                f"{row['median_microseconds_per_query']:.3f}",
            ]
            for row in representative
        ],
    )
    plots = "\n".join(
        f"- [`{Path(path).name}`](../{path})" for path in record["plots"]
    )
    random_no_win = {
        row["method"]: row["median_relative_error"]
        for row in selected_recall
        if row["key_regime"] == "random_gaussian"
    }
    return f"""# Phase 3 baseline separation

## Gate decision: {status}

{checks}

The same-dimension regime gives every method the same `d_key` and `d_value`; explicit methods may therefore use state growing with `K`. The equal-state-budget regime caps explicit key/value storage at the byte count of dense fp64 CSM `S` and `C`. Hebbian and linear attention use every write but consume less than that maximum. In the linear-functional experiment `K = d_key = d_value`, so CSM and explicit softmax each use exactly `16 * d_key^2` bytes.

| Gate measurement | Observed | Required |
|---|---:|---:|
| correlated compressed-baseline / CSM median error ratio | {observed['compressed_correlated_to_csm_median_error_ratio']:.3e} | >= {record['gate']['thresholds']['compressed_correlated_to_csm_error_ratio']:.1f} |
| equal-memory nonconvex CSM median absolute error | {observed['nonconvex_functional_csm_median_absolute_error']:.3e} | <= {record['gate']['thresholds']['nonconvex_functional_csm_median_error']:.1e} |
| equal-memory softmax / CSM nonconvex median error ratio | {observed['nonconvex_softmax_to_csm_median_error_ratio']:.3e} | >= {record['gate']['thresholds']['nonconvex_softmax_to_csm_error_ratio']:.1e} |
| minimum softmax error minus convex-hull distance | {observed['minimum_softmax_error_minus_hull_distance']:.3e} | >= -{record['gate']['thresholds']['convex_hull_numerical_tolerance']:.1e} |
| minimum softmax weight | {observed['minimum_softmax_weight']:.3e} | >= -{record['gate']['thresholds']['convex_hull_numerical_tolerance']:.1e} |
| maximum softmax weight-sum error | {observed['maximum_softmax_weight_sum_error']:.3e} | <= {record['gate']['thresholds']['convex_hull_numerical_tolerance']:.1e} |

## Associative recall

The table aggregates the minimum-epsilon, at-or-below-capacity cases under the equal byte budget. Softmax receives oracle selection over all committed temperatures for each dataset; this is intentionally favorable to it. Least squares is an explicit-pair oracle, not a compressed practical winner.

{recall_table}

The solver has a reproducible fidelity advantage over the compressed Hebbian and positive-feature linear-attention states for correlated keys. It does **not** dominate explicit retrieval: for random independent stored-key recall, median errors are CSM `{random_no_win['csm']:.3e}`, oracle-tuned softmax `{random_no_win['softmax']:.3e}`, and least squares `{random_no_win['least_squares']:.3e}`. At `K <= d_key`, explicit softmax fits the equal budget and is often the better stored-key mechanism. Above capacity, arbitrary-value recall fails for CSM and the least-squares projection, while budgeted explicit methods also omit pairs; no broad winner is claimed there.

### Value-dimension sweep

The following isolates correlated keys at or below capacity, minimum epsilon, and the equal state-byte budget. All three committed value dimensions are reported rather than averaged away.

{value_dimension_table}

## Linear-functional separation

Values are the standard basis, so the target is exactly `alpha`. Every normalized softmax output is therefore a simplex point. Negative coefficients, coefficients above one, and sums other than one put the target outside that convex hull; their Euclidean simplex-projection distance is a method-independent lower bound on softmax error. CSM and least squares can instead produce signed linear-span coefficients.

{functional_table}

The positive-simplex case is included as a no-structural-separation control: its hull distance is zero, so the convexity argument alone predicts no CSM advantage. The equal-memory claim is limited to the characterized nonconvex coefficient classes.

## State, operations, and latency

The following is the `d_key=64`, `d_value=16`, `K=64` prepared-read measurement. CSM timing uses a precomputed Cholesky factor, just as least squares uses a precomputed pseudoinverse; neither derived factor is counted as recurrent state. FLOPs are leading-operation estimates, not profiler counts. GPU timings use batches of {record['config']['latency_queries']} queries and are latency diagnostics rather than optimized-kernel claims.

{resource_table}

## Plots

{plots}

## Reproducibility

- git source checkpoint: `{record['git_commit']}`
- working tree dirty at experiment start: `{record['working_tree_dirty']}`
- config: [`configs/phase3_baselines.json`](../configs/phase3_baselines.json)
- seeds: `{record['config']['seeds']}`; deterministic regime-specific mixing is in source
- hardware: `{record['hardware']}`
- software: Python `{record['software']['python']}`, PyTorch `{record['software']['torch']}`, HIP `{record['software']['hip']}`, NumPy `{record['software']['numpy']}`
- wall-clock time: `{record['wall_clock_seconds']:.3f}` seconds
- peak allocated VRAM: `{record['peak_vram_bytes'] / 2**30:.6f}` GiB
- complete recall rows: [`phase3/associative_recall.csv`](phase3/associative_recall.csv)
- complete linear-functional rows: [`phase3/linear_functional.csv`](phase3/linear_functional.csv)
- latency rows: [`phase3/latency.csv`](phase3/latency.csv)
- machine-readable summary: [`phase3_metrics.json`](phase3_metrics.json)

## Scoped conclusion

{record['interpretation']}

This is a synthetic, unlearned comparison. It does not establish an NLP, throughput, learned-representation, or universal memory advantage.
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase3_baselines.json")
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--plots", type=Path, default=Path("plots/phase3"))
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    source_dirty = bool(git_output(["status", "--porcelain"]))
    config = json.loads(args.config.read_text())
    device = select_device(config["device"])
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    recall = recall_sweep(config, device)
    functional = functional_sweep(config, device)
    latency = latency_sweep(config, device)
    recall_summary = aggregate_recall(recall, min(config["epsilons"]))
    functional_summary = aggregate_functional(functional)
    value_dimension_summary = aggregate_value_dimension(
        recall, min(config["epsilons"])
    )
    gate, observed = gate_summary(recall, functional, config)
    plots = make_plots(recall, functional, latency, config, args.plots)
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
        "row_counts": {
            "recall": len(recall),
            "functional": len(functional),
            "latency": len(latency),
        },
        "recall_summary": recall_summary,
        "functional_summary": functional_summary,
        "value_dimension_summary": value_dimension_summary,
        "latency": latency,
        "observed": observed,
        "plots": plots,
        "gate": gate,
        "interpretation": (
            "A solver fidelity advantage survives the equal-state-byte comparison in the stated correlated compressed-memory and nonconvex linear-functional regimes. Explicit softmax and least squares match or beat CSM in important stored-key regimes, so the result is a separation, not universal superiority."
            if gate["passed"]
            else "The Phase 3 separation gate failed. Any solver-advantage claim must be rejected or narrowed; the complete negative record is retained."
        ),
    }
    args.results.mkdir(parents=True, exist_ok=True)
    write_csv(args.results / "phase3" / "associative_recall.csv", recall)
    write_csv(args.results / "phase3" / "linear_functional.csv", functional)
    write_csv(args.results / "phase3" / "latency.csv", latency)
    (args.results / "phase3_metrics.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    report = render_report(record)
    (args.results / "phase3_baseline_separation.md").write_text(report)
    print(report)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
