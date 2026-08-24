#!/usr/bin/env python3
"""Phase 4: noisy evidence, precision weighting, and uncertainty calibration."""

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
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from csm import (
    FP64GaussMarkovMemory,
    explicit_pair_state,
    hebbian_read_many,
    hebbian_state,
    softmax_read_many,
)


METHODS = ("csm", "simple_average", "hebbian", "softmax", "oracle_ridge")


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


def standard_noise(
    model: str,
    shape: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    """Unit-variance noise, including two deliberately non-Gaussian cases."""

    if model in ("linear_gaussian", "nonlinear"):
        return torch.randn(
            *shape, generator=generator, device=device, dtype=torch.float64
        )
    if model == "laplace":
        uniform = torch.rand(
            *shape, generator=generator, device=device, dtype=torch.float64
        ) - 0.5
        return (
            -torch.sign(uniform)
            * torch.log1p(-2.0 * torch.abs(uniform))
            / math.sqrt(2.0)
        )
    if model == "student_like":
        # A standardized Gaussian scale mixture: finite variance, far heavier tails.
        base = torch.randn(
            *shape, generator=generator, device=device, dtype=torch.float64
        )
        mixture = torch.rand(
            *shape, generator=generator, device=device, dtype=torch.float64
        ) < 0.05
        scales = torch.where(mixture, 5.0, 1.0)
        return base * scales / math.sqrt(0.95 + 0.05 * 25.0)
    raise ValueError(f"unknown data model: {model}")


def latent_values(
    model: str,
    inputs: Tensor,
    linear: Tensor,
    nonlinear: Tensor,
    nonlinear_scale: float,
) -> Tensor:
    values = inputs @ linear.mT
    if model == "nonlinear":
        values = values + nonlinear_scale * torch.sin(2.0 * (inputs @ nonlinear.mT))
    return values


def best_softmax(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    targets: Tensor,
    temperatures: list[float],
) -> tuple[Tensor, float]:
    state = explicit_pair_state(keys, values)
    best_error = math.inf
    best_reads: Tensor | None = None
    best_temperature = math.nan
    for temperature in temperatures:
        reads, _ = softmax_read_many(state, queries, temperature)
        error = torch.mean(torch.sum((reads - targets) ** 2, dim=1)).item()
        if error < best_error:
            best_error = error
            best_reads = reads
            best_temperature = temperature
    assert best_reads is not None
    return best_reads, best_temperature


def duplicate_sweep(
    config: dict[str, Any], device: torch.device
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    d_key = config["d_key"]
    d_value = config["d_value"]
    unique = config["unique_associations"]
    epsilon = config["duplicate_epsilon"]
    for model_index, model in enumerate(config["data_models"]):
        for sigma in config["noise_sigmas"]:
            for repeats in config["repeats"]:
                for seed in config["seeds"]:
                    generator = torch.Generator(device=device).manual_seed(
                        10_000_019 * seed + 10_007 * model_index + 101 * repeats
                    )
                    keys = torch.linalg.qr(
                        torch.randn(
                            d_key,
                            unique,
                            generator=generator,
                            device=device,
                            dtype=torch.float64,
                        ),
                        mode="reduced",
                    ).Q.mT
                    linear = torch.randn(
                        d_value,
                        d_key,
                        generator=generator,
                        device=device,
                        dtype=torch.float64,
                    ) / math.sqrt(d_key)
                    nonlinear = torch.randn(
                        d_value,
                        d_key,
                        generator=generator,
                        device=device,
                        dtype=torch.float64,
                    ) / math.sqrt(d_key)
                    targets = latent_values(
                        model,
                        keys,
                        linear,
                        nonlinear,
                        config["nonlinear_scale"],
                    )
                    repeated_keys = keys.repeat_interleave(repeats, dim=0)
                    repeated_targets = targets.repeat_interleave(repeats, dim=0)
                    values = repeated_targets + sigma * standard_noise(
                        model,
                        tuple(repeated_targets.shape),
                        generator,
                        device,
                    )
                    beta = torch.full(
                        (values.shape[0],),
                        sigma**-2,
                        device=device,
                        dtype=torch.float64,
                    )
                    memory = FP64GaussMarkovMemory(
                        d_key, d_value, epsilon=epsilon
                    )
                    state = memory.undiscounted_state(repeated_keys, values, beta)
                    csm = memory.read_many(state, keys)
                    identity = torch.eye(
                        d_key, device=device, dtype=torch.float64
                    )
                    oracle_weighted_keys = beta.unsqueeze(1) * repeated_keys
                    oracle_system = (
                        repeated_keys.mT @ oracle_weighted_keys
                        + epsilon * identity
                    )
                    oracle_cross = values.mT @ oracle_weighted_keys
                    oracle = (
                        torch.linalg.solve(oracle_system, oracle_cross.mT).mT
                        @ keys.mT
                    ).mT
                    average = values.reshape(unique, repeats, d_value).mean(dim=1)
                    hebbian = hebbian_read_many(
                        hebbian_state(repeated_keys, values), keys
                    )
                    softmax, selected_temperature = best_softmax(
                        repeated_keys,
                        values,
                        keys,
                        targets,
                        config["softmax_temperatures"],
                    )
                    outputs = {
                        "csm": csm,
                        "simple_average": average,
                        "hebbian": hebbian,
                        "softmax": softmax,
                        "oracle_ridge": oracle,
                    }
                    oracle_difference = (
                        torch.linalg.vector_norm(csm - oracle)
                        / torch.linalg.vector_norm(oracle).clamp_min(1e-300)
                    ).item()
                    for method, predictions in outputs.items():
                        per_query = torch.sum((predictions - targets) ** 2, dim=1)
                        rows.append(
                            {
                                "data_model": model,
                                "sigma": sigma,
                                "repeats": repeats,
                                "seed": seed,
                                "method": method,
                                "mean_squared_error": per_query.mean().item(),
                                "p90_squared_error": torch.quantile(
                                    per_query, 0.9
                                ).item(),
                                "selected_temperature": (
                                    selected_temperature
                                    if method == "softmax"
                                    else math.nan
                                ),
                                "csm_oracle_relative_difference": (
                                    oracle_difference
                                    if method == "csm"
                                    else math.nan
                                ),
                                "gaussian_unregularized_expected_risk": (
                                    sigma**2 * d_value / repeats
                                    if model == "linear_gaussian"
                                    else math.nan
                                ),
                            }
                        )
    return rows


def precision_sweep(
    config: dict[str, Any], device: torch.device
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    d_key = config["d_key"]
    d_value = config["d_value"]
    epsilon = config["duplicate_epsilon"]
    key = torch.zeros(d_key, device=device, dtype=torch.float64)
    key[0] = 1.0
    for model_index, model in enumerate(
        ("linear_gaussian", "laplace", "student_like")
    ):
        for pattern_index, pattern in enumerate(config["precision_patterns"]):
            sigmas = torch.tensor(pattern, device=device, dtype=torch.float64)
            beta = sigmas.pow(-2)
            for seed in config["seeds"]:
                generator = torch.Generator(device=device).manual_seed(
                    90_000_011 + seed * 10_003 + pattern_index * 101 + model_index
                )
                target = torch.randn(
                    d_value,
                    generator=generator,
                    device=device,
                    dtype=torch.float64,
                )
                values = target + sigmas.unsqueeze(1) * standard_noise(
                    model, (len(pattern), d_value), generator, device
                )
                keys = key.repeat(len(pattern), 1)
                weighted_memory = FP64GaussMarkovMemory(
                    d_key, d_value, epsilon=epsilon
                )
                weighted = weighted_memory.read(
                    weighted_memory.undiscounted_state(keys, values, beta), key
                )
                uniform = weighted_memory.read(
                    weighted_memory.undiscounted_state(keys, values), key
                )
                analytic = (beta.unsqueeze(1) * values).sum(0) / beta.sum()
                outputs = {
                    "precision_weighted_csm": weighted,
                    "uniform_csm": uniform,
                    "analytic_precision_mean": analytic,
                }
                for method, prediction in outputs.items():
                    rows.append(
                        {
                            "data_model": model,
                            "pattern": pattern_index,
                            "sigmas": ";".join(str(value) for value in pattern),
                            "seed": seed,
                            "method": method,
                            "squared_error": torch.sum(
                                (prediction - target) ** 2
                            ).item(),
                            "weighted_to_analytic_error": (
                                torch.linalg.vector_norm(weighted - analytic).item()
                                if method == "precision_weighted_csm"
                                else math.nan
                            ),
                            "gaussian_expected_weighted_risk": (
                                d_value / beta.sum().item()
                                if model == "linear_gaussian"
                                else math.nan
                            ),
                            "gaussian_expected_uniform_risk": (
                                d_value * sigmas.square().sum().item()
                                / len(pattern) ** 2
                                if model == "linear_gaussian"
                                else math.nan
                            ),
                        }
                    )

    # An explicit contradictory pair verifies that beta follows declared precision.
    target = torch.linspace(-0.3, 0.3, d_value, device=device, dtype=torch.float64)
    offset = torch.ones(d_value, device=device, dtype=torch.float64)
    values = torch.stack((target + offset, target - offset))
    keys = key.repeat(2, 1)
    beta = torch.tensor([25.0, 1.0], device=device, dtype=torch.float64)
    memory = FP64GaussMarkovMemory(d_key, d_value, epsilon=epsilon)
    weighted = memory.read(memory.undiscounted_state(keys, values, beta), key)
    uniform = memory.read(memory.undiscounted_state(keys, values), key)
    analytic = (beta.unsqueeze(1) * values).sum(0) / beta.sum()
    for method, prediction in {
        "precision_weighted_csm": weighted,
        "uniform_csm": uniform,
        "analytic_precision_consensus": analytic,
    }.items():
        conflicts.append(
            {
                "method": method,
                "distance_to_true": torch.linalg.vector_norm(
                    prediction - target
                ).item(),
                "distance_to_precision_consensus": torch.linalg.vector_norm(
                    prediction - analytic
                ).item(),
                "first_observation_precision": beta[0].item(),
                "second_observation_precision": beta[1].item(),
            }
        )
    return rows, conflicts


def confidence_sweep(
    config: dict[str, Any], device: torch.device
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    d_key = config["d_key"]
    d_value = config["d_value"]
    rank = config["observed_rank"]
    epsilon = config["epsilon"]
    sigma = config["confidence_noise_sigma"]
    repeats = 2
    for model_index, model in enumerate(config["data_models"]):
        for trial in range(config["confidence_trials"]):
            generator = torch.Generator(device=device).manual_seed(
                700_000_003 + model_index * 1_000_003 + trial * 10_007
            )
            basis = torch.linalg.qr(
                torch.randn(
                    d_key,
                    d_key,
                    generator=generator,
                    device=device,
                    dtype=torch.float64,
                )
            ).Q
            in_basis = basis[:, :rank].mT
            out_basis = basis[:, rank:].mT
            linear = torch.randn(
                d_value,
                d_key,
                generator=generator,
                device=device,
                dtype=torch.float64,
            ) / math.sqrt(epsilon)
            nonlinear = torch.randn(
                d_value,
                d_key,
                generator=generator,
                device=device,
                dtype=torch.float64,
            ) / math.sqrt(d_key)
            training_keys = in_basis.repeat_interleave(repeats, dim=0)
            training_targets = latent_values(
                model,
                training_keys,
                linear,
                nonlinear,
                config["nonlinear_scale"],
            )
            training_values = training_targets + sigma * standard_noise(
                model, tuple(training_targets.shape), generator, device
            )
            beta = torch.full(
                (training_keys.shape[0],),
                sigma**-2,
                device=device,
                dtype=torch.float64,
            )
            memory = FP64GaussMarkovMemory(d_key, d_value, epsilon=epsilon)
            state = memory.undiscounted_state(training_keys, training_values, beta)
            query_blocks = []
            distances = []
            count = config["queries_per_distance"]
            for distance in config["ood_distances"]:
                in_coefficients = torch.randn(
                    count,
                    rank,
                    generator=generator,
                    device=device,
                    dtype=torch.float64,
                )
                in_vectors = in_coefficients @ in_basis
                in_vectors /= torch.linalg.vector_norm(
                    in_vectors, dim=1, keepdim=True
                )
                out_coefficients = torch.randn(
                    count,
                    d_key - rank,
                    generator=generator,
                    device=device,
                    dtype=torch.float64,
                )
                out_vectors = out_coefficients @ out_basis
                out_vectors /= torch.linalg.vector_norm(
                    out_vectors, dim=1, keepdim=True
                )
                queries = (
                    math.sqrt(max(0.0, 1.0 - distance**2)) * in_vectors
                    + distance * out_vectors
                )
                query_blocks.append(queries)
                distances.extend([distance] * count)
            queries = torch.cat(query_blocks)
            targets = latent_values(
                model,
                queries,
                linear,
                nonlinear,
                config["nonlinear_scale"],
            )
            predictions, uncertainty = memory.read_many_with_confidence(
                state, queries
            )
            residuals = predictions - targets
            squared_errors = torch.sum(residuals.square(), dim=1)
            coordinate_coverage = (
                torch.abs(residuals)
                <= 1.959963984540054 * torch.sqrt(uncertainty).unsqueeze(1)
            ).double().mean(dim=1)
            for query_index in range(queries.shape[0]):
                rows.append(
                    {
                        "data_model": model,
                        "trial": trial,
                        "query_index": query_index,
                        "distance_outside_observed_span": distances[query_index],
                        "uncertainty": uncertainty[query_index].item(),
                        "predicted_total_mse": (
                            d_value * uncertainty[query_index].item()
                        ),
                        "actual_squared_error": squared_errors[query_index].item(),
                        "normalized_squared_error": (
                            squared_errors[query_index].item()
                            / max(
                                d_value * uncertainty[query_index].item(), 1e-300
                            )
                        ),
                        "coordinate_95_coverage": coordinate_coverage[
                            query_index
                        ].item(),
                    }
                )
    return rows


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(values: np.ndarray, targets: np.ndarray) -> float:
    return float(np.corrcoef(average_ranks(values), average_ranks(targets))[0, 1])


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = average_ranks(scores)
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def confidence_summaries(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    summary: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    selective: list[dict[str, Any]] = []
    unseen: list[dict[str, Any]] = []
    for model in config["data_models"]:
        subset = [row for row in rows if row["data_model"] == model]
        uncertainty = np.asarray([row["uncertainty"] for row in subset])
        errors = np.asarray([row["actual_squared_error"] for row in subset])
        predicted = np.asarray([row["predicted_total_mse"] for row in subset])
        coverage = np.asarray([row["coordinate_95_coverage"] for row in subset])
        threshold = np.quantile(errors, config["high_error_quantile"])
        labels = errors >= threshold
        summary.append(
            {
                "data_model": model,
                "spearman_uncertainty_error": spearman(uncertainty, errors),
                "high_error_auroc": roc_auc(uncertainty, labels),
                "empirical_to_predicted_mse_ratio": float(
                    errors.mean() / predicted.mean()
                ),
                "coordinate_95_coverage": float(coverage.mean()),
                "p95_normalized_squared_error": float(
                    np.quantile(errors / predicted, 0.95)
                ),
                "query_count": len(subset),
            }
        )
        ordering = np.argsort(uncertainty, kind="mergesort")
        for bin_index, indices in enumerate(
            np.array_split(ordering, config["calibration_bins"])
        ):
            calibration.append(
                {
                    "data_model": model,
                    "bin": bin_index,
                    "count": len(indices),
                    "mean_uncertainty": float(uncertainty[indices].mean()),
                    "mean_predicted_total_mse": float(predicted[indices].mean()),
                    "mean_actual_squared_error": float(errors[indices].mean()),
                    "empirical_to_predicted_ratio": float(
                        errors[indices].mean() / predicted[indices].mean()
                    ),
                }
            )
        for retained_fraction in config["selective_coverages"]:
            retained = ordering[: max(1, int(round(len(ordering) * retained_fraction)))]
            selective.append(
                {
                    "data_model": model,
                    "retained_fraction": retained_fraction,
                    "abstained_fraction": 1.0 - retained_fraction,
                    "mean_squared_error": float(errors[retained].mean()),
                    "root_mean_squared_error": float(
                        math.sqrt(errors[retained].mean())
                    ),
                    "retained_count": len(retained),
                }
            )
        for distance in config["ood_distances"]:
            distance_rows = [
                row
                for row in subset
                if row["distance_outside_observed_span"] == distance
            ]
            unseen.append(
                {
                    "data_model": model,
                    "distance_outside_observed_span": distance,
                    "mean_uncertainty": float(
                        np.mean([row["uncertainty"] for row in distance_rows])
                    ),
                    "mean_actual_squared_error": float(
                        np.mean(
                            [row["actual_squared_error"] for row in distance_rows]
                        )
                    ),
                    "query_count": len(distance_rows),
                }
            )
    return summary, calibration, selective, unseen


def aggregate_duplicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, int, str], list[float]] = defaultdict(list)
    for row in rows:
        groups[
            (row["data_model"], row["sigma"], row["repeats"], row["method"])
        ].append(row["mean_squared_error"])
    return [
        {
            "data_model": key[0],
            "sigma": key[1],
            "repeats": key[2],
            "method": key[3],
            "mean_squared_error": float(np.mean(values)),
            "median_squared_error": percentile(values, 0.5),
            "p90_squared_error": percentile(values, 0.9),
            "case_count": len(values),
        }
        for key, values in sorted(groups.items())
    ]


def gate_summary(
    duplicate_rows: list[dict[str, Any]],
    precision_rows: list[dict[str, Any]],
    confidence_summary: list[dict[str, Any]],
    selective: list[dict[str, Any]],
    unseen: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    oracle_difference = max(
        row["csm_oracle_relative_difference"]
        for row in duplicate_rows
        if row["method"] == "csm"
    )
    slopes = []
    for sigma in config["noise_sigmas"]:
        risks = []
        for repeats in config["repeats"]:
            risks.append(
                np.mean(
                    [
                        row["mean_squared_error"]
                        for row in duplicate_rows
                        if row["data_model"] == "linear_gaussian"
                        and row["method"] == "csm"
                        and row["sigma"] == sigma
                        and row["repeats"] == repeats
                    ]
                )
            )
        slopes.append(
            float(np.polyfit(np.log(config["repeats"]), np.log(risks), 1)[0])
        )
    by_precision_case: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for row in precision_rows:
        if row["data_model"] == "linear_gaussian" and row["method"] in (
            "precision_weighted_csm",
            "uniform_csm",
        ):
            by_precision_case[(row["pattern"], row["seed"])][row["method"]] = row[
                "squared_error"
            ]
    precision_ratio = np.mean(
        [values["uniform_csm"] for values in by_precision_case.values()]
    ) / np.mean(
        [values["precision_weighted_csm"] for values in by_precision_case.values()]
    )
    gaussian = next(
        row for row in confidence_summary if row["data_model"] == "linear_gaussian"
    )
    gaussian_selective = [
        row for row in selective if row["data_model"] == "linear_gaussian"
    ]
    full_risk = next(
        row["mean_squared_error"]
        for row in gaussian_selective
        if row["retained_fraction"] == 1.0
    )
    half_row = min(
        gaussian_selective, key=lambda row: abs(row["retained_fraction"] - 0.5)
    )
    selective_ratio = half_row["mean_squared_error"] / full_risk
    gaussian_unseen = [
        row for row in unseen if row["data_model"] == "linear_gaussian"
    ]
    monotonic_deltas = np.diff([row["mean_uncertainty"] for row in gaussian_unseen])
    observed = {
        "csm_oracle_max_relative_difference": oracle_difference,
        "gaussian_repeated_risk_slopes": slopes,
        "gaussian_repeated_median_risk_slope": float(np.median(slopes)),
        "gaussian_uniform_to_precision_weighted_risk_ratio": float(precision_ratio),
        "gaussian_confidence_spearman": gaussian["spearman_uncertainty_error"],
        "gaussian_high_error_auroc": gaussian["high_error_auroc"],
        "gaussian_empirical_to_predicted_mse_ratio": gaussian[
            "empirical_to_predicted_mse_ratio"
        ],
        "gaussian_half_coverage_to_full_risk_ratio": selective_ratio,
        "minimum_unseen_uncertainty_increment": float(monotonic_deltas.min()),
    }
    thresholds = config["gate"]
    slope = observed["gaussian_repeated_median_risk_slope"]
    checks = {
        "csm_matches_oracle_ridge": oracle_difference
        <= thresholds["csm_oracle_max_relative_difference"],
        "gaussian_duplicates_follow_inverse_n_risk": (
            thresholds["gaussian_repeated_risk_slope_min"]
            <= slope
            <= thresholds["gaussian_repeated_risk_slope_max"]
        ),
        "known_precision_reduces_risk": precision_ratio
        >= thresholds["precision_weighting_min_risk_ratio"],
        "gaussian_uncertainty_tracks_error": gaussian[
            "spearman_uncertainty_error"
        ]
        >= thresholds["gaussian_confidence_min_spearman"],
        "gaussian_uncertainty_detects_high_error": gaussian["high_error_auroc"]
        >= thresholds["gaussian_confidence_min_auroc"],
        "gaussian_uncertainty_is_calibrated": (
            thresholds["gaussian_calibration_ratio_min"]
            <= gaussian["empirical_to_predicted_mse_ratio"]
            <= thresholds["gaussian_calibration_ratio_max"]
        ),
        "gaussian_selective_prediction_reduces_risk": selective_ratio
        <= thresholds["gaussian_selective_max_half_to_full_risk"],
        "unseen_directions_increase_uncertainty": float(monotonic_deltas.min())
        >= -thresholds["unseen_uncertainty_monotonic_tolerance"],
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
    duplicate_summary: list[dict[str, Any]],
    confidence_summary: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    selective: list[dict[str, Any]],
    unseen: list[dict[str, Any]],
    config: dict[str, Any],
    directory: Path,
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    colors = dict(zip(METHODS, plt.cm.tab10.colors))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
    for axis, sigma in zip(axes, config["noise_sigmas"]):
        for method in METHODS:
            subset = [
                row
                for row in duplicate_summary
                if row["data_model"] == "linear_gaussian"
                and row["sigma"] == sigma
                and row["method"] == method
            ]
            axis.loglog(
                [row["repeats"] for row in subset],
                [row["mean_squared_error"] for row in subset],
                marker="o",
                label=method,
                color=colors[method],
            )
        axis.set_title(f"sigma={sigma}")
        axis.set_xlabel("duplicate observations")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("mean squared latent prediction error")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Noisy repeated evidence (linear-Gaussian)")
    fig.tight_layout()
    path = directory / "noisy_duplicates.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))
    for axis, model in zip(axes.ravel(), config["data_models"]):
        subset = [row for row in calibration if row["data_model"] == model]
        predicted = [row["mean_predicted_total_mse"] for row in subset]
        actual = [row["mean_actual_squared_error"] for row in subset]
        axis.plot(predicted, actual, marker="o")
        bound = [min(predicted + actual), max(predicted + actual)]
        axis.plot(bound, bound, linestyle="--", color="black", linewidth=1)
        axis.set_title(model)
        axis.set(xlabel="predicted MSE", ylabel="empirical MSE")
        axis.grid(True, alpha=0.25)
    fig.suptitle("Uncertainty calibration by model class")
    fig.tight_layout()
    path = directory / "calibration.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for model in config["data_models"]:
        subset = [row for row in unseen if row["data_model"] == model]
        axis.plot(
            [row["distance_outside_observed_span"] for row in subset],
            [row["mean_uncertainty"] for row in subset],
            marker="o",
            label=model,
        )
    axis.set(
        xlabel="query component outside observed key span",
        ylabel="mean c(q)",
        title="Uncertainty on progressively unseen directions",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "unseen_directions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for model in config["data_models"]:
        subset = [row for row in selective if row["data_model"] == model]
        axis.plot(
            [row["retained_fraction"] for row in subset],
            [row["mean_squared_error"] for row in subset],
            marker="o",
            label=model,
        )
    axis.set(
        xlabel="retained fraction (lowest uncertainty first)",
        ylabel="mean squared error",
        title="Selective prediction",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "selective_prediction.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    names = [row["data_model"] for row in confidence_summary]
    axes[0].bar(
        names,
        [row["spearman_uncertainty_error"] for row in confidence_summary],
    )
    axes[0].set_ylabel("Spearman(c(q), squared error)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(names, [row["high_error_auroc"] for row in confidence_summary])
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("AUROC for high-error query")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Uncertainty ranking inside and outside the model")
    fig.tight_layout()
    path = directory / "uncertainty_ranking.png"
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
    confidence_table = markdown_table(
        [
            "data model",
            "Spearman",
            "high-error AUROC",
            "actual/predicted MSE",
            "95% coordinate coverage",
            "p95 normalized error",
        ],
        [
            [
                row["data_model"],
                f"{row['spearman_uncertainty_error']:.3f}",
                f"{row['high_error_auroc']:.3f}",
                f"{row['empirical_to_predicted_mse_ratio']:.3f}",
                f"{row['coordinate_95_coverage']:.3f}",
                f"{row['p95_normalized_squared_error']:.3f}",
            ]
            for row in record["confidence_summary"]
        ],
    )
    minimum_sigma = min(record["config"]["noise_sigmas"])
    duplicate_table = markdown_table(
        ["model", "method", "n=1 MSE", "n=16 MSE", "risk ratio"],
        [
            [
                model,
                method,
                f"{next(row['mean_squared_error'] for row in record['duplicate_summary'] if row['data_model'] == model and row['sigma'] == minimum_sigma and row['repeats'] == 1 and row['method'] == method):.3e}",
                f"{next(row['mean_squared_error'] for row in record['duplicate_summary'] if row['data_model'] == model and row['sigma'] == minimum_sigma and row['repeats'] == 16 and row['method'] == method):.3e}",
                f"{next(row['mean_squared_error'] for row in record['duplicate_summary'] if row['data_model'] == model and row['sigma'] == minimum_sigma and row['repeats'] == 1 and row['method'] == method) / next(row['mean_squared_error'] for row in record['duplicate_summary'] if row['data_model'] == model and row['sigma'] == minimum_sigma and row['repeats'] == 16 and row['method'] == method):.2f}x",
            ]
            for model in record["config"]["data_models"]
            for method in ("csm", "simple_average", "hebbian", "softmax", "oracle_ridge")
        ],
    )
    unseen_table = markdown_table(
        ["outside-span component", "mean c(q)", "Gaussian mean squared error"],
        [
            [
                f"{row['distance_outside_observed_span']:.2f}",
                f"{row['mean_uncertainty']:.4f}",
                f"{row['mean_actual_squared_error']:.4f}",
            ]
            for row in record["unseen_summary"]
            if row["data_model"] == "linear_gaussian"
        ],
    )
    observed = record["observed"]
    plots = "\n".join(
        f"- [`{Path(path).name}`](../{path})" for path in record["plots"]
    )
    return f"""# Phase 4 uncertainty and noisy evidence

## Gate decision: {status}

{checks}

The pass decision applies only to the declared linear-Gaussian model. Laplace, Student-like scale-mixture, and nonlinear rows are stress tests whose degradation is reported, not evidence for Bayesian optimality.

| Gate measurement | Observed |
|---|---:|
| maximum CSM/oracle-ridge relative difference | {observed['csm_oracle_max_relative_difference']:.3e} |
| repeated-evidence log-risk slope (median across sigma) | {observed['gaussian_repeated_median_risk_slope']:.3f} |
| uniform / precision-weighted Gaussian risk | {observed['gaussian_uniform_to_precision_weighted_risk_ratio']:.3f}x |
| Gaussian uncertainty/error Spearman | {observed['gaussian_confidence_spearman']:.3f} |
| Gaussian high-error AUROC | {observed['gaussian_high_error_auroc']:.3f} |
| Gaussian actual/predicted MSE | {observed['gaussian_empirical_to_predicted_mse_ratio']:.3f} |
| roughly half-coverage / full-coverage risk | {observed['gaussian_half_coverage_to_full_risk_ratio']:.3f} |
| minimum uncertainty increment toward unseen span | {observed['minimum_unseen_uncertainty_increment']:.3e} |

## A. Noisy duplicates and conflicting observations

Every dataset stores {record['config']['unique_associations']} orthogonal associations repeatedly, varies observation noise, and evaluates the latent (noise-free) value. Softmax receives oracle temperature selection from the committed grid, which favors it. Oracle ridge is independently assembled from the weighted normal equations. At `sigma={minimum_sigma}`:

{duplicate_table}

In the Gaussian rows, CSM's empirical risk follows the predicted `1/n` law and matches ridge to fp64 resolution. Simple averaging is nearly identical in this orthogonal repeated-key special case. Hebbian memory sums duplicates instead of averaging them and therefore becomes worse as repeats grow. The explicit conflicting pair has precisions 25 and 1; its complete estimator-to-consensus distances are retained in `conflicting_observations.csv`.

## B. Known beta precision

Known inverse variances are passed as `beta`. Across all Gaussian precision patterns, uniform weighting has {observed['gaussian_uniform_to_precision_weighted_risk_ratio']:.3f} times the aggregate risk of precision-weighted CSM. The raw record also includes Laplace and Student-like noise. This establishes the expected weighted least-squares behavior; it does not show that a learned gate will infer correct precisions.

## C. Confidence, calibration, high-error detection, and abstention

Training keys span 8 of 16 key dimensions. Queries move continuously from that observed subspace into its orthogonal complement. `c(q)` is treated as uncertainty (larger means less confident), and predicted latent squared error is `d_value * c(q)`.

{confidence_table}

The Gaussian calibration ratio and 95% coordinate coverage are in-model checks. The non-Gaussian rows preserve equal variance but expose tail sensitivity through coverage and p95 normalized error. The nonlinear row violates the latent linear-map assumption; its calibration/ranking values characterize that misspecification and are not relabeled as Bayesian guarantees.

Selective prediction retains queries with the smallest `c(q)`. The committed curves report risk at retained fractions `{record['config']['selective_coverages']}`; this is regression risk rather than a classification "accuracy" surrogate.

## D. Missing and out-of-distribution directions

{unseen_table}

The uncertainty increase is algebraic and remains present for every value/noise model because it depends only on the key statistic `S`. Only the Gaussian correspondence between its magnitude and prediction risk is an optimal/calibrated claim.

## Plots

{plots}

## Reproducibility

- source checkpoint: `{record['git_commit']}`; dirty at experiment start: `{record['working_tree_dirty']}`
- config: [`configs/phase4_uncertainty.json`](../configs/phase4_uncertainty.json)
- device: `{record['hardware']}`
- software: Python `{record['software']['python']}`, PyTorch `{record['software']['torch']}`, HIP `{record['software']['hip']}`, NumPy `{record['software']['numpy']}`
- wall time: {record['wall_clock_seconds']:.3f} seconds; peak allocated VRAM: {record['peak_vram_bytes'] / 2**30:.6f} GiB
- raw data: [`phase4/noisy_duplicates.csv`](phase4/noisy_duplicates.csv), [`phase4/precision_weighting.csv`](phase4/precision_weighting.csv), [`phase4/conflicting_observations.csv`](phase4/conflicting_observations.csv), [`phase4/confidence_queries.csv`](phase4/confidence_queries.csv)
- derived data: [`phase4/calibration.csv`](phase4/calibration.csv), [`phase4/selective_prediction.csv`](phase4/selective_prediction.csv), [`phase4/unseen_directions.csv`](phase4/unseen_directions.csv)
- machine-readable record: [`phase4_metrics.json`](phase4_metrics.json)

## Scoped conclusion

{record['interpretation']}
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase4_uncertainty.json")
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--plots", type=Path, default=Path("plots/phase4"))
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    source_dirty = bool(git_output(["status", "--porcelain"]))
    config = json.loads(args.config.read_text())
    device = select_device(config["device"])
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    duplicate_rows = duplicate_sweep(config, device)
    precision_rows, conflicts = precision_sweep(config, device)
    confidence_rows = confidence_sweep(config, device)
    duplicate_summary = aggregate_duplicates(duplicate_rows)
    confidence_summary, calibration, selective, unseen = confidence_summaries(
        confidence_rows, config
    )
    gate, observed = gate_summary(
        duplicate_rows,
        precision_rows,
        confidence_summary,
        selective,
        unseen,
        config,
    )
    plots = make_plots(
        duplicate_summary,
        confidence_summary,
        calibration,
        selective,
        unseen,
        config,
        args.plots,
    )
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
            "duplicates": len(duplicate_rows),
            "precision": len(precision_rows),
            "conflicts": len(conflicts),
            "confidence": len(confidence_rows),
        },
        "duplicate_summary": duplicate_summary,
        "confidence_summary": confidence_summary,
        "calibration": calibration,
        "selective_prediction": selective,
        "unseen_summary": unseen,
        "observed": observed,
        "plots": plots,
        "gate": gate,
        "interpretation": (
            "Within the exactly linear-Gaussian model, CSM matches weighted ridge, follows inverse-repeat risk, and its posterior variance is calibrated and useful for ranking and selective prediction. Under heavy-tailed and nonlinear misspecification the same score remains an algebraic coverage diagnostic, but tail and calibration degradation in the tables must replace any Bayesian-optimality claim."
            if gate["passed"]
            else "The linear-Gaussian Phase 4 gate failed. The complete output is retained and no uncertainty-optimality claim is accepted."
        ),
    }
    args.results.mkdir(parents=True, exist_ok=True)
    write_csv(args.results / "phase4" / "noisy_duplicates.csv", duplicate_rows)
    write_csv(args.results / "phase4" / "precision_weighting.csv", precision_rows)
    write_csv(args.results / "phase4" / "conflicting_observations.csv", conflicts)
    write_csv(args.results / "phase4" / "confidence_queries.csv", confidence_rows)
    write_csv(args.results / "phase4" / "calibration.csv", calibration)
    write_csv(args.results / "phase4" / "selective_prediction.csv", selective)
    write_csv(args.results / "phase4" / "unseen_directions.csv", unseen)
    (args.results / "phase4_metrics.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    (args.results / "phase4_uncertainty_and_noise.md").write_text(
        render_report(record)
    )
    print(json.dumps(record["gate"], indent=2, sort_keys=True))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
