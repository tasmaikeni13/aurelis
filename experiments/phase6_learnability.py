#!/usr/bin/env python3
"""Phase 6: learn key/query/value feature maps on small synthetic tasks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from csm import EpisodicMemoryModel, geometry_metrics, orthogonality_penalty


METHODS = {
    "learned_csm": ("csm", False),
    "fixed_random_csm": ("csm", True),
    "learned_hebbian": ("hebbian", False),
    "learned_attention": ("attention", False),
}
REGRESSION_TASKS = {
    "in_context_linear_regression",
    "noisy_in_context_regression",
}


@dataclass(frozen=True)
class EpisodeBatch:
    support_keys: Tensor
    support_values: Tensor
    queries: Tensor
    targets: Tensor
    target_indices: Tensor | None


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


def _gather(values: Tensor, indices: Tensor) -> Tensor:
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch, indices].unsqueeze(1)


def _bipolar(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> Tensor:
    bits = torch.randint(0, 2, shape, generator=generator, device=device)
    return (2.0 * bits.float() - 1.0) / math.sqrt(shape[-1])


def generate_batch(
    task: str,
    config: dict[str, Any],
    generator: torch.Generator,
    device: torch.device,
) -> EpisodeBatch:
    batch = config["batch_size"]
    associations = config["associations"]
    raw_key = config["raw_key_dimension"]
    raw_value = config["raw_value_dimension"]
    target_indices = torch.randint(
        associations, (batch,), generator=generator, device=device
    )

    if task == "associative_recall":
        keys = F.normalize(
            torch.randn(
                batch,
                associations,
                raw_key,
                generator=generator,
                device=device,
            ),
            dim=-1,
        )
        values = _bipolar((batch, associations, raw_value), generator, device)
        return EpisodeBatch(
            keys, values, _gather(keys, target_indices), _gather(values, target_indices), target_indices
        )

    if task in ("selective_copy", "key_value_lookup"):
        base = torch.eye(raw_key, device=device)[:associations]
        keys = base.unsqueeze(0).expand(batch, -1, -1).clone()
        if task == "selective_copy":
            symbols = torch.randint(
                raw_value,
                (batch, associations),
                generator=generator,
                device=device,
            )
            values = F.one_hot(symbols, num_classes=raw_value).float()
        else:
            values = _bipolar((batch, associations, raw_value), generator, device)
        return EpisodeBatch(
            keys, values, _gather(keys, target_indices), _gather(values, target_indices), target_indices
        )

    if task == "correlated_key_lookup":
        common = F.normalize(
            torch.randn(batch, 1, raw_key, generator=generator, device=device),
            dim=-1,
        )
        keys = F.normalize(
            common
            + config["correlated_key_noise"]
            * torch.randn(
                batch,
                associations,
                raw_key,
                generator=generator,
                device=device,
            ),
            dim=-1,
        )
        query = _gather(keys, target_indices)
        query = F.normalize(
            query
            + config["query_noise"]
            * torch.randn(
                query.shape, generator=generator, device=device
            ),
            dim=-1,
        )
        values = _bipolar((batch, associations, raw_value), generator, device)
        return EpisodeBatch(keys, values, query, _gather(values, target_indices), target_indices)

    if task in REGRESSION_TASKS:
        signal = config["signal_dimension"]
        support_x = torch.randn(
            batch, associations, signal, generator=generator, device=device
        )
        query_x = torch.randn(batch, 1, signal, generator=generator, device=device)
        operator = torch.randn(
            batch, raw_value, signal, generator=generator, device=device
        ) / math.sqrt(signal)
        clean_values = torch.einsum("bvd,bkd->bkv", operator, support_x)
        target = torch.einsum("bvd,bqd->bqv", operator, query_x)
        observed_values = clean_values
        if task == "noisy_in_context_regression":
            observed_values = observed_values + config["regression_noise"] * torch.randn(
                observed_values.shape, generator=generator, device=device
            )
        support_keys = torch.zeros(batch, associations, raw_key, device=device)
        queries = torch.zeros(batch, 1, raw_key, device=device)
        support_keys[..., :signal] = support_x
        queries[..., :signal] = query_x
        return EpisodeBatch(support_keys, observed_values, queries, target, None)

    if task == "contextual_associative_recall":
        if associations > raw_key:
            raise ValueError("context identities require associations <= raw key dimension")
        identity = torch.eye(raw_key, device=device)[:associations]
        identity = config["context_signal_scale"] * identity
        support = identity.unsqueeze(0).expand(batch, -1, -1).clone()
        query_signal = _gather(support, target_indices)
        nuisance_start = associations
        support[..., nuisance_start:] += torch.randn(
            batch,
            associations,
            raw_key - nuisance_start,
            generator=generator,
            device=device,
        )
        query = query_signal.clone()
        query[..., nuisance_start:] += torch.randn(
            batch,
            1,
            raw_key - nuisance_start,
            generator=generator,
            device=device,
        )
        values = _bipolar((batch, associations, raw_value), generator, device)
        return EpisodeBatch(support, values, query, _gather(values, target_indices), target_indices)

    raise ValueError(f"unknown task: {task}")


def make_model(
    method: str, config: dict[str, Any], seed: int, device: torch.device
) -> EpisodicMemoryModel:
    kind, fixed = METHODS.get(method, ("csm", False))
    shared_key_query = method != "learned_csm_independent_query_ablation"
    torch.manual_seed(seed)
    model = EpisodicMemoryModel(
        config["raw_key_dimension"],
        config["raw_value_dimension"],
        config["key_dimension"],
        config["value_dimension"],
        config["hidden_dimension"],
        epsilon=config["epsilon"],
        kind=kind,
        fixed_random_features=fixed,
        shared_key_query=shared_key_query,
    ).to(device)
    return model


def gradient_norm(model: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += parameter.grad.detach().square().sum().item()
    return math.sqrt(squared)


@torch.no_grad()
def evaluate(
    model: EpisodicMemoryModel,
    task: str,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    predictions = []
    targets = []
    keys = []
    successes = []
    uncertainties = []
    for _ in range(config["evaluation_batches"]):
        episode = generate_batch(task, config, generator, device)
        output = model(
            episode.support_keys, episode.support_values, episode.queries
        )
        predictions.append(output.prediction)
        targets.append(episode.targets)
        keys.append(output.keys)
        if output.uncertainty is not None:
            uncertainties.append(output.uncertainty)
        if task == "selective_copy":
            # Symbols may repeat at several support positions.  A correct copied
            # symbol must not be rejected because it also occurs in another slot.
            decoded = torch.argmax(output.prediction[:, 0], dim=-1)
            expected = torch.argmax(episode.targets[:, 0], dim=-1)
            successes.append((decoded == expected).float())
        elif episode.target_indices is not None:
            normalized_prediction = F.normalize(output.prediction[:, 0], dim=-1)
            normalized_values = F.normalize(episode.support_values, dim=-1)
            decoded = torch.argmax(
                torch.einsum("bv,bkv->bk", normalized_prediction, normalized_values),
                dim=-1,
            )
            successes.append((decoded == episode.target_indices).float())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    encoded_keys = torch.cat(keys)
    mse = torch.mean((prediction - target) ** 2).item()
    normalized_mse = mse / max(torch.mean(target**2).item(), 1e-12)
    metrics: dict[str, Any] = {
        "mse": mse,
        "normalized_mse": normalized_mse,
        "success_rate": (
            torch.cat(successes).mean().item() if successes else math.nan
        ),
        "mean_uncertainty": (
            torch.cat(uncertainties).mean().item() if uncertainties else math.nan
        ),
    }
    metrics.update(geometry_metrics(encoded_keys, config["epsilon"]))
    return metrics, encoded_keys


def train_one(
    method: str,
    task: str,
    seed: int,
    config: dict[str, Any],
    device: torch.device,
    *,
    orthogonality_weight: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_index = config["tasks"].index(task)
    model_seed = 100_003 * seed + 1_009 * task_index + 17
    model = make_model(method, config, model_seed, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    generator = torch.Generator(device=device).manual_seed(model_seed + 9_000_011)
    curves: list[dict[str, Any]] = []
    gradient_norms = []
    started = time.perf_counter()
    for step in range(1, config["training_steps"] + 1):
        episode = generate_batch(task, config, generator, device)
        output = model(
            episode.support_keys, episode.support_values, episode.queries
        )
        retrieval_loss = torch.mean((output.prediction - episode.targets) ** 2)
        geometry_loss = orthogonality_penalty(output.keys)
        loss = retrieval_loss + orthogonality_weight * geometry_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = gradient_norm(model)
        gradient_norms.append(norm)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0)
        optimizer.step()
        if step == 1 or step % config["log_every"] == 0:
            curves.append(
                {
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "step": step,
                    "training_mse": retrieval_loss.item(),
                    "orthogonality_penalty": geometry_loss.item(),
                    "gradient_norm": norm,
                }
            )
    synchronize(device)
    evaluation, _ = evaluate(
        model,
        task,
        config,
        seed=20_000_003 + model_seed,
        device=device,
    )
    row = {
        "task": task,
        "task_type": "regression" if task in REGRESSION_TASKS else "discrete_retrieval",
        "method": method,
        "seed": seed,
        "beta": 1.0,
        "lambda": 1.0,
        "query_scale": float(torch.exp(model.log_query_scale).detach().item()),
        "shared_key_query": model.shared_key_query,
        "epsilon": config["epsilon"],
        "orthogonality_regularizer": orthogonality_weight,
        "training_seconds": time.perf_counter() - started,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "median_gradient_norm": percentile(gradient_norms, 0.5),
        "final_gradient_norm": gradient_norms[-1],
        **evaluation,
    }
    row["gram_eigenvalue_spectrum"] = ";".join(
        f"{value:.8g}" for value in evaluation["gram_eigenvalue_spectrum"]
    )
    return row, curves


def run_sweep(
    config: dict[str, Any], device: torch.device
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    # Natural, unregularized behavior is always run and recorded first.
    for task in config["tasks"]:
        for seed in config["seeds"]:
            for method in METHODS:
                row, method_curves = train_one(
                    method, task, seed, config, device
                )
                rows.append(row)
                curves.extend(method_curves)
    # Only after the complete natural sweep do we run the declared ablation.
    for task in config["orthogonality_ablation_tasks"]:
        for seed in config["seeds"]:
            row, method_curves = train_one(
                "learned_csm_orthogonality_ablation",
                task,
                seed,
                config,
                device,
                orthogonality_weight=config["orthogonality_ablation_weight"],
            )
            rows.append(row)
            curves.extend(method_curves)
    # Restore the failed two-chart formulation only after the entire natural
    # sweep, as a diagnostic that cannot contribute to the pass gate.
    for task in config["independent_query_ablation_tasks"]:
        for seed in config["seeds"]:
            row, method_curves = train_one(
                "learned_csm_independent_query_ablation",
                task,
                seed,
                config,
                device,
            )
            rows.append(row)
            curves.extend(method_curves)
    return rows, curves


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["method"])].append(row)
    return [
        {
            "task": key[0],
            "method": key[1],
            "mean_normalized_mse": float(
                np.mean([row["normalized_mse"] for row in values])
            ),
            "maximum_normalized_mse": max(
                row["normalized_mse"] for row in values
            ),
            "mean_success_rate": (
                float(np.mean([row["success_rate"] for row in values]))
                if math.isfinite(values[0]["success_rate"])
                else math.nan
            ),
            "minimum_success_rate": (
                min(row["success_rate"] for row in values)
                if math.isfinite(values[0]["success_rate"])
                else math.nan
            ),
            "mean_effective_rank": float(
                np.mean([row["effective_rank"] for row in values])
            ),
            "mean_effective_capacity_fraction": float(
                np.mean([row["effective_capacity_fraction"] for row in values])
            ),
            "median_system_condition": percentile(
                [row["system_condition_number"] for row in values], 0.5
            ),
            "mean_absolute_pairwise_cosine": float(
                np.mean([row["mean_absolute_pairwise_cosine"] for row in values])
            ),
            "case_count": len(values),
        }
        for key, values in sorted(groups.items())
    ]


def gate_summary(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    natural = [row for row in rows if row["method"] == "learned_csm"]
    discrete_minimum = min(
        row["success_rate"]
        for row in natural
        if row["task"] not in REGRESSION_TASKS
    )
    regression_maximum = max(
        row["normalized_mse"]
        for row in natural
        if row["task"] in REGRESSION_TASKS
    )
    seed_ratios = []
    for seed in config["seeds"]:
        learned = [
            row["normalized_mse"]
            for row in rows
            if row["seed"] == seed and row["method"] == "learned_csm"
        ]
        random = [
            row["normalized_mse"]
            for row in rows
            if row["seed"] == seed and row["method"] == "fixed_random_csm"
        ]
        seed_ratios.append(float(np.mean(learned) / np.mean(random)))
    minimum_capacity = min(
        row["effective_capacity_fraction"] for row in natural
    )
    thresholds = config["gate"]
    observed = {
        "minimum_learned_csm_discrete_success": discrete_minimum,
        "maximum_learned_csm_regression_normalized_mse": regression_maximum,
        "learned_to_random_aggregate_risk_by_seed": seed_ratios,
        "maximum_learned_to_random_aggregate_risk": max(seed_ratios),
        "minimum_natural_effective_capacity_fraction": minimum_capacity,
    }
    checks = {
        "learned_csm_learns_every_discrete_task_across_seeds": discrete_minimum
        >= thresholds["minimum_discrete_success"],
        "learned_csm_learns_regression_tasks_across_seeds": regression_maximum
        <= thresholds["maximum_regression_normalized_mse"],
        "learned_csm_outperforms_random_features_every_seed": max(seed_ratios)
        <= thresholds["maximum_seed_aggregate_learned_to_random_risk"],
        "natural_geometry_uses_nontrivial_capacity": minimum_capacity
        >= thresholds["minimum_effective_capacity_fraction"],
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    rows: list[dict[str, Any]],
    curves: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    config: dict[str, Any],
    directory: Path,
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    method_names = list(METHODS)
    colors = dict(zip(method_names, plt.cm.tab10.colors))

    fig, axis = plt.subplots(figsize=(10.5, 5.0))
    x = np.arange(len(config["tasks"]))
    width = 0.19
    for index, method in enumerate(method_names):
        values = [
            next(
                row["mean_normalized_mse"]
                for row in summary
                if row["task"] == task and row["method"] == method
            )
            for task in config["tasks"]
        ]
        axis.bar(x + (index - 1.5) * width, values, width, label=method)
    axis.set_yscale("log")
    axis.set_xticks(x)
    axis.set_xticklabels(config["tasks"], rotation=25, ha="right")
    axis.set_ylabel("mean normalized evaluation MSE")
    axis.set_title("Learned feature-map task performance")
    axis.grid(True, axis="y", which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "task_performance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    for method in method_names:
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in curves:
            if row["method"] == method:
                grouped[row["step"]].append(row["training_mse"])
        steps = sorted(grouped)
        axis.semilogy(
            steps,
            [np.median(grouped[step]) for step in steps],
            marker="o",
            label=method,
            color=colors[method],
        )
    axis.set(xlabel="optimization step", ylabel="median training MSE", title="Natural unregularized learning curves")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "learning_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    natural = [row for row in summary if row["method"] == "learned_csm"]
    axes[0].bar(
        [row["task"] for row in natural],
        [row["mean_effective_rank"] for row in natural],
    )
    axes[0].axhline(config["key_dimension"], color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("effective rank")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(
        [row["task"] for row in natural],
        [row["median_system_condition"] for row in natural],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("median cond(S + epsilon I)")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(True, axis="y", which="both", alpha=0.25)
    fig.suptitle("Unassisted learned CSM geometry")
    fig.tight_layout()
    path = directory / "natural_geometry.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    labels = []
    natural_values = []
    regularized_values = []
    for task in config["orthogonality_ablation_tasks"]:
        labels.append(task)
        natural_values.append(
            np.mean(
                [row["normalized_mse"] for row in rows if row["task"] == task and row["method"] == "learned_csm"]
            )
        )
        regularized_values.append(
            np.mean(
                [row["normalized_mse"] for row in rows if row["task"] == task and row["method"] == "learned_csm_orthogonality_ablation"]
            )
        )
    x = np.arange(len(labels))
    axis.bar(x - 0.18, natural_values, 0.36, label="natural")
    axis.bar(x + 0.18, regularized_values, 0.36, label="orthogonality ablation")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel("mean normalized MSE")
    axis.set_title("Geometry regularization tested only after natural runs")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = directory / "regularizer_ablation.png"
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
    performance = markdown_table(
        ["task", "method", "normalized MSE", "success", "effective rank", "condition"],
        [
            [
                row["task"],
                row["method"],
                f"{row['mean_normalized_mse']:.3f}",
                (f"{row['mean_success_rate']:.3f}" if math.isfinite(row["mean_success_rate"]) else "—"),
                f"{row['mean_effective_rank']:.2f}",
                f"{row['median_system_condition']:.2e}",
            ]
            for row in record["summary"]
            if row["method"] in METHODS
        ],
    )
    ablation = markdown_table(
        ["task", "natural MSE", "regularized MSE", "natural cosine", "regularized cosine"],
        [
            [
                task,
                f"{next(row['mean_normalized_mse'] for row in record['summary'] if row['task'] == task and row['method'] == 'learned_csm'):.3f}",
                f"{next(row['mean_normalized_mse'] for row in record['summary'] if row['task'] == task and row['method'] == 'learned_csm_orthogonality_ablation'):.3f}",
                f"{next(row['mean_absolute_pairwise_cosine'] for row in record['summary'] if row['task'] == task and row['method'] == 'learned_csm'):.3f}",
                f"{next(row['mean_absolute_pairwise_cosine'] for row in record['summary'] if row['task'] == task and row['method'] == 'learned_csm_orthogonality_ablation'):.3f}",
            ]
            for task in record["config"]["orthogonality_ablation_tasks"]
        ],
    )
    coordinate_ablation = markdown_table(
        ["task", "shared success", "independent success", "shared rank", "independent rank"],
        [
            [
                task,
                f"{next(row['mean_success_rate'] for row in record['summary'] if row['task'] == task and row['method'] == 'learned_csm'):.3f}",
                f"{next(row['mean_success_rate'] for row in record['summary'] if row['task'] == task and row['method'] == 'learned_csm_independent_query_ablation'):.3f}",
                f"{next(row['mean_effective_rank'] for row in record['summary'] if row['task'] == task and row['method'] == 'learned_csm'):.2f}",
                f"{next(row['mean_effective_rank'] for row in record['summary'] if row['task'] == task and row['method'] == 'learned_csm_independent_query_ablation'):.2f}",
            ]
            for task in record["config"]["independent_query_ablation_tasks"]
        ],
    )
    plots = "\n".join(f"- [`{Path(path).name}`](../{path})" for path in record["plots"])
    observed = record["observed"]
    return f"""# Phase 6 learned feature maps

## Gate decision: {status}

{checks}

No general language model is trained. Every row is a small synthetic episodic model with `beta=lambda=1`. The primary sweep has no geometry regularizer; the regularizer ablation was executed only after all natural runs completed.

| Gate measurement | Observed |
|---|---:|
| minimum learned-CSM discrete success across tasks/seeds | {observed['minimum_learned_csm_discrete_success']:.3f} |
| maximum learned-CSM regression normalized MSE | {observed['maximum_learned_csm_regression_normalized_mse']:.3f} |
| learned/random aggregate risk ratios by seed | {', '.join(f'{value:.3f}' for value in observed['learned_to_random_aggregate_risk_by_seed'])} |
| minimum natural effective-capacity fraction | {observed['minimum_natural_effective_capacity_fraction']:.3f} |

## Task and baseline results

{performance}

`fixed_random_csm` freezes tied random key/query features and a random value map while training only its output decoder, making it a favorable exact-query random-feature control. Hebbian and attention models receive learned encoders of the same size. Attention is not expected to lose on every task; the scientific gate is whether learned CSM features beat their own untrained/random representation and solve all tasks reproducibly.

### Matched-coordinate correction

The natural architecture uses one shared feature map for keys and queries plus one learned positive scalar query calibration. This is not an orthogonality constraint: the loss remains retrieval error alone, and the Gram spectrum is free to emerge. It enforces the mathematical requirement that a query evaluate the ridge operator in the coordinate chart in which it was fitted. The post-natural ablation restores independent key/query maps while holding every other setting fixed.

{coordinate_ablation}

The initial diagnostic implementation normalized two independent encoders and therefore violated this compatibility condition. Its selective-copy scorer additionally confused symbol identity with support-slot identity. Those diagnostic outputs are not used by this report or its gate.

## Natural representation geometry

Every seed-level row records the full mean Gram eigenvalue spectrum, pairwise cosine statistics, effective rank, minimum singular value, `cond(S+epsilon I)`, nominal and effective capacity fractions, gradient norms, epsilon, query scale, and retrieval error. Effective capacity is `effective_rank / min(K, d_key)`, the reachable rank of a `K`-association episode. The primary learned CSM results above are entirely unassisted by orthogonality loss.

### Explicit post-hoc regularizer ablation

{ablation}

This ablation is diagnostic, not part of the pass gate. If it is required to rescue a task, that fact is a limitation rather than evidence that gradient descent naturally found good geometry.

## Plots

{plots}

## Reproducibility

- source checkpoint: `{record['git_commit']}`; dirty at experiment start: `{record['working_tree_dirty']}`
- config: [`configs/phase6_learnability.json`](../configs/phase6_learnability.json)
- device: `{record['hardware']}`
- software: Python `{record['software']['python']}`, PyTorch `{record['software']['torch']}`, HIP `{record['software']['hip']}`, NumPy `{record['software']['numpy']}`
- wall time: {record['wall_clock_seconds']:.3f} seconds; peak allocated VRAM: {record['peak_vram_bytes'] / 2**30:.6f} GiB
- seed metrics: [`phase6/seed_metrics.csv`](phase6/seed_metrics.csv)
- learning curves: [`phase6/learning_curves.csv`](phase6/learning_curves.csv)
- machine-readable record: [`phase6_metrics.json`](phase6_metrics.json)

## Failure study and mathematical basis

The correction follows differentiable ridge meta-learning, which applies one feature extractor to both support and held-out examples and backpropagates through the closed-form solver. It is also the construction required by deep-kernel learning: one learned feature map defines both arguments of a kernel. Representation-collapse work motivates the separately labeled regularizer ablation, but no covariance or orthogonality penalty enters the natural gate.

- Bertinetto et al., [Meta-learning with differentiable closed-form solvers](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/bertinetto19meta-learning.pdf)
- Lee et al., [Meta-Learning With Differentiable Convex Optimization](https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Meta-Learning_With_Differentiable_Convex_Optimization_CVPR_2019_paper.html)
- Wilson et al., [Deep Kernel Learning](https://proceedings.mlr.press/v51/wilson16.html)
- Bardes et al., [VICReg](https://arxiv.org/abs/2105.04906)

## Scoped conclusion

{record['interpretation']}
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase6_learnability.json"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--plots", type=Path, default=Path("plots/phase6"))
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    source_dirty = bool(git_output(["status", "--porcelain"]))
    config = json.loads(args.config.read_text())
    device = select_device(config["device"])
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows, curves = run_sweep(config, device)
    summary = aggregate(rows)
    gate, observed = gate_summary(rows, config)
    plots = make_plots(rows, curves, summary, config, args.plots)
    synchronize(device)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "working_tree_dirty": source_dirty,
        "config_path": str(args.config),
        "config": config,
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "software": {"python": platform.python_version(), "torch": torch.__version__, "hip": torch.version.hip, "numpy": np.__version__},
        "wall_clock_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "row_counts": {"seed_metrics": len(rows), "learning_curves": len(curves)},
        "summary": summary,
        "observed": observed,
        "plots": plots,
        "gate": gate,
        "interpretation": (
            "Across every seed, the unregularized learned CSM solves all seven synthetic tasks and improves aggregate risk over its frozen random-feature control. The geometry tables state whether this happened through naturally separated keys; regularized rows are post-hoc ablations only."
            if gate["passed"]
            else "The Phase 6 learned-memory gate failed. The complete seed/task/baseline record is retained; learned representations are not accepted as reliable and Phase 7 must not claim a successful learned-memory foundation."
        ),
    }
    args.results.mkdir(parents=True, exist_ok=True)
    write_csv(args.results / "phase6" / "seed_metrics.csv", rows)
    write_csv(args.results / "phase6" / "learning_curves.csv", curves)
    (args.results / "phase6_metrics.json").write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=True) + "\n")
    (args.results / "phase6_learnability.md").write_text(render_report(record))
    print(json.dumps(record["gate"], indent=2, sort_keys=True))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
