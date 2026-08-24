#!/usr/bin/env python3
"""Phase 7: test whether learned beta and lambda retain Bayesian semantics."""

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

from csm import BoundedScalarGate, EpisodicMemoryModel


CATEGORIES = ("clean", "noisy", "corrupted", "distractor")
BETA_METHODS = ("beta_fixed_1", "beta_generic_value", "beta_learned_precision", "beta_oracle")
LAMBDA_METHODS = ("lambda_fixed_1", "lambda_fixed_0.95", "lambda_fixed_0.8", "lambda_learned_cue", "lambda_innovation_ablation", "lambda_oracle_change")
JOINT_METHODS = ("joint_fixed", "joint_learned", "joint_oracle")


@dataclass(frozen=True)
class EvidenceBatch:
    keys: Tensor
    values: Tensor
    queries: Tensor
    targets: Tensor
    cues: Tensor
    oracle_beta: Tensor
    categories: Tensor


@dataclass(frozen=True)
class StreamBatch:
    keys: Tensor
    values: Tensor
    queries: Tensor
    targets: Tensor
    changes: Tensor
    change_points: Tensor
    drift_cue: Tensor
    cues: Tensor | None = None
    oracle_beta: Tensor | None = None
    categories: Tensor | None = None


def git_output(args: list[str]) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def device_for(name: str) -> torch.device:
    return torch.device("cuda:0" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_for(config: dict[str, Any], seed: int, device: torch.device) -> EpisodicMemoryModel:
    torch.manual_seed(seed)
    return EpisodicMemoryModel(
        config["raw_key_dimension"], config["raw_value_dimension"],
        config["key_dimension"], config["value_dimension"], config["hidden_dimension"],
        epsilon=config["epsilon"], shared_key_query=True,
    ).to(device)


def gate_for(config: dict[str, Any], inputs: int, low: float, high: float, initial: float, device: torch.device) -> BoundedScalarGate:
    return BoundedScalarGate(inputs, config["gate_hidden_dimension"], minimum=low, maximum=high, initial_fraction=initial).to(device)


def cue_table(device: torch.device) -> Tensor:
    # reported sensor quality, cross-view consistency, and task relevance
    return torch.tensor([[1.0, 1.0, 1.0], [-1.0, 0.4, 1.0], [1.0, -1.0, 1.0], [0.0, 0.0, -1.0]], device=device)


def oracle_beta_table(device: torch.device) -> Tensor:
    return torch.tensor([4.0, 0.04, 0.001, 0.001], device=device)


def pad_keys(x: Tensor, raw_dimension: int) -> Tensor:
    out = x.new_zeros((*x.shape[:-1], raw_dimension))
    out[..., : x.shape[-1]] = x
    return out


def evidence_batch(config: dict[str, Any], gen: torch.Generator, device: torch.device) -> EvidenceBatch:
    cfg = config["evidence"]
    b, n, d, v = cfg["batch_size"], cfg["observations"], config["signal_dimension"], config["raw_value_dimension"]
    category = torch.randint(4, (b, n), generator=gen, device=device)
    key = F.normalize(torch.randn(b, 1, d, generator=gen, device=device), dim=-1)
    keys = pad_keys(key.expand(-1, n, -1), config["raw_key_dimension"])
    query = pad_keys(key, config["raw_key_dimension"])
    target = torch.randn(b, 1, v, generator=gen, device=device)
    noise = torch.randn(b, n, v, generator=gen, device=device)
    clean = target + cfg["clean_noise"] * noise
    noisy = target + cfg["noisy_noise"] * noise
    corrupt = -target + cfg["clean_noise"] * noise
    distract = torch.randn(b, n, v, generator=gen, device=device)
    values = torch.where((category == 0)[..., None], clean, torch.where((category == 1)[..., None], noisy, torch.where((category == 2)[..., None], corrupt, distract)))
    cues = cue_table(device)[category] + cfg["cue_noise"] * torch.randn(b, n, 3, generator=gen, device=device)
    return EvidenceBatch(keys, values, query, target, cues, oracle_beta_table(device)[category], category)


def evidence_forward(model: EpisodicMemoryModel, gate: BoundedScalarGate | None, method: str, batch: EvidenceBatch) -> tuple[Tensor, Tensor]:
    if method == "beta_fixed_1":
        weights = torch.ones_like(batch.oracle_beta)
        output = model(batch.keys, batch.values, batch.queries, beta=weights)
    elif method == "beta_oracle":
        weights = batch.oracle_beta
        output = model(batch.keys, batch.values, batch.queries, beta=weights)
    elif method == "beta_learned_precision":
        assert gate is not None
        weights = gate(batch.cues)
        output = model(batch.keys, batch.values, batch.queries, beta=weights)
    else:
        assert gate is not None
        weights = gate(batch.cues)
        output = model(batch.keys, batch.values, batch.queries, beta=torch.ones_like(weights), value_weight=weights)
    return output.prediction, weights


@torch.no_grad()
def eval_evidence(model: EpisodicMemoryModel, gate: BoundedScalarGate | None, method: str, config: dict[str, Any], seed: int, device: torch.device) -> dict[str, float]:
    gen = torch.Generator(device=device).manual_seed(seed)
    predictions, targets, weights, categories, oracle = [], [], [], [], []
    for _ in range(config["evaluation_batches"]):
        batch = evidence_batch(config, gen, device)
        prediction, weight = evidence_forward(model, gate, method, batch)
        predictions.append(prediction); targets.append(batch.targets); weights.append(weight); categories.append(batch.categories); oracle.append(batch.oracle_beta)
    pred, target = torch.cat(predictions), torch.cat(targets)
    weight, category, true_weight = torch.cat(weights), torch.cat(categories), torch.cat(oracle)
    mse = torch.mean((pred - target) ** 2).item()
    result = {"mse": mse, "normalized_mse": mse / torch.mean(target ** 2).item()}
    for index, name in enumerate(CATEGORIES):
        result[f"mean_weight_{name}"] = weight[category == index].mean().item()
    result["oracle_weight_correlation"] = float(np.corrcoef(weight.cpu().flatten(), true_weight.cpu().flatten())[0, 1]) if weight.std() > 1e-6 else 0.0
    return result


def train_evidence(method: str, seed: int, config: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_seed = 71_003 * seed + 101 * BETA_METHODS.index(method) + 7
    model = model_for(config, run_seed, device)
    gate = None
    if method in ("beta_generic_value", "beta_learned_precision"):
        gate = gate_for(config, 3, config["evidence"]["beta_minimum"], config["evidence"]["beta_maximum"], 0.25, device)
    params = list(model.parameters()) + ([] if gate is None else list(gate.parameters()))
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    gen = torch.Generator(device=device).manual_seed(run_seed + 1_000_003)
    curves, started = [], time.perf_counter()
    for step in range(1, config["evidence"]["training_steps"] + 1):
        batch = evidence_batch(config, gen, device)
        prediction, _ = evidence_forward(model, gate, method, batch)
        loss = torch.mean((prediction - batch.targets) ** 2)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 10.0); optimizer.step()
        if step == 1 or step % config["log_every"] == 0:
            curves.append({"experiment": "beta", "method": method, "seed": seed, "step": step, "loss": loss.item()})
    metrics = eval_evidence(model, gate, method, config, run_seed + 9_000_019, device)
    return {"experiment": "beta", "method": method, "seed": seed, "training_seconds": time.perf_counter() - started, **metrics}, curves


def drift_batch(config: dict[str, Any], section: str, gen: torch.Generator, device: torch.device, *, heterogeneous: bool) -> StreamBatch:
    cfg = config[section]; b, t, d, v = cfg["batch_size"], cfg["sequence_length"], config["signal_dimension"], config["raw_value_dimension"]
    cp = torch.randint(cfg["change_point_minimum"], cfg["change_point_maximum"] + 1, (b,), generator=gen, device=device)
    x = F.normalize(torch.randn(b, t, d, generator=gen, device=device), dim=-1)
    q = F.normalize(torch.randn(b, t, d, generator=gen, device=device), dim=-1)
    w1 = torch.randn(b, v, d, generator=gen, device=device) / math.sqrt(d)
    w2 = torch.randn(b, v, d, generator=gen, device=device) / math.sqrt(d)
    times = torch.arange(t, device=device)[None, :]
    current = torch.where((times < cp[:, None])[:, :, None, None], w1[:, None], w2[:, None])
    clean = torch.einsum("btvd,btd->btv", current, x)
    target = torch.einsum("btvd,btd->btv", current, q)
    changes = times == cp[:, None]
    cues = oracle_beta = categories = None
    if heterogeneous:
        category = torch.randint(4, (b, t), generator=gen, device=device)
        noise = torch.randn(b, t, v, generator=gen, device=device)
        values = torch.where((category == 0)[..., None], clean + cfg["clean_noise"] * noise,
                 torch.where((category == 1)[..., None], clean + cfg["noisy_noise"] * noise,
                 torch.where((category == 2)[..., None], -clean + cfg["clean_noise"] * noise,
                 torch.randn(b, t, v, generator=gen, device=device))))
        cues = cue_table(device)[category] + cfg["cue_noise"] * torch.randn(b, t, 3, generator=gen, device=device)
        oracle_beta = oracle_beta_table(device)[category]; categories = category
    else:
        values = clean + cfg["observation_noise"] * torch.randn(b, t, v, generator=gen, device=device)
    drift_cue = changes.float() + cfg["drift_cue_noise"] * torch.randn(b, t, generator=gen, device=device)
    return StreamBatch(pad_keys(x, config["raw_key_dimension"]), values, pad_keys(q, config["raw_key_dimension"]), target, changes, cp, drift_cue, cues, oracle_beta, categories)


def sequential_forward(model: EpisodicMemoryModel, lambda_gate: BoundedScalarGate | None, method: str, batch: StreamBatch, config: dict[str, Any], beta_gate: BoundedScalarGate | None = None) -> tuple[Tensor, Tensor, Tensor | None]:
    b, t = batch.keys.shape[:2]
    state = model.initial_state(b, batch.keys)
    predictions, lambdas, betas = [], [], []
    for index in range(t):
        key = model.encode_keys(batch.keys[:, index:index + 1])[:, 0]
        value = model.encode_values(batch.values[:, index:index + 1])[:, 0]
        query = model.encode_queries(batch.queries[:, index:index + 1])
        pre_latent, uncertainty = model.read_state(state, key[:, None])
        residual = torch.mean((model.decode(pre_latent)[:, 0] - batch.values[:, index]) ** 2, dim=-1)
        innovation = torch.stack((torch.log1p(residual), torch.log1p(uncertainty[:, 0])), dim=-1).detach()
        if method in ("lambda_learned_cue", "joint_learned"):
            assert lambda_gate is not None
            lambda_input = batch.drift_cue[:, index:index + 1] if batch.cues is None else torch.cat((batch.drift_cue[:, index:index + 1], batch.cues[:, index]), dim=-1)
            decay = lambda_gate(lambda_input)
        elif method == "lambda_innovation_ablation":
            assert lambda_gate is not None
            decay = lambda_gate(innovation)
        elif method in ("lambda_oracle_change", "joint_oracle"):
            decay = torch.where(batch.changes[:, index], torch.full_like(residual, config["drift"]["lambda_minimum"]), torch.ones_like(residual))
        elif method.startswith("lambda_fixed_"):
            decay = torch.full_like(residual, float(method.removeprefix("lambda_fixed_").replace("_", ".")))
        else:
            decay = torch.ones_like(residual)
        if batch.cues is None:
            beta = torch.ones_like(decay)
        elif method == "joint_learned":
            assert beta_gate is not None
            beta = beta_gate(batch.cues[:, index])
        elif method == "joint_oracle":
            assert batch.oracle_beta is not None
            beta = batch.oracle_beta[:, index]
        else:
            beta = torch.ones_like(decay)
        state = model.write_state(state, key, value, beta, decay)
        latent, _ = model.read_state(state, query)
        predictions.append(model.decode(latent)[:, 0]); lambdas.append(decay); betas.append(beta)
    return torch.stack(predictions, dim=1), torch.stack(lambdas, dim=1), (torch.stack(betas, dim=1) if batch.cues is not None else None)


def stream_metrics(prediction: Tensor, lambdas: Tensor, batch: StreamBatch, warmup: int, betas: Tensor | None = None) -> dict[str, Any]:
    b, t = lambdas.shape; times = torch.arange(t, device=lambdas.device)[None, :]; rel = times - batch.change_points[:, None]
    error = torch.mean((prediction - batch.targets) ** 2, dim=-1); power = torch.mean(batch.targets ** 2, dim=-1).clamp_min(1e-8)
    def risk(mask: Tensor) -> float:
        return (error[mask].mean() / power[mask].mean()).item()
    valid = times.expand(b, -1) >= warmup
    stable = valid & (rel.abs() > 2); post = (rel >= 0) & (rel < 4)
    drift_signal = batch.changes[valid].float().cpu().numpy(); forgetting = (1.0 - lambdas[valid]).detach().cpu().numpy()
    corr = float(np.corrcoef(drift_signal, forgetting)[0, 1]) if np.std(forgetting) > 1e-6 else 0.0
    offsets = list(range(-8, 12)); error_trace, lambda_trace = [], []
    for offset in offsets:
        mask = rel == offset
        error_trace.append(risk(mask)); lambda_trace.append(lambdas[mask].mean().item())
    result: dict[str, Any] = {
        "normalized_mse": risk(valid), "stationary_normalized_mse": risk(stable), "postchange_normalized_mse": risk(post),
        "mean_stationary_lambda": lambdas[stable].mean().item(), "mean_change_lambda": lambdas[batch.changes].mean().item(),
        "drift_forgetting_correlation": corr, "trace_offsets": ";".join(map(str, offsets)),
        "error_trace": ";".join(f"{x:.8g}" for x in error_trace), "lambda_trace": ";".join(f"{x:.8g}" for x in lambda_trace),
    }
    if betas is not None and batch.categories is not None and batch.oracle_beta is not None:
        for index, name in enumerate(CATEGORIES):
            result[f"mean_weight_{name}"] = betas[batch.categories == index].mean().item()
        result["oracle_weight_correlation"] = float(np.corrcoef(betas.detach().cpu().flatten(), batch.oracle_beta.cpu().flatten())[0, 1]) if betas.std() > 1e-6 else 0.0
    return result


@torch.no_grad()
def eval_stream(model: EpisodicMemoryModel, lambda_gate: BoundedScalarGate | None, method: str, config: dict[str, Any], section: str, seed: int, device: torch.device, beta_gate: BoundedScalarGate | None = None) -> dict[str, Any]:
    gen = torch.Generator(device=device).manual_seed(seed); collected = []
    for _ in range(config["evaluation_batches"]):
        batch = drift_batch(config, section, gen, device, heterogeneous=section == "joint")
        prediction, lambdas, betas = sequential_forward(model, lambda_gate, method, batch, config, beta_gate)
        collected.append(stream_metrics(prediction, lambdas, batch, config[section]["warmup"], betas))
    scalar_keys = [key for key, value in collected[0].items() if isinstance(value, float)]
    result = {key: float(np.mean([row[key] for row in collected])) for key in scalar_keys}
    for key in ("trace_offsets", "error_trace", "lambda_trace"):
        if key == "trace_offsets": result[key] = collected[0][key]
        else:
            arrays = [np.fromstring(row[key], sep=";") for row in collected]
            result[key] = ";".join(f"{x:.8g}" for x in np.mean(arrays, axis=0))
    return result


def train_stream(method: str, seed: int, config: dict[str, Any], section: str, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    methods = LAMBDA_METHODS if section == "drift" else JOINT_METHODS
    run_seed = 91_009 * seed + 211 * methods.index(method) + (13 if section == "drift" else 29)
    model = model_for(config, run_seed, device); lambda_gate = beta_gate = None
    if method in ("lambda_learned_cue", "lambda_innovation_ablation", "joint_learned"):
        inputs = 2 if method == "lambda_innovation_ablation" else (1 if section == "drift" else 4)
        lambda_gate = gate_for(config, inputs, config["drift"]["lambda_minimum"], 1.0, 0.95, device)
    if method == "joint_learned":
        beta_gate = gate_for(config, 3, config["evidence"]["beta_minimum"], config["evidence"]["beta_maximum"], 0.25, device)
    params = list(model.parameters()) + ([] if lambda_gate is None else list(lambda_gate.parameters())) + ([] if beta_gate is None else list(beta_gate.parameters()))
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    gen = torch.Generator(device=device).manual_seed(run_seed + 2_000_003); curves, started = [], time.perf_counter()
    for step in range(1, config[section]["training_steps"] + 1):
        batch = drift_batch(config, section, gen, device, heterogeneous=section == "joint")
        prediction, _, _ = sequential_forward(model, lambda_gate, method, batch, config, beta_gate)
        loss = torch.mean((prediction[:, config[section]["warmup"]:] - batch.targets[:, config[section]["warmup"]:]) ** 2)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 10.0); optimizer.step()
        if step == 1 or step % config["log_every"] == 0:
            curves.append({"experiment": section, "method": method, "seed": seed, "step": step, "loss": loss.item()})
    metrics = eval_stream(model, lambda_gate, method, config, section, run_seed + 8_000_021, device, beta_gate)
    return {"experiment": section, "method": method, "seed": seed, "training_seconds": time.perf_counter() - started, **metrics}, curves


def separate_gate(beta_rows: list[dict[str, Any]], drift_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["gate"]; checks: dict[str, bool] = {}; observed: dict[str, Any] = {}
    beta_ratios = []
    for seed in config["seeds"]:
        learned = next(r for r in beta_rows if r["seed"] == seed and r["method"] == "beta_learned_precision")
        fixed = next(r for r in beta_rows if r["seed"] == seed and r["method"] == "beta_fixed_1")
        beta_ratios.append(learned["normalized_mse"] / fixed["normalized_mse"])
    learned_beta = [r for r in beta_rows if r["method"] == "beta_learned_precision"]
    clean_noisy = min(r["mean_weight_clean"] - r["mean_weight_noisy"] for r in learned_beta)
    noisy_bad = min(r["mean_weight_noisy"] - max(r["mean_weight_corrupted"], r["mean_weight_distractor"]) for r in learned_beta)
    beta_corr = min(r["oracle_weight_correlation"] for r in learned_beta)
    checks["learned_beta_improves_fixed_every_seed"] = max(beta_ratios) <= thresholds["maximum_beta_learned_to_fixed_risk"]
    checks["learned_beta_orders_reliability"] = clean_noisy >= thresholds["minimum_clean_noisy_beta_gap"] and noisy_bad >= thresholds["minimum_noisy_bad_beta_gap"] and beta_corr >= thresholds["minimum_beta_oracle_correlation"]
    learned_fixed_ratios, post_ratios = [], []
    for seed in config["seeds"]:
        learned = next(r for r in drift_rows if r["seed"] == seed and r["method"] == "lambda_learned_cue")
        fixed = [r for r in drift_rows if r["seed"] == seed and r["method"].startswith("lambda_fixed_")]
        fixed_one = next(r for r in fixed if r["method"] == "lambda_fixed_1")
        learned_fixed_ratios.append(learned["normalized_mse"] / min(r["normalized_mse"] for r in fixed))
        post_ratios.append(learned["postchange_normalized_mse"] / fixed_one["postchange_normalized_mse"])
    learned_lambda = [r for r in drift_rows if r["method"] == "lambda_learned_cue"]
    stable_min = min(r["mean_stationary_lambda"] for r in learned_lambda); change_max = max(r["mean_change_lambda"] for r in learned_lambda); drift_corr = min(r["drift_forgetting_correlation"] for r in learned_lambda)
    checks["learned_lambda_beats_fixed_tradeoffs_every_seed"] = max(learned_fixed_ratios) <= thresholds["maximum_lambda_learned_to_best_fixed_risk"] and max(post_ratios) <= thresholds["maximum_lambda_postchange_to_fixed_one_risk"]
    checks["learned_lambda_tracks_drift"] = stable_min >= thresholds["minimum_stationary_lambda"] and change_max <= thresholds["maximum_change_lambda"] and drift_corr >= thresholds["minimum_drift_correlation"]
    observed.update(beta_risk_ratios=beta_ratios, minimum_clean_noisy_beta_gap=clean_noisy, minimum_noisy_bad_beta_gap=noisy_bad, minimum_beta_oracle_correlation=beta_corr, lambda_to_best_fixed_risk_ratios=learned_fixed_ratios, lambda_postchange_to_fixed_one_ratios=post_ratios, minimum_stationary_lambda=stable_min, maximum_change_lambda=change_max, minimum_drift_correlation=drift_corr)
    return {"passed": all(checks.values()), "checks": checks, "observed": observed, "thresholds": thresholds}


def joint_gate(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    ratios = []
    for seed in config["seeds"]:
        learned = next(r for r in rows if r["seed"] == seed and r["method"] == "joint_learned")
        fixed = next(r for r in rows if r["seed"] == seed and r["method"] == "joint_fixed")
        ratios.append(learned["normalized_mse"] / fixed["normalized_mse"])
    learned = [r for r in rows if r["method"] == "joint_learned"]
    checks = {
        "joint_gates_improve_fixed_every_seed": max(ratios) <= config["gate"]["maximum_joint_learned_to_fixed_risk"],
        "joint_beta_remains_ordered": min(r["mean_weight_clean"] - max(r["mean_weight_corrupted"], r["mean_weight_distractor"]) for r in learned) > 0,
        "joint_lambda_remains_drift_sensitive": min(r["mean_stationary_lambda"] - r["mean_change_lambda"] for r in learned) > 0 and min(r["drift_forgetting_correlation"] for r in learned) > 0,
    }
    return {"passed": all(checks.values()), "checks": checks, "observed": {"joint_learned_to_fixed_risk_ratios": ratios}}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def mean_by(rows: list[dict[str, Any]], method: str, metric: str) -> float:
    return float(np.mean([r[metric] for r in rows if r["method"] == method]))


def plots(beta: list[dict[str, Any]], drift: list[dict[str, Any]], joint: list[dict[str, Any]], directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True); paths = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(BETA_METHODS, [mean_by(beta, m, "normalized_mse") for m in BETA_METHODS]); axes[0].tick_params(axis="x", rotation=25); axes[0].set_ylabel("normalized MSE"); axes[0].set_title("Evidence-quality risk")
    learned = [r for r in beta if r["method"] == "beta_learned_precision"]
    axes[1].bar(CATEGORIES, [float(np.mean([r[f"mean_weight_{c}"] for r in learned])) for c in CATEGORIES]); axes[1].set_ylabel("learned beta"); axes[1].set_title("Learned evidence precision")
    fig.tight_layout(); path = directory / "evidence_precision.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(LAMBDA_METHODS, [mean_by(drift, m, "normalized_mse") for m in LAMBDA_METHODS]); axes[0].tick_params(axis="x", rotation=28); axes[0].set_ylabel("normalized MSE"); axes[0].set_title("Drift risk")
    for method in ("lambda_fixed_1", "lambda_learned_cue", "lambda_innovation_ablation", "lambda_oracle_change"):
        values = [np.fromstring(r["lambda_trace"], sep=";") for r in drift if r["method"] == method]
        offsets = np.fromstring(next(r["trace_offsets"] for r in drift if r["method"] == method), sep=";")
        axes[1].plot(offsets, np.mean(values, axis=0), marker="o", label=method)
    axes[1].axvline(0, color="black", linestyle="--"); axes[1].set(xlabel="steps from change", ylabel="lambda", title="Forgetting response"); axes[1].legend(fontsize=8)
    fig.tight_layout(); path = directory / "drift_lambda.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))
    fig, axis = plt.subplots(figsize=(7, 4.5)); axis.bar(JOINT_METHODS, [mean_by(joint, m, "normalized_mse") for m in JOINT_METHODS]); axis.set_ylabel("normalized MSE"); axis.set_title("Joint evidence and drift"); fig.tight_layout(); path = directory / "joint_gating.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))
    return paths


def table(headers: list[str], rows: Iterable[Iterable[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |", *("| " + " | ".join(row) + " |" for row in rows)])


def report(record: dict[str, Any]) -> str:
    separate, joint = record["separate_gate"], record["joint_gate"]
    checks = "\n".join(f"- {'PASS' if ok else 'FAIL'}: `{name}`" for name, ok in {**separate["checks"], **joint["checks"]}.items())
    beta_table = table(["method", "risk", "clean beta", "noisy beta", "corrupt beta", "distractor beta"], ([m, f"{mean_by(record['beta_rows'], m, 'normalized_mse'):.3f}", *(f"{mean_by(record['beta_rows'], m, f'mean_weight_{c}'):.3f}" for c in CATEGORIES)] for m in BETA_METHODS))
    drift_table = table(["method", "risk", "post-change risk", "stationary lambda", "change lambda", "drift corr"], ([m, f"{mean_by(record['drift_rows'], m, 'normalized_mse'):.3f}", f"{mean_by(record['drift_rows'], m, 'postchange_normalized_mse'):.3f}", f"{mean_by(record['drift_rows'], m, 'mean_stationary_lambda'):.3f}", f"{mean_by(record['drift_rows'], m, 'mean_change_lambda'):.3f}", f"{mean_by(record['drift_rows'], m, 'drift_forgetting_correlation'):.3f}"] for m in LAMBDA_METHODS))
    joint_table = table(["method", "risk", "clean beta", "bad beta", "stationary lambda", "change lambda"], ([m, f"{mean_by(record['joint_rows'], m, 'normalized_mse'):.3f}", f"{mean_by(record['joint_rows'], m, 'mean_weight_clean'):.3f}", f"{max(mean_by(record['joint_rows'], m, 'mean_weight_corrupted'), mean_by(record['joint_rows'], m, 'mean_weight_distractor')):.3f}", f"{mean_by(record['joint_rows'], m, 'mean_stationary_lambda'):.3f}", f"{mean_by(record['joint_rows'], m, 'mean_change_lambda'):.3f}"] for m in JOINT_METHODS))
    status = "PASS" if record["gate_passed"] else "FAIL"
    return f"""# Phase 7 learned Bayesian gates

## Gate decision: {status}

{checks}

All models use the successful Phase 6 shared key/query feature chart. The evidence and drift experiments are completed and gated separately before joint training begins.

## Evidence precision beta

{beta_table}

`beta_learned_precision` weights both sufficient statistics, as required by weighted least squares. `beta_generic_value` is an intentionally unconstrained control that weights only the cross-statistic. The gate sees noisy observable sensor-quality, consistency, and relevance cues—not the oracle category or target. Absolute beta has a global scale interaction with epsilon and the decoder, so ordering, oracle correlation, and risk are interpreted more strongly than raw magnitude.

## Drift lambda

{drift_table}

The primary learned gate sees a noisy, locally observable drift cue, so it remains a token-emitted quantity compatible with the affine scan. `lambda_innovation_ablation` instead sees the detached pre-write innovation and posterior uncertainty. Its failure is retained because a single innovation is not a changepoint posterior. Exact causal inference of hidden changes requires an additional run-length belief state; lambda is the action conditional on such evidence, not the detector itself. The oracle receives the true change point. Fixed values expose the stationary/adaptation tradeoff. This distinction follows Bayesian online changepoint detection, which explicitly maintains a posterior over run length, and variable-forgetting-factor RLS, which treats forgetting adaptation as an additional mechanism rather than a consequence of naming a scalar lambda.

## Joint identifiability

{joint_table}

In the joint experiment the lambda gate additionally sees evidence-quality cues, allowing it to distinguish a low-quality outlier from persistent operator drift. The tables report whether beta ordering and lambda drift sensitivity survive joint optimization; names alone are not treated as semantics.

## Failure study and mathematical basis

- Adams and MacKay, [Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742)
- Leung and So, [Gradient-based variable forgetting factor RLS](https://www.sciencedirect.com/science/article/pii/S0165168403000379)

The initial innovation-only run failed the lambda gate and correctly prevented joint training. A 120-step observable-cue probe recovered change response but missed the stationary-lambda threshold; the final run gives every drift and joint method the same 360-step budget without changing any gate threshold. These failed assumptions and probes are recorded in `EXPERIMENT_LOG.md`; their invalid partial artifacts are not gate inputs.

## Plots

- [`evidence_precision.png`](../plots/phase7/evidence_precision.png)
- [`drift_lambda.png`](../plots/phase7/drift_lambda.png)
- [`joint_gating.png`](../plots/phase7/joint_gating.png)

## Reproducibility

- commit at run start: `{record['git_commit']}`; dirty: `{record['working_tree_dirty']}`
- device: `{record['hardware']}`; Python `{record['software']['python']}`; PyTorch `{record['software']['torch']}`
- wall time: {record['wall_clock_seconds']:.2f}s; peak VRAM: {record['peak_vram_bytes'] / 2**30:.6f} GiB
- config: [`configs/phase7_gating.json`](../configs/phase7_gating.json)
- raw seed rows: [`phase7/`](phase7/)
- machine-readable record: [`phase7_metrics.json`](phase7_metrics.json)

## Scoped conclusion

{record['interpretation']}
"""


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=Path("configs/phase7_gating.json")); parser.add_argument("--results", type=Path, default=Path("results")); parser.add_argument("--plots", type=Path, default=Path("plots/phase7")); args = parser.parse_args()
    config = json.loads(args.config.read_text()); device = device_for(config["device"]); dirty = bool(git_output(["status", "--porcelain"])); started = time.perf_counter()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    beta_rows: list[dict[str, Any]] = []; drift_rows: list[dict[str, Any]] = []; joint_rows: list[dict[str, Any]] = []; curves: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        for method in BETA_METHODS:
            row, part = train_evidence(method, seed, config, device); beta_rows.append(row); curves.extend(part)
    for seed in config["seeds"]:
        for method in LAMBDA_METHODS:
            row, part = train_stream(method, seed, config, "drift", device); drift_rows.append(row); curves.extend(part)
    separate = separate_gate(beta_rows, drift_rows, config)
    if separate["passed"]:
        for seed in config["seeds"]:
            for method in JOINT_METHODS:
                row, part = train_stream(method, seed, config, "joint", device); joint_rows.append(row); curves.extend(part)
        joint = joint_gate(joint_rows, config)
    else:
        joint = {"passed": False, "checks": {"joint_runs_only_after_separate_gates": False}, "observed": {}}
    synchronize(device); args.results.mkdir(parents=True, exist_ok=True)
    write_csv(args.results / "phase7" / "evidence_seed_metrics.csv", beta_rows); write_csv(args.results / "phase7" / "drift_seed_metrics.csv", drift_rows); write_csv(args.results / "phase7" / "learning_curves.csv", curves)
    if joint_rows: write_csv(args.results / "phase7" / "joint_seed_metrics.csv", joint_rows)
    plot_paths = plots(beta_rows, drift_rows, joint_rows, args.plots) if joint_rows else []
    passed = separate["passed"] and joint["passed"]
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_output(["rev-parse", "HEAD"]), "working_tree_dirty": dirty, "config": config, "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(), "software": {"python": platform.python_version(), "torch": torch.__version__, "hip": torch.version.hip, "numpy": np.__version__}, "wall_clock_seconds": time.perf_counter() - started, "peak_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0, "beta_rows": beta_rows, "drift_rows": drift_rows, "joint_rows": joint_rows, "curves_count": len(curves), "plots": plot_paths, "separate_gate": separate, "joint_gate": joint, "gate_passed": passed, "interpretation": ("Learned precision and forgetting gates reproducibly improve risk over fixed gates while preserving the predicted reliability ordering and drift response; joint training retains both signals without claiming exact parameter identifiability." if passed else "The Phase 7 gating gate failed. No Bayesian semantic claim is accepted; the complete separate-experiment diagnostics are retained for assumption-level analysis before any correction.")}
    (args.results / "phase7_metrics.json").write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=True) + "\n"); (args.results / "phase7_gating.md").write_text(report(record) if joint_rows else "# Phase 7 learned Bayesian gates\n\nSeparate-gate failure; see `phase7_metrics.json`. Joint training was not run.\n")
    print(json.dumps({"separate_gate": separate, "joint_gate": joint, "passed": passed}, indent=2, sort_keys=True)); return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
