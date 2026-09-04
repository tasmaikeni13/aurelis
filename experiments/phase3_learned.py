"""Phase 3 experiment runner: Learned features and episodic routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor

from aurelis.curriculum import CurriculumBatch, CurriculumGenerator
from aurelis.nn_phase3 import (
    CumulativeLeastSquaresBlock,
    GatedDeltaBlock,
    LearnedAurelisBlock,
    LearnedSumBlock,
    LocalOnlyBlock,
    Phase3SequenceModel,
    RemoteOnlyBlock,
    compute_effective_rank,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def compute_auroc(scores: Tensor, targets: Tensor) -> float:
    scores = scores.flatten().float().cpu()
    targets = targets.flatten().float().cpu()
    pos = scores[targets == 1]
    neg = scores[targets == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    all_scores = torch.cat([pos, neg])
    ranks = torch.argsort(torch.argsort(all_scores)) + 1
    rank_pos = ranks[:n_pos].sum().item()
    u = rank_pos - (n_pos * (n_pos + 1)) / 2
    return float(u / (n_pos * n_neg))


def compute_r2(x: Tensor, y: Tensor) -> float:
    x = x.flatten().float().cpu()
    y = y.flatten().float().cpu()
    if len(x) < 2:
        return 0.0
    cov = torch.cov(torch.stack([x, y]))
    var_x = cov[0, 0]
    var_y = cov[1, 1]
    if var_x < 1e-8 or var_y < 1e-8:
        return 0.0
    r = cov[0, 1] / torch.sqrt(var_x * var_y)
    return float((r**2).item())


def instantiate_model(name: str, cfg: dict[str, Any], device: torch.device) -> Phase3SequenceModel:
    arch = cfg["architecture"]
    d_in = arch["d_in"]
    d_model = arch["d_model"]
    d_out = arch["d_out"]
    heads = arch["heads"]
    d_key = arch["d_key"]
    d_value = arch["d_value"]
    window = arch["window"]
    prior = arch["prior"]

    if name == "aurelis_e" or name == "analytic_plus_episodic_gate":
        block = LearnedAurelisBlock(
            d_model, heads, d_key, d_value, window, prior=prior,
            gate_mode="aurelis_e", shared_charts=True, learned_evidence=True
        )
    elif name == "aurelis_b" or name == "analytic_gate_no_override":
        block = LearnedAurelisBlock(
            d_model, heads, d_key, d_value, window, prior=prior,
            gate_mode="aurelis_b", shared_charts=True, learned_evidence=True
        )
    elif name == "local_only":
        block = LocalOnlyBlock(d_model, heads, d_key, d_value, window)
    elif name == "remote_only":
        block = RemoteOnlyBlock(d_model, heads, d_key, d_value, window, prior=prior)
    elif name == "learned_sum":
        block = LearnedSumBlock(d_model, heads, d_key, d_value, window, prior=prior)
    elif name == "gated_delta":
        block = GatedDeltaBlock(d_model, heads, d_key, d_value)
    elif name == "cumulative_least_squares":
        block = CumulativeLeastSquaresBlock(d_model, heads, d_key, d_value, prior=prior)
    elif name == "independent_charts":
        block = LearnedAurelisBlock(
            d_model, heads, d_key, d_value, window, prior=prior,
            gate_mode="aurelis_e", shared_charts=False, learned_evidence=True
        )
    elif name == "fixed_evidence":
        block = LearnedAurelisBlock(
            d_model, heads, d_key, d_value, window, prior=prior,
            gate_mode="aurelis_e", shared_charts=True, learned_evidence=False
        )
    elif name == "fixed_gate_0":
        block = LearnedAurelisBlock(
            d_model, heads, d_key, d_value, window, prior=prior,
            gate_mode="fixed_0", shared_charts=True, learned_evidence=True
        )
    elif name == "fixed_gate_1":
        block = LearnedAurelisBlock(
            d_model, heads, d_key, d_value, window, prior=prior,
            gate_mode="fixed_1", shared_charts=True, learned_evidence=True
        )
    elif name == "learned_sigmoid_gate_no_analytic":
        block = LearnedAurelisBlock(
            d_model, heads, d_key, d_value, window, prior=prior,
            gate_mode="learned_sigmoid", shared_charts=True, learned_evidence=True
        )
    elif name == "cache_overlap_double_counting":
        block = LearnedAurelisBlock(
            d_model, heads, d_key, d_value, window, prior=prior,
            gate_mode="aurelis_e", shared_charts=True, learned_evidence=True, cache_overlap=True
        )
    elif name == "random_frozen_features":
        block = LearnedAurelisBlock(
            d_model, heads, d_key, d_value, window, prior=prior,
            gate_mode="aurelis_e", shared_charts=True, learned_evidence=True, frozen_features=True
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

    return Phase3SequenceModel(d_in, d_model, d_out, block).to(device)


def train_model(
    model: Phase3SequenceModel,
    model_name: str,
    gen: CurriculumGenerator,
    cfg: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    train_cfg = cfg["training"]
    steps = train_cfg["steps"]
    b_size_per_task = train_cfg["batch_size_per_task"]
    length = train_cfg["train_length"]
    lr = train_cfg["learning_rate"]
    wd = train_cfg["weight_decay"]
    cue_loss_weight = train_cfg.get("cue_loss_weight", 0.2)

    if model_name == "random_frozen_features":
        return []

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.2)

    training_logs = []
    start_time = time.time()

    model.train()
    for step in range(steps):
        batch = gen.generate_balanced_batch(batch_size_per_task=b_size_per_task, length=length)
        optimizer.zero_grad()
        pred, diag = model(batch.x)

        task_loss = ((pred - batch.y) ** 2 * batch.mask.unsqueeze(-1)).sum() / batch.mask.sum()

        # Observable cue auxiliary objective for episodic router
        if "e_t" in diag and model_name in ("aurelis_e", "analytic_plus_episodic_gate", "independent_charts"):
            cue_feat = batch.x[:, -1, -1]  # Observable cue at query token
            e_t_query = diag["e_t"][:, :, -1].mean(dim=1)
            cue_loss = F.binary_cross_entropy(e_t_query, cue_feat)
            loss = task_loss + cue_loss_weight * cue_loss
        else:
            cue_loss = torch.tensor(0.0, device=device)
            loss = task_loss

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 50 == 0 or step == steps - 1:
            erank = diag.get("erank_kq", 0.0)
            training_logs.append({
                "step": step + 1,
                "task_loss": float(task_loss.item()),
                "cue_loss": float(cue_loss.item()),
                "total_loss": float(loss.item()),
                "grad_norm": float(grad_norm),
                "erank": float(erank),
                "elapsed_s": time.time() - start_time,
            })

    return training_logs


def evaluate_model(
    model: Phase3SequenceModel,
    model_name: str,
    gen: CurriculumGenerator,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    eval_cfg = cfg["evaluation"]
    batches_n = eval_cfg["eval_batches"]
    b_size = eval_cfg["eval_batch_size"]
    lengths = eval_cfg["held_out_lengths"]
    noise_stds = eval_cfg["held_out_noise_stds"]
    windows = eval_cfg["held_out_windows"]
    default_win = cfg["architecture"]["window"]

    model.eval()
    task_losses: dict[str, list[float]] = {
        "noisy_linear_regression": [],
        "recent_associative_copy": [],
        "remote_structured_recall": [],
        "mixed_latent_and_exception": [],
        "selective_copy_and_shift": [],
        "cache_boundary_recall": [],
        "overcapacity_conflicting_no_context_negatives": [],
    }

    # Task 4 breakdown
    exception_losses: list[float] = []
    latent_losses: list[float] = []
    gate_exceptions: list[float] = []
    gate_latents: list[float] = []
    e_t_exceptions: list[float] = []
    e_t_latents: list[float] = []
    all_e_t: list[Tensor] = []
    all_cues: list[Tensor] = []

    # Boundary ages
    boundary_losses: dict[int, list[float]] = {
        default_win - 1: [],
        default_win: [],
        default_win + 1: [],
        default_win + 2: [],
    }

    erank_val = 0.0

    with torch.no_grad():
        for b_idx in range(batches_n):
            eval_len = lengths[b_idx % len(lengths)]
            noise_std = noise_stds[b_idx % len(noise_stds)]
            eval_win = windows[b_idx % len(windows)]

            # 1. Noisy linear regression
            b1 = gen.generate_task1_noisy_linear(b_size, eval_len, noise_std=noise_std)
            p1, d1 = model(b1.x)
            l1 = ((p1 - b1.y) ** 2 * b1.mask.unsqueeze(-1)).sum() / b1.mask.sum()
            task_losses["noisy_linear_regression"].append(float(l1.item()))
            erank_val = d1.get("erank_kq", erank_val)

            # 2. Recent copy
            b2 = gen.generate_task2_recent_copy(b_size, eval_len, window=eval_win)
            p2, _ = model(b2.x)
            l2 = ((p2 - b2.y) ** 2 * b2.mask.unsqueeze(-1)).sum() / b2.mask.sum()
            task_losses["recent_associative_copy"].append(float(l2.item()))

            # 3. Remote recall
            b3 = gen.generate_task3_remote_recall(b_size, eval_len, rank=4, window=eval_win)
            p3, _ = model(b3.x)
            l3 = ((p3 - b3.y) ** 2 * b3.mask.unsqueeze(-1)).sum() / b3.mask.sum()
            task_losses["remote_structured_recall"].append(float(l3.item()))

            # 4. Mixed exception
            b4 = gen.generate_task4_mixed_exception(b_size, eval_len, window=eval_win)
            p4, d4 = model(b4.x)
            l4 = ((p4 - b4.y) ** 2 * b4.mask.unsqueeze(-1)).sum() / b4.mask.sum()
            task_losses["mixed_latent_and_exception"].append(float(l4.item()))

            cue = b4.metadata["observable_cue"]
            is_ex = cue == 1.0
            is_lat = cue == 0.0

            if is_ex.sum() > 0:
                l_ex = ((p4[is_ex, -1] - b4.y[is_ex, -1]) ** 2).mean()
                exception_losses.append(float(l_ex.item()))
            if is_lat.sum() > 0:
                l_lat = ((p4[is_lat, -1] - b4.y[is_lat, -1]) ** 2).mean()
                latent_losses.append(float(l_lat.item()))

            if "gate" in d4:
                g_query = d4["gate"][:, :, -1].mean(dim=1)
                if is_ex.sum() > 0:
                    gate_exceptions.append(float(g_query[is_ex].mean().item()))
                if is_lat.sum() > 0:
                    gate_latents.append(float(g_query[is_lat].mean().item()))

            if "e_t" in d4:
                e_query = d4["e_t"][:, :, -1].mean(dim=1)
                if is_ex.sum() > 0:
                    e_t_exceptions.append(float(e_query[is_ex].mean().item()))
                if is_lat.sum() > 0:
                    e_t_latents.append(float(e_query[is_lat].mean().item()))
                all_e_t.append(e_query.cpu())
                all_cues.append(cue.cpu())

            # 5. Selective copy
            b5 = gen.generate_task5_selective_copy(b_size, eval_len, window=eval_win)
            p5, _ = model(b5.x)
            l5 = ((p5 - b5.y) ** 2 * b5.mask.unsqueeze(-1)).sum() / b5.mask.sum()
            task_losses["selective_copy_and_shift"].append(float(l5.item()))

            # 6. Cache boundary recall
            for age in (default_win - 1, default_win, default_win + 1, default_win + 2):
                b6 = gen.generate_task6_cache_boundary(b_size, 32, age=age, window=default_win)
                p6, _ = model(b6.x)
                l6 = ((p6 - b6.y) ** 2 * b6.mask.unsqueeze(-1)).sum() / b6.mask.sum()
                boundary_losses[age].append(float(l6.item()))
            task_losses["cache_boundary_recall"].append(float(sum(boundary_losses[default_win].copy()) / len(boundary_losses[default_win])))

            # 7. Negatives
            b7 = gen.generate_task7_negatives(b_size, eval_len, window=eval_win)
            p7, _ = model(b7.x)
            l7 = ((p7 - b7.y) ** 2 * b7.mask.unsqueeze(-1)).sum() / b7.mask.sum()
            task_losses["overcapacity_conflicting_no_context_negatives"].append(float(l7.item()))

    # Summaries
    family_means = {k: float(sum(v) / len(v)) for k, v in task_losses.items()}
    aggregate_risk = float(sum(family_means.values()) / len(family_means))

    boundary_means = {age: float(sum(vals) / len(vals)) for age, vals in boundary_losses.items()}
    base_age = default_win - 1
    boundary_degradation = float(
        max(boundary_means[default_win], boundary_means[default_win + 1], boundary_means[default_win + 2]) - boundary_means[base_age]
    )

    auroc = 0.5
    cue_r2 = 0.0
    if all_e_t and all_cues:
        cat_e = torch.cat(all_e_t)
        cat_c = torch.cat(all_cues)
        auroc = compute_auroc(cat_e, cat_c)
        cue_r2 = compute_r2(cat_e, cat_c)

    return {
        "model_name": model_name,
        "family_means": family_means,
        "aggregate_risk": aggregate_risk,
        "erank": float(erank_val),
        "exception_mse": float(sum(exception_losses) / max(len(exception_losses), 1)),
        "latent_mse": float(sum(latent_losses) / max(len(latent_losses), 1)),
        "mean_gate_exception": float(sum(gate_exceptions) / max(len(gate_exceptions), 1)),
        "mean_gate_latent": float(sum(gate_latents) / max(len(gate_latents), 1)),
        "mean_et_exception": float(sum(e_t_exceptions) / max(len(e_t_exceptions), 1)),
        "mean_et_latent": float(sum(e_t_latents) / max(len(e_t_latents), 1)),
        "episodic_auroc": float(auroc),
        "cue_r2": float(cue_r2),
        "boundary_means": boundary_means,
        "boundary_degradation": boundary_degradation,
    }


def generate_plots(results: dict[str, Any], output_dir: Path) -> list[Path]:
    plots_dir = REPO_ROOT / "plots" / "phase3"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    # 1. Task Family Performance Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    families = list(results["seeds_summary"]["aurelis_e"]["family_means"].keys())
    short_families = [
        "1. Linear", "2. Copy", "3. Remote", "4. Mixed", "5. Select", "6. Bound", "7. Neg"
    ]
    compare_models = ["aurelis_e", "aurelis_b", "local_only", "remote_only", "learned_sum", "gated_delta", "random_frozen_features"]
    bar_width = 0.12
    x = range(len(families))

    for idx, m_name in enumerate(compare_models):
        if m_name in results["seeds_summary"]:
            means = [results["seeds_summary"][m_name]["family_means"][f] for f in families]
            offset = (idx - len(compare_models) / 2) * bar_width
            ax.bar([xi + offset for xi in x], means, bar_width, label=m_name)

    ax.set_xticks(list(x))
    ax.set_xticklabels(short_families)
    ax.set_ylabel("Mean Squared Error (Held-out Test)")
    ax.set_title("Performance Across 7 Task Families: Matched Models and Baselines")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    p1 = plots_dir / "task_family_performance.png"
    plt.savefig(p1, dpi=200)
    plt.close()
    generated_plots.append(p1)

    # 2. Episodic Cue Calibration and Override
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    e_summ = results["seeds_summary"]["aurelis_e"]
    b_summ = results["seeds_summary"]["aurelis_b"]

    categories = ["Latent Trend", "Episodic Exception"]
    e_mses = [e_summ["latent_mse"], e_summ["exception_mse"]]
    b_mses = [b_summ["latent_mse"], b_summ["exception_mse"]]
    bx = [0, 1]
    axes[0].bar([i - 0.15 for i in bx], b_mses, width=0.3, label="AURELIS-B (Analytic Only)", color="steelblue")
    axes[0].bar([i + 0.15 for i in bx], e_mses, width=0.3, label="AURELIS-E (Episodic Override)", color="crimson")
    axes[0].set_xticks(bx)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Exception vs Latent Reconstruction Error")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    gates_b = [b_summ["mean_gate_latent"], b_summ["mean_gate_exception"]]
    gates_e = [e_summ["mean_gate_latent"], e_summ["mean_gate_exception"]]
    axes[1].bar([i - 0.15 for i in bx], gates_b, width=0.3, label="AURELIS-B Gate", color="steelblue")
    axes[1].bar([i + 0.15 for i in bx], gates_e, width=0.3, label="AURELIS-E Gate", color="crimson")
    axes[1].set_xticks(bx)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel("Mean Gate Value")
    axes[1].set_title(f"Routing Gate Calibration (AUROC = {e_summ['episodic_auroc']:.3f})")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    p2 = plots_dir / "episodic_cue_calibration.png"
    plt.savefig(p2, dpi=200)
    plt.close()
    generated_plots.append(p2)

    # 3. Shared vs Independent Charts Effective Rank
    fig, ax = plt.subplots(figsize=(8, 5))
    shared_erank = results["seeds_summary"]["aurelis_e"]["erank"]
    indep_erank = results["seeds_summary"]["independent_charts"]["erank"]
    shared_risk = results["seeds_summary"]["aurelis_e"]["aggregate_risk"]
    indep_risk = results["seeds_summary"]["independent_charts"]["aggregate_risk"]

    models_bar = ["Shared Charts (AURELIS-E)", "Independent Charts (Wk != Wq)"]
    eranks = [shared_erank, indep_erank]
    risks = [shared_risk, indep_risk]

    x_pos = [0, 1]
    ax.bar(x_pos, eranks, width=0.4, color=["teal", "coral"], label="Effective Rank (erank)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models_bar)
    ax.set_ylabel("Effective Rank")
    ax.set_title(f"Shared vs Independent Charts (Shared Risk: {shared_risk:.3f}, Indep Risk: {indep_risk:.3f})")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    p3 = plots_dir / "shared_vs_independent_spectra.png"
    plt.savefig(p3, dpi=200)
    plt.close()
    generated_plots.append(p3)

    # 4. Cache-boundary continuity
    fig, ax = plt.subplots(figsize=(8, 5))
    win = results["config"]["architecture"]["window"]
    ages = [win - 1, win, win + 1, win + 2]
    age_labels = ["w-1 (Recent)", "w (Boundary)", "w+1 (Handoff)", "w+2 (Remote)"]

    aurelis_bnd = [results["seeds_summary"]["aurelis_e"]["boundary_means"][str(a)] for a in ages]
    local_bnd = [results["seeds_summary"]["local_only"]["boundary_means"][str(a)] for a in ages]
    remote_bnd = [results["seeds_summary"]["remote_only"]["boundary_means"][str(a)] for a in ages]

    ax.plot(range(4), aurelis_bnd, marker="o", linewidth=2, color="crimson", label="AURELIS-E (Continuous Handoff)")
    ax.plot(range(4), local_bnd, marker="s", linewidth=2, linestyle="--", color="gray", label="Local Only (Collapses past w)")
    ax.plot(range(4), remote_bnd, marker="^", linewidth=2, linestyle=":", color="steelblue", label="Remote Only")

    ax.set_xticks(range(4))
    ax.set_xticklabels(age_labels)
    ax.set_ylabel("MSE")
    ax.set_title("Memory Retention Across Delayed Handoff Boundary")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    p4 = plots_dir / "cache_boundary_continuity.png"
    plt.savefig(p4, dpi=200)
    plt.close()
    generated_plots.append(p4)

    return generated_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 learned neural models and evaluation.")
    parser.add_argument("--config", type=str, default="configs/phase3_learned.json")
    parser.add_argument("--output", type=str, default="results/phase3")
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    output_dir = REPO_ROOT / args.output
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = config["seeds"]
    model_names = config["models"] + config["ablations"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"=== Starting Phase 3: {len(model_names)} models x {len(seeds)} seeds on {device} ===")

    all_seed_results: dict[int, dict[str, dict[str, Any]]] = {}
    training_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        torch.manual_seed(seed)
        gen = CurriculumGenerator(
            d_in=config["architecture"]["d_in"],
            d_out=config["architecture"]["d_out"],
            d_feat=config["architecture"]["d_feat"],
            default_window=config["architecture"]["window"],
            seed=seed,
            device=device,
        )
        all_seed_results[seed] = {}

        for m_name in model_names:
            print(f"  Training model: {m_name} (seed {seed})...")
            model = instantiate_model(m_name, config, device)
            t_logs = train_model(model, m_name, gen, config, device)
            for t_log in t_logs:
                training_rows.append({"seed": seed, "model": m_name, **t_log})

            eval_res = evaluate_model(model, m_name, gen, config, device)
            all_seed_results[seed][m_name] = eval_res
            eval_rows.append({"seed": seed, **eval_res})
            print(f"    Done: Agg Risk = {eval_res['aggregate_risk']:.4f}, erank = {eval_res['erank']:.2f}")

    # Compute seeds summary across 5 seeds
    seeds_summary: dict[str, Any] = {}
    for m_name in model_names:
        agg_risks = [all_seed_results[s][m_name]["aggregate_risk"] for s in seeds]
        eranks = [all_seed_results[s][m_name]["erank"] for s in seeds]
        ex_mses = [all_seed_results[s][m_name]["exception_mse"] for s in seeds]
        lat_mses = [all_seed_results[s][m_name]["latent_mse"] for s in seeds]
        gate_exs = [all_seed_results[s][m_name]["mean_gate_exception"] for s in seeds]
        gate_lats = [all_seed_results[s][m_name]["mean_gate_latent"] for s in seeds]
        aurocs = [all_seed_results[s][m_name]["episodic_auroc"] for s in seeds]
        r2s = [all_seed_results[s][m_name]["cue_r2"] for s in seeds]
        degradations = [all_seed_results[s][m_name]["boundary_degradation"] for s in seeds]

        family_means: dict[str, float] = {}
        for fam in config["gates"]["family_thresholds"].keys():
            family_means[fam] = float(sum(all_seed_results[s][m_name]["family_means"][fam] for s in seeds) / len(seeds))

        boundary_means: dict[str, float] = {}
        win = config["architecture"]["window"]
        for age in (win - 1, win, win + 1, win + 2):
            boundary_means[str(age)] = float(sum(all_seed_results[s][m_name]["boundary_means"][age] for s in seeds) / len(seeds))

        seeds_summary[m_name] = {
            "aggregate_risk": float(sum(agg_risks) / len(agg_risks)),
            "erank": float(sum(eranks) / len(eranks)),
            "exception_mse": float(sum(ex_mses) / len(ex_mses)),
            "latent_mse": float(sum(lat_mses) / len(lat_mses)),
            "mean_gate_exception": float(sum(gate_exs) / len(gate_exs)),
            "mean_gate_latent": float(sum(gate_lats) / len(gate_lats)),
            "episodic_auroc": float(sum(aurocs) / len(aurocs)),
            "cue_r2": float(sum(r2s) / len(r2s)),
            "boundary_means": boundary_means,
            "boundary_degradation": float(sum(degradations) / len(degradations)),
            "family_means": family_means,
        }

    # PASS Gate Checks
    gate_thresholds = config["gates"]["family_thresholds"]
    gate1_pass = True
    for s in seeds:
        for fam, thresh in gate_thresholds.items():
            if all_seed_results[s]["aurelis_e"]["family_means"][fam] > thresh:
                gate1_pass = False

    # Gate 2: Improves over frozen random features for every seed
    gate2_pass = all(
        all_seed_results[s]["aurelis_e"]["aggregate_risk"] < all_seed_results[s]["random_frozen_features"]["aggregate_risk"]
        for s in seeds
    )

    # Gate 3: Shared chart beats independent chart and retains usable erank
    gate3_pass = (
        seeds_summary["aurelis_e"]["aggregate_risk"] < seeds_summary["independent_charts"]["aggregate_risk"]
        and seeds_summary["aurelis_e"]["erank"] >= config["gates"]["effective_rank_min"]
    )

    # Gate 4: AURELIS-B calibrated; AURELIS-E materially improves exact exception copy without unacceptable anti-copy degradation
    e_ex = seeds_summary["aurelis_e"]["exception_mse"]
    b_ex = seeds_summary["aurelis_b"]["exception_mse"]
    e_lat = seeds_summary["aurelis_e"]["latent_mse"]
    b_lat = seeds_summary["aurelis_b"]["latent_mse"]
    ex_ratio = b_ex / max(e_ex, 1e-6)
    anticopy_deg = max(0.0, e_lat - b_lat)
    gate4_pass = (
        ex_ratio >= config["gates"]["exception_copy_improvement_ratio_min"]
        and anticopy_deg <= config["gates"]["anticopy_degradation_max"]
    )

    # Gate 5: Observable cue explains override (AUROC >= 0.90, R2 >= 0.80)
    gate5_pass = (
        seeds_summary["aurelis_e"]["episodic_auroc"] >= config["gates"]["episodic_auroc_min"]
        and seeds_summary["aurelis_e"]["cue_r2"] >= config["gates"]["cue_correlation_r2_min"]
    )

    # Gate 6: Handoff boundary degradation within tolerance
    gate6_pass = seeds_summary["aurelis_e"]["boundary_degradation"] <= config["gates"]["boundary_degradation_max"]

    # Gate 7: All seeds reported and finite
    gate7_pass = len(seeds) >= 5 and all(
        math.isfinite(all_seed_results[s]["aurelis_e"]["aggregate_risk"]) for s in seeds
    )

    all_gates_pass = (
        gate1_pass and gate2_pass and gate3_pass and gate4_pass and gate5_pass and gate6_pass and gate7_pass
    )

    results_data = {
        "status": "PASS" if all_gates_pass else "FAIL",
        "timestamp": datetime.now(UTC).isoformat(),
        "config": config,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "seeds": seeds,
        "checks": {
            "gate_1_solves_every_task_family": gate1_pass,
            "gate_2_improves_over_frozen_random_features": gate2_pass,
            "gate_3_shared_beats_independent_charts": gate3_pass,
            "gate_4_aurelis_e_exception_isolated_from_bayes": gate4_pass,
            "gate_5_observable_cue_explains_override": gate5_pass,
            "gate_6_handoff_boundary_within_tolerance": gate6_pass,
            "gate_7_all_seeds_reported_finite": gate7_pass,
        },
        "summary": {
            "aurelis_e_aggregate_risk": seeds_summary["aurelis_e"]["aggregate_risk"],
            "frozen_aggregate_risk": seeds_summary["random_frozen_features"]["aggregate_risk"],
            "independent_charts_aggregate_risk": seeds_summary["independent_charts"]["aggregate_risk"],
            "shared_erank": seeds_summary["aurelis_e"]["erank"],
            "independent_erank": seeds_summary["independent_charts"]["erank"],
            "exception_improvement_ratio": float(ex_ratio),
            "anticopy_degradation": float(anticopy_deg),
            "episodic_auroc": seeds_summary["aurelis_e"]["episodic_auroc"],
            "cue_r2": seeds_summary["aurelis_e"]["cue_r2"],
            "boundary_degradation": seeds_summary["aurelis_e"]["boundary_degradation"],
        },
        "seeds_summary": seeds_summary,
        "seed_details": all_seed_results,
    }

    # Write metrics.json
    (output_dir / "metrics.json").write_text(json.dumps(results_data, indent=2) + "\n", encoding="utf-8")

    # Write raw rows
    with (raw_dir / "models_training.jsonl").open("w", encoding="utf-8") as f:
        for row in training_rows:
            f.write(json.dumps(row) + "\n")

    with (raw_dir / "evaluation_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row) + "\n")

    # Generate plots
    print("Generating plots...")
    generate_plots(results_data, output_dir)

    print(f"\n=== Experiment Finished. Status: {results_data['status']} ===")
    for gate, passed in results_data["checks"].items():
        print(f"  {gate}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
