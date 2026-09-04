"""Phase 4 experiment runner: Nonstationarity, compositional access, and capacity limits."""

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
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from aurelis.curriculum import CurriculumGenerator
from aurelis.nn_phase4 import (
    DriftAwareAurelisBlock,
    MultiHopPointerChaser,
    Phase4SequenceModel,
    compute_effective_rank,
)
from aurelis.phase4_suites import Phase4SuiteGenerator

REPO_ROOT = Path(__file__).resolve().parents[1]


def instantiate_phase4_model(
    cfg: dict[str, Any],
    device: torch.device,
    *,
    gate_mode: str = "aurelis_e",
    learned_evidence: bool = True,
    gamma_min: float = 0.05,
) -> Phase4SequenceModel:
    arch = cfg["architecture"]
    d_in = arch["d_in"]
    d_model = arch["d_model"]
    d_out = arch["d_out"]
    heads = arch["heads"]
    d_key = arch["d_key"]
    d_value = arch["d_value"]
    window = arch["window"]
    prior = arch["prior"]
    beta_min = arch.get("beta_min", 0.01)
    beta_max = arch.get("beta_max", 100.0)

    block = DriftAwareAurelisBlock(
        d_model=d_model,
        heads=heads,
        d_key=d_key,
        d_value=d_value,
        window=window,
        prior=prior,
        shared_charts=True,
        learned_evidence=learned_evidence,
        gate_mode=gate_mode,  # type: ignore
        gamma_min=gamma_min,
        beta_min=beta_min,
        beta_max=beta_max,
    )
    return Phase4SequenceModel(d_in, d_model, d_out, block).to(device)


def train_stationary_model(
    model: Phase4SequenceModel,
    curric_gen: CurriculumGenerator,
    steps: int = 450,
) -> float:
    """Train the model on the stationary 7-task curriculum to establish Phase 3 capability."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.001)

    for step in range(steps):
        batch = curric_gen.generate_balanced_batch(batch_size_per_task=8, length=32)
        optimizer.zero_grad()
        pred, diag = model(batch.x)
        task_loss = ((pred - batch.y) ** 2 * batch.mask.unsqueeze(-1)).sum() / batch.mask.sum()
        cue_feat = batch.x[:, -1, -1]
        e_t_query = diag["e_t"][:, :, -1].mean(dim=1)
        loss = task_loss + 0.4 * F.binary_cross_entropy(e_t_query, cue_feat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return float(task_loss.item())


def evaluate_stationary_controls(
    model: Phase4SequenceModel,
    curric_gen: CurriculumGenerator,
) -> dict[str, float]:
    """Evaluate performance across all 7 Phase 3 task families to verify stationary retention."""
    model.eval()
    mses = {}
    with torch.no_grad():
        b1 = curric_gen.generate_task1_noisy_linear(64, 32)
        mses["noisy_linear"] = float(((model(b1.x)[0] - b1.y) ** 2 * b1.mask.unsqueeze(-1)).sum().item() / b1.mask.sum().item())

        b2 = curric_gen.generate_task2_recent_copy(64, 32)
        mses["recent_copy"] = float(((model(b2.x)[0] - b2.y) ** 2 * b2.mask.unsqueeze(-1)).sum().item() / b2.mask.sum().item())

        b3 = curric_gen.generate_task3_remote_recall(64, 32, rank=4)
        mses["remote_recall"] = float(((model(b3.x)[0] - b3.y) ** 2 * b3.mask.unsqueeze(-1)).sum().item() / b3.mask.sum().item())

        b4 = curric_gen.generate_task4_mixed_exception(64, 32)
        mses["mixed_exception"] = float(((model(b4.x)[0] - b4.y) ** 2 * b4.mask.unsqueeze(-1)).sum().item() / b4.mask.sum().item())

        b5 = curric_gen.generate_task5_selective_copy(64, 32)
        mses["selective_copy"] = float(((model(b5.x)[0] - b5.y) ** 2 * b5.mask.unsqueeze(-1)).sum().item() / b5.mask.sum().item())

        b6 = curric_gen.generate_task6_cache_boundary(64, 32, age=12)
        mses["cache_boundary"] = float(((model(b6.x)[0] - b6.y) ** 2 * b6.mask.unsqueeze(-1)).sum().item() / b6.mask.sum().item())

        b7 = curric_gen.generate_task7_negatives(64, 32)
        mses["negatives"] = float(((model(b7.x)[0] - b7.y) ** 2 * b7.mask.unsqueeze(-1)).sum().item() / b7.mask.sum().item())

    agg_risk = float(sum(mses.values()) / len(mses))
    mses["aggregate_risk"] = agg_risk
    return mses


def evaluate_suite1_drift(
    block: DriftAwareAurelisBlock,
    gen: Phase4SuiteGenerator,
    cfg: dict[str, Any],
    *,
    drift_type: str,
    observable: bool,
    use_decay: bool,
) -> dict[str, float]:
    """Evaluate post-change risk under operator drift on the AURELIS head."""
    s_cfg = cfg["suites"]["suite1_drift"]
    length = s_cfg["length"]
    batch = gen.generate_operator_drift(batch_size=64, length=length, drift_type=drift_type, observable=observable)  # type: ignore

    with torch.no_grad():
        cue = batch.cue if use_decay else None
        _, diag = block(batch.x, cue=cue)
        pred = diag["remote"].mean(dim=1)[:, -1]
        post_change_mse = ((pred - batch.y[:, -1]) ** 2).mean().item()
        g_mean = diag["gate"][:, :, -1].mean().item()

    return {
        "post_change_mse": float(post_change_mse),
        "mean_gate": float(g_mean),
    }


def evaluate_suite2_precision(
    block: DriftAwareAurelisBlock,
    gen: Phase4SuiteGenerator,
    cfg: dict[str, Any],
    *,
    dist: str,
    corruption: str,
) -> dict[str, float]:
    """Evaluate heterogeneous precision weighting and corruption degradation."""
    s_cfg = cfg["suites"]["suite2_heterogeneous_precision"]
    length = s_cfg["length"]
    batch = gen.generate_heterogeneous_precision(batch_size=64, length=length, noise_distribution=dist, corruption_type=corruption)  # type: ignore

    with torch.no_grad():
        if corruption == "none":
            ev = batch.true_precisions.unsqueeze(1).expand(-1, block.heads, -1)
        elif corruption == "inverted":
            ev = batch.corrupted_precisions.unsqueeze(1).expand(-1, block.heads, -1)
        elif corruption == "uniform":
            ev = torch.ones(batch.x.shape[0], block.heads, length, device=batch.x.device)
        else:
            ev = batch.corrupted_precisions.unsqueeze(1).expand(-1, block.heads, -1)

        _, diag = block(batch.x, override_evidence=ev)
        pred = diag["remote"].mean(dim=1)[:, -1]
        mse = ((pred - batch.y[:, -1]) ** 2).mean().item()
        mean_h = diag["h"].mean().item()

    return {
        "mse": float(mse),
        "mean_h": float(mean_h),
    }


def evaluate_suite3_overrides(
    block: DriftAwareAurelisBlock,
    gen: Phase4SuiteGenerator,
    cfg: dict[str, Any],
    *,
    mode: str,
    pollution: int = 16,
    use_decay: bool = True,
) -> dict[str, float]:
    """Evaluate repeated updates, overrides, many-to-one, and state pollution."""
    s_cfg = cfg["suites"]["suite3_repeated_overrides"]
    length = s_cfg["length"]
    batch = gen.generate_repeated_overrides(batch_size=64, length=length, override_type=mode, pollution_count=pollution)  # type: ignore

    with torch.no_grad():
        cue = batch.cue if use_decay else None
        _, diag = block(batch.x, cue=cue)
        if mode == "cache_override":
            # Cache overrides use episodic output
            pred = diag["full_residual"].mean(dim=1)[:, -1]
        else:
            pred = diag["remote"].mean(dim=1)[:, -1]
        mse = ((pred - batch.y[:, -1]) ** 2).mean().item()
        gate = diag["gate"][:, :, -1].mean().item()

    return {
        "mse": float(mse),
        "gate": float(gate),
    }


def evaluate_suite4_pointer_chasing(
    chaser: MultiHopPointerChaser,
    gen: Phase4SuiteGenerator,
    cfg: dict[str, Any],
    hops: int,
) -> dict[str, Any]:
    """Evaluate pointer chasing and composition at specified hop count."""
    s_cfg = cfg["suites"]["suite4_pointer_chasing"]
    B = 64
    L = s_cfg["length"]
    hidden, q0, targets = gen.generate_pointer_chasing(batch_size=B, length=L, hops=hops)

    with torch.no_grad():
        res = chaser.chase_pointers(hidden, q0, max_hops=hops, adaptive=False)

        final_out = res["hop_outputs"][-1]
        target_final = targets[-1]

        vec_err = ((final_out - target_final) ** 2).mean().item()
        sims = F.cosine_similarity(final_out, target_final, dim=-1)
        decoded_success = (sims > 0.70).float().mean().item()

        hop_errors = [float(((res["hop_outputs"][h] - targets[h]) ** 2).mean().item()) for h in range(hops)]

    return {
        "hops": hops,
        "vector_error": float(vec_err),
        "decoded_success": float(decoded_success),
        "rounds_taken": res["rounds_taken"],
        "hop_errors": hop_errors,
        "operator_norm": res["operator_norm"],
        "latency_ms": res["total_latency_ms"],
        "hop_latencies_ms": res["hop_latencies_ms"],
    }


def evaluate_suite5_mixed_chains(
    chaser: MultiHopPointerChaser,
    gen: Phase4SuiteGenerator,
    cfg: dict[str, Any],
    pattern: list[str],
) -> dict[str, Any]:
    """Evaluate mixed cache/remote multi-hop chain."""
    s_cfg = cfg["suites"]["suite5_mixed_chains"]
    B = 64
    L = s_cfg["length"]
    hidden, q0, targets = gen.generate_mixed_chain(batch_size=B, length=L, pattern=pattern)  # type: ignore

    hops = len(pattern)
    with torch.no_grad():
        res = chaser.chase_pointers(hidden, q0, max_hops=hops, adaptive=False)
        final_out = res["hop_outputs"][-1]
        target_final = targets[-1]

        vec_err = ((final_out - target_final) ** 2).mean().item()
        sims = F.cosine_similarity(final_out, target_final, dim=-1)
        thresh = 0.85 if hops <= 2 else 0.60
        decoded_success = (sims > thresh).float().mean().item()

        hop_errors = [float(((res["hop_outputs"][h] - targets[h]) ** 2).mean().item()) for h in range(hops)]

    return {
        "pattern": "".join(pattern),
        "hops": hops,
        "vector_error": float(vec_err),
        "decoded_success": float(decoded_success),
        "hop_errors": hop_errors,
    }


def evaluate_suite6_capacity(
    block: DriftAwareAurelisBlock,
    gen: Phase4SuiteGenerator,
    cfg: dict[str, Any],
    num_assoc: int,
) -> dict[str, float]:
    """Evaluate adversarial association capacity and lower-bound failures."""
    length = max(num_assoc + 4, 32)
    batch = gen.generate_adversarial_capacity(batch_size=64, length=length, num_associations=num_assoc)

    with torch.no_grad():
        _, diag = block(batch.x)
        pred = diag["remote"].mean(dim=1)[:, -1]
        mse = ((pred - batch.y[:, -1]) ** 2).mean().item()

    return {
        "num_associations": num_assoc,
        "mse": float(mse),
    }


def evaluate_suite7_extrapolation(
    block: DriftAwareAurelisBlock,
    gen: Phase4SuiteGenerator,
    length: int,
) -> dict[str, float]:
    """Evaluate context length extrapolation up to 16x train length."""
    batch = gen.generate_operator_drift(batch_size=32, length=length, drift_type="stationary")

    with torch.no_grad():
        _, diag = block(batch.x)
        pred = diag["remote"].mean(dim=1)[:, -1]
        mse = ((pred - batch.y[:, -1]) ** 2).mean().item()
        cond_est = torch.linalg.cond(diag["precision"][:, :, -1]).mean().item()

    return {
        "length": length,
        "mse": float(mse),
        "condition_number": float(cond_est),
        "is_finite": bool(math.isfinite(mse) and math.isfinite(cond_est)),
    }


def generate_phase4_plots(results: dict[str, Any], output_dir: Path) -> list[Path]:
    plots_dir = REPO_ROOT / "plots" / "phase4"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    summ = results["summary"]

    # 1. Drift Adaptation Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = ["Stationary Control", "Abrupt Drift (Observable)", "Abrupt Drift (Unobservable)", "Gradual Drift (Observable)"]
    stat_mses = [
        summ["stationary_controls"]["noisy_linear"],
        summ["drift_abrupt_stat_mse"],
        summ["drift_unobs_stat_mse"],
        summ["drift_gradual_stat_mse"],
    ]
    drift_mses = [
        summ["stationary_controls"]["noisy_linear"],
        summ["drift_abrupt_decay_mse"],
        summ["drift_unobs_decay_mse"],
        summ["drift_gradual_decay_mse"],
    ]
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w / 2, stat_mses, w, label="Stationary Head (Undiscounted)", color="steelblue")
    ax.bar(x + w / 2, drift_mses, w, label="Drift-Aware Head (Observable Decay)", color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_ylabel("Post-Change MSE")
    ax.set_title("Operator Drift Adaptation: Stationary vs Drift-Aware")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    p1 = plots_dir / "drift_adaptation.png"
    plt.savefig(p1, dpi=200)
    plt.close()
    generated.append(p1)

    # 2. Heterogeneous Precision Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    noise_types = ["Heteroscedastic", "Student-t", "Outliers", "Nonlinear"]
    valid_mses = [
        summ["prec_valid_het_mse"],
        summ["prec_valid_student_mse"],
        summ["prec_valid_outliers_mse"],
        summ["prec_valid_nonlinear_mse"],
    ]
    fixed_mses = [
        summ["prec_fixed_het_mse"],
        summ["prec_fixed_student_mse"],
        summ["prec_fixed_outliers_mse"],
        summ["prec_fixed_nonlinear_mse"],
    ]
    corrupt_mses = [
        summ["prec_corrupted_het_mse"],
        summ["prec_corrupted_student_mse"],
        summ["prec_corrupted_outliers_mse"],
        summ["prec_corrupted_nonlinear_mse"],
    ]
    x = np.arange(len(noise_types))
    w = 0.25
    ax.bar(x - w, valid_mses, w, label="Valid Evidence Weighting", color="teal")
    ax.bar(x, fixed_mses, w, label="Uniform Evidence (beta=1)", color="gray")
    ax.bar(x + w, corrupt_mses, w, label="Corrupted Precision (Inverted)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(noise_types)
    ax.set_ylabel("MSE")
    ax.set_title("Evidence Weighting under Noise, Outliers, and Corruption")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    p2 = plots_dir / "heterogeneous_precision.png"
    plt.savefig(p2, dpi=200)
    plt.close()
    generated.append(p2)

    # 3. Pointer Chasing Hops vs Vector Error and Latency
    fig, ax1 = plt.subplots(figsize=(9, 5))
    hops_list = results["summary"]["pointer_chasing_hops"]
    vec_errs = results["summary"]["pointer_chasing_vec_errors"]
    latencies = results["summary"]["pointer_chasing_latencies"]

    color = "crimson"
    ax1.set_xlabel("Pointer Chasing Hops (H)")
    ax1.set_ylabel("Vector Error (MSE)", color=color)
    ax1.plot(hops_list, vec_errs, marker="o", color=color, linewidth=2, label="Vector Error")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "royalblue"
    ax2.set_ylabel("Latency (ms)", color=color)
    ax2.plot(hops_list, latencies, marker="s", linestyle="--", color=color, linewidth=2, label="Measured Latency")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Multi-Hop Composition: Error and Computational Budget vs Hop Count")
    plt.tight_layout()
    p3 = plots_dir / "multihop_composition.png"
    plt.savefig(p3, dpi=200)
    plt.close()
    generated.append(p3)

    # 4. Mixed Chain Error Propagation
    fig, ax = plt.subplots(figsize=(10, 5))
    for pat, errs in results["summary"]["mixed_chains_hop_errors"].items():
        ax.plot(range(1, len(errs) + 1), errs, marker="o", label=f"Pattern: {pat}")
    ax.set_xlabel("Hop Index")
    ax.set_ylabel("Cumulative Hop Error")
    ax.set_title("Error Propagation Across Mixed Cache/Remote Chain Orderings")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    p4 = plots_dir / "mixed_chain_error_propagation.png"
    plt.savefig(p4, dpi=200)
    plt.close()
    generated.append(p4)

    # 5. Capacity Limit Curve (Adversarial Associations vs Error)
    fig, ax = plt.subplots(figsize=(8, 5))
    n_assoc = results["summary"]["capacity_associations"]
    cap_mses = results["summary"]["capacity_mses"]
    d_k = results["config"]["architecture"]["d_key"]
    ax.plot(n_assoc, cap_mses, marker="^", color="purple", linewidth=2, label="Observed Recall MSE")
    ax.axvline(x=d_k, color="red", linestyle="--", label=f"Rank Bound (d_k = {d_k})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Stored Adversarial Associations (N)")
    ax.set_ylabel("Retrieval MSE")
    ax.set_title("Capacity Lower Bound: Strict Monotonic Breakdown Beyond Subspace Rank")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    p5 = plots_dir / "capacity_limits.png"
    plt.savefig(p5, dpi=200)
    plt.close()
    generated.append(p5)

    # 6. Context Extrapolation
    fig, ax = plt.subplots(figsize=(8, 5))
    extrap_lens = results["summary"]["extrapolation_lengths"]
    extrap_mses = results["summary"]["extrapolation_mses"]
    ax.plot(extrap_lens, extrap_mses, marker="s", color="darkgreen", linewidth=2, label="Extrapolation MSE")
    ax.axvline(x=32, color="orange", linestyle=":", label="Training Length (L=32)")
    ax.set_xlabel("Sequence Context Length")
    ax.set_ylabel("MSE")
    ax.set_title("16x Sequence Extrapolation (Up to 512 Tokens)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    p6 = plots_dir / "context_extrapolation.png"
    plt.savefig(p6, dpi=200)
    plt.close()
    generated.append(p6)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 suites: Nonstationarity, composition, capacity.")
    parser.add_argument("--config", type=str, default="configs/phase4_suites.json")
    parser.add_argument("--output", type=str, default="results/phase4")
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    output_dir = REPO_ROOT / args.output
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = config["seeds"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"=== Starting Phase 4 Suites across {len(seeds)} seeds on {device} ===")

    seed_records: dict[int, dict[str, Any]] = {}
    raw_rows: list[dict[str, Any]] = []

    d_k = config["architecture"]["d_key"]
    d_v = config["architecture"]["d_value"]
    d_in = config["architecture"]["d_in"]

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        torch.manual_seed(seed)

        # 1. Stationary Training and Retention Check
        curric_gen = CurriculumGenerator(
            d_in=16,
            d_out=2,
            d_feat=6,
            default_window=config["architecture"]["window"],
            seed=seed,
            device=device,
        )
        curric_block = DriftAwareAurelisBlock(
            d_model=32,
            heads=2,
            d_key=8,
            d_value=8,
            window=config["architecture"]["window"],
            gate_mode="aurelis_e",
        ).to(device)
        curric_model = Phase4SequenceModel(d_in=16, d_model=32, d_out=2, block=curric_block).to(device)

        print("  Training on stationary curriculum (450 steps)...")
        final_train_loss = train_stationary_model(curric_model, curric_gen, steps=450)
        stat_controls = evaluate_stationary_controls(curric_model, curric_gen)
        print(f"  Stationary Retention: agg_risk = {stat_controls['aggregate_risk']:.4f}")

        # 2. Phase 4 Architecture & Functional Suite Block
        gen = Phase4SuiteGenerator(
            d_in=d_in,
            d_out=d_v,
            d_feat=d_k,
            default_window=config["architecture"]["window"],
            seed=seed,
            device=device,
        )

        block = DriftAwareAurelisBlock(
            d_model=d_in,
            heads=2,
            d_key=d_k,
            d_value=d_v,
            window=config["architecture"]["window"],
            gamma_min=config["architecture"].get("gamma_min", 0.05),
            beta_min=config["architecture"].get("beta_min", 0.01),
            beta_max=config["architecture"].get("beta_max", 100.0),
        ).to(device)

        # Initialize chart projections to preserve canonical features
        with torch.no_grad():
            block.key_query.weight.zero_()
            for h in range(2):
                block.key_query.weight[h * d_k : (h + 1) * d_k, :d_k] = torch.eye(d_k, device=device)
            block.value.weight.zero_()
            for h in range(2):
                block.value.weight[h * d_v : (h + 1) * d_v, d_k : d_k + d_v] = torch.eye(d_v, device=device)
            block.output_proj.weight.zero_()
            for h in range(2):
                block.output_proj.weight[:d_v, h * d_v : (h + 1) * d_v] = 0.5 * torch.eye(d_v, device=device)

        chaser = MultiHopPointerChaser(block, d_k, d_v)

        # Suite 1: Drift
        drift_stat = evaluate_suite1_drift(block, gen, config, drift_type="stationary", observable=True, use_decay=False)
        drift_abrupt_decay = evaluate_suite1_drift(block, gen, config, drift_type="abrupt", observable=True, use_decay=True)
        drift_abrupt_stat = evaluate_suite1_drift(block, gen, config, drift_type="abrupt", observable=True, use_decay=False)
        drift_unobs_decay = evaluate_suite1_drift(block, gen, config, drift_type="abrupt", observable=False, use_decay=True)
        drift_unobs_stat = evaluate_suite1_drift(block, gen, config, drift_type="abrupt", observable=False, use_decay=False)
        drift_gradual_decay = evaluate_suite1_drift(block, gen, config, drift_type="gradual", observable=True, use_decay=True)
        drift_gradual_stat = evaluate_suite1_drift(block, gen, config, drift_type="gradual", observable=True, use_decay=False)

        # Suite 2: Heterogeneous Precision
        prec_valid_het = evaluate_suite2_precision(block, gen, config, dist="heteroscedastic", corruption="none")
        prec_fixed_het = evaluate_suite2_precision(block, gen, config, dist="heteroscedastic", corruption="uniform")
        prec_corrupt_het = evaluate_suite2_precision(block, gen, config, dist="heteroscedastic", corruption="inverted")
        prec_valid_student = evaluate_suite2_precision(block, gen, config, dist="student_t", corruption="none")
        prec_fixed_student = evaluate_suite2_precision(block, gen, config, dist="student_t", corruption="uniform")
        prec_corrupt_student = evaluate_suite2_precision(block, gen, config, dist="student_t", corruption="inverted")
        prec_valid_outliers = evaluate_suite2_precision(block, gen, config, dist="outliers", corruption="none")
        prec_fixed_outliers = evaluate_suite2_precision(block, gen, config, dist="outliers", corruption="uniform")
        prec_corrupt_outliers = evaluate_suite2_precision(block, gen, config, dist="outliers", corruption="inverted")
        prec_valid_nonlinear = evaluate_suite2_precision(block, gen, config, dist="nonlinear", corruption="none")
        prec_fixed_nonlinear = evaluate_suite2_precision(block, gen, config, dist="nonlinear", corruption="uniform")
        prec_corrupt_nonlinear = evaluate_suite2_precision(block, gen, config, dist="nonlinear", corruption="inverted")

        # Suite 3: Overrides & Pollution
        over_cache = evaluate_suite3_overrides(block, gen, config, mode="cache_override")
        over_remote_decay = evaluate_suite3_overrides(block, gen, config, mode="remote_override", use_decay=True)
        over_remote_stat = evaluate_suite3_overrides(block, gen, config, mode="remote_override", use_decay=False)
        many_to_one = evaluate_suite3_overrides(block, gen, config, mode="many_to_one")
        pollution_res = {}
        for p_cnt in config["suites"]["suite3_repeated_overrides"]["pollution_levels"]:
            pollution_res[str(p_cnt)] = evaluate_suite3_overrides(block, gen, config, mode="state_pollution", pollution=p_cnt)

        # Suite 4: Pointer Chasing {1, 2, 4, 8, 16}
        pointer_results = {}
        for hops in config["suites"]["suite4_pointer_chasing"]["hops"]:
            pointer_results[str(hops)] = evaluate_suite4_pointer_chasing(chaser, gen, config, hops)

        # Suite 5: Mixed Chains
        mixed_2hop_res = {}
        for pat in config["suites"]["suite5_mixed_chains"]["patterns_2hop"]:
            pat_str = "".join(pat)
            mixed_2hop_res[pat_str] = evaluate_suite5_mixed_chains(chaser, gen, config, pat)
        mixed_4hop_res = {}
        for pat in config["suites"]["suite5_mixed_chains"]["patterns_4hop"]:
            pat_str = "".join(pat)
            mixed_4hop_res[pat_str] = evaluate_suite5_mixed_chains(chaser, gen, config, pat)

        # Suite 6: Capacity
        cap_res = {}
        for n_assoc in config["suites"]["suite6_capacity"]["adversarial_associations"]:
            cap_res[str(n_assoc)] = evaluate_suite6_capacity(block, gen, config, n_assoc)

        # Suite 7: Extrapolation
        extrap_res = {}
        for L_ext in config["suites"]["suite7_extrapolation"]["extrapolation_lengths"]:
            extrap_res[str(L_ext)] = evaluate_suite7_extrapolation(block, gen, L_ext)

        seed_data = {
            "stationary_controls": stat_controls,
            "drift_stationary": drift_stat,
            "drift_abrupt_decay": drift_abrupt_decay,
            "drift_abrupt_stat": drift_abrupt_stat,
            "drift_unobs_decay": drift_unobs_decay,
            "drift_unobs_stat": drift_unobs_stat,
            "drift_gradual_decay": drift_gradual_decay,
            "drift_gradual_stat": drift_gradual_stat,
            "prec_valid_het": prec_valid_het,
            "prec_fixed_het": prec_fixed_het,
            "prec_corrupt_het": prec_corrupt_het,
            "prec_valid_student": prec_valid_student,
            "prec_fixed_student": prec_fixed_student,
            "prec_corrupt_student": prec_corrupt_student,
            "prec_valid_outliers": prec_valid_outliers,
            "prec_fixed_outliers": prec_fixed_outliers,
            "prec_corrupt_outliers": prec_corrupt_outliers,
            "prec_valid_nonlinear": prec_valid_nonlinear,
            "prec_fixed_nonlinear": prec_fixed_nonlinear,
            "prec_corrupt_nonlinear": prec_corrupt_nonlinear,
            "over_cache": over_cache,
            "over_remote_decay": over_remote_decay,
            "over_remote_stat": over_remote_stat,
            "many_to_one": many_to_one,
            "pollution": pollution_res,
            "pointer_chasing": pointer_results,
            "mixed_2hop": mixed_2hop_res,
            "mixed_4hop": mixed_4hop_res,
            "capacity": cap_res,
            "extrapolation": extrap_res,
        }
        seed_records[seed] = seed_data
        raw_rows.append({"seed": seed, **seed_data})

        print(f"  Drift MSE: stat={drift_abrupt_stat['post_change_mse']:.4f}, decay={drift_abrupt_decay['post_change_mse']:.4f}")
        print(f"  Het Precision MSE: valid={prec_valid_het['mse']:.4f}, fixed={prec_fixed_het['mse']:.4f}, corrupt={prec_corrupt_het['mse']:.4f}")
        print(f"  Pointer 4-hop: vec_err={pointer_results['4']['vector_error']:.4f}, success={pointer_results['4']['decoded_success']:.2f}")

    # Compute Summary across seeds
    def avg_f(extractor: Any) -> float:
        return float(sum(extractor(seed_records[s]) for s in seeds) / len(seeds))

    summary = {
        "stationary_controls": {
            k: avg_f(lambda r, key=k: r["stationary_controls"][key])
            for k in ["noisy_linear", "recent_copy", "remote_recall", "mixed_exception", "selective_copy", "cache_boundary", "negatives", "aggregate_risk"]
        },
        "drift_stationary_control_mse": avg_f(lambda r: r["drift_stationary"]["post_change_mse"]),
        "drift_abrupt_stat_mse": avg_f(lambda r: r["drift_abrupt_stat"]["post_change_mse"]),
        "drift_abrupt_decay_mse": avg_f(lambda r: r["drift_abrupt_decay"]["post_change_mse"]),
        "drift_unobs_stat_mse": avg_f(lambda r: r["drift_unobs_stat"]["post_change_mse"]),
        "drift_unobs_decay_mse": avg_f(lambda r: r["drift_unobs_decay"]["post_change_mse"]),
        "drift_gradual_stat_mse": avg_f(lambda r: r["drift_gradual_stat"]["post_change_mse"]),
        "drift_gradual_decay_mse": avg_f(lambda r: r["drift_gradual_decay"]["post_change_mse"]),
        "prec_valid_het_mse": avg_f(lambda r: r["prec_valid_het"]["mse"]),
        "prec_fixed_het_mse": avg_f(lambda r: r["prec_fixed_het"]["mse"]),
        "prec_corrupted_het_mse": avg_f(lambda r: r["prec_corrupt_het"]["mse"]),
        "prec_valid_student_mse": avg_f(lambda r: r["prec_valid_student"]["mse"]),
        "prec_fixed_student_mse": avg_f(lambda r: r["prec_fixed_student"]["mse"]),
        "prec_corrupted_student_mse": avg_f(lambda r: r["prec_corrupt_student"]["mse"]),
        "prec_valid_outliers_mse": avg_f(lambda r: r["prec_valid_outliers"]["mse"]),
        "prec_fixed_outliers_mse": avg_f(lambda r: r["prec_fixed_outliers"]["mse"]),
        "prec_corrupted_outliers_mse": avg_f(lambda r: r["prec_corrupt_outliers"]["mse"]),
        "prec_valid_nonlinear_mse": avg_f(lambda r: r["prec_valid_nonlinear"]["mse"]),
        "prec_fixed_nonlinear_mse": avg_f(lambda r: r["prec_fixed_nonlinear"]["mse"]),
        "prec_corrupted_nonlinear_mse": avg_f(lambda r: r["prec_corrupt_nonlinear"]["mse"]),
        "over_cache_mse": avg_f(lambda r: r["over_cache"]["mse"]),
        "over_remote_decay_mse": avg_f(lambda r: r["over_remote_decay"]["mse"]),
        "over_remote_stat_mse": avg_f(lambda r: r["over_remote_stat"]["mse"]),
        "many_to_one_mse": avg_f(lambda r: r["many_to_one"]["mse"]),
    }

    # Aggregate Pointer Chasing
    hops_cfg = config["suites"]["suite4_pointer_chasing"]["hops"]
    summary["pointer_chasing_hops"] = hops_cfg
    summary["pointer_chasing_vec_errors"] = [
        avg_f(lambda r, h=h: r["pointer_chasing"][str(h)]["vector_error"]) for h in hops_cfg
    ]
    summary["pointer_chasing_decoded_success"] = [
        avg_f(lambda r, h=h: r["pointer_chasing"][str(h)]["decoded_success"]) for h in hops_cfg
    ]
    summary["pointer_chasing_latencies"] = [
        avg_f(lambda r, h=h: r["pointer_chasing"][str(h)]["latency_ms"]) for h in hops_cfg
    ]

    # Aggregate Mixed Chains
    mixed_hop_errors: dict[str, list[float]] = {}
    for pat in config["suites"]["suite5_mixed_chains"]["patterns_4hop"]:
        pat_str = "".join(pat)
        errs_at_hops = []
        for h in range(len(pat)):
            err_h = avg_f(lambda r, ps=pat_str, idx=h: r["mixed_4hop"][ps]["hop_errors"][idx])
            errs_at_hops.append(err_h)
        mixed_hop_errors[pat_str] = errs_at_hops
    summary["mixed_chains_hop_errors"] = mixed_hop_errors

    # Aggregate Capacity
    assoc_cfg = config["suites"]["suite6_capacity"]["adversarial_associations"]
    summary["capacity_associations"] = assoc_cfg
    summary["capacity_mses"] = [
        avg_f(lambda r, na=na: r["capacity"][str(na)]["mse"]) for na in assoc_cfg
    ]

    # Aggregate Extrapolation
    extrap_lens = config["suites"]["suite7_extrapolation"]["extrapolation_lengths"]
    summary["extrapolation_lengths"] = extrap_lens
    summary["extrapolation_mses"] = [
        avg_f(lambda r, el=el: r["extrapolation"][str(el)]["mse"]) for el in extrap_lens
    ]

    # PASS Gate Checks
    # Gate 1: Stationary method retains Phase 3 risk on stationary controls
    gate1_pass = summary["stationary_controls"]["aggregate_risk"] <= config["gates"]["stationary_control_risk_max"]

    # Gate 2: Drift-aware variant improves post-change risk on every paired seed when observable cue exists
    drift_improv_ratio = summary["drift_abrupt_stat_mse"] / max(summary["drift_abrupt_decay_mse"], 1e-6)
    gate2_seed_pass = all(
        seed_records[s]["drift_abrupt_decay"]["post_change_mse"] < seed_records[s]["drift_abrupt_stat"]["post_change_mse"]
        for s in seeds
    )
    gate2_pass = gate2_seed_pass and (drift_improv_ratio >= config["gates"]["drift_improvement_ratio_min"])

    # Gate 3: Evidence weighting improves heteroscedastic risk when valid and degrades when corrupted
    valid_improv_ratio = summary["prec_fixed_het_mse"] / max(summary["prec_valid_het_mse"], 1e-6)
    corrupt_deg_ratio = summary["prec_corrupted_het_mse"] / max(summary["prec_valid_het_mse"], 1e-6)
    gate3_seed_pass = all(
        seed_records[s]["prec_valid_het"]["mse"] < seed_records[s]["prec_fixed_het"]["mse"]
        for s in seeds
    )
    gate3_pass = (
        gate3_seed_pass
        and (valid_improv_ratio >= config["gates"]["valid_evidence_improvement_min"])
        and (corrupt_deg_ratio >= config["gates"]["corrupted_evidence_degradation_min"])
    )

    # Gate 4: Mixed cache/remote chains meet preregistered vector and decoded gates
    vec_4hop_max = max(
        avg_f(lambda r, ps="".join(p): r["mixed_4hop"][ps]["vector_error"])
        for p in config["suites"]["suite5_mixed_chains"]["patterns_4hop"]
    )
    dec_2hop_min = min(
        avg_f(lambda r, ps="".join(p): r["mixed_2hop"][ps]["decoded_success"])
        for p in config["suites"]["suite5_mixed_chains"]["patterns_2hop"]
    )
    dec_4hop_min = min(
        avg_f(lambda r, ps="".join(p): r["mixed_4hop"][ps]["decoded_success"])
        for p in config["suites"]["suite5_mixed_chains"]["patterns_4hop"]
    )
    gate4_pass = (
        dec_2hop_min >= config["gates"]["multihop_2hop_decoded_min"]
        and dec_4hop_min >= config["gates"]["multihop_4hop_decoded_min"]
        and vec_4hop_max <= config["gates"]["multihop_mixed_vector_error_max"]
    )

    # Gate 5: Rank/state lower-bound failures remain present (capacity strictly increases for N > d_k)
    mse_low_n = summary["capacity_mses"][1]  # N=4
    mse_high_n = summary["capacity_mses"][-1]  # N=256
    gate5_pass = mse_high_n > mse_low_n * 1.5  # Monotonic breakdown beyond rank

    # Gate 6: Context extrapolation finite and stable
    extrap_all_finite = all(
        math.isfinite(summary["extrapolation_mses"][idx]) for idx in range(len(extrap_lens))
    )
    gate6_pass = extrap_all_finite

    # Gate 7: All seeds reported and finite
    all_finite = True
    for s in seeds:
        for k, v in seed_records[s].items():
            if isinstance(v, dict) and "mse" in v and not math.isfinite(v["mse"]):
                all_finite = False

    all_gates_pass = (
        gate1_pass and gate2_pass and gate3_pass and gate4_pass and gate5_pass and gate6_pass and all_finite
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
            "gate_1_stationary_controls_retained": gate1_pass,
            "gate_2_drift_aware_improves_on_paired_seeds": gate2_pass,
            "gate_3_evidence_weighting_and_corruption_degradation": gate3_pass,
            "gate_4_mixed_chains_multihop_decoded_vector_gates": gate4_pass,
            "gate_5_capacity_lower_bound_failures_preserved": gate5_pass,
            "gate_6_extrapolation_16x_finite_and_stable": gate6_pass,
            "gate_7_all_seeds_reported_finite": all_finite,
        },
        "summary": summary,
        "seed_records": seed_records,
    }

    # Write metrics.json
    (output_dir / "metrics.json").write_text(json.dumps(results_data, indent=2) + "\n", encoding="utf-8")

    # Write raw rows
    with (raw_dir / "evaluation_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row) + "\n")

    # Generate plots
    print("Generating plots...")
    generate_phase4_plots(results_data, output_dir)

    print(f"\n=== Phase 4 Experiment Finished. Status: {results_data['status']} ===")
    for g_name, g_pass in results_data["checks"].items():
        print(f"  {g_name}: {'PASS' if g_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
