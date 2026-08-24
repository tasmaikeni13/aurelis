#!/usr/bin/env python3
"""Phase 10: seeded 25M--50M natural-language controlled comparison."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from csm import TASKS, bytes_to_tensor, diagnostic_corpus, wikitext_bytes
from phase9_tiny_lm import (
    actual_state_bytes,
    config_digest,
    decode_metrics,
    device_for,
    evaluate_diagnostics,
    evaluate_natural,
    scaling_metrics,
    synchronize,
    train_model,
)


def git_output(arguments: list[str]) -> str:
    return subprocess.run(["git", *arguments], check=True, capture_output=True, text=True).stdout.strip()


@torch.no_grad()
def decode_scaling(
    model: torch.nn.Module,
    validation: Tensor,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    cfg = config["evaluation"]
    batch_size = cfg["decode_scaling_batch_size"]
    decode_tokens = cfg["decode_scaling_tokens"]
    rows = []
    model.eval()
    for context in cfg["decode_scaling_context_lengths"]:
        prompt = validation[: batch_size * context].view(batch_size, context)
        states = None
        logits = None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for position in range(context):
                logits, states = model.step(prompt[:, position], states, position)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for offset in range(decode_tokens):
                token = logits[:, -1].argmax(dim=-1)
                logits, states = model.step(token, states, context + offset)
        synchronize(device)
        elapsed = time.perf_counter() - started
        total = batch_size * decode_tokens
        rows.append(
            {
                "architecture": model.architecture,
                "seed": seed,
                "context_length": context,
                "decode_tokens": decode_tokens,
                "decode_microseconds_per_token": elapsed * 1e6 / total,
                "decode_tokens_per_second": total / elapsed,
                "actual_state_bytes": actual_state_bytes(states),
                "decode_peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], digits: int = 3) -> list[str]:
    lines = ["| " + " | ".join(label for _, label in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            cells.append(f"{value:.{digits}f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def aggregate_summaries(rows: list[dict[str, Any]], architectures: list[str]) -> list[dict[str, Any]]:
    output = []
    fields = (
        "parameters",
        "validation_loss",
        "validation_perplexity",
        "training_tokens_per_second",
        "peak_vram_bytes",
        "training_seconds",
        "decode_tokens_per_second",
        "decode_microseconds_per_token",
        "decode_actual_state_bytes",
    )
    for architecture in architectures:
        subset = [row for row in rows if row["architecture"] == architecture]
        record: dict[str, Any] = {"architecture": architecture, "seeds": len(subset)}
        for field in fields:
            values = [float(row[field]) for row in subset]
            record[f"{field}_mean"] = statistics.mean(values)
            record[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(record)
    return output


def aggregate_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["architecture"], row["context_length"])].append(row)
    output = []
    for (architecture, context), subset in sorted(groups.items()):
        losses = [row["validation_loss"] for row in subset]
        output.append(
            {
                "architecture": architecture,
                "context_length": context,
                "validation_loss_mean": statistics.mean(losses),
                "validation_loss_std": statistics.stdev(losses) if len(losses) > 1 else 0.0,
                "validation_perplexity_mean": statistics.mean(row["validation_perplexity"] for row in subset),
            }
        )
    return output


def aggregate_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["architecture"], row["length_regime"], row["task"])].append(row)
    output = []
    for (architecture, regime, task), subset in sorted(groups.items()):
        output.append(
            {
                "architecture": architecture,
                "length_regime": regime,
                "task": task,
                "token_accuracy_mean": statistics.mean(row["token_accuracy"] for row in subset),
                "token_accuracy_std": statistics.stdev(row["token_accuracy"] for row in subset),
                "exact_rate_mean": statistics.mean(row["teacher_forced_exact_rate"] for row in subset),
                "autoregressive_exact_mean": statistics.mean(row["autoregressive_exact_rate"] for row in subset),
            }
        )
    return output


def aggregate_decode_scaling(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["architecture"], row["context_length"])].append(row)
    return [
        {
            "architecture": architecture,
            "context_length": context,
            "decode_us_mean": statistics.mean(row["decode_microseconds_per_token"] for row in subset),
            "decode_us_std": statistics.stdev(row["decode_microseconds_per_token"] for row in subset),
            "state_bytes_mean": statistics.mean(row["actual_state_bytes"] for row in subset),
            "peak_vram_mean": statistics.mean(row["decode_peak_vram_bytes"] for row in subset),
        }
        for (architecture, context), subset in sorted(groups.items())
    ]


def make_report(
    config: dict[str, Any], metadata: dict[str, Any], summaries: list[dict[str, Any]],
    context_rows: list[dict[str, Any]], diagnostics: list[dict[str, Any]],
    decode_rows: list[dict[str, Any]], gates: dict[str, bool], effects: dict[str, Any],
) -> str:
    decision = "PASS" if all(gates.values()) else "STOP"
    summary_agg = aggregate_summaries(summaries, config["architectures"])
    context_agg = aggregate_context(context_rows)
    diagnostic_agg = aggregate_diagnostics(diagnostics)
    decode_agg = aggregate_decode_scaling(decode_rows)
    long_diagnostics = [row for row in diagnostic_agg if row["length_regime"] == "long_context"]
    lines = [
        "# Phase 10 small natural-language comparison", "", f"## Scaling gate decision: {decision}", "",
        *[f"- {'PASS' if value else 'FAIL'}: `{name}`" for name, value in gates.items()], "",
        "## Pre-registration", "",
        f"Generation `{config['experimental_generation']}`; config SHA-256 `{metadata['config_sha256']}`. Parameter counts, 256-byte context, AdamW, cosine schedule, 16,384 batch tokens, 100M-token budget, seeds `{config['seeds']}`, evaluations, and exclusion criteria were fixed before training.", "",
        "Exclusion criteria: " + "; ".join(config["exclusion_criteria"]) + f". Excluded runs: {metadata['excluded_runs']}. No failed seed was replaced.", "",
        f"The hybrid was not run: {config['hybrid_omission_reason']} This follows Phase 10's conditional hybrid requirement rather than silently dropping a necessary comparator.", "",
        "## Corpus and matched training", "",
        f"Both architectures train on the same deterministic first {config['dataset']['training_subset_bytes']:,} UTF-8 bytes of pinned raw WikiText-103, mixed with the same 10% diagnostic stream. Each seed/architecture sees the same {summaries[0]['training_tokens']:,}-token budget and optimizer protocol. Training stopped at the preregistered initial 100M budget; no 200M–300M extension was needed to interpret the first gate.", "",
        *table(summary_agg, [("architecture", "architecture"), ("parameters_mean", "parameters"), ("validation_perplexity_mean", "val PPL mean"), ("validation_perplexity_std", "val PPL sd"), ("training_tokens_per_second_mean", "train tok/s"), ("peak_vram_bytes_mean", "peak VRAM B"), ("training_seconds_mean", "train seconds"), ("decode_microseconds_per_token_mean", "decode us/token"), ("decode_actual_state_bytes_mean", "decode state B")]), "",
        f"Across paired seeds, CSM-minus-Transformer validation-loss differences were `{effects['paired_validation_loss_differences']}` (mean `{effects['paired_validation_loss_difference_mean']:.4f}`). This exposes seed noise rather than presenting one checkpoint as an architecture effect.", "",
        "## Same-checkpoint context evaluation", "",
        *table(context_agg, [("architecture", "architecture"), ("context_length", "context"), ("validation_loss_mean", "loss mean"), ("validation_loss_std", "loss sd"), ("validation_perplexity_mean", "PPL mean")]), "",
        "Every row evaluates the same final checkpoint at a different sequence length; checkpoints are not fine-tuned per context.", "",
        "## Downstream and long-context memory probes", "",
        *table(long_diagnostics, [("architecture", "architecture"), ("task", "task"), ("token_accuracy_mean", "token acc mean"), ("token_accuracy_std", "token acc sd"), ("exact_rate_mean", "exact mean"), ("autoregressive_exact_mean", "AR exact mean")]), "",
        "The complete trained-length and long-context tables, including all seeds, are in [`phase10/diagnostics.csv`](phase10/diagnostics.csv). No claim here rests on perplexity alone.", "",
        "## Incremental decode and state scaling", "",
        *table(decode_agg, [("architecture", "architecture"), ("context_length", "prompt context"), ("decode_us_mean", "decode us/token"), ("decode_us_std", "latency sd"), ("state_bytes_mean", "live state B"), ("peak_vram_mean", "peak VRAM B")]), "",
        f"At the longest prompt, the CSM/Transformer live-state ratio is `{effects['long_context_state_ratio']:.6f}`. CSM state growth from the shortest to longest prompt is `{effects['csm_state_growth_ratio']:.6f}x`; Transformer KV state grows with context. This is the qualifying efficiency advantage, while the latency table retains the current CSM kernel tax.", "",
        "## Interpretation and scaling decision", "",
        "The controlled comparison separates three facts: natural-text likelihood, probe behavior, and systems cost. The CSM is allowed to pass the scaling gate through a compelling state-scaling advantage even if perplexity or wall-clock is worse; those losses remain visible and constrain any claim. Further token scaling should target the factorization/decode bottleneck and only proceed if the state advantage matters for the intended context regime.", "",
        "## Reproducibility", "",
        f"- commit at run start: `{metadata['commit']}`; dirty: `{metadata['dirty']}`",
        f"- device: `{metadata['device']}`; Python `{metadata['python']}`; PyTorch `{metadata['torch']}`",
        f"- total wall time: `{metadata['wall_seconds']:.2f}s`; corpus bytes loaded: `{metadata['natural_training_bytes']}`",
        "- config: [`configs/phase10_small_nlp.json`](../configs/phase10_small_nlp.json)",
        "- raw records: [`phase10/`](phase10/); machine record: [`phase10_metrics.json`](phase10_metrics.json)", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase10_small_nlp.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/phase10"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    device = device_for(config["device"])
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    cache = Path(config["dataset"]["cache"])
    natural_bytes = wikitext_bytes(cache, config["dataset"]["configuration"], "train", maximum_bytes=config["dataset"]["training_subset_bytes"])
    validation_bytes = wikitext_bytes(cache, config["dataset"]["configuration"], "validation")
    synthetic_bytes = diagnostic_corpus(10_700_001, config["dataset"]["diagnostic_training_bytes"])
    natural = bytes_to_tensor(natural_bytes, device)
    validation = bytes_to_tensor(validation_bytes, device)
    diagnostic = bytes_to_tensor(synthetic_bytes, device)
    summaries: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    scaling: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    excluded_runs: list[str] = []
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in config["seeds"]:
        run_config = copy.deepcopy(config)
        run_config["seed"] = seed
        for architecture in config["architectures"]:
            run_started = time.perf_counter()
            print(f"starting seed={seed} architecture={architecture}", flush=True)
            model, summary, run_curves = train_model(architecture, run_config, natural, diagnostic, device)
            summary["seed"] = seed
            summary.update(evaluate_natural(model, validation, config["training"]["sequence_length"], config["evaluation"]["natural_validation_tokens"], device))
            summary.update(decode_metrics(model, validation, run_config, device))
            summaries.append(summary)
            for row in run_curves:
                row["seed"] = seed
            curves.extend(run_curves)
            run_diagnostics = evaluate_diagnostics(model, run_config, device)
            for row in run_diagnostics:
                row["seed"] = seed
            diagnostics.extend(run_diagnostics)
            run_scaling = scaling_metrics(model, validation, run_config, device)
            for row in run_scaling:
                row["seed"] = seed
            scaling.extend(run_scaling)
            for context in config["evaluation"]["validation_context_lengths"]:
                row = evaluate_natural(model, validation, context, config["evaluation"]["context_validation_tokens"], device)
                context_rows.append({"architecture": architecture, "seed": seed, "context_length": context, **row})
            decode_rows.extend(decode_scaling(model, validation, run_config, device, seed))
            checkpoint = args.checkpoint_dir / f"{architecture}_seed{seed}.pt"
            torch.save({"architecture": architecture, "seed": seed, "config": config, "model": model.state_dict(), "summary": summary}, checkpoint)
            reloaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if len(reloaded["model"]) != len(model.state_dict()):
                excluded_runs.append(f"{architecture}/seed{seed}: checkpoint reload mismatch")
            print(f"completed seed={seed} architecture={architecture} wall={time.perf_counter()-run_started:.1f}s val_loss={summary['validation_loss']:.4f}", flush=True)
            del model, reloaded
            if device.type == "cuda":
                torch.cuda.empty_cache()
    parameters = [row["parameters"] for row in summaries]
    expected_runs = len(config["seeds"]) * len(config["architectures"])
    paired_differences = []
    for seed in config["seeds"]:
        transformer = next(row for row in summaries if row["seed"] == seed and row["architecture"] == "transformer")
        csm = next(row for row in summaries if row["seed"] == seed and row["architecture"] == "csm")
        paired_differences.append(csm["validation_loss"] - transformer["validation_loss"])
    longest = max(config["evaluation"]["decode_scaling_context_lengths"])
    shortest = min(config["evaluation"]["decode_scaling_context_lengths"])
    transformer_long = statistics.mean(row["actual_state_bytes"] for row in decode_rows if row["architecture"] == "transformer" and row["context_length"] == longest)
    csm_long = statistics.mean(row["actual_state_bytes"] for row in decode_rows if row["architecture"] == "csm" and row["context_length"] == longest)
    csm_short = statistics.mean(row["actual_state_bytes"] for row in decode_rows if row["architecture"] == "csm" and row["context_length"] == shortest)
    effects = {
        "paired_validation_loss_differences": [round(value, 6) for value in paired_differences],
        "paired_validation_loss_difference_mean": statistics.mean(paired_differences),
        "paired_validation_loss_difference_std": statistics.stdev(paired_differences),
        "long_context_state_ratio": csm_long / transformer_long,
        "csm_state_growth_ratio": csm_long / csm_short,
    }
    transformer_mean_loss = statistics.mean(row["validation_loss"] for row in summaries if row["architecture"] == "transformer")
    csm_mean_loss = statistics.mean(row["validation_loss"] for row in summaries if row["architecture"] == "csm")
    gates = {
        "all_preregistered_runs_completed": len(summaries) == expected_runs and not excluded_runs,
        "three_seed_critical_comparison": len(config["seeds"]) >= config["gate"]["required_seeds"],
        "models_in_25m_to_50m_range": min(parameters) >= config["gate"]["minimum_parameters"] and max(parameters) <= config["gate"]["maximum_parameters"],
        "parameter_counts_matched": max(parameters) / min(parameters) <= config["gate"]["maximum_parameter_ratio"],
        "token_and_batch_budgets_matched": len({row["training_tokens"] for row in summaries}) == 1 and len({row["batch_tokens"] for row in summaries}) == 1,
        "no_nonfinite_training_steps": all(row["nonfinite_steps"] <= config["gate"]["maximum_nonfinite_steps"] for row in summaries),
        "natural_text_loss_not_catastrophic": csm_mean_loss <= config["gate"]["maximum_csm_to_transformer_validation_loss"] * transformer_mean_loss,
        "compelling_decode_state_scaling_advantage": effects["long_context_state_ratio"] <= config["gate"]["maximum_long_context_state_ratio"] and effects["csm_state_growth_ratio"] <= config["gate"]["maximum_csm_state_growth_ratio"],
        "all_context_and_probe_evaluations_present": len(context_rows) == expected_runs * len(config["evaluation"]["validation_context_lengths"]) and len(diagnostics) == expected_runs * len(TASKS) * 2,
    }
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_output(["rev-parse", "HEAD"]),
        "dirty": bool(git_output(["status", "--porcelain"])),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config_sha256": config_digest(config),
        "natural_training_bytes": len(natural_bytes),
        "natural_validation_bytes": len(validation_bytes),
        "diagnostic_training_bytes": len(synthetic_bytes),
        "excluded_runs": excluded_runs,
        "wall_seconds": time.perf_counter() - started,
    }
    output = args.output / "phase10"
    write_csv(output / "seed_metrics.csv", summaries)
    write_csv(output / "learning_curves.csv", curves)
    write_csv(output / "diagnostics.csv", diagnostics)
    write_csv(output / "context_evaluation.csv", context_rows)
    write_csv(output / "sequence_scaling.csv", scaling)
    write_csv(output / "decode_scaling.csv", decode_rows)
    payload = {"metadata": metadata, "config": config, "gates": gates, "effects": effects, "summaries": summaries, "context_evaluation": context_rows, "diagnostics": diagnostics, "sequence_scaling": scaling, "decode_scaling": decode_rows}
    (args.output / "phase10_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output / "phase10_small_nlp.md").write_text(make_report(config, metadata, summaries, context_rows, diagnostics, decode_rows, gates, effects))
    print(json.dumps({"decision": "PASS" if all(gates.values()) else "STOP", "gates": gates, "effects": effects, "summaries": summaries, "metadata": metadata}, indent=2))
    if not all(gates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
