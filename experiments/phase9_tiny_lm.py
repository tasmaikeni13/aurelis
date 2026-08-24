#!/usr/bin/env python3
"""Phase 9: matched tiny decoder language-model optimization experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from torch.nn import functional as F

from csm import (
    TASKS,
    TinyDecoderLM,
    bytes_to_tensor,
    diagnostic_corpus,
    diagnostic_examples,
    sample_token_batch,
    wikitext_bytes,
)


def git_output(arguments: list[str]) -> str:
    return subprocess.run(["git", *arguments], check=True, capture_output=True, text=True).stdout.strip()


def device_for(name: str) -> torch.device:
    return torch.device("cuda:0" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def build_model(config: dict[str, Any], architecture: str, device: torch.device) -> TinyDecoderLM:
    cfg = config["model"]
    return TinyDecoderLM(
        vocabulary_size=cfg["vocabulary_size"],
        width=cfg["width"],
        layers=cfg["layers"],
        heads=cfg["heads"],
        feedforward_width=cfg["feedforward_width"],
        architecture=architecture,  # type: ignore[arg-type]
        csm_key_dimension=cfg["csm_key_dimension"],
        csm_epsilon=cfg["csm_epsilon"],
        local_window=cfg["local_attention_window"],
    ).to(device)


def learning_rate(step: int, total: int, warmup: int, minimum_fraction: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return minimum_fraction + (1.0 - minimum_fraction) * cosine


def train_model(
    architecture: str,
    config: dict[str, Any],
    natural: Tensor,
    diagnostic: Tensor,
    device: torch.device,
) -> tuple[TinyDecoderLM, dict[str, Any], list[dict[str, Any]]]:
    seed = config["seed"]
    architecture_index = config["architectures"].index(architecture)
    torch.manual_seed(seed + architecture_index * 1009)
    model = build_model(config, architecture, device)
    cfg = config["training"]
    batch_tokens = cfg["batch_size"] * cfg["sequence_length"]
    steps = math.ceil(cfg["tokens"] / batch_tokens)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda index: learning_rate(
            index, steps, cfg["warmup_steps"], cfg["minimum_learning_rate_fraction"]
        ),
    )
    generator = torch.Generator(device=device).manual_seed(seed + architecture_index * 1009 + 1)
    curves: list[dict[str, Any]] = []
    nonfinite_steps = 0
    gradient_norms: list[float] = []
    first_window: list[float] = []
    final_window: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        inputs, targets = sample_token_batch(
            natural,
            diagnostic,
            cfg["batch_size"],
            cfg["sequence_length"],
            config["dataset"]["diagnostic_batch_fraction"],
            generator,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = F.cross_entropy(logits.float().reshape(-1, model.vocabulary_size), targets.reshape(-1))
        if not torch.isfinite(loss):
            nonfinite_steps += 1
            continue
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
        if not torch.isfinite(gradient_norm):
            nonfinite_steps += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        scheduler.step()
        scalar_loss = loss.item()
        gradient_norms.append(float(gradient_norm))
        if step < max(10, steps // 20):
            first_window.append(scalar_loss)
        if step >= steps - max(10, steps // 20):
            final_window.append(scalar_loss)
        if step == 0 or (step + 1) % cfg["log_every"] == 0 or step + 1 == steps:
            curves.append(
                {
                    "architecture": architecture,
                    "step": step + 1,
                    "tokens": (step + 1) * batch_tokens,
                    "loss": scalar_loss,
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
    synchronize(device)
    elapsed = time.perf_counter() - started
    summary = {
        "architecture": architecture,
        "parameters": model.parameter_count(),
        "training_tokens": steps * batch_tokens,
        "optimizer_steps": steps,
        "batch_tokens": batch_tokens,
        "training_seconds": elapsed,
        "training_tokens_per_second": steps * batch_tokens / elapsed,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "nonfinite_steps": nonfinite_steps,
        "initial_window_loss": statistics.mean(first_window),
        "final_window_loss": statistics.mean(final_window),
        "loss_reduction_fraction": 1.0 - statistics.mean(final_window) / statistics.mean(first_window),
        "gradient_norm_mean": statistics.mean(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
    }
    return model, summary, curves


@torch.no_grad()
def evaluate_natural(
    model: TinyDecoderLM,
    tokens: Tensor,
    sequence_length: int,
    token_budget: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    evaluated = 0
    width = sequence_length + 1
    rows = min(token_budget // sequence_length, (tokens.numel() - 1) // width)
    batch_size = min(64, rows)
    for start_row in range(0, rows, batch_size):
        count = min(batch_size, rows - start_row)
        starts = torch.arange(start_row, start_row + count, device=device) * width
        positions = torch.arange(width, device=device)
        batch = tokens[starts[:, None] + positions]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(batch[:, :-1])
        loss = F.cross_entropy(
            logits.float().reshape(-1, model.vocabulary_size),
            batch[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        evaluated += count * sequence_length
    mean_loss = total_loss / evaluated
    return {
        "validation_tokens": evaluated,
        "validation_loss": mean_loss,
        "validation_perplexity": math.exp(mean_loss),
    }


@torch.no_grad()
def evaluate_diagnostics(
    model: TinyDecoderLM,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    cfg = config["evaluation"]
    records: list[dict[str, Any]] = []
    for length_name, examples in (
        ("trained_length", diagnostic_examples(config["seed"] + 900_001, cfg["diagnostic_examples_per_task"])),
        ("long_context", diagnostic_examples(config["seed"] + 900_002, cfg["long_diagnostic_examples_per_task"], long=True)),
    ):
        grouped: dict[str, list[Any]] = defaultdict(list)
        for example in examples:
            grouped[example.task].append(example)
        for task in TASKS:
            exact = token_correct = token_total = 0
            negative_log_likelihood = 0.0
            autoregressive_exact = 0
            task_examples = grouped[task]
            for index, example in enumerate(task_examples):
                sequence = bytes_to_tensor(example.prompt + example.target, device)[None]
                inputs = sequence[:, :-1]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(inputs)
                start = len(example.prompt) - 1
                relevant = logits[:, start : start + len(example.target)].float()
                target = sequence[:, len(example.prompt) :]
                predictions = relevant.argmax(dim=-1)
                exact += int(torch.equal(predictions, target))
                token_correct += int((predictions == target).sum())
                token_total += target.numel()
                negative_log_likelihood += F.cross_entropy(
                    relevant.reshape(-1, model.vocabulary_size), target.reshape(-1), reduction="sum"
                ).item()
                if index < cfg["autoregressive_examples_per_task"]:
                    prompt = bytes_to_tensor(example.prompt, device)[None]
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        generated = model.generate(prompt, len(example.target))
                    autoregressive_exact += int(
                        bytes(generated[0, -len(example.target) :].cpu().tolist()) == example.target
                    )
            records.append(
                {
                    "architecture": model.architecture,
                    "length_regime": length_name,
                    "task": task,
                    "examples": len(task_examples),
                    "teacher_forced_exact_rate": exact / len(task_examples),
                    "token_accuracy": token_correct / token_total,
                    "target_nll": negative_log_likelihood / token_total,
                    "autoregressive_examples": min(cfg["autoregressive_examples_per_task"], len(task_examples)),
                    "autoregressive_exact_rate": autoregressive_exact / min(cfg["autoregressive_examples_per_task"], len(task_examples)),
                }
            )
    return records


def timed(function: Any, device: torch.device, warmup: int, repetitions: int) -> tuple[float, float, int]:
    for _ in range(warmup):
        function()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    measurements = []
    for _ in range(repetitions):
        synchronize(device)
        started = time.perf_counter()
        function()
        synchronize(device)
        measurements.append((time.perf_counter() - started) * 1000)
    return statistics.median(measurements), statistics.pstdev(measurements) / statistics.mean(measurements), torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0


@torch.no_grad()
def scaling_metrics(
    model: TinyDecoderLM,
    validation: Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    cfg = config["evaluation"]
    records = []
    for context in cfg["scaling_context_lengths"]:
        batch = validation[: cfg["scaling_batch_size"] * context].view(cfg["scaling_batch_size"], context)

        def forward() -> Tensor:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                return model(batch)

        milliseconds, cv, peak = timed(forward, device, cfg["scaling_warmup"], cfg["scaling_repetitions"])
        tokens = batch.numel()
        records.append(
            {
                "architecture": model.architecture,
                "context_length": context,
                "batch_size": cfg["scaling_batch_size"],
                "forward_milliseconds": milliseconds,
                "timing_cv": cv,
                "tokens_per_second": tokens / (milliseconds / 1000),
                "peak_vram_bytes": peak,
                "recurrent_state_bytes": model.recurrent_state_bytes(cfg["scaling_batch_size"], context),
            }
        )
    return records


def actual_state_bytes(state: Any) -> int:
    if state is None:
        return 0
    if isinstance(state, Tensor):
        return state.numel() * state.element_size()
    if isinstance(state, (list, tuple)):
        return sum(actual_state_bytes(item) for item in state)
    if hasattr(state, "__dataclass_fields__"):
        return sum(actual_state_bytes(getattr(state, name)) for name in state.__dataclass_fields__)
    return 0


@torch.no_grad()
def decode_metrics(
    model: TinyDecoderLM,
    validation: Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    cfg = config["evaluation"]
    batch_size, prompt_length, decode_tokens = cfg["decode_batch_size"], cfg["decode_prompt_length"], cfg["decode_tokens"]
    prompt = validation[: batch_size * prompt_length].view(batch_size, prompt_length)
    states = None
    logits = None
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for position in range(prompt_length):
            logits, states = model.step(prompt[:, position], states, position)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for offset in range(decode_tokens):
            assert logits is not None
            token = logits[:, -1].argmax(dim=-1)
            logits, states = model.step(token, states, prompt_length + offset)
    synchronize(device)
    elapsed = time.perf_counter() - started
    total = batch_size * decode_tokens
    return {
        "decode_batch_size": batch_size,
        "decode_prompt_length": prompt_length,
        "decode_tokens": decode_tokens,
        "decode_milliseconds": elapsed * 1000,
        "decode_tokens_per_second": total / elapsed,
        "decode_microseconds_per_token": elapsed * 1e6 / total,
        "decode_peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "decode_actual_state_bytes": actual_state_bytes(states),
    }


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
        cells = [f"{row.get(key):.{digits}f}" if isinstance(row.get(key), float) else str(row.get(key, "")) for key, _ in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def make_report(
    config: dict[str, Any], metadata: dict[str, Any], summaries: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]], scaling: list[dict[str, Any]], gates: dict[str, bool],
) -> str:
    decision = "PASS" if all(gates.values()) else "FAIL"
    diagnostic_aggregate = []
    for architecture in config["architectures"]:
        for regime in ("trained_length", "long_context"):
            subset = [row for row in diagnostics if row["architecture"] == architecture and row["length_regime"] == regime]
            diagnostic_aggregate.append({"architecture": architecture, "length_regime": regime, "exact": statistics.mean(row["teacher_forced_exact_rate"] for row in subset), "token_accuracy": statistics.mean(row["token_accuracy"] for row in subset), "autoregressive_exact": statistics.mean(row["autoregressive_exact_rate"] for row in subset)})
    lines = [
        "# Phase 9 tiny language-model optimization", "", f"## Gate decision: {decision}", "",
        *[f"- {'PASS' if value else 'FAIL'}: `{name}`" for name, value in gates.items()], "",
        "## Pre-registration and protocol", "",
        f"Experimental generation `{config['experimental_generation']}` and config SHA-256 `{metadata['config_sha256']}` were fixed before checkpoint evaluation. All variants use the same byte tokenizer, sampled token budget, batch-token count, AdamW settings, LR schedule, gradient clipping, WikiText split, and diagnostic mixture. No architecture was modified after test results.", "",
        "The natural corpus is the pinned raw WikiText-2 training/validation parquet release. Synthetic streams are separate and deterministic, and cover all six specified probe families. The 10M-token run is intentionally not expanded because Phase 9 is a stability/mechanism gate, not scaling.", "",
        "## Natural-text training and systems behavior", "",
        *table(summaries, [("architecture", "architecture"), ("parameters", "parameters"), ("training_tokens", "tokens"), ("initial_window_loss", "initial loss"), ("final_window_loss", "final loss"), ("validation_perplexity", "val PPL"), ("gradient_norm_max", "max grad"), ("nonfinite_steps", "NaN/Inf"), ("training_tokens_per_second", "train tok/s"), ("peak_vram_bytes", "peak VRAM B"), ("decode_microseconds_per_token", "decode us/token"), ("decode_actual_state_bytes", "decode state B")]), "",
        "Validation perplexity is byte-level and should only be compared within this protocol. Peak VRAM includes model, optimizer, and training activations; decode state is measured from live incremental state tensors after the timed decode.", "",
        "## Diagnostic memory probes", "",
        *table(diagnostic_aggregate, [("architecture", "architecture"), ("length_regime", "regime"), ("exact", "mean exact"), ("token_accuracy", "token accuracy"), ("autoregressive_exact", "AR exact")]), "",
        "Task-level associative recall, variable tracking, repeated-name recall, exact-value retrieval, in-context regression, and multi-hop results are retained in [`phase9/diagnostics.csv`](phase9/diagnostics.csv). Teacher-forced exactness requires every target byte to be correct; AR exactness greedily generates a preregistered subset.", "",
        "## Sequence-length scaling", "",
        *table(scaling, [("architecture", "architecture"), ("context_length", "context"), ("forward_milliseconds", "forward ms"), ("tokens_per_second", "tokens/s"), ("peak_vram_bytes", "peak VRAM B"), ("recurrent_state_bytes", "state/cache B")]), "",
        "CSM recurrent-state bytes are constant in context length. Transformer cache bytes grow linearly, while hybrid local-attention cache is window bounded and its CSM states remain fixed. Training-time CSM prefix matrices still grow linearly and are included in measured peak VRAM.", "",
        "## Architectural versus kernel findings", "",
        "Validation loss and diagnostic accuracy diagnose architectural learning behavior; throughput, VRAM, and decode latency diagnose the current kernel tax. In particular, a useful diagnostic result does not erase the factorization cost measured in Phase 8, and a throughput deficit is not labeled a representation failure.", "",
        "## Reproducibility", "",
        f"- commit at run start: `{metadata['commit']}`; dirty: `{metadata['dirty']}`",
        f"- device: `{metadata['device']}`; Python `{metadata['python']}`; PyTorch `{metadata['torch']}`",
        f"- wall time: `{metadata['wall_seconds']:.2f}s`; tokenizer: raw UTF-8 bytes (vocabulary 256)",
        "- config: [`configs/phase9_tiny_lm.json`](../configs/phase9_tiny_lm.json)",
        "- raw rows: [`phase9/`](phase9/); machine record: [`phase9_metrics.json`](phase9_metrics.json)", "",
        "## Scoped conclusion", "",
        "The gate asks whether CSM or its hybrid trains stably, learns nontrivial natural-text structure, and exhibits a matched-parameter targeted memory advantage. It does not claim Transformer parity or authorize larger scaling unless those facts are all observed.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase9_tiny_lm.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/phase9"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    device = device_for(config["device"])
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    dataset_cfg = config["dataset"]
    cache = Path(dataset_cfg["cache"])
    natural_bytes = wikitext_bytes(cache, dataset_cfg["configuration"], "train")
    validation_bytes = wikitext_bytes(cache, dataset_cfg["configuration"], "validation")
    synthetic_bytes = diagnostic_corpus(config["seed"] + 700_001, dataset_cfg["diagnostic_training_bytes"])
    natural = bytes_to_tensor(natural_bytes, device)
    validation = bytes_to_tensor(validation_bytes, device)
    diagnostic = bytes_to_tensor(synthetic_bytes, device)
    summaries: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    scaling: list[dict[str, Any]] = []
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for architecture in config["architectures"]:
        model, summary, architecture_curves = train_model(architecture, config, natural, diagnostic, device)
        summary.update(evaluate_natural(model, validation, config["training"]["sequence_length"], config["evaluation"]["natural_validation_tokens"], device))
        summary.update(decode_metrics(model, validation, config, device))
        summaries.append(summary)
        curves.extend(architecture_curves)
        diagnostics.extend(evaluate_diagnostics(model, config, device))
        scaling.extend(scaling_metrics(model, validation, config, device))
        torch.save({"architecture": architecture, "config": config, "model": model.state_dict(), "summary": summary}, args.checkpoint_dir / f"{architecture}.pt")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    parameters = [row["parameters"] for row in summaries]
    transformer = next(row for row in summaries if row["architecture"] == "transformer")
    candidates = [row for row in summaries if row["architecture"] in ("csm", "hybrid")]
    task_advantages = []
    for candidate in ("csm", "hybrid"):
        for task in TASKS:
            candidate_row = next(row for row in diagnostics if row["architecture"] == candidate and row["task"] == task and row["length_regime"] == "long_context")
            transformer_row = next(row for row in diagnostics if row["architecture"] == "transformer" and row["task"] == task and row["length_regime"] == "long_context")
            task_advantages.append(candidate_row["token_accuracy"] - transformer_row["token_accuracy"])
    gates = {
        "models_in_5m_to_20m_range": min(parameters) >= config["gate"]["minimum_parameters"] and max(parameters) <= config["gate"]["maximum_parameters"],
        "parameter_counts_reasonably_matched": max(parameters) / min(parameters) <= config["gate"]["maximum_parameter_ratio"],
        "csm_or_hybrid_numerically_stable": any(row["nonfinite_steps"] <= config["gate"]["maximum_nonfinite_steps"] for row in candidates),
        "csm_or_hybrid_optimizes": any(row["loss_reduction_fraction"] >= config["gate"]["minimum_loss_reduction_fraction"] for row in candidates),
        "natural_text_loss_not_catastrophic": min(row["validation_loss"] for row in candidates) <= config["gate"]["maximum_csm_to_transformer_validation_loss"] * transformer["validation_loss"],
        "matched_parameter_targeted_memory_advantage": max(task_advantages) >= config["gate"]["minimum_targeted_accuracy_advantage"],
        "all_diagnostic_families_evaluated": len(diagnostics) == len(config["architectures"]) * len(TASKS) * 2,
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
        "wall_seconds": time.perf_counter() - started,
    }
    output = args.output / "phase9"
    write_csv(output / "seed_metrics.csv", summaries)
    write_csv(output / "learning_curves.csv", curves)
    write_csv(output / "diagnostics.csv", diagnostics)
    write_csv(output / "sequence_scaling.csv", scaling)
    payload = {"metadata": metadata, "config": config, "gates": gates, "summaries": summaries, "diagnostics": diagnostics, "scaling": scaling}
    (args.output / "phase9_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output / "phase9_tiny_lm.md").write_text(make_report(config, metadata, summaries, diagnostics, scaling, gates))
    print(json.dumps({"decision": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "summaries": summaries, "metadata": metadata}, indent=2))
    if not all(gates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
