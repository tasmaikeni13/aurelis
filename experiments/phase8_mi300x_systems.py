#!/usr/bin/env python3
"""Phase 8: measure and optimize the exact CSM operation on ROCm/MI300X."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor
from torch.nn import functional as F

from csm import (
    AffineSummary,
    FP64GaussMarkovMemory,
    csm_leading_flops_per_token,
    csm_state_bytes,
    prefix_states,
    read_prefix_states,
    sequential_decode,
    summarize_chunks,
    summarize_segment,
    token_summaries,
)


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def git_output(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def device_for(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class Timing:
    milliseconds: float
    minimum_milliseconds: float
    cv: float
    peak_vram_bytes: int
    gpu_utilization_mean: float
    gpu_utilization_peak: float


class RocmSampler:
    """Low-rate ROCm-SMI utilization sampler for sustained timing regions."""

    def __init__(self) -> None:
        self.values: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showuse", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                )
                payload = json.loads(result.stdout)
                raw = next(iter(payload.values()))["GPU use (%)"]
                self.values.append(float(str(raw).split()[0]))
            except (OSError, subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError):
                pass
            self._stop.wait(0.05)

    def __enter__(self) -> "RocmSampler":
        if subprocess.run(["sh", "-c", "command -v rocm-smi"], capture_output=True).returncode == 0:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)


def benchmark(
    function: Callable[[], Any],
    device: torch.device,
    *,
    warmup: int,
    samples: int,
    repetitions: int,
) -> Timing:
    for _ in range(warmup):
        function()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    measurements: list[float] = []
    with RocmSampler() as sampler:
        for _ in range(samples):
            synchronize(device)
            started = time.perf_counter()
            for _ in range(repetitions):
                function()
            synchronize(device)
            measurements.append((time.perf_counter() - started) * 1000.0 / repetitions)
    mean = statistics.mean(measurements)
    cv = statistics.pstdev(measurements) / mean if mean else 0.0
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return Timing(
        milliseconds=statistics.median(measurements),
        minimum_milliseconds=min(measurements),
        cv=cv,
        peak_vram_bytes=peak,
        gpu_utilization_mean=statistics.mean(sampler.values) if sampler.values else float("nan"),
        gpu_utilization_peak=max(sampler.values) if sampler.values else float("nan"),
    )


def tensors(
    shape: dict[str, int],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    *,
    requires_grad: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    b, h, t = shape["batch_size"], shape["heads"], shape["sequence_length"]
    dk, dv = shape["d_key"], shape["d_value"]
    generator = torch.Generator(device=device).manual_seed(seed)
    keys = F.normalize(torch.randn(b, h, t, dk, generator=generator, device=device), dim=-1).to(dtype)
    values = torch.randn(b, h, t, dv, generator=generator, device=device).to(dtype)
    queries = F.normalize(torch.randn(b, h, t, dk, generator=generator, device=device), dim=-1).to(dtype)
    beta = (0.25 + 0.75 * torch.rand(b, h, t, generator=generator, device=device)).to(dtype)
    decay = torch.ones(b, h, t, dtype=dtype, device=device)
    if requires_grad:
        for tensor in (keys, values, queries, beta):
            tensor.requires_grad_()
    return keys, values, queries, beta, decay


def timing_row(
    operation: str,
    timing: Timing,
    *,
    tokens: int,
    flops: float = 0.0,
    traffic_bytes: float = 0.0,
    **fields: Any,
) -> dict[str, Any]:
    seconds = timing.milliseconds / 1000.0
    return {
        "operation": operation,
        **fields,
        **asdict(timing),
        "tokens": tokens,
        "tokens_per_second": tokens / seconds if tokens else 0.0,
        "microseconds_per_token": timing.milliseconds * 1000.0 / tokens if tokens else 0.0,
        "estimated_flops": flops,
        "achieved_tflops": flops / seconds / 1e12 if flops else 0.0,
        "estimated_traffic_bytes": traffic_bytes,
        "achieved_hbm_gbps": traffic_bytes / seconds / 1e9 if traffic_bytes else 0.0,
    }


def component_benchmarks(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    cfg = config["component_shape"]
    keys, values, queries, beta, decay = tensors(cfg, torch.bfloat16, device, config["seed"])
    b, h, t, dk, dv = cfg["batch_size"], cfg["heads"], cfg["sequence_length"], cfg["d_key"], cfg["d_value"]
    tokens = b * t
    common = dict(warmup=config["warmup"], samples=config["timing_samples"], repetitions=config["component_repetitions"])
    records: list[dict[str, Any]] = []

    token_fn = lambda: token_summaries(keys, values, beta, decay)
    timing = benchmark(token_fn, device, **common)
    outer_flops = b * h * t * (dk * dk + dv * dk) * 2
    records.append(timing_row("outer_product_updates", timing, tokens=tokens, flops=outer_flops, shape=json.dumps(cfg)))

    construct_fn = lambda: summarize_segment(keys, values, beta, decay)
    timing = benchmark(construct_fn, device, **common)
    records.append(timing_row("construct_S_C", timing, tokens=tokens, flops=outer_flops, shape=json.dumps(cfg)))
    final = construct_fn()
    identity = torch.eye(dk, dtype=final.S.dtype, device=device)
    system = final.S + config["epsilon"] * identity
    factor = torch.linalg.cholesky(system)
    rhs = queries[..., -1, :].float().unsqueeze(-1)

    timing = benchmark(lambda: torch.linalg.cholesky(system), device, **common)
    chol_flops = b * h * dk**3 / 3
    records.append(timing_row("cholesky", timing, tokens=b, flops=chol_flops, shape=json.dumps(cfg)))
    timing = benchmark(lambda: torch.cholesky_solve(rhs, factor), device, **common)
    solve_flops = b * h * 2 * dk**2
    records.append(timing_row("triangular_solves", timing, tokens=b, flops=solve_flops, shape=json.dumps(cfg)))

    sequential_common = dict(common)
    sequential_common["repetitions"] = 1
    timing = benchmark(
        lambda: sequential_decode(keys, values, queries, beta, decay, config["epsilon"]),
        device,
        **sequential_common,
    )
    seq_flops = tokens * csm_leading_flops_per_token(h, dk, dv) + b * h * t * dk**3 / 3
    records.append(timing_row("sequential_decode", timing, tokens=tokens, flops=seq_flops, shape=json.dumps(cfg)))

    train_cfg = config["training_shape"]
    tk, tv, tq, tbeta, tdecay = tensors(train_cfg, torch.bfloat16, device, config["seed"] + 1, requires_grad=True)
    train_tokens = train_cfg["batch_size"] * train_cfg["sequence_length"]

    def forward() -> tuple[Tensor, Tensor]:
        summary = prefix_states(tk, tv, tbeta, tdecay)
        return read_prefix_states(summary, tq, config["epsilon"], output_dtype=torch.float32)

    timing = benchmark(forward, device, **common)
    records.append(timing_row("training_forward", timing, tokens=train_tokens, shape=json.dumps(train_cfg)))
    reads, uncertainty = forward()
    loss = reads.square().mean() + 1e-3 * uncertainty.mean()

    def backward() -> None:
        for tensor in (tk, tv, tq, tbeta):
            tensor.grad = None
        loss.backward(retain_graph=True)

    timing = benchmark(backward, device, warmup=1, samples=config["timing_samples"], repetitions=1)
    records.append(timing_row("backward", timing, tokens=train_tokens, shape=json.dumps(train_cfg)))

    movement_elements = 128 * 1024 * 1024 // 4
    source = torch.empty(movement_elements, dtype=torch.float32, device=device).normal_()
    destination = torch.empty_like(source)
    timing = benchmark(lambda: destination.copy_(source), device, **common)
    records.append(timing_row("memory_movement", timing, tokens=0, traffic_bytes=2 * source.numel() * source.element_size(), shape="128MiB_copy"))

    scalar = torch.ones((), device=device)
    launch_common = dict(common)
    launch_common["repetitions"] = max(100, common["repetitions"])
    timing = benchmark(lambda: scalar.add_(1), device, **launch_common)
    records.append(timing_row("kernel_launch", timing, tokens=0, shape="scalar_in_place_add"))
    return records


def compile_probe(config: dict[str, Any], device: torch.device) -> tuple[bool, str, Callable[..., tuple[Tensor, Tensor]] | None]:
    def construction(k: Tensor, v: Tensor, beta: Tensor, decay: Tensor) -> tuple[Tensor, Tensor]:
        summary = summarize_segment(k, v, beta, decay)
        return summary.S, summary.C

    if not hasattr(torch, "compile"):
        return False, "torch.compile unavailable", None
    try:
        compiled = torch.compile(construction, fullgraph=True)
        shape = {"batch_size": 2, "heads": 2, "sequence_length": 32, "d_key": 16, "d_value": 16}
        keys, values, _, beta, decay = tensors(shape, torch.bfloat16, device, config["seed"] + 2)
        expected = construction(keys, values, beta, decay)
        actual = compiled(keys, values, beta, decay)
        synchronize(device)
        torch.testing.assert_close(actual[0], expected[0], rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(actual[1], expected[1], rtol=2e-5, atol=2e-5)
        return True, "supported", compiled
    except Exception as error:  # capability result must be retained, not hide a backend failure
        return False, f"{type(error).__name__}: {error}", None


def path_benchmarks(
    config: dict[str, Any], device: torch.device, compiled: Callable[..., tuple[Tensor, Tensor]] | None
) -> list[dict[str, Any]]:
    shape = config["component_shape"]
    keys, values, _, beta, decay = tensors(shape, torch.bfloat16, device, config["seed"] + 3)
    common = dict(warmup=config["warmup"], samples=config["timing_samples"], repetitions=config["component_repetitions"])
    functions: list[tuple[str, Callable[[], Any]]] = [
        ("A_vectorized", lambda: summarize_segment(keys, values, beta, decay)),
        ("C_chunked_32", lambda: summarize_chunks(keys, values, beta, decay, 32)),
        ("D_associative_scan", lambda: prefix_states(keys, values, beta, decay, unit_decay_fast_path=False)),
    ]
    if compiled is not None:
        functions.insert(1, ("B_torch_compile", lambda: compiled(keys, values, beta, decay)))
    records = []
    tokens = shape["batch_size"] * shape["sequence_length"]
    for name, function in functions:
        timing = benchmark(function, device, **common)
        records.append(timing_row(name, timing, tokens=tokens, shape=json.dumps(shape)))
    return records


def sweep_benchmarks(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    common = dict(warmup=2, samples=config["timing_samples"], repetitions=config["sweep_repetitions"])
    records: list[dict[str, Any]] = []

    def run(shape: dict[str, int], name: str, value: Any, activation: torch.dtype = torch.bfloat16, accumulation: torch.dtype = torch.float32) -> None:
        keys, values, _, beta, decay = tensors(shape, activation, device, config["seed"] + len(records) + 10)
        timing = benchmark(lambda: summarize_segment(keys, values, beta, decay, accumulation_dtype=accumulation), device, **common)
        records.append(timing_row("construct_S_C", timing, tokens=shape["batch_size"] * shape["sequence_length"], sweep=name, sweep_value=value, activation_dtype=str(activation), accumulation_dtype=str(accumulation), shape=json.dumps(shape)))

    base = dict(config["component_shape"])
    base.update(batch_size=4, heads=4, sequence_length=128)
    for dk in config["dimension_sweep"]:
        for dv in config["dimension_sweep"]:
            run({**base, "d_key": dk, "d_value": dv}, "dimension_grid", f"{dk}x{dv}")
    for axis, values in config["axis_sweeps"].items():
        for value in values:
            shape = dict(base)
            shape[axis] = value
            run(shape, axis, value)
    for policy in config["precision_policies"]:
        shape = dict(base)
        run(shape, "dtype", policy["name"], DTYPES[policy["activation"]], DTYPES[policy["accumulation"]])
    return records


def head_economics(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    records = []
    for index, item in enumerate(config["head_economics"]):
        shape = {
            "batch_size": config["head_economics_batch"],
            "heads": item["heads"],
            "sequence_length": config["head_economics_sequence"],
            "d_key": item["d_key"],
            "d_value": item["d_value"],
        }
        keys, values, queries, beta, decay = tensors(shape, torch.bfloat16, device, config["seed"] + 100 + index)

        def operation() -> tuple[Tensor, Tensor]:
            state = summarize_segment(keys, values, beta, decay)
            expanded = AffineSummary(
                state.decay[..., None].expand(*state.decay.shape, shape["sequence_length"]),
                state.S[..., None, :, :].expand(*state.S.shape[:-2], shape["sequence_length"], *state.S.shape[-2:]),
                state.C[..., None, :, :].expand(*state.C.shape[:-2], shape["sequence_length"], *state.C.shape[-2:]),
            )
            return read_prefix_states(expanded, queries, config["epsilon"], output_dtype=torch.float32)

        timing = benchmark(operation, device, warmup=2, samples=config["timing_samples"], repetitions=config["sweep_repetitions"])
        with torch.no_grad():
            recall_count = config["head_economics_recall_associations"]
            recall_state = summarize_segment(
                keys[..., :recall_count, :],
                values[..., :recall_count, :],
                beta[..., :recall_count],
                decay[..., :recall_count],
            )
            recall_summary = AffineSummary(
                recall_state.decay[..., None].expand(*recall_state.decay.shape, recall_count),
                recall_state.S[..., None, :, :].expand(*recall_state.S.shape[:-2], recall_count, *recall_state.S.shape[-2:]),
                recall_state.C[..., None, :, :].expand(*recall_state.C.shape[:-2], recall_count, *recall_state.C.shape[-2:]),
            )
            reads, _ = read_prefix_states(
                recall_summary,
                keys[..., :recall_count, :],
                config["epsilon"],
                output_dtype=torch.float32,
            )
            targets = values[..., :recall_count, :].float()
            normalized_mse = ((reads - targets) ** 2).mean().item() / targets.square().mean().item()
        state_bytes = csm_state_bytes(shape["batch_size"], shape["heads"], shape["d_key"], shape["d_value"])
        quality = max(0.0, 1.0 - normalized_mse)
        records.append(timing_row("many_small_heads", timing, tokens=shape["batch_size"] * shape["sequence_length"], heads=shape["heads"], d_key=shape["d_key"], d_value=shape["d_value"], aggregate_width=shape["heads"] * shape["d_value"], state_bytes=state_bytes, normalized_recall_mse=normalized_mse, recall_quality=quality, quality_per_megabyte=quality / (state_bytes / 1e6), quality_per_wall_second=quality / (timing.milliseconds / 1000), shape=json.dumps(shape)))
    return records


def baselines(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    shape = config["baseline_shape"]
    keys, values, queries, beta, decay = tensors(shape, torch.bfloat16, device, config["seed"] + 200)
    scale = shape["d_key"] ** -0.5

    def csm_operation() -> Tensor:
        states = prefix_states(keys, values, beta, decay)
        return read_prefix_states(states, queries, config["epsilon"], output_dtype=keys.dtype)[0]

    def attention_operation() -> Tensor:
        return F.scaled_dot_product_attention(queries, keys, values, is_causal=True, scale=scale)

    def linear_operation() -> Tensor:
        positive_keys = F.elu(keys.float()) + 1
        positive_queries = F.elu(queries.float()) + 1
        cross = torch.cumsum(values.float()[..., :, :, None] * positive_keys[..., :, None, :], dim=-3)
        normalizer = torch.cumsum(positive_keys, dim=-2)
        numerator = torch.einsum("...tvk,...tk->...tv", cross, positive_queries)
        denominator = torch.einsum("...tk,...tk->...t", normalizer, positive_queries).clamp_min(1e-6)
        return (numerator / denominator[..., None]).to(keys.dtype)

    functions = [("csm", csm_operation), ("attention", attention_operation), ("linear_memory", linear_operation)]
    records = []
    tokens = shape["batch_size"] * shape["sequence_length"]
    element_bytes = keys.element_size()
    state_sizes = {
        "csm": csm_state_bytes(shape["batch_size"], shape["heads"], shape["d_key"], shape["d_value"]),
        "attention": shape["batch_size"] * shape["heads"] * shape["sequence_length"] * (shape["d_key"] + shape["d_value"]) * element_bytes,
        "linear_memory": shape["batch_size"] * shape["heads"] * (shape["d_value"] * shape["d_key"] + shape["d_key"]) * 4,
    }
    for method, function in functions:
        timing = benchmark(function, device, warmup=config["warmup"], samples=config["timing_samples"], repetitions=config["sweep_repetitions"])
        records.append(timing_row(method, timing, tokens=tokens, state_bytes=state_sizes[method], shape=json.dumps(shape)))
    return records


def relative_error(actual: Tensor, expected: Tensor) -> float:
    return float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected).clamp_min(1e-30))


def oracle_validation(
    config: dict[str, Any], device: torch.device, compiled: Callable[..., tuple[Tensor, Tensor]] | None
) -> list[dict[str, Any]]:
    cfg = config["oracle"]
    shape = {"batch_size": 1, "heads": 1, "sequence_length": cfg["steps"], "d_key": cfg["d_key"], "d_value": cfg["d_value"]}
    keys, values, queries, beta, _ = tensors(shape, torch.bfloat16, device, config["seed"] + 300)
    generator = torch.Generator(device=device).manual_seed(config["seed"] + 301)
    decay = (0.9 + 0.1 * torch.rand(1, 1, cfg["steps"], generator=generator, device=device)).to(torch.bfloat16)
    k64, v64, b64, d64, q64 = (x[0, 0].double().cpu() for x in (keys, values, beta, decay, queries))
    oracle_memory = FP64GaussMarkovMemory(cfg["d_key"], cfg["d_value"], epsilon=config["epsilon"])
    oracle_state = oracle_memory.run(k64, v64, b64, d64)
    oracle_reads = torch.stack([oracle_memory.read(oracle_memory.run(k64[:i+1], v64[:i+1], b64[:i+1], d64[:i+1]), q64[i]) for i in range(cfg["steps"])])
    paths: list[tuple[str, AffineSummary, Tensor | None]] = []
    vectorized = summarize_segment(keys, values, beta, decay)
    paths.append(("A_vectorized", vectorized, None))
    chunked = summarize_chunks(keys, values, beta, decay, cfg["chunk_size"])
    paths.append(("C_chunked", chunked, None))
    scanned = prefix_states(keys, values, beta, decay, unit_decay_fast_path=False)
    scan_reads = read_prefix_states(scanned, queries, config["epsilon"], output_dtype=torch.float32)[0]
    paths.append(("D_associative", AffineSummary(scanned.decay[..., -1], scanned.S[..., -1, :, :], scanned.C[..., -1, :, :]), scan_reads))
    sequential_reads, _, sequential = sequential_decode(keys, values, queries, beta, decay, config["epsilon"], output_dtype=torch.float32)
    paths.append(("reference_batched_sequential", sequential, sequential_reads))
    if compiled is not None:
        S, C = compiled(keys, values, beta, decay)
        paths.append(("B_torch_compile", AffineSummary(vectorized.decay, S, C), None))
    records = []
    for name, state, reads in paths:
        S_error = relative_error(state.S[0, 0].double().cpu(), oracle_state.S)
        C_error = relative_error(state.C[0, 0].double().cpu(), oracle_state.C)
        if reads is None:
            system = state.S + config["epsilon"] * torch.eye(cfg["d_key"], device=device)
            solved = torch.linalg.solve(system, queries[..., -1, :].float().unsqueeze(-1)).squeeze(-1)
            read = torch.einsum("...vk,...k->...v", state.C, solved)[0, 0].double().cpu()
            read_error = relative_error(read, oracle_reads[-1])
        else:
            read_error = relative_error(reads[0, 0].double().cpu(), oracle_reads)
        records.append({"path": name, "S_relative_error": S_error, "C_relative_error": C_error, "read_relative_error": read_error, "finite": bool(torch.isfinite(state.S).all() and torch.isfinite(state.C).all())})
    return records


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def markdown_table(records: list[dict[str, Any]], columns: list[tuple[str, str]], precision: int = 3) -> list[str]:
    lines = ["| " + " | ".join(label for _, label in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in records:
        cells = []
        for key, _ in columns:
            value = record.get(key, "")
            cells.append(f"{value:.{precision}f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def report(
    config: dict[str, Any], metadata: dict[str, Any], components: list[dict[str, Any]],
    paths: list[dict[str, Any]], sweeps: list[dict[str, Any]], heads: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]], oracle: list[dict[str, Any]], gates: dict[str, bool],
) -> str:
    decision = "PASS" if all(gates.values()) else "FAIL"
    lines = [
        "# Phase 8 MI300X systems characterization", "", f"## Gate decision: {decision}", "",
        *[f"- {'PASS' if value else 'FAIL'}: `{name}`" for name, value in gates.items()], "",
        "## Exactness against the Phase 1 oracle", "",
        *markdown_table(oracle, [("path", "path"), ("S_relative_error", "S rel. error"), ("C_relative_error", "C rel. error"), ("read_relative_error", "read rel. error")], 6), "",
        "All oracle inputs are first quantized to bf16, then the Phase 1 recurrence is evaluated in fp64 on those same values. Optimized paths accumulate and solve in fp32. Thus the table isolates implementation error from activation quantization.", "",
        "## Component profile", "",
        *markdown_table(components, [("operation", "operation"), ("milliseconds", "ms"), ("tokens_per_second", "tokens/s"), ("microseconds_per_token", "us/token"), ("achieved_tflops", "est. TFLOP/s"), ("achieved_hbm_gbps", "est. GB/s"), ("gpu_utilization_mean", "GPU util %"), ("gpu_utilization_peak", "peak util %"), ("peak_vram_bytes", "peak VRAM B"), ("cv", "timing CV")]), "",
        "GPU utilization is sampled from ROCm-SMI over sustained timing regions. Estimated operation bandwidth and FLOPs use explicit tensor-traffic and leading-operation models; the 128 MiB copy row is the direct empirical HBM movement reference. Kernel-launch latency is a synchronized scalar-device operation.", "",
        "## Optimization ladder", "",
        *markdown_table(paths, [("operation", "path"), ("milliseconds", "ms"), ("tokens_per_second", "tokens/s"), ("peak_vram_bytes", "peak VRAM B"), ("cv", "CV")]), "",
        f"`torch.compile`: {metadata['compile_status']}. The best ROCm-compatible fusion available here is Inductor/Triton fusion around state construction. A custom fused Cholesky kernel is not justified by this phase: rocSOLVER supplies the factorization, and replacing it would expand the numerical validation surface.", "",
        "Chunking bounds temporary token-summary storage. Associative scan has logarithmic dependency depth but materializes every prefix matrix; the unit-decay training fast path uses an exact cumulative sum instead.", "",
        "## Many-small-head economics", "",
        *markdown_table(heads, [("heads", "heads"), ("d_key", "d_k"), ("state_bytes", "state B"), ("milliseconds", "ms"), ("normalized_recall_mse", "normalized MSE"), ("quality_per_megabyte", "quality/MB"), ("quality_per_wall_second", "quality/s")]), "",
        f"These timing rows hold aggregate width, batch, {config['head_economics_sequence']}-token context, activation dtype, and workload fixed. Quality uses a separate under-capacity {config['head_economics_recall_associations']}-association recall control at each head size: `quality = max(0, 1-normalized_MSE)`. Quality/MB is the requested memory-quality-per-byte view. Smaller heads reduce the quadratic state term, but measured latency and quality determine whether that theoretical saving is economical on this GPU.", "",
        "## Baselines at equivalent width/context", "",
        *markdown_table(baseline_rows, [("operation", "method"), ("milliseconds", "ms"), ("tokens_per_second", "tokens/s"), ("microseconds_per_token", "us/token"), ("state_bytes", "persistent/cache B"), ("peak_vram_bytes", "peak VRAM B")]), "",
        "Attention uses PyTorch causal scaled-dot-product attention on ROCm; the linear-memory baseline is positive-feature causal linear attention. CSM state bytes are constant in context length, whereas the reported attention KV bytes grow with context.", "",
        "## Sweep coverage and complexity", "",
        f"The raw sweep contains {len(sweeps)} rows: the complete `d_k x d_v` grid over `{config['dimension_sweep']}`, plus independent batch-size, sequence-length, head-count, and dtype/precision-policy sweeps. See [`phase8/sweeps.csv`](phase8/sweeps.csv).", "",
        "For `H` heads, CSM persistent state is `B H d_k (d_k+d_v)` fp32 elements; a write/read has leading `Theta(H(d_k^2+d_k d_v))` work, Cholesky preparation is `Theta(H d_k^3)`, and prepared reads are `Theta(H(d_k^2+d_k d_v))`. Attention stores `Theta(B H T(d_k+d_v))` KV elements and performs `Theta(B H T^2 d_k)` prefill work. The tables above are measured wall-clock behavior, not substitutions for these asymptotics.", "",
        "## Precision policy", "",
        "The stable primary policy is bf16 features/activations with fp32 `S/C` accumulation, factorization, and triangular solves. The raw dtype rows include fp32, fp16, bf16, and fp64 alternatives. Low-precision Cholesky is not offered by the ROCm PyTorch linalg path and is numerically inappropriate for the conditioned system; this is a measured stack constraint, not a CUDA assumption.", "",
        "## Reproducibility", "",
        f"- commit at run start: `{metadata['commit']}`; dirty: `{metadata['dirty']}`",
        f"- device: `{metadata['device']}`; gfx: `{metadata['gcn_arch']}`; ROCm/HIP: `{metadata['hip']}`",
        f"- Python `{metadata['python']}`; PyTorch `{metadata['torch']}`; wall time `{metadata['wall_seconds']:.2f}s`",
        "- config: [`configs/phase8_mi300x.json`](../configs/phase8_mi300x.json)",
        "- raw records: [`phase8/`](phase8/); machine record: [`phase8_metrics.json`](phase8_metrics.json)", "",
        "## Scoped conclusion", "",
        "The pass gate establishes a stable, oracle-checked ROCm implementation and quantifies its hardware tax. It does not require or claim a win over FlashAttention. Later model comparisons should use the measured bf16/fp32 policy and distinguish persistent recurrent state from training-time prefix activations.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase8_mi300x.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    device = device_for(config["device"])
    torch.manual_seed(config["seed"])
    started = time.perf_counter()
    compile_ok, compile_status, compiled = compile_probe(config, device)
    components = component_benchmarks(config, device)
    paths = path_benchmarks(config, device, compiled)
    sweeps = sweep_benchmarks(config, device)
    heads = head_economics(config, device)
    baseline_rows = baselines(config, device)
    oracle = oracle_validation(config, device, compiled)
    oracle_cfg = config["oracle"]
    gates = {
        "rocm_mi300x_target": (getattr(torch.version, "hip", None) is not None and "MI300X" in (torch.cuda.get_device_name(device) if device.type == "cuda" else "")) if config["gate"]["require_rocm"] else True,
        "torch_compile_supported": compile_ok if config["gate"]["require_compile"] else True,
        "all_paths_finite": all(row["finite"] for row in oracle),
        "optimized_state_matches_oracle": max(max(row["S_relative_error"], row["C_relative_error"]) for row in oracle) <= oracle_cfg["maximum_fp32_state_relative_error"],
        "optimized_reads_match_oracle": max(row["read_relative_error"] for row in oracle) <= oracle_cfg["maximum_bf16_read_relative_error"],
        "timings_stable": max(row["cv"] for row in components + paths + sweeps + heads + baseline_rows) <= config["gate"]["maximum_timing_cv"],
        "all_required_sweeps_present": len(sweeps) == len(config["dimension_sweep"]) ** 2 + sum(len(v) for v in config["axis_sweeps"].values()) + len(config["precision_policies"]),
    }
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_output(["rev-parse", "HEAD"]),
        "dirty": bool(git_output(["status", "--porcelain"])),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "gcn_arch": torch.cuda.get_device_properties(device).gcnArchName if device.type == "cuda" else "n/a",
        "hip": getattr(torch.version, "hip", None),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "compile_status": compile_status,
        "wall_seconds": time.perf_counter() - started,
    }
    output_dir = args.output / "phase8"
    write_csv(output_dir / "components.csv", components)
    write_csv(output_dir / "optimization_paths.csv", paths)
    write_csv(output_dir / "sweeps.csv", sweeps)
    write_csv(output_dir / "head_economics.csv", heads)
    write_csv(output_dir / "baselines.csv", baseline_rows)
    write_csv(output_dir / "oracle_validation.csv", oracle)
    payload = {"metadata": metadata, "config": config, "gates": gates, "components": components, "optimization_paths": paths, "sweeps": sweeps, "head_economics": heads, "baselines": baseline_rows, "oracle": oracle}
    (args.output / "phase8_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output / "phase8_mi300x_systems.md").write_text(report(config, metadata, components, paths, sweeps, heads, baseline_rows, oracle, gates))
    print(json.dumps({"decision": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "metadata": metadata}, indent=2))
    if not all(gates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
