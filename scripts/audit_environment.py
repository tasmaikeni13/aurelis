#!/usr/bin/env python3
"""Reproducible ROCm environment audit for Phase 0."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def command_output(arguments: list[str]) -> str:
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"
    output = (completed.stdout + completed.stderr).strip()
    return output if output else f"exit_code={completed.returncode}"


def synchronize(torch_module) -> None:
    torch_module.cuda.synchronize()


def test_dtype(torch_module, dtype, device: str) -> dict[str, Any]:
    result: dict[str, Any] = {"supported": False}
    try:
        torch_module.manual_seed(0)
        left = torch_module.randn(256, 256, device=device, dtype=dtype)
        right = torch_module.randn(256, 256, device=device, dtype=dtype)
        output = left @ right
        synchronize(torch_module)
        result.update(
            supported=True,
            finite=bool(torch_module.isfinite(output).all().item()),
            output_dtype=str(output.dtype),
        )
    except Exception as error:  # environment probe must report, not abort
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def test_torch_compile(torch_module, device: str) -> dict[str, Any]:
    result: dict[str, Any] = {"usable": False}
    try:
        def workload(left, right):
            return torch_module.relu(left @ right + 0.25)

        compiled = torch_module.compile(workload, fullgraph=True)
        left = torch_module.randn(512, 512, device=device)
        right = torch_module.randn(512, 512, device=device)
        expected = workload(left, right)
        actual = compiled(left, right)
        synchronize(torch_module)
        result.update(
            usable=bool(torch_module.allclose(actual, expected, rtol=1e-4, atol=1e-4)),
            max_absolute_error=float((actual - expected).abs().max().item()),
            backend="inductor",
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def test_triton(torch_module, device: str) -> dict[str, Any]:
    result: dict[str, Any] = {"usable": False}
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x_pointer, y_pointer, output_pointer, size: tl.constexpr, BLOCK: tl.constexpr):
            offsets = tl.arange(0, BLOCK)
            mask = offsets < size
            x = tl.load(x_pointer + offsets, mask=mask)
            y = tl.load(y_pointer + offsets, mask=mask)
            tl.store(output_pointer + offsets, x + y, mask=mask)

        size = 1024
        x = torch_module.randn(size, device=device)
        y = torch_module.randn(size, device=device)
        output = torch_module.empty_like(x)
        add_kernel[(1,)](x, y, output, size=size, BLOCK=1024)
        synchronize(torch_module)
        result.update(
            usable=bool(torch_module.allclose(output, x + y)),
            version=triton.__version__,
            max_absolute_error=float((output - x - y).abs().max().item()),
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def gemm_benchmark(
    torch_module, dtype, device: str, size: int, repetitions: int
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "matrix_size": size,
        "repetitions": repetitions,
        "dtype": str(dtype),
    }
    try:
        torch_module.manual_seed(0)
        left = torch_module.randn(size, size, device=device, dtype=dtype)
        right = torch_module.randn(size, size, device=device, dtype=dtype)
        for _ in range(3):
            _ = left @ right
        synchronize(torch_module)
        start = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        samples_ms: list[float] = []
        for _ in range(repetitions):
            start.record()
            output = left @ right
            end.record()
            synchronize(torch_module)
            samples_ms.append(float(start.elapsed_time(end)))
        median_ms = sorted(samples_ms)[len(samples_ms) // 2]
        tflops = 2.0 * size**3 / (median_ms / 1000.0) / 1e12
        result.update(
            usable=True,
            median_ms=median_ms,
            min_ms=min(samples_ms),
            max_ms=max(samples_ms),
            median_tflops=tflops,
            finite=bool(torch_module.isfinite(output).all().item()),
        )
    except Exception as error:
        result.update(usable=False, error=f"{type(error).__name__}: {error}")
    return result


def collect(args: argparse.Namespace) -> dict[str, Any]:
    import psutil
    import torch

    audit: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "cpu": {
            "model": platform.processor() or command_output(["uname", "-p"]),
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
        },
        "host_ram": {
            "total_bytes": psutil.virtual_memory().total,
            "available_bytes": psutil.virtual_memory().available,
        },
        "rocm": {
            "version_file": Path("/opt/rocm/.info/version").read_text().strip()
            if Path("/opt/rocm/.info/version").exists()
            else "unavailable",
            "hipcc": command_output(["hipcc", "--version"]),
            "rocm_smi": command_output(
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--showdriverversion"]
            ),
        },
        "pytorch": {
            "version": torch.__version__,
            "hip_version": torch.version.hip,
            "cuda_namespace_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        },
    }

    if not torch.cuda.is_available():
        audit["fatal"] = "PyTorch cannot access a ROCm GPU"
        return audit

    device = "cuda:0"
    properties = torch.cuda.get_device_properties(0)
    audit["gpu"] = {
        "name": torch.cuda.get_device_name(0),
        "architecture": getattr(properties, "gcnArchName", "unavailable"),
        "total_vram_bytes": properties.total_memory,
        "multiprocessor_count": properties.multi_processor_count,
    }
    torch.cuda.reset_peak_memory_stats()
    audit["dtype_support"] = {
        "bf16": test_dtype(torch, torch.bfloat16, device),
        "fp16": test_dtype(torch, torch.float16, device),
        "fp32": test_dtype(torch, torch.float32, device),
        "fp64": test_dtype(torch, torch.float64, device),
    }
    audit["torch_compile"] = test_torch_compile(torch, device)
    audit["triton"] = test_triton(torch, device)
    audit["gemm_sanity"] = [
        gemm_benchmark(torch, dtype, device, args.gemm_size, args.repetitions)
        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64)
    ]
    audit["audit_peak_vram_bytes"] = torch.cuda.max_memory_allocated()
    return audit


def render_text(audit: dict[str, Any]) -> str:
    cpu = audit["cpu"]
    host_ram = audit["host_ram"]
    pytorch = audit["pytorch"]
    lines = [
        "CSM PHASE 0 ENVIRONMENT AUDIT",
        "=================================",
        f"timestamp_utc: {audit['timestamp_utc']}",
        f"python: {audit['python']}",
        f"platform: {audit['platform']}",
        f"kernel: {audit['kernel']}",
        f"cpu_model: {cpu['model']}",
        f"cpu_physical_cores: {cpu['physical_cores']}",
        f"cpu_logical_cores: {cpu['logical_cores']}",
        f"host_ram_total_gib: {host_ram['total_bytes'] / 2**30:.3f}",
        f"host_ram_available_gib: {host_ram['available_bytes'] / 2**30:.3f}",
        f"rocm_version: {audit['rocm']['version_file']}",
        f"torch_version: {pytorch['version']}",
        f"torch_hip_version: {pytorch['hip_version']}",
        f"torch_cuda_api_namespace_available: {pytorch['cuda_namespace_available']}",
        f"torch_device_count: {pytorch['device_count']}",
    ]
    if "gpu" in audit:
        gpu = audit["gpu"]
        lines.extend(
            [
                f"gpu_model: {gpu['name']}",
                f"gpu_architecture: {gpu['architecture']}",
                f"gpu_vram_total_gib: {gpu['total_vram_bytes'] / 2**30:.3f}",
                f"gpu_compute_units: {gpu['multiprocessor_count']}",
                f"torch_compile_usable: {audit['torch_compile']['usable']}",
                f"torch_compile_detail: {json.dumps(audit['torch_compile'], sort_keys=True)}",
                f"triton_rocm_usable: {audit['triton']['usable']}",
                f"triton_detail: {json.dumps(audit['triton'], sort_keys=True)}",
            ]
        )
        for name, result in audit["dtype_support"].items():
            lines.append(f"{name}_gemm_support: {json.dumps(result, sort_keys=True)}")
        for result in audit["gemm_sanity"]:
            lines.append(f"gemm_sanity: {json.dumps(result, sort_keys=True)}")
        lines.append(
            f"audit_peak_vram_gib: {audit['audit_peak_vram_bytes'] / 2**30:.3f}"
        )
    if "fatal" in audit:
        lines.append(f"fatal: {audit['fatal']}")
    lines.extend(
        [
            "",
            "RAW HIPCC",
            "---------",
            audit["rocm"]["hipcc"],
            "",
            "RAW ROCM-SMI",
            "------------",
            audit["rocm"]["rocm_smi"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("environment.txt"))
    parser.add_argument(
        "--json-output", type=Path, default=Path("results/environment.json")
    )
    parser.add_argument("--gemm-size", type=int, default=8192)
    parser.add_argument("--repetitions", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    started = time.perf_counter()
    audit = collect(args)
    audit["wall_clock_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_text(audit))
    args.json_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(render_text(audit))
    return 1 if "fatal" in audit else 0


if __name__ == "__main__":
    raise SystemExit(main())

