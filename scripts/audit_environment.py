#!/usr/bin/env python3
"""Non-destructive AURELIS host, ROCm, PyTorch, and GEMM audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


REPO = Path(__file__).resolve().parents[1]
OFFICIAL_SOURCES = [
    {
        "title": "ROCm 7.2.4 compatibility matrix",
        "url": "https://rocm.docs.amd.com/en/docs-7.2.4/compatibility/compatibility-matrix.html",
        "decision": "gfx942 is supported and PyTorch 2.7.1, 2.8.0, and 2.9.1 are listed for ROCm 7.2.4.",
    },
    {
        "title": "AMD MI300 series workload optimization",
        "url": "https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/workload.html",
        "decision": "Measure first; synchronize device timing; compare eager, Inductor, and Triton only when justified.",
    },
    {
        "title": "rocSOLVER LAPACK functions",
        "url": "https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html",
        "decision": "Use positive-definite Cholesky factorization and solves for the reference path.",
    },
    {
        "title": "PyTorch HIP semantics",
        "url": "https://docs.pytorch.org/docs/main/notes/hip.html",
        "decision": "Use the torch.cuda API namespace on ROCm and identify HIP with torch.version.hip.",
    },
]


def command(args: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(args, text=True, capture_output=True, check=False)
    except OSError as exc:
        return {"command": args, "available": False, "error": repr(exc)}
    return {
        "command": args,
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def file_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gemm_health(dtype: torch.dtype, size: int, repetitions: int) -> dict[str, Any]:
    device = torch.device("cuda")
    try:
        left = torch.randn(size, size, device=device, dtype=dtype)
        right = torch.randn(size, size, device=device, dtype=dtype)
        for _ in range(2):
            result = left @ right
        synchronize()
        samples: list[float] = []
        for _ in range(repetitions):
            started = time.perf_counter()
            result = left @ right
            synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
        median_ms = statistics.median(samples)
        return {
            "supported": True,
            "finite": bool(torch.isfinite(result).all().cpu()),
            "matrix_size": size,
            "repetitions": repetitions,
            "samples_ms": samples,
            "median_ms": median_ms,
            "median_tflops": 2.0 * size**3 / (median_ms / 1000.0) / 1e12,
            "output_dtype": str(result.dtype),
        }
    except Exception as exc:
        return {"supported": False, "error": repr(exc), "matrix_size": size}


def compile_health() -> dict[str, Any]:
    if not hasattr(torch, "compile"):
        return {"available": False, "reason": "torch.compile is absent"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = torch.randn(256, device=device)

    def operation(value: torch.Tensor) -> torch.Tensor:
        return torch.sin(value) + value.square()

    try:
        started = time.perf_counter()
        compiled = torch.compile(operation, backend="inductor")
        actual = compiled(sample)
        synchronize()
        compile_seconds = time.perf_counter() - started
        difference = float((actual - operation(sample)).abs().max().cpu())
        return {
            "available": True,
            "usable": bool(torch.isfinite(actual).all().cpu()),
            "compile_and_first_run_seconds": compile_seconds,
            "max_absolute_error": difference,
        }
    except Exception as exc:
        return {"available": True, "usable": False, "error": repr(exc)}


def profiler_health() -> dict[str, Any]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.profiler.profile(activities=activities) as profile:
            value = torch.randn(128, device="cuda" if torch.cuda.is_available() else "cpu")
            _ = value.square().sum()
            synchronize()
        return {"available": True, "event_count": len(profile.key_averages())}
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def cholesky_health() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "reason": "no HIP device"}
    try:
        matrix = torch.randn(8, 16, 16, device="cuda", dtype=torch.float64)
        positive = matrix @ matrix.mT + 0.5 * torch.eye(16, device="cuda", dtype=torch.float64)
        factor = torch.linalg.cholesky(positive)
        rhs = torch.randn(8, 16, 2, device="cuda", dtype=torch.float64)
        solution = torch.cholesky_solve(rhs, factor)
        residual = torch.linalg.vector_norm(positive @ solution - rhs, dim=-2).max()
        synchronize()
        return {
            "available": True,
            "finite": bool(torch.isfinite(solution).all().cpu()),
            "max_residual": float(residual.cpu()),
        }
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def library_paths(pattern: str) -> list[str]:
    paths: list[str] = []
    for root in (Path("/opt/rocm"), Path("/usr/lib")):
        if root.exists():
            paths.extend(str(path) for path in root.rglob(pattern))
    return sorted(set(paths))[:40]


def package_inventory() -> list[str]:
    return sorted(
        f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )


def build_record(gemm_size: int, repetitions: int) -> dict[str, Any]:
    git_status = command(["git", "status", "--short"])
    git_head = command(["git", "rev-parse", "HEAD"])
    status_lines = git_status.get("stdout", "").splitlines()
    rocm_versions = {
        str(path): file_text(path)
        for path in sorted(Path("/opt/rocm").glob("core-*/.info/version"))
    }
    torch_packages = package_inventory()
    forbidden = [
        item
        for item in torch_packages
        if any(term in item.lower() for term in ("nvidia", "cublas", "cudnn"))
    ]
    hip_available = bool(torch.cuda.is_available() and torch.version.hip)
    device: dict[str, Any] = {"hip_available": hip_available}
    if hip_available:
        properties = torch.cuda.get_device_properties(0)
        device.update(
            {
                "count": torch.cuda.device_count(),
                "name": torch.cuda.get_device_name(0),
                "architecture": torch.cuda.get_device_capability(0),
                "total_memory_bytes": properties.total_memory,
                "multiprocessor_count": properties.multi_processor_count,
            }
        )
        torch.cuda.reset_peak_memory_stats()

    gemm: dict[str, Any] = {}
    if hip_available:
        for name, dtype in (
            ("bf16", torch.bfloat16),
            ("fp16", torch.float16),
            ("fp32", torch.float32),
            ("fp64", torch.float64),
        ):
            gemm[name] = gemm_health(dtype, gemm_size, repetitions)

    record = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "command": ".venv/bin/python scripts/audit_environment.py",
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "cpu": command(["lscpu"]),
            "memory": command(["free", "-b"]),
            "git_commit": git_head.get("stdout"),
            "git_dirty_state": {
                "dirty": bool(status_lines),
                "path_count": len(status_lines),
                "status_sha256": hashlib.sha256("\n".join(status_lines).encode()).hexdigest(),
            },
            "host_rocm_versions": rocm_versions,
            "hipcc": command(["hipcc", "--version"]),
            "rocm_smi": command(
                ["rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "vram"]
            ),
            "rocminfo": command(["rocminfo"]),
        },
        "pytorch": {
            "version": torch.__version__,
            "hip_version": torch.version.hip,
            "cuda_build_version": torch.version.cuda,
            "cuda_api_namespace_available": hasattr(torch, "cuda"),
            "device": device,
            "torch_compile": compile_health(),
            "profiler": profiler_health(),
            "triton": {
                "installed": any(item.lower().startswith("triton==") for item in torch_packages),
                "packages": [item for item in torch_packages if item.lower().startswith("triton==")],
            },
            "packages": torch_packages,
        },
        "libraries": {
            "rocblas": library_paths("librocblas.so*"),
            "rocsolver": library_paths("librocsolver.so*"),
            "cholesky_solve_health": cholesky_health(),
        },
        "gemm_health": gemm,
        "dependency_policy": {
            "forbidden_accelerator_packages": forbidden,
            "passes": not forbidden and hip_available and torch.version.cuda is None,
        },
        "compatibility": {
            "accessed_utc_date": "2026-08-29",
            "official_sources": OFFICIAL_SOURCES,
            "assessment": (
                "The installed AMD wheel reports PyTorch 2.8.0 and HIP 7.0.2, while "
                "the host exposes newer multi-version ROCm user-space directories. "
                "PyTorch 2.8 and gfx942 appear in AMD's current ROCm 7.2.4 matrix, "
                "but this exact mixed host/wheel tuple is treated as measured rather "
                "than assumed compatible. All Phase 0 claims are limited to the "
                "recorded health checks."
            ),
        },
    }
    if hip_available:
        record["pytorch"]["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    record["status"] = "PASS" if (
        record["dependency_policy"]["passes"]
        and all(row.get("supported") and row.get("finite") for row in gemm.values())
        and record["libraries"]["cholesky_solve_health"].get("finite")
        and record["pytorch"]["torch_compile"].get("usable")
        and record["pytorch"]["profiler"].get("available")
        and record["libraries"]["rocblas"]
        and record["libraries"]["rocsolver"]
    ) else "FAIL"
    return record


def text_report(record: dict[str, Any]) -> str:
    device = record["pytorch"]["device"]
    lines = [
        "AURELIS PHASE 0 ENVIRONMENT AUDIT",
        "=================================",
        f"status: {record['status']}",
        f"timestamp_utc: {record['timestamp_utc']}",
        f"git_commit: {record['environment']['git_commit']}",
        f"git_dirty_state: {json.dumps(record['environment']['git_dirty_state'], sort_keys=True)}",
        f"python: {record['environment']['python']}",
        f"kernel: {record['environment']['kernel']}",
        f"host_rocm_versions: {json.dumps(record['environment']['host_rocm_versions'], sort_keys=True)}",
        f"torch_version: {record['pytorch']['version']}",
        f"torch_hip_version: {record['pytorch']['hip_version']}",
        f"torch_cuda_build_version: {record['pytorch']['cuda_build_version']}",
        f"torch_cuda_api_namespace_available: {record['pytorch']['cuda_api_namespace_available']}",
        f"gpu_name: {device.get('name')}",
        f"gpu_total_memory_bytes: {device.get('total_memory_bytes')}",
        f"gpu_multiprocessor_count: {device.get('multiprocessor_count')}",
        f"torch_compile: {json.dumps(record['pytorch']['torch_compile'], sort_keys=True)}",
        f"profiler: {json.dumps(record['pytorch']['profiler'], sort_keys=True)}",
        f"triton: {json.dumps(record['pytorch']['triton'], sort_keys=True)}",
        f"rocblas_paths: {json.dumps(record['libraries']['rocblas'])}",
        f"rocsolver_paths: {json.dumps(record['libraries']['rocsolver'])}",
        f"cholesky_solve_health: {json.dumps(record['libraries']['cholesky_solve_health'], sort_keys=True)}",
    ]
    for dtype, result in record["gemm_health"].items():
        lines.append(f"gemm_{dtype}: {json.dumps(result, sort_keys=True)}")
    lines.extend(
        [
            f"dependency_policy: {json.dumps(record['dependency_policy'], sort_keys=True)}",
            "",
            "COMPATIBILITY ASSESSMENT",
            "------------------------",
            record["compatibility"]["assessment"],
            "",
            "OFFICIAL SOURCES",
            "----------------",
        ]
    )
    for source in record["compatibility"]["official_sources"]:
        lines.append(f"- {source['title']}: {source['url']} — {source['decision']}")
    lines.extend(
        [
            "",
            "RAW HIPCC",
            "---------",
            record["environment"]["hipcc"].get("stdout", ""),
            record["environment"]["hipcc"].get("stderr", ""),
            "",
            "RAW ROCM-SMI",
            "------------",
            record["environment"]["rocm_smi"].get("stdout", ""),
            record["environment"]["rocm_smi"].get("stderr", ""),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=REPO / "results/phase0/environment.json")
    parser.add_argument("--text", type=Path, default=REPO / "environment.txt")
    parser.add_argument("--gemm-size", type=int, default=int(os.environ.get("AURELIS_GEMM_SIZE", "2048")))
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    record = build_record(args.gemm_size, args.repetitions)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    args.text.write_text(text_report(record))
    if record["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
