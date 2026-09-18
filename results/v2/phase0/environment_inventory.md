# AURELIS-R v2 Phase 0: Runtime Environment & Device Inventory

**Generated UTC:** `2026-09-18T06:45:54.083436+00:00`  
**Theory Revision:** `v2.0`  
**Git HEAD:** `9921cfb207c7712102d2a02439a77b49dbecf74a`  
**Working Tree Status:** `DIRTY (2 paths)`

---

## 1. Hardware & Platform Inventory (Runtime Enumerated)

- **System OS:** Linux 5.19.0-1022-gcp (x86_64)
- **Host Kernel:** `Linux t1v-n-49ba2866-w-0 5.19.0-1022-gcp #24~22.04.1-Ubuntu SMP Sun Apr 23 09:51:08 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux`
- **CPU Architecture:** AMD EPYC 7B12 (240 logical processors, 2 sockets, 60 cores/socket, 2 threads/core)
- **Host RAM:** 400.0 GiB total physical memory (~392 GiB available)
- **Swap Space:** 0 B
- **NUMA Topology:** 2 NUMA nodes (Node 0: CPUs 0-59, 120-179; Node 1: CPUs 60-119, 180-239)
- **PCI Accelerator Device Nodes:** ["/dev/accel0", "/dev/accel1", "/dev/accel2", "/dev/accel3"]
- **TPU Runtime Status:** Host contains /dev/accel nodes from GCP TPU VM instance, but TPU slice coordinator is unreachable/inactive; libtpu cannot initialize local TPU runtime. Valid runtime execution is strictly CPU.

> [!IMPORTANT]
> In accordance with AUTONOMY_PROTOCOL.md, all devices are returned by runtime enumeration. While the host is an instance with physical Google Device 005e PCI interfaces (`/dev/accel0-3`), the TPU slice coordinator service is unreachable in this standalone worker session. Therefore, the active compute device for all executable workloads is strictly runtime-enumerated as **CPU**.

---

## 2. Software & Runtime Environments

- **Python Runtime:** `3.10.12 (main, Aug 31 2026, 10:18:17) [GCC 11.4.0]` at `/usr/bin/python3`
- **PyTorch:** Version `2.14.0+cpu`, CUDA Available: `False`, Devices: `cpu`
- **JAX:** Version `0.6.2`, Platform: `cpu`, Active Devices: `['TFRT_CPU_0']`
- **Lean 4 Toolchain:** `leanprover/lean4:v4.19.0`
  - `lean --version`: `Lean (version 4.19.0, x86_64-unknown-linux-gnu, commit 6caaee842e94, Release)`
  - `lake --version`: `Lake version 5.0.0-6caaee8 (Lean version 4.19.0)`
- **Mathlib Pin:** `v4.19.0` (`commit c44e0c8ee63ca166450922a373c7409c5d26b00b`)
- **CPU GEMM Benchmark:** 1024x1024 Float32 GEMM in `76.5 ms` (`28.07 GFLOPS`)

---

## 3. Enumeration Integrity Gate Verdict

- Every claimed device is returned by runtime enumeration: **PASS**
- No hard-coded hardware labels or TPU pod assumptions: **PASS**
- Overall Environment Gate: **PASS**
