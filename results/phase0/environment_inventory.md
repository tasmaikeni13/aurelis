# AURELIS Phase 0: Runtime Environment & Hardware Inventory

**Generated UTC:** `2026-09-19T07:02:04.040729+00:00`  
**Execution Substrate:** `CPU`  
**Active Enumerated Devices:** `JAX:cpu(cpu)`

---

## 1. Host Hardware

- **CPU Model:** AMD EPYC 7B12
- **Logical Cores:** 240
- **Total RAM:** 400Gi (430004359168 bytes)
- **Operating System:** Linux 5.19.0-1022-gcp (x86_64)

---

## 2. Python & Runtimes

- **Python:** `3.10.12 (main, Aug 31 2026, 10:18:17) [GCC 11.4.0]` (/usr/bin/python3)
- **PyTorch Available:** `True` (v2.14.0+cpu)
  - CUDA Available: `False`
  - Devices: []
- **JAX Available:** `True` (v0.6.2)
  - Default Backend: `cpu`
  - Devices: [{'id': 0, 'platform': 'cpu', 'device_kind': 'cpu', 'client_type': 'cpu'}]

---

## 3. Formal Verification Toolchain

- **Lean Toolchain:** `leanprover/lean4:v4.19.0`
- **Lake Version:** `Lake version 5.0.0-6caaee8 (Lean version 4.19.0)`
- **Mathlib Pin:** `c44e0c8ee63ca166450922a373c7409c5d26b00b` (https://github.com/leanprover-community/mathlib4.git)

---

## 4. Substrate Verification

Direct runtime introspection confirmed CPU execution environment. No unverified hardware is claimed.
