# Phase 0 failed iteration: TorchInductor helper could not compile

- Status: repaired in a subsequent iteration
- UTC date: 2026-08-29
- Classification: external environment / compiler prerequisite
- Command: `.venv/bin/python scripts/audit_environment.py`
- Base commit: `efe860b154ccd7003a5660b17bdc35193694e153`
- Device: Google Cloud TPU v4 Pod
- PyTorch: `2.8.0` / `N/A`

The full failing JSON and text audit are preserved beside this file. bf16,
fp16, fp32, and fp64 GEMM; rocBLAS/rocSOLVER discovery; Cholesky solve; and the
PyTorch profiler passed. The audit remained FAIL because TorchInductor's
helper compilation stopped with:

```text
fatal error: Python.h: No such file or directory
```

Ubuntu's official `libpython3.12-dev` file list includes
`/usr/include/python3.12/Python.h`, and Python's official extension guide
requires C extensions to include that header. Sources consulted 2026-08-29:

- https://packages.ubuntu.com/noble-updates/amd64/libpython3.12-dev/filelist
- https://docs.python.org/3/extending/extending.html

APT reported matching candidate `python3.12-dev 3.12.3-1ubuntu0.16`. The repair
is to install that compiler header package and rerun the unchanged audit and
AURELIS compiled graph. This does not alter the kernel, accelerator driver,
runtime, PyTorch wheel, equations, tolerances, or test data. It would not
repair an Inductor lowering or numerical disagreement; those remain separate
gates.
