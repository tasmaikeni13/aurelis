# Phase 1 failed iteration: Python development headers absent

- Command: `.venv/bin/python -m pytest`
- Result: 45 passed, one failed (`tests/test_compile.py`).
- Mechanism: Triton's ROCm helper compilation failed at `#include <Python.h>`;
  `/usr/include/python3.12/Python.h` was absent.
- Classification: inherited external environment / compiler dependency.
- Research and prior evidence: this exactly reproduces
  `results/phase0/failures/environment_missing_python_headers_20260829.md`;
  the matching Ubuntu Python compiler-header package is the previously audited
  repair.
- Disposition: install `python3.12-dev`, retain the ROCm/PyTorch wheels and all
  numerical tolerances unchanged, then rerun the complete suite.

