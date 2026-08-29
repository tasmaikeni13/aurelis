# Phase 0 failed iteration: implicit experiment dtype

- Status: repaired in a subsequent iteration
- UTC date: 2026-08-29
- Classification: implementation / reproducibility
- Command: `.venv/bin/python experiments/phase0_reference.py`
- Base commit: `efe860b154ccd7003a5660b17bdc35193694e153`
- Seed at first case: `20260829`
- Device: CPU
- Intended oracle dtype: fp64

## Frozen symptom

```text
RuntimeError: Expected b and A to have the same dtype, but found b of type Float and A of type Double instead.
```

The exception arose in `torch.cholesky_solve(rhs, factor)`: the immutable
state correctly defaulted to fp64, while standalone random tensors inherited
PyTorch's process default fp32. The unit suite had set fp64 globally and
therefore did not expose this process-boundary bug.

## Research and repair

PyTorch's official `torch.cholesky_solve` documentation defines both right-hand
side `B` and Cholesky factor `L` as floating/complex tensors of the same linear
system. Source consulted 2026-08-29:
https://docs.pytorch.org/docs/2.9/generated/torch.cholesky_solve.html

The experiment now passes `dtype=torch.float64` to every generated tensor and
identity. The state API is unchanged. The predicted effect is a fully fp64
standalone experiment independent of global default dtype. This repair would
not conceal a conditioning or factorization failure; those remain separate
assertions and expected-failure tests.
