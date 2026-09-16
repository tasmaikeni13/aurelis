# Phase 0 failed iteration: Inductor-generated prefix-scan kernel

- Status: repaired by separating compiled prepared-head execution from eager
  exact prefix construction
- UTC date: 2026-08-29
- Classification: hardware/kernel / compiler lowering
- Command: `.venv/bin/python benchmarks/phase0_components.py`
- Config: `configs/phase0_benchmark.json`
- Seed: `20260829`
- Device: Google Cloud TPU v4 Pod
- Dtype: fp32, with CPU/fp64 oracle
- PyTorch/Triton: `2.8.0` /
  `N/A` / `3.4.0`

## Frozen failure

Eager vectorized prefix construction and its backward pass completed. The
first Inductor compile of the same all-prefix graph failed while compiling the
generated cumulative-sum kernel:

```text
triton.compiler.errors.CompilationError: at 1:0:
def triton_per_fused_cumsum_mul_0(...):
^
AttributeError("type object 'constexpr' has no attribute '_flatten_ir_types'")
```

The complete tool traceback classified this as
`torch._inductor.exc.InductorError: SubprocException`. No result or tolerance
was evaluated after the compiler exception.

## Research and repair

Documentation notes that TorchInductor emits Triton kernels on accelerators while also
calling accelerator libraries. PyTorch's compiler guidance recommends isolating the
tensor region that is amenable to compilation when another region is not
supported. Current Triton source defines `_flatten_ir_types` in the compiler
type hierarchy, so the missing attribute in the paired 3.4 wheel is treated as
a generated-kernel/compiler compatibility failure, not an AURELIS equation
failure. Sources consulted 2026-08-29:

- https://docs.pytorch.org/docs/stable/torch.compiler.html
- https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html
- https://github.com/triton-lang/triton/blob/main/python/triton/language/core.py

The repair keeps exact all-prefix construction as an eager, independently
tested training reference and compiles the prepared AURELIS head—the
factorization, solves, local attention, residual outputs, covariance router,
and backward graph—with `fullgraph=True`. This predicts removal of the failing
generated prefix-scan kernel while preserving the architecture's complete read
and gradient mechanism. It does not claim the all-prefix scan is compiled, and
that limitation remains in the PASS record.
