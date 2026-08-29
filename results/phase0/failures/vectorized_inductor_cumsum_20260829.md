# Phase 0 failed iteration: Inductor-generated prefix-scan kernel

- Status: repaired by separating compiled prepared-head execution from eager
  exact prefix construction
- UTC date: 2026-08-29
- Classification: hardware/kernel / compiler lowering
- Command: `.venv/bin/python benchmarks/phase0_components.py`
- Config: `configs/phase0_benchmark.json`
- Seed: `20260829`
- Device: AMD Instinct MI300X VF (`gfx942`)
- Dtype: fp32, with CPU/fp64 oracle
- PyTorch/HIP/Triton: `2.8.0+rocm7.0.2.git245bf6ed` /
  `7.0.51831-7c9236b16` / `3.4.0+rocm7.0.2.gitf9e5bf54`

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

AMD documents that TorchInductor emits Triton kernels on AMD GPUs while also
calling ROCm libraries. PyTorch's compiler guidance recommends isolating the
tensor region that is amenable to compilation when another region is not
supported. Current Triton source defines `_flatten_ir_types` in the compiler
type hierarchy, so the missing attribute in the paired 3.4 wheel is treated as
a generated-kernel/compiler compatibility failure, not an AURELIS equation
failure. Sources consulted 2026-08-29:

- https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/workload.html
- https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html
- https://github.com/triton-lang/triton/blob/main/python/triton/language/core.py

The repair keeps exact all-prefix construction as an eager, independently
tested training reference and compiles the prepared AURELIS head—the
factorization, solves, local attention, residual outputs, covariance router,
and backward graph—with `fullgraph=True`. This predicts removal of the failing
generated prefix-scan kernel while preserving the architecture's complete read
and gradient mechanism. It does not claim the all-prefix scan is compiled, and
that limitation remains in the PASS record.
