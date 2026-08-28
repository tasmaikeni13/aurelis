# Phase 0 — AURELIS migration and ROCm reference substrate

Read `aurelis.md`, `research/LITERATURE_REVIEW.md`, every file in `lean/`, and
`phases/AUTONOMY_PROTOCOL.md` completely before acting. Execute the shared
failure-repair loop until this phase passes.

This phase turns the theory repository into an AURELIS implementation research
repository. Do not train a language model or claim accelerator superiority.

## 0.1 Remove the obsolete identity

Erase the old CSM identity from the working tree, not from Git history:

- rename the Python package and all imports to `aurelis`;
- replace project metadata, module/class names, docs, configs, scripts, tests,
  result schemas, plot labels, comments, and environment headers;
- delete stale experiments/results/plots whose equations or claims do not
  represent AURELIS; do not cosmetically relabel old data;
- preserve only reusable mechanisms after proving their equations match the
  new paper; and
- make case-insensitive `rg` for the old acronym and expanded name return no
  tracked-working-tree matches outside an explicitly generated migration audit
  that quotes the search term. The `.git` directory is out of scope.

The authoritative manuscript is `aurelis.md`; there must be no second legacy
paper.

## 0.2 Build independent reference paths

Create a transparent fp64 CPU implementation with immutable state for:

- the delayed FIFO handoff and exact occurrence partition;
- `P=Lambda+sum beta kk^T`, `C=sum beta vk^T`;
- Cholesky/solve reads without explicit inversion;
- local causal softmax with shared key/value weights;
- remote, full-residual, AURELIS-B, and AURELIS-E outputs;
- `h,V_R,V_H,K_RH,g_raw,g_B`, plus diagnostic residuals;
- batched/multi-head shapes and empty/warm-up cache behavior; and
- autograd through keys, queries, values, evidence, temperature, projections,
  and any learned episodic responsibility.

Create an independently assembled historical oracle from the full prefix. The
streaming and oracle paths must not share state-update logic. Keep a tiny
dimension-capped explicit inverse only as a test oracle.

## 0.3 Build the ROCm/MI300X substrate

The server target is one AMD Instinct MI300X under ROCm. Inspect and record
actual state before installation or optimization:

- GPU name/architecture, VRAM, ROCm/HIP, driver, PyTorch, Python, kernel, host
  RAM/CPU, and git state;
- `torch.version.hip`, `torch.cuda.is_available()`, dtype support, rocBLAS,
  rocSOLVER, TorchInductor, Triton/ROCm, and profiler availability;
- bf16/fp16/fp32/fp64 GEMM health checks with synchronized timing; and
- installed versus officially compatible versions.

PyTorch deliberately reuses `torch.cuda` on ROCm. This API name is allowed;
NVIDIA libraries, CUDA wheels, CUDA toolkit assumptions, `nvcc`, and
NVIDIA-only kernels are not. Detect ROCm with `torch.version.hip`. Consult
current official AMD ROCm, rocSOLVER, and PyTorch HIP documentation before
choosing versions or kernel paths.

Implement reproducible scripts for:

- non-destructive environment bootstrap and audit;
- CPU/fp64 reference tests;
- eager and `torch.compile` AURELIS forward/backward;
- sequential decode with ring-buffer handoff and stable factor maintenance;
- vectorized exact training reference;
- component benchmarks for outer updates, local attention, factorization,
  triangular solves, routing, and full head; and
- an optional Triton/ROCm prototype only after the eager/Inductor path is
  correct. A custom kernel is not a phase requirement if measurement shows it
  is unjustified.

## 0.4 Required repository contract

Create or replace `README.md`, `CLAIMS.md`, `RESEARCH_PLAN.md`,
`EXPERIMENT_LOG.md`, `environment.txt`, `pyproject.toml`, `requirements.txt`,
and structured `src/`, `tests/`, `experiments/`, `configs/`, `results/`,
`plots/`, and `scripts/` content. The claim registry must distinguish theorem,
Lean coverage, numerical evidence, and pending empirical claims.

Define one command that runs environment audit, Python unit/property tests,
Lean build, and a small AURELIS reference experiment in fail-fast order.

## PASS gates

- No stale identity remains by the scoped case-insensitive search.
- The historical oracle and streaming path agree in fp64 over random lengths,
  windows, dimensions, evidence, empty-cache, handoff-boundary, repeated-key,
  near-singular, and over-capacity cases.
- The cache and remote occurrence IDs form a disjoint exhaustive partition at
  every step.
- Cholesky, dense solve, and capped inverse agree inside conditioned domains;
  expected failures outside them are retained.
- Autograd and gradcheck cover every declared differentiable input.
- AURELIS-B gate matches dense one-dimensional variance minimization; AURELIS-E
  exact one-hot hits pass.
- Eager, compiled, and any custom ROCm paths agree with fp64 after dtype-aware
  tolerances.
- MI300X/ROCm is measured, with no NVIDIA dependency and no unsupported version
  assumption.
- Full Lean build and all Phase 0 tests pass from the documented command.
- `results/phase0/PASS.md` satisfies the shared PASS record.
