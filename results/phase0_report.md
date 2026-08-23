# Phase 0 report

## 1. Environment audit

Phase 0 environment gate: **PASS**.

| Item | Observed |
|---|---|
| GPU | AMD Instinct MI300X VF, `gfx942:sramecc+:xnack-` |
| Visible VRAM | 191.688 GiB (205,822,885,888 bytes from ROCm SMI) |
| ROCm | 7.0.2 |
| PyTorch | 2.8.0+rocm7.0.2.git245bf6ed |
| PyTorch HIP | 7.0.51831-7c9236b16 |
| Python | 3.12.3 |
| `torch.compile` | usable with Inductor; exact match in audit workload |
| Triton/ROCm | usable; direct add kernel max absolute error `2.3841858e-7` |
| Dtypes | bf16, fp16, fp32, and fp64 GEMMs all supported and finite |
| Median 8192 GEMM sanity | bf16 617.1, fp16 590.0, fp32 120.3, fp64 69.7 TFLOP/s |
| Host | 20 physical/logical cores; 235.948 GiB RAM |
| Audit peak VRAM | 2.387 GiB |

The exact raw audit, driver output, timing samples, and versions are in [`environment.txt`](../environment.txt) and [`environment.json`](environment.json). GEMM measurements are basic sanity checks with no tuning, power control, clock control, or comparative protocol.

The repository-local `.venv` keeps PyTorch and research dependencies isolated. The only host packages added were Ubuntu’s `python3.12-venv` and `elan`; no ROCm component, driver, kernel, or system Python package was upgraded.

## 2. Extracted falsifiable claims

[`CLAIMS.md`](../CLAIMS.md) contains 20 concrete claims with experiments, metrics, expected behavior, and failure interpretations. Phase 0 specifically covers:

- Gauss–Markov sufficient-statistic recurrence and PSD invariants.
- interpolation and finite-epsilon error versus key geometry.
- posterior uncertainty `c_t(q)`, including the corrected non-unit unwritten-direction formula.
- noisy observations, beta precision weighting, and ridge risk.
- signed linear-functional reads and convex-hull limits.
- ricochet reads with the corrected explicit contraction premise for the `H epsilon_1` simplification.
- lambda forgetting, affine scan composition, exact solve/Cholesky behavior, conditioning, and reduced precision.

The first-principles audit corrected three material overstatements in the theory at commit `223afde`: the randomized lower-bound appendix now uses valid bitwise Fano; the duplicate-noise smoothing comparison is scoped to evidence-blind heteroscedastic averaging; and normalized value columns no longer incorrectly imply a contractive ricochet operator. It also corrected `c_t(q)=epsilon^-1 ||q||^2` on unwritten directions and made clear that `c` is uncertainty, not increasing confidence.

## 3. Planned dependency graph

```text
environment + claims + records
            |
            v
fp64 recurrence ── historical recomputation
       |                    |
       ├── solve/oracle ────┤
       └── PSD/autograd ────┘
                    |
                    v
          Phase 1 reference gate
                    |
        future synthetic claim gate
                    |
        future learned-memory gate
                    |
                    v
       only then: NLP-scale proposal
```

The authoritative full graph and gate definitions are in [`RESEARCH_PLAN.md`](../RESEARCH_PLAN.md).

## 4. Identified implementation risks

The highest Phase 1 risks are false agreement through shared code, inverse use hidden in the reference path, suffix-decay indexing, conditioning being mistaken for algebraic failure, inverted interpretation of `c(q)`, and overclaiming Lean coverage. Controls are independent implementations, a dimension-capped oracle, closed-form cases, condition-number logging, explicit variance terminology, and a theorem-by-theorem proof coverage map. The complete risk register is in [`RESEARCH_PLAN.md`](../RESEARCH_PLAN.md).

## 5. Recommendation

**Phase 1 can begin.** The host exposes a working fp64 ROCm PyTorch stack, all required repository structure and records exist, and no environment blocker remains. This recommendation is the Phase 0 boundary decision; Phase 1 proceeds because the user explicitly requested both phases, not because Phase 0 automatically authorizes later work.

Phase 1 may validate only the transparent reference equations. It must not train a language model, optimize kernels, add learned encoders, implement the cascade, or proceed beyond its own report.

