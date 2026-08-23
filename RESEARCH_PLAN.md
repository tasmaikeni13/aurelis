# CSM falsification research plan

## Objective and scope

The repository is organized to disprove claims early. Phase 0 audits the environment and names falsifiers. Phase 1 implements only the fp64 Gauss–Markov reference and tests equation fidelity. Phase 2 attacks interpolation, conditioning, and capacity; Phase 3 compares auditable baselines under dimension and byte fairness. No language model, learned encoder, optimized scan, dyadic cascade, or approximate inverse belongs in these phases.

## Dependency graph

```text
P0-A theory extraction ─────┐
P0-B ROCm environment audit ├──> P0 gate: reference work is feasible
P0-C record schema/risks ───┘                 │
                                              v
                              P1-A fp64 recurrence + recomputation
                                      │               │
                                      v               v
                              P1-B solve/oracle    P1-C PSD + autograd
                                      └──────┬────────┘
                                             v
                                  P1 gate: equations agree
                                             │
                                             v
                              P2 interpolation/capacity gate
                                             │
                                             v
                               P3 fair-baseline separation
                                             │
                    ┌────────────────────────┴──────────────────────┐
                    v                                               v
       remaining synthetic claim suite                  future learned-memory suite
                    └────────────────────────┬──────────────────────┘
                                             v
                             synthetic + learned gates both pass
                                             │
                                             v
                                  NLP-scale work may be proposed
```

The graph through the Phase 3 gate is executed in the current scope. The remaining downstream nodes are shown solely to make the hard dependency explicit.

## Gates

### Phase 0 gate

- MI300X visible through PyTorch HIP.
- fp64 reference operations and Cholesky available.
- `torch.compile` and Triton status measured rather than assumed.
- bf16/fp16/fp32 support and a GEMM sanity measurement recorded.
- claims, falsifiers, risks, deterministic seed rule, and experiment schema committed.

### Phase 1 pass gate

- Dimensions `{2,4,8,16,32}` all exercised.
- Well-conditioned fp64 sequential and historical-recompute states/reads agree within configured numerical thresholds.
- S is symmetric PSD within a scale-aware tolerance.
- Cholesky and `torch.linalg.solve` agree with a tiny direct-inverse oracle.
- Autograd reaches keys, values, beta, lambda, and queries; `gradcheck` passes.
- All named pathologies have explicit assertions and quantitative output.
- `results/phase1_report.md`, CSV metrics, JSON record, and plots are generated from one pinned config.
- Lean build passes for its declared proof coverage. Lean-incomplete claims remain labeled unproved rather than failed.

### Phase 2 pass gate

- Full-row-rank recall visibly approaches interpolation as epsilon decreases.
- The finite-epsilon bound is compared only in its valid domain, with exact-real and fp64 effects distinguished.
- Dependent and over-capacity breakdown is retained, measurable, and explained through rank.
- All dimensions, loads, seeds, key/value regimes, quantiles, worst cases, and required plots are recorded.

### Phase 3 pass gate

- Hebbian, explicit softmax, positive-feature linear attention, and least-squares oracle baselines are equation-tested.
- Same-dimension and equal-state-byte regimes are both reported.
- A fidelity separation survives equal-byte comparison in a clearly characterized regime.
- Random, correlated, near-collision, capacity, epsilon, and value-dimension sweeps are retained.
- Positive, negative, above-one, and nonunit-sum linear-functional queries are tested against convex-hull lower bounds.
- FLOPs, state bytes, reasonable latency diagnostics, and regimes with no CSM win are reported.

### Non-negotiable NLP gate

**NO NLP SCALE EXPERIMENT is allowed until both the complete synthetic-memory gate and a later learned-memory gate pass.** Phase 3 is still synthetic and unlearned, so its pass is not sufficient.

## Determinism policy

- Every test receives a fixed seed via `tests/conftest.py`.
- Experiments use the seeds in committed JSON plus fixed specialized seeds visible in source.
- fp64 is the reference dtype; reduced precision is compared against it, never treated as its own oracle.
- GPU synchronization brackets timing and measurement boundaries.
- Results include commit, dirty status, hardware/software, wall time, peak VRAM, config, metrics, plots, and interpretation.

## Implementation risks and controls

| Risk | Consequence | Control |
|---|---|---|
| Sequential and recomputation paths accidentally share logic | False agreement | Separate suffix-weight construction in `recompute_state`; tiny inverse oracle is separate again. |
| Direct inverse leaks into production code | Misleading numerical stability | Inverse exists only in `direct_inverse_oracle`, dimension-capped at 32; reads use Cholesky or solve. |
| beta/lambda off-by-one | Wrong forgetting semantics | Closed-form two-write test and full suffix-product recomputation. |
| Near-collinear keys make a correct algorithm look broken | Misdiagnosed theory | Record condition number and compare fp32 with fp64 under explicit epsilon. |
| Cholesky failure at tiny epsilon | Abrupt experiment abort | Pathology cases use fp64, record conditioning, and distinguish expected representational limits from recurrence errors. |
| “confidence” direction is misread | Router learns inverted behavior | Document `c(q)` as variance: smaller is more confident; regression test includes non-unit unwritten query. |
| Stochastic Monte Carlo noise | False pass/fail | Fixed seeds, enough trials, analytic predictions, and ratio rather than a single realization. |
| Lean build failure is treated as theorem refutation | Incorrect theory edits | Diagnose encoding, assumptions, and proof completeness first; require a faithful counterexample or missing-premise demonstration before changing math. |
| Lean proves an encoding weaker than the prose | False assurance | Maintain `lean/PROOF_COVERAGE.md` with explicit status boundaries. |
| ROCm API is mistaken for CUDA dependency | Portability regression | Use PyTorch’s ROCm `torch.cuda` namespace only; no CUDA toolkit or NVIDIA package. |
| Performance sanity is marketed as benchmark | Invalid conclusion | Environment GEMM values are health checks, not tuned or comparative results. |
| Dirty worktree obscures provenance | Irreproducible report | Record tested commit and dirty flag; checkpoint implementation before generated results. |

## Automation

```bash
scripts/bootstrap.sh
scripts/run_phase01.sh
.venv/bin/python experiments/phase2_interpolation.py
.venv/bin/python experiments/phase3_baseline_separation.py
```

The first command creates the local environment from pinned requirements. The second reruns the environment audit, Python tests, Lean proofs, and Phase 1 measurements in fail-fast order. The final two commands reproduce the separately gated Phase 2 and Phase 3 records.
