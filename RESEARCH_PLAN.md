# CSM falsification research plan

## Objective and scope

The repository is organized to disprove claims early. Phase 0 audits the environment and names falsifiers. Phase 1 implements only the fp64 Gauss–Markov reference and tests equation fidelity. Phase 2 attacks interpolation, conditioning, and capacity; Phase 3 compares auditable baselines under dimension and byte fairness; Phase 4 tests noisy evidence and uncertainty; Phase 5 tests chained adaptive reads; Phase 6 tests learned feature maps; and Phase 7 tests learned evidence and forgetting gates. No general language model, optimized scan, dyadic cascade, or approximate inverse is part of the completed scope.

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
                                             v
                              P4 uncertainty/noise calibration
                                             │
                                             v
                              P5 multi-hop functional graphs
                                             │
                                             v
                               P6 learned feature maps
                                             │
                                             v
                           P7 learned beta/lambda gates
                                             │
                                             v
                             synthetic + learned gates both pass
                                             │
                                             v
                                  NLP-scale work may be proposed
```

The graph through the Phase 7 gate is executed in the current scope. The NLP node is shown solely to make the hard dependency explicit; no NLP experiment has been run.

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

### Phase 4 pass gate

- CSM agrees with independently assembled weighted ridge in the linear-Gaussian setting.
- Repeated Gaussian observations follow inverse-count risk and known `beta` precision improves heteroscedastic risk.
- `c(q)` has meaningful Spearman and high-error AUROC, is calibrated in-model, and supports lower-risk selective prediction.
- Missing and progressively out-of-span directions increase uncertainty.
- Laplace, Student-like, and nonlinear misspecification rows are retained and described as degradation, never Bayesian-optimality evidence.

### Phase 5 pass gate

- Controlled codes reproduce pointer chasing through `H in {1,2,4,8,16}` against one unchanged state.
- Per-hop, accumulated, decoded, confidence, conditioning, capacity, epsilon, and operator-norm diagnostics are retained.
- The full geometric propagation bound is checked; many-to-one amplification above norm one is visible.
- One-read and equal-adaptive-read softmax comparisons distinguish round count from fidelity.
- FLOPs and prepared latency are reported separately from the one-layer architectural claim.

### Phase 6 pass gate

- Learned CSM features solve all seven synthetic tasks across every seed and beat the frozen random-feature CSM on aggregate risk for every seed.
- The primary key/query parameterization uses one learned coordinate chart; an independent-query chart is retained as a failure ablation rather than silently repaired.
- Regression normalized MSE, discrete success, full Gram spectra, effective rank, reachable capacity, conditioning, query calibration, and gradient norms are retained.
- No orthogonality or covariance regularizer enters the natural gate; regularization is tested only afterward as an explicit ablation.
- Hebbian and attention controls receive learned encoders of the same size, and attention is not required to lose.

### Phase 7 pass gate

- Learned beta improves fixed beta on varying-quality evidence across every seed, preserves reliability ordering, and correlates with oracle precision.
- Learned lambda improves both fixed-lambda tradeoffs and post-change risk across every seed, remains near one in stationary periods, and falls at observed changes.
- An innovation-only gate is retained as a failed detector ablation: a forgetting action is not itself a changepoint posterior.
- The primary scan-compatible lambda gate consumes a noisy locally observable drift cue; an exact hidden-changepoint model would require an additional run-length belief state.
- Joint beta/lambda training runs only after both separate gates pass and must retain both semantic orderings while improving the fixed-gate control.

### Non-negotiable NLP gate

**NO NLP SCALE EXPERIMENT is allowed until both the complete synthetic-memory gate and the learned-memory gate pass.** Both prerequisites now pass through Phase 7. This permits a future NLP proposal; it is not evidence of NLP value and does not automatically authorize one.

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
| Model-free confidence is called calibrated | Invalid statistical claim | Gate calibration only on prior-matched linear-Gaussian data; retain heavy-tailed and nonlinear degradation separately. |
| Decoded hop success hides vector drift | False exactness claim | Record vector, accumulated, and nearest-code errors together at every configured hop. |
| Unit successor codes are assumed contractive | Incorrect `H epsilon_1` bound | Measure the exact small reference operator norm and test many-to-one graphs with `L > 1`. |
| Adaptive rounds are conflated with runtime | Architectural success marketed as efficiency | Record layer-depth semantics separately from FLOPs and synchronized prepared latency. |
| Key and query encoders learn incompatible charts | The solver fits one coordinate system and is evaluated in another | Share the feature map, learn only a scalar query calibration, and retain independent encoders as a failure ablation. |
| Retrieval success is misread as proof of high rank | A degenerate task may pass with collapsed features | Record spectrum, effective rank, minimum singular value, cosine statistics, and reachable capacity separately from task loss. |
| Capacity is divided by an unreachable rank | A correct low-load representation looks artificially collapsed | Normalize effective rank by `min(K,d_key)`, while retaining nominal `effective_rank/d_key` separately. |
| Repeated symbols make slot and symbol accuracy disagree | Selective-copy scorer reports a false failure | Score the decoded target symbol; retain vector error independently. |
| Lambda is treated as a hidden changepoint detector | Bayesian tempering semantics are overstated and scan compatibility is lost | Distinguish the forgetting action from detection; use observable cues for the scan-compatible test and retain the innovation-only ablation. |
| Gate methods receive unequal optimization opportunity | Undertraining masquerades as a theory failure | Use the same 360-step budget for every drift and joint method; keep thresholds fixed. |

## Automation

```bash
scripts/bootstrap.sh
scripts/run_phase01.sh
.venv/bin/python experiments/phase2_interpolation.py
.venv/bin/python experiments/phase3_baseline_separation.py
.venv/bin/python experiments/phase4_uncertainty_and_noise.py
.venv/bin/python experiments/phase5_multihop.py
.venv/bin/python experiments/phase6_learnability.py
.venv/bin/python experiments/phase7_gating.py
```

The first command creates the local environment from pinned requirements. The second reruns the environment audit, Python tests, Lean proofs, and Phase 1 measurements in fail-fast order. The remaining commands reproduce the separately gated Phase 2–7 records.
