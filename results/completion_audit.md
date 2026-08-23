# Phase 0/1 completion audit

This audit maps every explicit requirement in `phase0.md` and `phase1.md` to authoritative repository evidence. “Pass” is limited to the current scope.

## Phase 0

| Requirement | Evidence | Decision |
|---|---|---|
| Read theory and extract testable claims | `CLAIMS.md` has 20 falsifiable claims spanning every named Phase 0 topic; corrected theory at commit `223afde` | PASS |
| GPU model and VRAM | `environment.txt`, `results/environment.json`: MI300X VF, 191.688 GiB | PASS |
| ROCm/PyTorch/HIP/Python versions | exact values in both environment records | PASS |
| `torch.compile` | Inductor audit output, zero max error | PASS |
| Triton/ROCm usability | direct Triton kernel, finite output, max error `2.384e-7` | PASS |
| bf16/fp16/fp32 support | all GEMM probes supported and finite; fp64 also recorded | PASS |
| GEMM sanity | seven measured 8192 runs per dtype in `results/environment.json` | PASS |
| host RAM and CPU cores | 235.948 GiB, 20 physical/logical cores | PASS |
| required repository tree | README, plan, claims, log, environment, src/tests/experiments/configs/results/plots/scripts all populated | PASS |
| exact claims-table columns | `CLAIMS.md` header matches all seven required names | PASS |
| hard NLP rule | README, claims, and research plan state the rule verbatim | PASS |
| experiment-record format | nine required fields plus timestamp/dirty status in `EXPERIMENT_LOG.md` | PASS |
| deterministic seeds | autouse test fixture, committed config, explicit experiment generators | PASS |
| five-part Phase 0 report | `results/phase0_report.md` sections 1–5 | PASS |
| no automatic progression | report records that explicit user authorization, not Phase 0 itself, allowed Phase 1 | PASS |

## Phase 1

| Requirement | Evidence | Decision |
|---|---|---|
| smallest transparent PyTorch reference | `src/csm/memory.py`; no learned parameters or optimized/cascade code | PASS |
| explicit fp64 version | `FP64GaussMarkovMemory`; dtype tests | PASS |
| no inverse in production paths | Cholesky/default and solve paths; inverse isolated in `direct_inverse_oracle` | PASS |
| tiny direct-inverse oracle | dimension cap 32 and rejection test at 33 | PASS |
| dimensions 2,4,8,16,32 | unit parameterization and 25-case GPU experiment CSV | PASS |
| random beta, lambda, epsilon | randomized beta/lambda tests and configured epsilon/conditioning sweeps | PASS |
| sequential vs historical recomputation | independent suffix-product implementation; max state error `8.882e-16` | PASS |
| PSD verification | eigenspectrum tests/CSV plus Lean matrix proof; min observed `-2.856e-16` | PASS |
| autograd | gradients reach keys, values, beta, lambda, query and remain finite | PASS |
| gradcheck | unit test and experiment internal gradcheck pass | PASS |
| repeated keys | unit assertion and quantitative pathology record | PASS |
| nearly collinear keys | oracle comparison and quantitative pathology record | PASS |
| beta=0 | exact skip-write test and pathology record | PASS |
| lambda=1 | direct undiscounted-sum equality test | PASS |
| lambda<1 | closed-form suffix-decay test and random recurrence suite | PASS |
| tiny epsilon | interpolation sweep and well-conditioned exactness test | PASS |
| very large beta | finite state/read pathology plus `1e150` state test | PASS |
| zero values | exact zero-read test and pathology record | PASS |
| single observation | closed-form read/confidence equality test | PASS |
| plots/errors | four inspected plots, five CSV files, full JSON record | PASS |
| required report | `results/phase1_report.md` | PASS |
| fp64 pass gate | state/read thresholds, PSD, gradcheck, pathologies all pass | PASS |
| do not change definitions to pass | recurrence equations unchanged; theory edits correct separate overstated claims and missing premises | PASS |
| no work beyond Phase 1 | no language model, learned encoder, dyadic cascade implementation, approximate inverse, or optimized kernel | PASS |

## Formal proof interpretation

`lake build` proves the exact statements listed in `lean/PROOF_COVERAGE.md`, including affine associativity, recurrence aggregation, exact simultaneous `(S,C)` action, actual finite-matrix PSD preservation, positive definiteness of `S+epsilon I`, invertibility, and scalar one-key identities. Claims outside that map are not labeled proved. Build failures encountered during development were syntax/tactic/API issues; none supplied a mathematical counterexample. The manuscript changes were instead justified by explicit first-principles derivations and executable counterexamples.

## Final decision

Phase 0: **PASS**. Phase 1: **PASS**. All downstream gates remain closed.

