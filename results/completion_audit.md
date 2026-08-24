# Phase 0–5 completion audit

This audit maps every explicit requirement in `phases/phase0.md` through `phases/phase5.md` to authoritative repository evidence. “Pass” is limited to the current scope.

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
| Phase 1 stopped at its gate | the Phase 1 checkpoint contains no language model, learned encoder, dyadic cascade, approximate inverse, or optimized kernel; later work began only after explicit user authorization | PASS |

## Phase 2

| Requirement | Evidence | Decision |
|---|---|---|
| dimensions `8,16,32,64,128` | committed config and all 12,600 CSV rows | PASS |
| loads from `0.125` through `2.0` | eight committed loads on both sides of `K=d_key` | PASS |
| logarithmic epsilon sweep and multiple seeds | seven epsilon values from `1e-12` to `1`, seeds `[0,1,2]` | PASS |
| orthogonal/approximately orthogonal keys | `orthogonal` generator is exact QR under capacity and normalized random above | PASS |
| random normalized Gaussian keys | `random_gaussian` regime and tests | PASS |
| correlated keys | common-component regime at correlation `0.9` | PASS |
| near-collinear and duplicate keys | both retained in complete sweep; no filtering | PASS |
| Gaussian, one-hot, binary-like values | all three regimes in every key/epsilon dataset | PASS |
| recall/exact-rate measurements | mean/median/p90/p99/max relative error, Frobenius error, exact rate | PASS |
| Gram conditioning and minimum eigenvalue | rank, minimum eigenvalue, Gram/system condition in every row | PASS |
| `c(q)` versus actual error | three `c(q)` aggregates plus inspected scatter plot | PASS |
| error versus epsilon/load/conditioning | raw fields and five required inspected plots | PASS |
| independent `K<=d_key` limit | median `2.214321e-12`, p99 `6.195306e-8` at minimum epsilon | PASS |
| compare finite-epsilon theorem bound | 6,363 applicable rows; exact ratio and explicit fp64 roundoff allowance both retained | PASS |
| no hidden loose seeds | 1,800 datasets / 12,600 rows; median, quantiles, worst cases; all seeds retained | PASS |
| dependent/over-capacity breakdown | median low-epsilon separation ratio `3.034e11`, explained as rank limitation | PASS |
| required five plots | epsilon, load, minimum Gram eigenvalue, `c(q)`, load/conditioning heatmap inspected | PASS |
| required report | `results/phase2_interpolation_report.md` | PASS |
| failed gate preserved | strict fp64/exact-real comparison failure preserved at `2f7dba6` and logged | PASS |
| do not proceed automatically | Phase 3 proceeded only because the user explicitly requested both phases | PASS |

## Phase 3

| Requirement | Evidence | Decision |
|---|---|---|
| auditable Hebbian baseline | exact `C=V^T K`, `read=Q C^T`; equation test | PASS |
| explicit normalized dot-product/softmax | explicit pair state, oracle temperature grid, normalized-weight tests | PASS |
| simple linear-attention baseline | positive `ELU+1` features, numerator/normalizer statistics, convex-weight test | PASS |
| ridge/least-squares oracle | Moore–Penrose explicit-pair oracle and interpolation test | PASS |
| same key/value dimension fairness | all five methods across the full sweep; same-dimension plot | PASS |
| same total state-byte fairness | explicit pairs capped at dense CSM byte budget; all 8,505 equal-budget recall rows audited within ceiling | PASS |
| random independent recall | `random_gaussian` rows and report no-win table | PASS |
| highly correlated and almost-colliding keys | `correlated` and `near_collinear` rows/plots | PASS |
| capacity sweep | seven loads through `2.0`, both fairness plots | PASS |
| epsilon sweep | three values and inspected CSM epsilon plot | PASS |
| value-dimension sweep | `4,16,64`, explicit report table | PASS |
| positive/negative/>1/nonunit coefficient queries | five named patterns with stored coefficient extrema/sums | PASS |
| target `sum alpha_i v_i` | standard-basis values make target exactly `alpha`; code stores direct target errors | PASS |
| convex-combination constraint | simplex-projection lower bound, weight nonnegativity/sum diagnostics, Lean theorems | PASS |
| equal-memory separation | `K=d_key=d_value` makes CSM and softmax bytes exactly equal in all 180 audited comparison rows | PASS |
| FLOPs/query and state bytes | every raw row plus representative report table | PASS |
| wall-clock latency where reasonable | 15 prepared batched-read records, 10 warmups/50 repetitions, inspected plot | PASS |
| document no-win regimes | explicit softmax/least squares beat or match stored recall; CSM reference slower; over-capacity failures scoped | PASS |
| required report | `results/phase3_baseline_separation.md` | PASS |
| pass gate | four precommitted checks pass; advantage limited to characterized equal-budget regimes | PASS |

## Phase 4

| Requirement | Evidence | Decision |
|---|---|---|
| latent linear operators and `v_i = W* k_i + sigma noise` | explicit latent matrices and model-specific value generator in `experiments/phase4_uncertainty_and_noise.py`; 4,800 duplicate rows | PASS |
| varying observation noise | committed sigma sweep `[0.1,0.5,1.0]`, raw CSV, inspected plot | PASS |
| repeated same association | repeats `[1,2,4,8,16]`; Gaussian slope `-1.018`; complete per-method rows | PASS |
| conflicting observations | explicit precisions 25 and 1; three estimator/consensus rows in `conflicting_observations.csv` | PASS |
| heteroscedastic evidence | three committed variance patterns under Gaussian, Laplace, and Student-like noise | PASS |
| missing directions in key space | observed rank 8 in a 16-dimensional key space, stated in config/report | PASS |
| OOD query directions | outside-span components `[0,.25,.5,.75,1]`; 23,040 query rows and inspected plot | PASS |
| `beta` represents true precision | every heteroscedastic write uses inverse variance; analytic weighted mean comparison error retained | PASS |
| noisy-duplicate CSM comparison | CSM, simple average, Hebbian, oracle-tuned explicit softmax, and independent ridge all present in raw/summary tables | PASS |
| CSM matches oracle ridge | maximum relative difference `5.535e-16` | PASS |
| weighting reduces prediction risk | uniform/precision-weighted Gaussian aggregate risk ratio `23.206` | PASS |
| record `c(q)` and actual squared error | every confidence-query CSV row has uncertainty, predicted MSE, actual error, normalized error, and coverage | PASS |
| Spearman uncertainty/error | Gaussian `0.704`; all misspecified values retained | PASS |
| calibration plots | eight uncertainty-quantile bins per model; raw calibration CSV and visually inspected four-panel plot | PASS |
| risk conditioned on confidence quantile | calibration and unseen-direction summaries report empirical risk by ordered uncertainty groups | PASS |
| high-error AUROC | Gaussian `0.875`; threshold definition and all model summaries recorded | PASS |
| selective prediction | exact retained fractions `[1,.8,.6,.5,.4,.2]`; Gaussian half/full risk ratio `0.404` | PASS |
| unseen distance increases uncertainty | mean `c(q)` rises `0.1111,0.1667,0.3333,0.6111,1.0`; minimum increment `0.0556` | PASS |
| exactly linear-Gaussian data | prior-matched Gaussian latent/data generation and 5,760 confidence queries | PASS |
| Laplace noise | unit-variance Laplace generation, full duplicate/precision/confidence records | PASS |
| Student-like heavy-tailed noise | standardized Gaussian scale mixture, full records; p95 normalized error `2.559` vs Gaussian `2.302` | PASS |
| nonlinear latent functions | sinusoidal nonlinear latent term; calibration ratio `1.164` and coverage `0.934` retained | PASS |
| no outside-model Bayesian claim | report gates only linear-Gaussian rows and explicitly scopes every misspecified interpretation | PASS |
| initial failed gate retained | `phase4_initial_strict_gate_failure.md` and `EXPERIMENT_LOG.md` preserve the seven-of-eight strict run and correction | PASS |
| pass gate | eight corrected, requirement-aligned checks pass; in-model behavior and useful confidence established | PASS |
| required report | `results/phase4_uncertainty_and_noise.md` plus JSON, seven CSV files, and five inspected plots | PASS |

## Phase 5

| Requirement | Evidence | Decision |
|---|---|---|
| pointer-chasing / functional graphs | permutation and many-to-one successor maps in every dataset; 12,960 raw hop rows | PASS |
| one state stores `node -> successor(node)` | values are successor codes; CSM state is constructed once and unchanged for the complete chain | PASS |
| adaptive recurrence `q_(j+1)=read(q_j)` | reusable `csm_chained_reads` primitive, equation-level tests, and prepared-factor experiment loop | PASS |
| sweep stored edges and `d_key` | dimensions `[16,32,64]`; loads through `1.5`; exact edge count in every row | PASS |
| epsilon sweep | `[1e-8,1e-4,1e-2,1e-1]`; inspected accumulation plot | PASS |
| key conditioning | orthogonal, random, correlation `0.8`, correlation `0.98`; Gram/system condition and coherence retained | PASS |
| `H in {1,2,4,8,16}` | all values represented for every method/dataset | PASS |
| `K/d_key` sweep | controlled loads `[.25,.5,1]`; random loads `[.5,1,1.5]`; capacity domain explicit | PASS |
| controlled orthogonal codes | QR-orthonormal codes restricted to the mathematically feasible `K<=d_key` domain | PASS |
| random nonorthogonal representations | random normalized and two correlated random regimes; no learned-encoder claim | PASS |
| success by hop | nearest-cosine decode and success rate in every configured hop row | PASS |
| per-hop and accumulated error | mean/p90/max endpoint vector error and accumulated prefix error in every row | PASS |
| operator norms | exact small-reference `||C(S+epsilon I)^-1||_2`, up to `3.0` on controlled many-to-one graphs | PASS |
| confidence across hops | mean `q^T A^-1 q` in every CSM hop row | PASS |
| total FLOPs | leading total estimate per row and representative report table | PASS |
| latency | 90 synchronized prepared fp64 single-query chain measurements across dimensions, geometry, methods, and H | PASS |
| one softmax access | `softmax_one` performs one access and compares directly with every H-hop target | PASS |
| equal adaptive softmax reads | `softmax_repeated` performs H accesses and reaches 100% on controlled codes | PASS |
| equivalent depth | adaptive read count and minimum layer depth explicit in raw and latency rows; report explains H attention layers versus one CSM read loop | PASS |
| architecture/systems separation | controlled architectural pass reported alongside much slower CSM reference latency; no efficiency win claimed | PASS |
| controlled expected behavior | decoded success `1.0` across all controlled gate rows; maximum H=16 vector error `1.600e-7` | PASS |
| propagation bound | full `epsilon_1 sum L^j` checked; maximum relative excess `7.973e-9` within fp64 tolerance | PASS |
| amplification diagnosis | many-to-one norm above one tested; report attributes failures to epsilon, geometry, capacity, and amplification | PASS |
| required report | `results/phase5_multihop.md` plus JSON, two CSV files, and four inspected plots | PASS |

## Formal proof interpretation

`lake build` proves the exact statements listed in `lean/PROOF_COVERAGE.md`, including affine associativity, recurrence aggregation, exact simultaneous `(S,C)` action, finite-matrix PSD preservation, positive definiteness/invertibility, scalar ridge-factor properties, finite softmax normalization, and impossibility of matching negative, above-one, or nonunit-sum targets with normalized nonnegative weights. Claims outside that map—including the full matrix norm bound and floating-point backward stability—are not labeled proved. The one Lean failure in this work was the missing `noncomputable` marker for real division; it was a compilation issue, not a counterexample. No failed proof was treated as evidence that mathematics was false.

## Final decision

Phase 0: **PASS**. Phase 1: **PASS**. Phase 2: **PASS**. Phase 3: **PASS**. Phase 4: **PASS**. Phase 5: **PASS**. Learned-memory and NLP-scale gates remain closed.
