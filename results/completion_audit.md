# Phase 0–10 completion audit

This audit maps every explicit requirement in `phases/phase0.md` through `phases/phase7.md` to authoritative repository evidence. “Pass” is limited to the current scope.

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

## Phase 6

| Requirement | Evidence | Decision |
|---|---|---|
| first learned-encoder phase, no general language model | `src/csm/learning.py` and synthetic episode generators only; report states scope | PASS |
| associative recall | three seeds and all four primary methods in `seed_metrics.csv`; learned-CSM minimum gate includes task | PASS |
| copy/selective-copy | selective-copy rows score target symbol rather than support slot; learned CSM aggregate success `1.0` | PASS |
| key-value lookup | three seeds/four methods; learned CSM aggregate success `0.979` | PASS |
| correlated-key lookup | correlation-controlled generator, geometry fields, and post-hoc regularizer ablation | PASS |
| in-context linear regression | three seeds/four methods; learned CSM aggregate normalized MSE `0.213` | PASS |
| noisy in-context regression | three seeds/four methods; learned CSM aggregate normalized MSE `0.297` | PASS |
| contextual associative recall | random raw-to-context mapping; shared-chart success `0.992` versus independent-chart ablation `0.376` | PASS |
| learn key/query/value maps | differentiable key/query chart, value encoder, output decoder; exact parameter-free memory solve | PASS |
| initially beta=lambda=1 | all Phase 6 state construction uses fixed unit beta/lambda | PASS |
| no natural orthogonality regularizer | all 84 primary task/method/seed rows precede and exclude geometry regularization | PASS |
| explicit post-natural regularizer ablation | six additional rows for correlated/contextual CSM after natural sweep; never used by gate | PASS |
| learned CSM comparison | learned CSM, frozen random-feature CSM, learned Hebbian, and learned attention on every task/seed | PASS |
| multiple seeds | `[0,1,2]`, all retained | PASS |
| Gram spectrum and cosine geometry | full mean eigenvalue spectrum plus pairwise cosine mean/std/max in every seed row | PASS |
| effective rank and singular value | effective rank, reachable/nominal capacity fractions, and minimum singular value in every seed row | PASS |
| conditioning, gradients, epsilon, retrieval | `cond(S+epsilon I)`, parameter-group gradient norms, epsilon, query scale, MSE, normalized MSE, and success | PASS |
| matched coordinate correction | manuscript Lemma 5.2a, scalar Lean theorem, shared-chart model, and retained independent-query ablation | PASS |
| learned tasks across seeds | minimum discrete success `0.958984`; maximum regression normalized MSE `0.312310` | PASS |
| outperform random representation | learned/random aggregate risk ratios `[0.207611,0.155187,0.136485]` | PASS |
| natural nontrivial geometry | minimum effective rank / reachable rank `0.303686`, above fixed `0.2` threshold | PASS |
| initial failure studied, not hidden | failed chart/scorer/denominator assumptions in report and `EXPERIMENT_LOG.md`; invalid generated files excluded | PASS |
| required report | `results/phase6_learnability.md`, JSON, two CSVs, and four inspected plots | PASS |

## Phase 7

| Requirement | Evidence | Decision |
|---|---|---|
| use successful Phase 6 architecture | all experiments use the shared key/query feature chart and same batched CSM recurrence | PASS |
| controlled evidence reliability | clean, noisy, corrupt, and irrelevant categories with noisy observable quality/consistency/relevance cues | PASS |
| learned beta | bounded positive gate weights both `S` and `C`; category means and risk retained per seed | PASS |
| beta comparisons | fixed one, cross-statistic-only generic scalar, learned precision, and oracle precision, 12 rows | PASS |
| beta reliability ordering | minimum clean/noisy gap `2.002178`; minimum noisy/bad gap `0.371851`; oracle correlation `0.961677` | PASS |
| beta risk improvement | learned/fixed risk ratios `[0.047014,0.047265,0.040675]` | PASS |
| changing latent operator | streams use controlled per-episode change points and before/after latent operators | PASS |
| learned lambda behavior | minimum stationary lambda `0.882197`; maximum change lambda `0.021273`; minimum drift correlation `0.864607` | PASS |
| lambda comparisons | fixed `1`, `0.95`, `0.8`, learned observable cue, innovation-only ablation, and oracle change, 18 rows | PASS |
| lambda adaptation benefit | learned/best-fixed risk ratios `[0.476293,0.474996,0.463929]`; post-change/fixed-one ratios all at most `0.524410` | PASS |
| changepoint semantics not forced | innovation-only failure retained; manuscript distinguishes forgetting action from detector and extra run-length state | PASS |
| affine-scan compatibility scoped | primary gate consumes token-local noisy drift cue; state-dependent innovation gate explicitly excluded from simple affine scan claim | PASS |
| joint only after separate success | source branches on the separate gate before producing any of the nine joint rows | PASS |
| joint benefit and semantics | learned/fixed risk ratios `[0.767455,0.630988,0.628516]`; beta ordering and lambda drift sensitivity pass | PASS |
| unintended usage recorded | raw absolute gate values, oracle correlations, stationary/change response, generic-gate behavior, and identifiability caveat retained | PASS |
| equal training opportunity | every drift and joint method receives 360 steps; thresholds unchanged after 120-step undertraining probe | PASS |
| failed runs studied, not hidden | innovation-only and 120-step failures logged; invalid partial artifacts excluded; innovation method remains a final ablation | PASS |
| required report | `results/phase7_gating.md`, JSON, four CSVs, and three inspected plots | PASS |

## Phase 8

| Requirement | Evidence | Decision |
|---|---|---|
| benchmark on available AMD MI300X under ROCm | `phase8_metrics.json` identifies MI300X VF, gfx942, HIP 7.0; hardware gate passes | PASS |
| preserve mathematical operation | `src/csm/systems.py` expresses token/segment affine actions; 10 systems tests compare recurrence/every prefix and gradients | PASS |
| outer-product state updates | `components.csv`: synchronized `outer_product_updates` latency, FLOPs, utilization, VRAM | PASS |
| construction of S and C | separate `construct_S_C` component row and full sweep rows | PASS |
| Cholesky factorization | separate `cholesky` component with leading FLOPs and latency | PASS |
| triangular solves | separate `triangular_solves` component | PASS |
| sequential decode | inclusive write/read loop measured at `45.172 us/token` in retained report | PASS |
| training forward and backward | separate rows; 0.470 ms forward and 0.998 ms backward in retained report | PASS |
| memory movement | 128 MiB copy row with 4.241 TB/s estimated achieved movement | PASS |
| GPU utilization | ROCm-SMI mean/peak samples retained per timing row; sequential decode mean 93.5%, peak 97% in final raw record | PASS |
| HBM bandwidth | direct copy bandwidth plus estimated per-operation traffic fields retained | PASS |
| achieved FLOPs | explicit leading flop counts and achieved TFLOP/s retained where defensible | PASS |
| kernel launch overhead | synchronized scalar launch row, `0.004 ms` | PASS |
| peak VRAM | every row records peak allocation; maximum 730,496,512 bytes | PASS |
| tokens/sec and decode microseconds/token | normalized fields in all applicable component/path/baseline rows | PASS |
| exact `d_k` sweep | complete grid over `{16,32,64,128}` | PASS |
| exact `d_v` sweep | complete grid over `{16,32,64,128}` | PASS |
| batch-size sweep | `[1,8,32]` independent-axis rows | PASS |
| sequence-length sweep | `[32,128,512]` independent-axis rows | PASS |
| head-count sweep | `[1,8,32]` plus fixed-width head-economics rows `[1,2,4,8]` | PASS |
| dtype sweep | fp32, bf16, fp16, and fp64 policies retained | PASS |
| requested bf16/fp32 precision policy | bf16 features with fp32 S/C, Cholesky, and solves is primary; alternatives measured | PASS |
| A vectorized PyTorch | `A_vectorized` path, 0.184 ms in retained report | PASS |
| B torch.compile | supported on ROCm; `B_torch_compile` path, 0.244 ms and no retained speed win | PASS |
| C chunked processing | exact `summarize_chunks`; chunk-32 latency and reduced temporary allocation retained | PASS |
| D associative segment summaries | tested associative composition and Hillis--Steele prefix scan; every prefix matches oracle | PASS |
| E best justified ROCm fusion | Inductor/Triton fuses construction; report explains why custom Cholesky fusion is not justified | PASS |
| do not assume CUDA-only strategy | runtime detects HIP/gfx; code uses ROCm PyTorch namespace, Inductor, and rocSOLVER linalg | PASS |
| every optimized path versus Phase 1 oracle | five path rows; max read relative error below `3e-6` on quantized-equal inputs | PASS |
| attention baseline | causal PyTorch scaled-dot-product attention at matched width/context | PASS |
| recurrent/linear-memory baseline | positive-feature causal linear memory measured at same shape | PASS |
| theoretical complexity | report states state, write/read, factorization, and attention asymptotics | PASS |
| actual MI300X behavior | synchronized wall-clock, VRAM, utilization, bandwidth, and throughput tables | PASS |
| memory quality per byte and wall-clock | head-economics rows report normalized recall quality/MB and quality/s | PASS |
| test many-small-head claim | fixed aggregate width: 8x16 heads retain 0.997683 quality with 1/8 the state bytes and lower measured latency than 1x128 | PASS |
| stable implementation pass gate | all seven precommitted systems/oracle/stability/coverage gates pass | PASS |
| required report | `results/phase8_mi300x_systems.md`, JSON, and six CSVs | PASS |

## Phase 9

| Requirement | Evidence | Decision |
|---|---|---|
| 5M–20M decoder-only models | all variants 5.13M–6.96M; causal/full-vs-step tests pass | PASS |
| manageable tokenizer | deterministic raw UTF-8 byte vocabulary of 256 | PASS |
| modest real-text data | checksum-pinned raw WikiText-2 train/validation splits | PASS |
| diagnostic synthetic sequences | deterministic corpus covers all six named families | PASS |
| A small Transformer | 5,574,400 parameters; full causal SDPA and incremental KV path | PASS |
| B CSM sequence mixer | 5,130,096 parameters; exact fp32 prefix states/solves and incremental state | PASS |
| C local-attention/CSM hybrid | 5,383,984 parameters; alternating local-attention and CSM layers | PASS |
| D recurrent/linear-memory baseline | 6,961,408-parameter stacked GRU mixer | PASS |
| parameter counts reasonably matched | max/min ratio 1.357 below precommitted 1.4 | PASS |
| optimizer matched | AdamW, LR schedule, warmup, decay, and clipping identical | PASS |
| token budget matched | every variant trains 10,002,432 tokens | PASS |
| batch tokens matched | every variant uses 8,192 batch tokens | PASS |
| start 10M–30M; expand only after stability | initial 10M run completed and intentionally stopped as mechanism gate | PASS |
| training loss | complete learning curves and initial/final window summaries | PASS |
| validation perplexity | byte perplexities 8.192/9.897/9.163/5.795 retained | PASS |
| gradient stability | per-log gradient norms, mean/max summaries, clipping, and zero nonfinite steps | PASS |
| NaN/Inf frequency | explicit `nonfinite_steps=0` for every architecture | PASS |
| tokens/sec | per-run measured throughput | PASS |
| peak VRAM | per-run training peak; maximum 2,991,553,024 bytes | PASS |
| recurrent-state bytes | calculated and live measured state bytes; CSM 688,128 versus Transformer 4,587,520 at decode prompt | PASS |
| decode latency | true incremental step timing; all four variants retained | PASS |
| sequence-length scaling | context `[32,64,128,256,512]` for every architecture | PASS |
| associative recall probe | trained/long rows plus autoregressive subset | PASS |
| variable tracking probe | trained/long rows plus autoregressive subset | PASS |
| repeated-name recall probe | trained/long rows; pure CSM long token-accuracy advantage 0.0588 over Transformer | PASS |
| exact-value retrieval probe | trained/long rows; best CSM-family long advantage 0.0833 | PASS |
| in-context regression probe | trained/long rows; pure CSM long advantage 0.0435 | PASS |
| multi-hop probe | trained/long rows; failures retained | PASS |
| no post-test architecture edit | generation ID/config SHA recorded before evaluation; no later generation | PASS |
| stable optimization question | CSM and hybrid loss reductions 0.511/0.527; zero nonfinite steps | PASS |
| useful natural representations | CSM validation loss 2.292, far below random-byte `ln(256)` and within preregistered Transformer ratio | PASS |
| targeted memory capability beyond parameter count | pure CSM has fewer parameters than Transformer and wins three preregistered long task token-accuracy comparisons | PASS |
| ordinary LM loss not catastrophic | best CSM-family validation-loss ratio to Transformer below 1.3 gate | PASS |
| architecture versus kernel diagnosis | report separates validation/diagnostics from throughput/VRAM/latency | PASS |
| pass gate | all seven precommitted Phase 9 gates pass | PASS |
| required report | `results/phase9_tiny_lm.md`, JSON, and four CSVs | PASS |

## Phase 10

| Requirement | Evidence | Decision |
|---|---|---|
| target 25M–50M parameters | Transformer 29,499,904; CSM 27,468,544 | PASS |
| start around 100M tokens | each of six runs trains 100,007,936 tokens | PASS |
| extend only if healthy/informative | stopped at preregistered initial 100M after first gate became interpretable; no unsupported 200M–300M claim | PASS |
| reproducible documented corpus | deterministic first 100,000,000 UTF-8 bytes of checksum-pinned raw WikiText-103 | PASS |
| matched Transformer | three seeds, identical budget/protocol | PASS |
| strongest prior CSM | pure Phase 9 CSM configuration scaled in width/depth; 16-dimensional small heads retained | PASS |
| conditional hybrid | omitted with recorded reason: Phase 9 pure CSM passed without hybridization and hybrid was weaker on aggregate long diagnostics | PASS |
| preregister parameter counts | config generation and SHA precede evaluation; counts fall inside gate | PASS |
| preregister context | 256 training context | PASS |
| preregister optimizer/LR schedule | AdamW, `3e-4`, 300 warmup, cosine/min fraction fixed | PASS |
| preregister batch/training tokens | 16,384 batch tokens and 100M target fixed | PASS |
| preregister seeds/evaluations/exclusions | seeds `[0,1,2]`, context/probe/scaling lists, four exclusions fixed | PASS |
| enough seeds | three paired seeds; no exclusion/replacement | PASS |
| validation perplexity | per-seed and mean/std; 3.261 Transformer versus 3.328 CSM mean | PASS |
| downstream memory probes | every six-family trained/long row for every seed | PASS |
| long-context probes | expanded synthetic prompts plus 512/1024 natural-context evaluations | PASS |
| tokens/sec | mean 715,420 Transformer versus 69,161 CSM | PASS |
| peak VRAM | mean 5.09 GB Transformer versus 12.50 GB CSM; per-seed rows retained | PASS |
| training wall-clock | per-seed and mean seconds; total experiment 5,116.907 seconds | PASS |
| inference throughput | prefill scaling and incremental decode tokens/s retained | PASS |
| decode latency | actual incremental latency at prompt 128/512/2048 and standard prompt 256 | PASS |
| state-memory cost | actual live state object bytes; no estimate substituted for qualifying ratio | PASS |
| same checkpoint several contexts | each final checkpoint evaluated at 128/256/512/1024 without fine-tuning | PASS |
| no perplexity-only claim | diagnostics, context behavior, wall-clock, VRAM, latency, and state all jointly reported | PASS |
| architecture effects versus seed noise | paired loss gaps `[0.021880,0.009817,0.028710]`, mean/std retained | PASS |
| compelling qualifying advantage | at prompt 2048 CSM live state 786,432 bytes versus 67,371,008; ratio 0.011673 and growth exactly 1.0x | PASS |
| countervailing failures retained | CSM slower training/decode, higher training VRAM, slightly worse perplexity, mixed probes | PASS |
| scaling gate | all nine precommitted Phase 10 checks pass through decode/state scaling advantage | PASS |
| required report | `results/phase10_small_nlp.md`, JSON, and six CSVs | PASS |

## Formal proof interpretation

`lake build` proves the exact statements listed in `lean/PROOF_COVERAGE.md`, including affine associativity, recurrence aggregation, exact simultaneous `(S,C)` action, finite-matrix PSD preservation, positive definiteness/invertibility, scalar ridge-factor properties, the one-key mismatched-query error decomposition, finite softmax normalization, and impossibility of matching negative, above-one, or nonunit-sum targets with normalized nonnegative weights. Claims outside that map—including the full matrix mismatch norm bound and floating-point backward stability—are not labeled proved. A Lean compilation or missing-premise issue is not treated as a mathematical counterexample.

## Final decision

Phase 0: **PASS**. Phase 1: **PASS**. Phase 2: **PASS**. Phase 3: **PASS**. Phase 4: **PASS**. Phase 5: **PASS**. Phase 6: **PASS**. Phase 7: **PASS**. Phase 8: **PASS**. Phase 9: **PASS**. Phase 10 scaling gate: **PASS**. The Phase 10 advantage is context-independent incremental state; it is explicitly not a training-speed, decode-latency, training-memory, or unqualified quality win.
