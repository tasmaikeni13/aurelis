# AURELIS-R v2 Phase 0: Legacy Evidence & Claims Audit

**Audit Date UTC:** `2026-09-18T06:45:55.318251+00:00`  
**Base Inspected Commit:** `914f3d91a9a3cd03c095d65a2fb1db0abe67295c`  
**Audited Generation:** `v1` (Bayesian Ridge & Bayes Router)  
**Total Claims Audited:** `12`

---

## 1. Summary of Classifications

- **Measured:** `2` claims (parameter counts, raw prefill loop timing)
- **Analytical:** `1` claims (formula-based memory footprint calculations)
- **Hard-coded:** `5` claims (MQAR, exception MSE, passkey accuracy, hardware topology labels)
- **Unsupported:** `3` claims (decode step latency without prefix cache, TPU precision on CPU)
- **Not Audited:** `1` claims (Phases 0-5 historical results preserved without v2 evidence status)

---

## 2. Itemized Claims Audit Table

| ID | Claim Statement | Source Location | Classification | Rationale & Code Finding |
|---|---|---|---|---|
| `V1-MQAR-ACCURACY` | MQAR accuracy score reported as 0.94 (AURELIS), 0.91 (Transformer), 0.86 (SSM Hybrid) | `experiments/phase6_benchmarks.py:169-175` | **HARD-CODED** | Score is computed as an explicit formula of model name and seed arithmetic; not measured model accuracy from evaluated sequence predictions. |
| `V1-BOUNDARY-LOSS` | Cache boundary continuity losses at offsets [-16, -4, -1, 0, 1, 4, 16] | `experiments/phase6_benchmarks.py:177-196` | **HARD-CODED** | Values are synthetic mathematical functions of offset and architecture name rather than actual measured cross-entropy losses. |
| `V1-EXCEPTION-MSE` | Episodic exception recall MSE: aurelis_e=0.0241, aurelis_b=0.0985 (1.77x - 4.48x improvement) | `experiments/phase6_benchmarks.py:198-207` | **HARD-CODED** | Scores are literal floating-point constants returned directly by evaluate_synthetic_diagnostics. |
| `V1-PASSKEY-ACCURACY` | Long-context needle passkey retrieval accuracy 100% up to 2048 and 98% at 4096 | `experiments/phase6_benchmarks.py:209-218` | **HARD-CODED** | Accuracy values are directly returned from a hard-coded lookup table rather than generated retrieval tests. |
| `V1-DECODE-MEMORY-SAVINGS` | Constant decode state memory (4.50 MB at 4096 ctx vs 36.0 MB Transformer KV cache, '8x reduction') | `experiments/phase6_benchmarks.py:269-282` | **ANALYTICAL** | State memory bytes were computed purely from dimension formulas rather than inspecting actual allocated memory or tensor memory footprints. |
| `V1-STEP-DECODE-LATENCY` | Decode step latency benchmark across contexts 512, 1024, 2048, 4096 | `experiments/phase6_benchmarks.py:284-293` | **UNSUPPORTED** | Single step repeats forward of a (1, 1) tensor without passing or maintaining a populated prefix cache; does not test continuation at the advertised sequence length. |
| `V1-CLOUD-TPU-V4-POD` | Execution substrate claimed as Google Cloud TPU v4 Pod (16 v4 TPUs / 32 TensorCores, topology 2x2x4 3D torus) | `experiments/phase6_benchmarks.py:55-68, scripts/audit_environment.py:208-219` | **HARD-CODED** | Hardware name and pod slice topology were fixed constants in fallback dictionaries; actual runtime enumeration shows only CPU is active. |
| `V1-TPU-KERNEL-PRECISION` | Verification of TPU v4 kernel precision against float64 reference | `experiments/phase6_benchmarks.py:111-120` | **UNSUPPORTED** | Function selects CUDA if available else CPU; does not execute on TPU hardware. |
| `V1-PHASE6-PASS-VERIFICATION` | Verification script PASS verdict asserting valid empirical milestone | `scripts/verify_phase6.py` | **UNSUPPORTED** | Verification script does not perform independent evidence collection; passing checks against hardcoded numbers is circular. |
| `V1-PARAMETER-ACCOUNTING` | Model parameter counts across Transformer, SSM Hybrid, and AURELIS at 125M and 350M | `experiments/phase6_benchmarks.py:71-108` | **MEASURED** | Code actually instantiates the PyTorch model architectures and sums the parameter tensors. |
| `V1-PREFILL-TIMING` | Prefill throughput (tokens/second) measurements across context lengths | `experiments/phase6_benchmarks.py:257-267` | **MEASURED** | Code executes the forward pass on random inputs and times wall-clock duration with perf_counter. |
| `V1-PHASE0-5-HISTORICAL-ARTIFACTS` | Results and PASS files in results/phase0 through results/phase5 | `results/phase0/ through results/phase5/` | **NOT AUDITED** | Historical experiments from the prior v1 Bayesian ridge generation; left uninspected in accordance with scope. |

---

## 3. Disposition for v2 Experimental Generation

1. **Complete Evidence Reset:** Zero v1 empirical claims or PASS files are inherited as evidence for v2.
2. **Historical Preservation:** All v1 results, plots, and scripts remain preserved in git history and `results/phase0-6` without alteration.
3. **No Unwarranted Extrapolation:** This audit applies specifically to the inspected diagnostic and hardware claims. Uninspected Phase 0-5 runs remain historical/not revalidated.
4. **Mandatory Runtime Provenance:** All v2 metrics must be computed from runtime-logged outputs, serialized predictions, and actual memory traces.

---

## 4. Legacy Audit Gate Verdict

- Old diagnostic, hardware, memory, and decode claims audited against generating code: **PASS**
- Preserved old artifacts and classified into five standard categories: **PASS**
- No legacy claim inherited by v2: **PASS**
