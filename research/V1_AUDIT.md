# V1 source and evidence audit

Audit date: 2026-09-16. Inspected base commit:
914f3d91a9a3cd03c095d65a2fb1db0abe67295c.
This was a source audit, not a reproduction of every prior experiment or a
measurement of current hardware. Python/JAX/scripts/results were left unchanged
at the user's request. Line references below refer to the inspected source.

## Concrete findings

| Source | Finding | Consequence |
|---|---|---|
| src/aurelis/functional.py, _solve and prepared_aurelis_head | Standard path calls Cholesky and solves dense key-space systems | Typical fresh factorization costs O(d_key³); factor reuse does not remove all update/read costs |
| src/aurelis/models/jax_aurelis.py:43 | Materializes per-token outer products, prefix precision/cross states | O(L d_key²+L d_value d_key) intermediates |
| src/aurelis/models/jax_aurelis.py:66 | Builds full bhij score tensor then applies a local mask | Local semantics do not produce local computational complexity |
| experiments/phase6_benchmarks.py:169 | MQAR score chosen from architecture name and seed arithmetic | Not measured model accuracy |
| experiments/phase6_benchmarks.py:177 | Boundary losses are manual functions of offset/model name | Not measured losses |
| experiments/phase6_benchmarks.py:198 | Exception and latent MSEs are literal constants | Cannot support episodic improvement or denoising claims |
| experiments/phase6_benchmarks.py:209 | Passkey accuracies are assigned by context/model | Not generated retrieval measurements |
| experiments/phase6_benchmarks.py:269 | State bytes computed from fixed dimensions | Analytical estimates, not memory traces |
| experiments/phase6_benchmarks.py:284 | Decode timing repeats m(single_step) with no prefix cache | Does not measure continuation at the advertised context |
| experiments/phase6_benchmarks.py:55 | Hardware name and total HBM are fixed; topology has fallback defaults | Labels alone cannot demonstrate execution on a TPU pod |
| experiments/phase6_benchmarks.py:111 | “TPU precision” routine chooses CPU/CUDA tensors and unspecified default float dtype | Does not by itself establish TPU or fp64 correspondence |
| scripts/verify_phase6.py | Generates publication-style PASS claims from these metric fields | PASS checks are not independent evidence for those claims |

The prefill loop does time model calls, and parameter accounting instantiates
models and counts parameters. That is a different level of evidence from the
assigned diagnostics; this audit does not label every measurement fabricated.
It also does not establish what physical device ran historical jobs.

## Publication action

Current README, paper, plan, and claim registry no longer present these Phase 6
values as learned-quality or deployment evidence. No v1 results/plots were
deleted or rewritten. Prior source/paper versions remain in git history.

Earlier phase results not individually reproduced in this audit are classified
as historical/not revalidated, not automatically invalid. Existing Lean
statements compile under their explicit hypotheses. They never proved GPU
speed, arbitrary exact recall, or the numerical provenance of Phase 6.

## Research response

V2 removes the ridge state from the normal path, requires true window kernels
and structured delta chunks, adds independent oracles and actual cached decode,
and makes all v2 performance/quality claims pending. New results use a separate
results/v2 namespace. Future scripts must compute metrics from saved outputs,
real checkpoints, actual device enumeration, and measured state/transfer bytes.
