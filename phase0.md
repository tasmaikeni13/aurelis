You are beginning an empirical research program on Conjugate State Machines (CSMs).

This is Phase 0 only. Do NOT implement the full model and do NOT train a language model.

Read conjugate-state-machines.md completely and treat it as the theory specification. Extract the concrete claims that can be empirically tested, especially:

- Definition 5.1 Gauss–Markov memory
- Theorem 5.2 interpolation / finite-epsilon recall
- confidence c_t(q)
- noisy-observation / ridge-regression behavior
- linear-functional queries
- multi-hop / ricochet reads
- beta evidence weighting
- lambda forgetting/drift
- associative scan composition
- Cholesky-based decoding
- conditioning and finite-precision predictions

Set up a research repository whose purpose is falsification, not benchmark chasing.

Environment:
- AMD server
- 1x AMD MI300X
- ROCm
- no CUDA assumptions

First inspect and record:
- GPU model
- available VRAM
- ROCm version
- PyTorch version
- HIP version exposed by PyTorch
- Python version
- whether torch.compile works
- whether Triton/ROCm support is usable in this environment
- bf16/fp16/fp32 support
- basic GEMM throughput sanity
- available host RAM and CPU cores

Do not upgrade the system destructively unless necessary.

Create:

README.md
RESEARCH_PLAN.md
CLAIMS.md
EXPERIMENT_LOG.md
environment.txt
src/
tests/
experiments/
configs/
results/
plots/
scripts/

CLAIMS.md must contain a table:

claim_id
theoretical_claim
experiment_that_can_falsify_it
metric
expected_behavior
failure_interpretation
manuscript_section

Define a strict rule:
NO NLP SCALE EXPERIMENT is allowed until the synthetic and learned-memory gates pass.

Also define an experiment record format containing:
git commit
config
seed
hardware
software versions
wall-clock time
peak VRAM
metrics
plots
interpretation

Use deterministic seeds wherever practical.

At the end, produce a Phase 0 report containing:
1. environment audit
2. extracted falsifiable claims
3. planned experiment dependency graph
4. identified implementation risks
5. recommendation whether Phase 1 can begin

Do not proceed to Phase 1 automatically.
