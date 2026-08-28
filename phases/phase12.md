This is Phase 12.

This is a confirmatory scaling experiment, not an architecture-discovery experiment.

Only run this phase if Phases 0-11 justify it.

Goal:
Evaluate a frozen CSM design at approximately 125M parameters over approximately 1B training tokens on FineWeb-Edu, against a strong matched baseline.

Freeze BEFORE training:

- architecture
- parameter count
- tokenizer
- data preprocessing
- token budget
- optimizer
- LR schedule
- context length
- batch size / batch tokens
- initialization policy
- evaluation suite
- checkpoint schedule
- seeds
- primary metrics

Do not redesign CSM during the main confirmatory run.

Primary metrics:

1. validation perplexity
2. targeted associative-memory performance
3. long-context retrieval
4. in-context regression/learning probes
5. multi-hop probes if applicable
6. tokens/sec
7. wall-clock time
8. peak VRAM
9. recurrent memory bytes
10. decode latency versus context length

Analyze memory quality at equal:
- parameters
- training tokens
- recurrent-state bytes
- wall-clock budget where possible

Throughout training periodically record:
- key Gram spectra
- S conditioning
- confidence distributions
- beta/lambda statistics
- gradient norms
- per-layer memory utilization

Include negative results.

After completion, compare empirical observations against the original claims table created in Phase 0.

Classify every claim as:
SUPPORTED
PARTIALLY SUPPORTED
FALSIFIED
INCONCLUSIVE

Write:
results/phase12_1b_confirmatory.md

Do not alter the original claim wording after seeing results.
