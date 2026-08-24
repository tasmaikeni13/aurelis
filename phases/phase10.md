This is Phase 10.

Goal:
Run the first meaningful natural-language controlled comparison after all mechanism gates have passed.

Target model scale:
approximately 25M-50M parameters.

Target token budget:
start around 100M tokens.
Only extend to approximately 200M-300M if runs are healthy and informative.

Dataset:
a reproducible subset of FineWeb-Edu or another clearly documented corpus.

Architectures:
1. matched Transformer
2. strongest CSM configuration from previous phases
3. strongest CSM hybrid if hybridization proved necessary

Pre-register:
- parameter counts
- context length
- optimizer
- LR schedule
- batch tokens
- training tokens
- seeds
- evaluation tasks
- exclusion criteria

Run at least enough seeds to distinguish architecture effects from obvious seed noise for the critical comparisons.

Report:
- validation perplexity
- downstream memory probes
- long-context probes
- tokens/sec
- peak VRAM
- training wall-clock
- inference throughput
- decode latency
- state-memory cost

Also evaluate the same checkpoint at several context lengths.

No claims based solely on perplexity.

PASS GATE FOR SCALING:
A CSM configuration must demonstrate at least one compelling advantage:
- better matched-budget memory performance,
- better long-context behavior,
- useful confidence behavior,
- better decode/state scaling,
or
- competitive perplexity with a meaningful efficiency advantage.

If there is no such advantage, stop scaling and investigate.

Write:
results/phase10_small_nlp.md
