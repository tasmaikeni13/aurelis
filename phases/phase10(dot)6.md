This is Phase 10.6.

Goal:

Determine whether the Phase 10 conclusions survive transfer from the byte-level
WikiText experimental environment to the actual language-model data/tokenization
regime intended for Phase 11.

DO NOT increase model size beyond approximately 25M-35M parameters.

Use:
- FineWeb-Edu or the exact corpus pipeline intended for the later scale run
- a standard subword tokenizer suitable for the eventual 125M experiment
- identical tokenizer and data order for CSM and Transformer

Do not mix synthetic diagnostic samples into the natural-language training
stream for the primary comparison.

Synthetic probes remain separate evaluations.

Pre-register:
- tokenizer
- corpus revision / dataset identifiers
- filtering
- data order
- context length
- model architecture
- parameters
- optimizer
- token budget
- seeds
- evaluation suite

Run approximately 100M natural-language tokens first.

Use the strongest systems implementation from Phase 10.5.

Architectures:
1. matched Transformer
2. pure CSM

Optionally include the Phase 9 hybrid only as a secondary comparator.

Use at least 2 seeds initially.
A third seed is justified if the comparison is close.

Prefer context length 512 or 1024 rather than the previous 256 if hardware allows,
because the CSM state advantage is specifically a long-context hypothesis.

Measure:

QUALITY
- train loss
- validation loss/perplexity
- loss-versus-token curves

MEMORY CAPABILITY
- associative recall
- exact-value retrieval
- repeated-name recall
- variable tracking
- in-context regression
- multi-hop

Do not inject these tasks into the primary training data.

SYSTEMS
- training tokens/sec
- peak VRAM
- decode latency at 128 through >=16384 context
- live-state/cache bytes

CSM INTERNAL DIAGNOSTICS
- key Gram spectra
- effective rank
- condition numbers
- beta distributions
- query/key alignment
- uncertainty distributions

The purpose is transfer, not architecture search.

Do not heavily retune CSM relative to Transformer.

Produce:

results/phase10_6_fineweb_transfer.md

GO TO PHASE 11 only if:

1. CSM trains stably under the target tokenizer/data regime;
2. its constant-state advantage remains intact;
3. its language loss remains plausibly competitive rather than diverging;
4. learned key geometry remains healthy;
5. at least one architecture-level advantage survives outside the synthetic-
   mixture WikiText environment;
6. Phase 10.5 has removed or clearly characterized avoidable systems bottlenecks.

If these conditions fail, do not spend the 125M budget.
