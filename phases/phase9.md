This is Phase 9.

Goal:
Determine whether a CSM layer can participate in ordinary language-model optimization at all.

This is NOT a scaling experiment.

Build a small decoder-only model roughly in the 5M-20M parameter range.

Use a manageable tokenizer and a modest real-text dataset plus diagnostic synthetic sequences.

Train several architecture variants:

A. small Transformer baseline
B. CSM-based sequence mixer
C. hybrid model containing both local attention and CSM
D. relevant simple recurrent/linear-memory baseline

Keep:
- parameter count reasonably matched
- optimizer matched
- token budget matched
- batch token count matched where practical

Start with approximately 10M-30M training tokens.

Only expand toward ~50M-100M tokens after proving training stability.

Measure:
- training loss
- validation perplexity
- gradient stability
- NaN/Inf frequency
- tokens/sec
- peak VRAM
- recurrent-state bytes
- decode latency
- sequence-length scaling

Also evaluate diagnostic tasks separately:
- associative recall
- variable tracking
- repeated-name recall
- exact-value retrieval
- in-context regression
- multi-hop synthetic probes

Do not modify architecture after seeing test-set results without recording a new experimental generation.

The objective of Phase 9 is NOT to beat the Transformer.

Questions:

1. Does optimization remain stable?
2. Does CSM learn useful representations from natural text?
3. Are memory advantages visible on targeted probes?
4. Does ordinary language-model loss catastrophically regress?
5. Which failure is architectural versus kernel-related?

PASS GATE:
CSM or a CSM hybrid must train stably and demonstrate a measurable memory capability that cannot be explained solely by parameter count.

Write:
results/phase9_tiny_lm.md
