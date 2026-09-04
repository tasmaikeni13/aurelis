# Phase 8 — 350M Medium-Scale Pretraining Study (3.0B FineWeb-Edu tokens on 8x MI300X)

Start only after Phase 7 PASS. Read all prior artifacts and
`phases/AUTONOMY_PROTOCOL.md`. Execute the failure-repair loop until PASS.

This phase tests whether the AURELIS architecture scales effectively to medium model
capacity (350M parameters) and a multi-billion token pretraining regime (3.0 Billion
FineWeb-Edu tokens) on an 8x AMD Instinct MI300X cluster. It evaluates whether the
dual-memory mechanism (recent cache + discounted remote state) retains its quality,
stability, and inference advantages over standard Transformer and linear recurrent baselines.

## Frozen design

Preregister one experimental generation with:

- **Architecture scale**: **350M parameters** (`d_model=1024`, 16 heads, `d_k=64, d_v=64`, 24 layers, vocabulary size 50257 / standard tiktoken/GPT-2 tokenizer);
- **Dataset & token budget**: **3.0 Billion training tokens** on the **FineWeb-Edu** corpus (`HuggingFaceFW/fineweb-edu`) per model;
- **Distributed hardware**: **8x AMD Instinct MI300X** (PyTorch DDP / FSDP with ROCm);
- **Comparators**:
  - Standard Transformer baseline (causal self-attention with RoPE / RMSNorm);
  - Published-style hybrid comparator (Gated DeltaNet / Kimi-style linear attention);
  - Cumulative least-squares baseline;
  - Strongest learned AURELIS variant (AURELIS-E with shared charts and observable cue discounting);
- **Training protocol**: Identical FineWeb-Edu corpus shards, AdamW optimizer, cosine decay schedule, batch tokens, context length `2048`, precision policy (bf16 with fp32 precision accumulation), checkpoint cadence, and validation harness;
- Parameter, FLOP, and memory state reconciliation across all comparators;
- Fixed primary and secondary claims with confidence intervals across paired seeds.

## Evaluation

1. **Pretraining dynamics & validation**:
   - Validation perplexity on held-out FineWeb-Edu validation split;
   - Training loss convergence, gradient norm stability, and condition number tracking.

2. **Long-context retrieval & extrapolation**:
   - Passkey retrieval and multi-needle in-a-haystack (NIAH) across context lengths `{2048, 4096, 8192, 16384}`;
   - Associative memory recall and multi-hop pointer chasing at extended scale.

3. **Systems & inference Pareto efficiency**:
   - Prefill throughput (tokens/sec) and decoding throughput (tokens/sec per sequence) on 8x MI300X;
   - Peak VRAM allocation and active decoding state footprint (AURELIS constant-size state vs Transformer KV-cache linear growth).

## Failure repair

If a run diverges, suffers quality degradation, or fails to produce competitive results:
1. Freeze trace, checkpoint, and numerical state;
2. Classify the failure (e.g. numerical conditioning of Cholesky solve at depth 24, learning rate / warm-up schedule, precision accumulation, or representation capacity);
3. Research primary literature (distributed linear attention, stabilized Kalman filters, spectral normalization);
4. Derive mathematical and architectural repairs (e.g., adaptive pre-norm, block-wise solve regularization, tempered evidence scaling);
5. Formalize deterministic claims in Lean 4 (`lake build` clean);
6. Verify against fp64 CPU oracle and regression tests;
7. Rerun all models and seeds under symmetric budgets.

## PASS gates

- All preregistered models and paired seeds complete the full 3.0B token budget or the phase remains failed.
- AURELIS validation loss is non-inferior within the preregistered margin to the strongest hybrid comparator.
- AURELIS demonstrates a statistically significant long-context retrieval advantage on multi-needle tests at contexts >= 4096 tokens.
- AURELIS achieves an end-to-end decoding throughput and memory Pareto advantage at context lengths >= 4096 tokens due to eliminating the $O(L)$ KV cache.
- Zero nonfinite loss spikes or unhandled gradient explosions during distributed 8x MI300X training.
- All inherited gates and Lean build pass, and `results/phase8/PASS.md` satisfies the shared PASS record.
