# Phase 9 tiny language-model optimization

## Gate decision: PASS

- PASS: `models_in_5m_to_20m_range`
- PASS: `parameter_counts_reasonably_matched`
- PASS: `csm_or_hybrid_numerically_stable`
- PASS: `csm_or_hybrid_optimizes`
- PASS: `natural_text_loss_not_catastrophic`
- PASS: `matched_parameter_targeted_memory_advantage`
- PASS: `all_diagnostic_families_evaluated`

## Pre-registration and protocol

Experimental generation `phase9-preregistered-g1` and config SHA-256 `b53186ef06940c5d60bcafe43f62b642031b99217065f385ebbf5ea9adfcd493` were fixed before checkpoint evaluation. All variants use the same byte tokenizer, sampled token budget, batch-token count, AdamW settings, LR schedule, gradient clipping, WikiText split, and diagnostic mixture. No architecture was modified after test results.

The natural corpus is the pinned raw WikiText-2 training/validation parquet release. Synthetic streams are separate and deterministic, and cover all six specified probe families. The 10M-token run is intentionally not expanded because Phase 9 is a stability/mechanism gate, not scaling.

## Natural-text training and systems behavior

| architecture | parameters | tokens | initial loss | final loss | val PPL | max grad | NaN/Inf | train tok/s | peak VRAM B | decode us/token | decode state B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transformer | 5574400 | 10002432 | 4.386 | 1.958 | 8.192 | 24.166 | 0 | 485541.898 | 1085869056 | 529.925 | 4587520 |
| csm | 5130096 | 10002432 | 4.396 | 2.148 | 9.897 | 38.185 | 0 | 121558.106 | 2991553024 | 1626.394 | 688128 |
| hybrid | 5383984 | 10002432 | 4.416 | 2.089 | 9.163 | 149.491 | 0 | 269970.038 | 2187012608 | 1025.463 | 1343488 |
| recurrent | 6961408 | 10002432 | 4.412 | 1.637 | 5.795 | 57.046 | 0 | 77714.191 | 1574324224 | 892.262 | 14336 |

Validation perplexity is byte-level and should only be compared within this protocol. Peak VRAM includes model, optimizer, and training activations; decode state is measured from live incremental state tensors after the timed decode.

## Diagnostic memory probes

| architecture | regime | mean exact | token accuracy | AR exact |
| --- | --- | --- | --- | --- |
| transformer | trained_length | 0.035 | 0.229 | 0.042 |
| transformer | long_context | 0.014 | 0.126 | 0.000 |
| csm | trained_length | 0.007 | 0.166 | 0.000 |
| csm | long_context | 0.042 | 0.138 | 0.000 |
| hybrid | trained_length | 0.000 | 0.176 | 0.000 |
| hybrid | long_context | 0.000 | 0.101 | 0.000 |
| recurrent | trained_length | 0.035 | 0.265 | 0.042 |
| recurrent | long_context | 0.028 | 0.146 | 0.000 |

Task-level associative recall, variable tracking, repeated-name recall, exact-value retrieval, in-context regression, and multi-hop results are retained in [`phase9/diagnostics.csv`](phase9/diagnostics.csv). Teacher-forced exactness requires every target byte to be correct; AR exactness greedily generates a preregistered subset.

## Sequence-length scaling

| architecture | context | forward ms | tokens/s | peak VRAM B | state/cache B |
| --- | --- | --- | --- | --- | --- |
| transformer | 32 | 2.174 | 58874.964 | 329721344 | 917504 |
| transformer | 64 | 2.068 | 123806.010 | 330442240 | 1835008 |
| transformer | 128 | 2.059 | 248685.295 | 332146176 | 3670016 |
| transformer | 256 | 2.090 | 489984.271 | 335816192 | 7340032 |
| transformer | 512 | 2.092 | 978952.521 | 343156224 | 14680064 |
| csm | 32 | 7.504 | 17057.565 | 601903104 | 688128 |
| csm | 64 | 7.478 | 34234.499 | 611975168 | 688128 |
| csm | 128 | 7.831 | 65379.048 | 632119296 | 688128 |
| csm | 256 | 8.803 | 116319.409 | 672407552 | 688128 |
| csm | 512 | 10.544 | 194228.585 | 752984064 | 688128 |
| hybrid | 32 | 5.109 | 25053.380 | 623386624 | 819200 |
| hybrid | 64 | 4.941 | 51811.437 | 633458688 | 1343488 |
| hybrid | 128 | 5.005 | 102294.085 | 653602816 | 1343488 |
| hybrid | 256 | 5.436 | 188382.191 | 693891072 | 1343488 |
| hybrid | 512 | 6.101 | 335703.859 | 774467584 | 1343488 |
| recurrent | 32 | 12.813 | 9990.147 | 627566080 | 28672 |
| recurrent | 64 | 21.587 | 11858.741 | 628160512 | 28672 |
| recurrent | 128 | 40.313 | 12700.689 | 629602304 | 28672 |
| recurrent | 256 | 75.950 | 13482.528 | 633122304 | 28672 |
| recurrent | 512 | 147.056 | 13926.710 | 640462336 | 28672 |

CSM recurrent-state bytes are constant in context length. Transformer cache bytes grow linearly, while hybrid local-attention cache is window bounded and its CSM states remain fixed. Training-time CSM prefix matrices still grow linearly and are included in measured peak VRAM.

## Architectural versus kernel findings

Validation loss and diagnostic accuracy diagnose architectural learning behavior; throughput, VRAM, and decode latency diagnose the current kernel tax. In particular, a useful diagnostic result does not erase the factorization cost measured in Phase 8, and a throughput deficit is not labeled a representation failure.

## Reproducibility

- commit at run start: `c728a210077d073e9180ef683c9e85f166ecfa87`; dirty: `True`
- device: `AMD Instinct MI300X VF`; Python `3.12.3`; PyTorch `2.8.0+rocm7.0.2.git245bf6ed`
- wall time: `351.27s`; tokenizer: raw UTF-8 bytes (vocabulary 256)
- config: [`configs/phase9_tiny_lm.json`](../configs/phase9_tiny_lm.json)
- raw rows: [`phase9/`](phase9/); machine record: [`phase9_metrics.json`](phase9_metrics.json)

## Scoped conclusion

The gate asks whether CSM or its hybrid trains stably, learns nontrivial natural-text structure, and exhibits a matched-parameter targeted memory advantage. It does not claim Transformer parity or authorize larger scaling unless those facts are all observed.
