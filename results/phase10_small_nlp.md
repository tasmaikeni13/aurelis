# Phase 10 small natural-language comparison

## Scaling gate decision: PASS

- PASS: `all_preregistered_runs_completed`
- PASS: `three_seed_critical_comparison`
- PASS: `models_in_25m_to_50m_range`
- PASS: `parameter_counts_matched`
- PASS: `token_and_batch_budgets_matched`
- PASS: `no_nonfinite_training_steps`
- PASS: `natural_text_loss_not_catastrophic`
- PASS: `compelling_decode_state_scaling_advantage`
- PASS: `all_context_and_probe_evaluations_present`

## Pre-registration

Generation `phase10-preregistered-g1`; config SHA-256 `8a4141f09296333ab0bce2ae731462221f48717751c3f438def48893f542c81a`. Parameter counts, 256-byte context, AdamW, cosine schedule, 16,384 batch tokens, 100M-token budget, seeds `[0, 1, 2]`, evaluations, and exclusion criteria were fixed before training.

Exclusion criteria: dataset checksum mismatch; fewer than the preregistered training tokens; any nonfinite loss or gradient step; checkpoint serialization or reload failure. Excluded runs: []. No failed seed was replaced.

The hybrid was not run: Phase 9 pure CSM passed without hybridization and had stronger aggregate long-context diagnostics than the hybrid. This follows Phase 10's conditional hybrid requirement rather than silently dropping a necessary comparator.

## Corpus and matched training

Both architectures train on the same deterministic first 100,000,000 UTF-8 bytes of pinned raw WikiText-103, mixed with the same 10% diagnostic stream. Each seed/architecture sees the same 100,007,936-token budget and optimizer protocol. Training stopped at the preregistered initial 100M budget; no 200M–300M extension was needed to interpret the first gate.

| architecture | parameters | val PPL mean | val PPL sd | train tok/s | peak VRAM B | train seconds | decode us/token | decode state B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transformer | 29499904.000 | 3.261 | 0.026 | 715419.901 | 5090231466.667 | 139.929 | 591.817 | 18874368.000 |
| csm | 27468544.000 | 3.328 | 0.039 | 69161.383 | 12501047978.667 | 1446.020 | 1937.583 | 1572864.000 |

Across paired seeds, CSM-minus-Transformer validation-loss differences were `[0.02188, 0.009817, 0.02871]` (mean `0.0201`). This exposes seed noise rather than presenting one checkpoint as an architecture effect.

## Same-checkpoint context evaluation

| architecture | context | loss mean | loss sd | PPL mean |
| --- | --- | --- | --- | --- |
| csm | 128 | 1.225 | 0.011 | 3.403 |
| csm | 256 | 1.202 | 0.011 | 3.329 |
| csm | 512 | 2.595 | 0.082 | 13.432 |
| csm | 1024 | 3.487 | 0.317 | 33.758 |
| transformer | 128 | 1.208 | 0.006 | 3.348 |
| transformer | 256 | 1.184 | 0.007 | 3.267 |
| transformer | 512 | 2.752 | 0.217 | 15.919 |
| transformer | 1024 | 3.677 | 0.739 | 48.036 |

Every row evaluates the same final checkpoint at a different sequence length; checkpoints are not fine-tuned per context.

## Downstream and long-context memory probes

| architecture | task | token acc mean | token acc sd | exact mean | AR exact mean |
| --- | --- | --- | --- | --- | --- |
| csm | associative_recall | 0.099 | 0.009 | 0.000 | 0.000 |
| csm | exact_value_retrieval | 0.113 | 0.006 | 0.000 | 0.000 |
| csm | in_context_regression | 0.418 | 0.058 | 0.031 | 0.000 |
| csm | multi_hop | 0.083 | 0.048 | 0.083 | 0.083 |
| csm | repeated_name_recall | 0.838 | 0.005 | 0.323 | 0.250 |
| csm | variable_tracking | 0.130 | 0.048 | 0.000 | 0.000 |
| transformer | associative_recall | 0.156 | 0.041 | 0.000 | 0.000 |
| transformer | exact_value_retrieval | 0.102 | 0.014 | 0.000 | 0.000 |
| transformer | in_context_regression | 0.478 | 0.038 | 0.052 | 0.083 |
| transformer | multi_hop | 0.042 | 0.018 | 0.042 | 0.083 |
| transformer | repeated_name_recall | 0.826 | 0.024 | 0.365 | 0.167 |
| transformer | variable_tracking | 0.120 | 0.059 | 0.021 | 0.000 |

The complete trained-length and long-context tables, including all seeds, are in [`phase10/diagnostics.csv`](phase10/diagnostics.csv). No claim here rests on perplexity alone.

## Incremental decode and state scaling

| architecture | prompt context | decode us/token | latency sd | live state B | peak VRAM B |
| --- | --- | --- | --- | --- | --- |
| csm | 128 | 3918.782 | 257.647 | 786432 | 1549897045.333 |
| csm | 512 | 3961.867 | 283.613 | 786432 | 1549897045.333 |
| csm | 2048 | 3869.034 | 146.269 | 786432 | 1549897045.333 |
| transformer | 128 | 1205.384 | 70.592 | 4456448 | 1492938240 |
| transformer | 512 | 1147.684 | 40.607 | 17039360 | 1527465472 |
| transformer | 2048 | 1227.656 | 23.955 | 67371008 | 1619611136 |

At the longest prompt, the CSM/Transformer live-state ratio is `0.011673`. CSM state growth from the shortest to longest prompt is `1.000000x`; Transformer KV state grows with context. This is the qualifying efficiency advantage, while the latency table retains the current CSM kernel tax.

## Interpretation and scaling decision

The controlled comparison separates three facts: natural-text likelihood, probe behavior, and systems cost. The CSM is allowed to pass the scaling gate through a compelling state-scaling advantage even if perplexity or wall-clock is worse; those losses remain visible and constrain any claim. Further token scaling should target the factorization/decode bottleneck and only proceed if the state advantage matters for the intended context regime.

## Reproducibility

- commit at run start: `c728a210077d073e9180ef683c9e85f166ecfa87`; dirty: `True`
- device: `AMD Instinct MI300X VF`; Python `3.12.3`; PyTorch `2.8.0+rocm7.0.2.git245bf6ed`
- total wall time: `5116.91s`; corpus bytes loaded: `100000000`
- config: [`configs/phase10_small_nlp.json`](../configs/phase10_small_nlp.json)
- raw records: [`phase10/`](phase10/); machine record: [`phase10_metrics.json`](phase10_metrics.json)
