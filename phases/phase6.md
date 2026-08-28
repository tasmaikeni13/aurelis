# Phase 6 — Tiny language-model viability

Start only after Phase 5 PASS. Read all prior evidence and
`phases/AUTONOMY_PROTOCOL.md`. This is the first natural-language phase; it is
forbidden if any synthetic, learned, formal, or systems prerequisite is not
currently passing. Execute the failure-repair loop until PASS.

## Preregistration

Before training, freeze an experimental generation containing:

- checksum-pinned tokenizer and openly licensed corpus;
- train/validation/test splits with decontamination checks;
- parameter range `5M–20M`, context, optimizer, schedule, batch tokens, token
  budget, seeds, checkpoint/evaluation cadence, and exclusions;
- architecture equations and implementation commit; and
- primary natural loss plus six targeted diagnostic families.

Any architecture or evaluator change increments the generation and reruns all
models from scratch.

## Models

Train matched Transformer, sliding-window Transformer, Gated DeltaNet,
cumulative least-squares remote model, a published-style local/recurrent
hybrid, AURELIS-B, and AURELIS-E. Include remote-only and local-only ablations.
Match parameter count, tokens, optimizer, schedule, batch tokens, corpus, and
tuning opportunity; report residual mismatches.

## Diagnostics

Evaluate trained and longer-than-trained contexts on:

1. multi-query associative recall by age, load, and distractors;
2. exact recent copy and cache-boundary copy;
3. latent denoising versus observed exception tasks;
4. induction/selective copy and local shifts;
5. mixed recent/remote pointer chains; and
6. natural-text loss sliced by repeated n-grams, document distance, and
   context length, including an autoregressive subset.

Report validation loss, per-task metrics, endpoint outputs, gate calibration,
episodic AUROC, rank/conditioning, tokens/s, training peak VRAM, live decode
state, prefill/decode latency, wall time, and all seeds.

## PASS gates

- Every model completes the same preregistered token budget without hidden
  restarts or nonfinite steps; failures remain visible.
- AURELIS meaningfully learns natural text and its validation loss is within a
  preregistered small-model tolerance of the matched Transformer and strongest
  hybrid on every seed.
- AURELIS has a statistically supported matched-parameter advantage on at
  least one targeted mixed recent/remote diagnostic, not merely state bytes.
- AURELIS-E improves exact exceptions over AURELIS-B while AURELIS-B retains
  better or calibrated latent denoising; the target distinction survives
  learned language features.
- Longer-context evaluation does not show catastrophic collapse, and any
  boundary-age discontinuity meets its threshold.
- Systems advantages are reported separately from quality and reproduce the
  Phase 5 operating point.
- All inherited gates and Lean build pass, and
  `results/phase6/PASS.md` satisfies the shared PASS record.
