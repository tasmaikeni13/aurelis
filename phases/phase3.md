# Phase 3 — Learned features and episodic routing

Start only after Phase 2 PASS. Read all prior evidence and
`phases/AUTONOMY_PROTOCOL.md`. Use the failure-repair loop until all gates pass.

This is the first learned-memory phase. No natural-language scale run is
allowed yet.

## Models

Train matched small models with shared key/query feature charts and learned
value, evidence, temperature, and episodic-responsibility projections. Include
local-only, remote-only, learned-sum, Gated DeltaNet, cumulative least-squares,
AURELIS-B, and AURELIS-E. Give controls the same encoder capacity and
optimization opportunity.

Retain these ablations:

- independent key/query charts;
- fixed versus learned evidence;
- fixed gates `0` and `1`;
- analytic gate without episodic override;
- learned sigmoid gate without analytic features;
- analytic-plus-episodic gate;
- cache overlap/double counting as a known-invalid covariance ablation; and
- random frozen features.

## Curriculum

Train and evaluate across at least seven task families:

1. noisy linear and affine in-context regression;
2. recent exact associative copy;
3. remote structured recall within rank;
4. mixed latent-denoising and episodic-exception targets with an observable
   task cue;
5. selective copy and local token shift;
6. cache-boundary recall at ages `w-1,w,w+1,w+2`; and
7. over-capacity, conflicting-write, and no-relevant-context negatives.

Use held-out dimensions, loads, temperatures, noise distributions, windows,
and sequence lengths. Include at least five paired seeds. Measure task metrics,
endpoint/routed risks, gate calibration by target type, AUROC for episodic
responsibility, spectra/effective rank, conditioning, gradient norms, and
failure rate.

## Self-correction focus

If representation rank collapses, do not immediately add orthogonality loss.
Research and diagnose normalization, parameterization, optimization, and task
identifiability first. Any regularizer becomes a named ablation with a derived
objective and test. If the episodic router lacks a future-query signal, add an
observable cue or revise the claim; do not let it consume hidden labels.

## PASS gates

- Learned AURELIS solves every task family above a preregistered threshold on
  every seed, including negative/no-write cases.
- It improves over frozen random features on aggregate risk for every seed.
- Shared-chart AURELIS beats the independent-chart failure ablation and retains
  usable effective rank.
- AURELIS-B is calibrated on latent tasks; AURELIS-E materially improves exact
  exception copy without unacceptable degradation on anti-copy cases.
- The observable episodic cue, not hidden evaluator metadata, explains the
  override.
- Handoff-boundary degradation is within a declared tolerance or a researched,
  proved, and tested repair is adopted.
- Every seed, including failures and nonfinite runs, is reported.
- Inherited gates and Lean build pass, and
  `results/phase3/PASS.md` satisfies the shared PASS record.
