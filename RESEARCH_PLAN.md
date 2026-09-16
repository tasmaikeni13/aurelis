# AURELIS-R research plan

The question is whether a solve-free recurrent predictor can reduce the cost
of retrieving exact history under an explicit local attention-error budget.
The current deliverable is the new paper, formal core, and phase instructions;
scripts/model implementations must now be written to match them.

Start with [the implementation contract](phases/IMPLEMENTATION_CONTRACT.md),
[the adaptive change-impact protocol](phases/CHANGE_IMPACT_PROTOCOL.md), and
[the phase index](phases/README.md). All v2 empirical phases are pending and
are tracked in [the revision manifest](results/v2/REVISION_MANIFEST.yaml).

1. Audit evidence and register resource/quality/SLO budgets.
2. Establish independent mathematics, streaming semantics, and certificate
   arithmetic before making performance claims.
3. Falsify the recurrent contribution against strong recurrence-free
   completion, sparse attention, and actual hybrid baselines.
4. Optimize kernels only after a distinct mechanism survives.
5. Train real models, measure actual predictions/caches, integrate serving,
   and scale only within the authorized compute envelope.
6. Reproduce and publish the supported result, including a negative result
   if recurrence or certification fails to justify its cost.

The primary rejection criterion is recurrence-free dominance at matched quality,
error, and total service cost. Other stop conditions are loose certificates
forcing frequent full reads, index/transfer overhead erasing gains, poor trained
quality, invalid numerical certification, and unacceptable p99 latency.

A bounded-state approximate model and an archive-backed locally certified model
are different contracts. Do not claim archive storage vanished, finite softmax
copies arbitrary values exactly, or a local head certificate guarantees a
generated answer. Do not rename a failed hypothesis as a successful one.
