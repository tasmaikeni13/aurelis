This is Phase 7.

Goal:
Determine whether the Bayesian interpretation of CSM gates leads to learnable and useful behavior.

Use the successful Phase 6 architecture.

Experiment 1: evidence precision beta

Create observations with controlled reliability.
Some tokens contain:
- clean evidence
- noisy evidence
- irrelevant distractors
- corrupted values

Make beta_t learned.

Test whether learned beta:
- increases on reliable observations
- decreases on corrupted/irrelevant observations
- improves downstream risk

Compare with:
- beta = 1
- unconstrained generic scalar gate
- oracle beta based on true noise

Experiment 2: drift lambda

Create streams where latent operator W changes at controlled change points.

Learn lambda_t.

Test whether:
- lambda approaches 1 in stationary periods
- forgetting increases after distribution changes
- adaptation speed improves versus lambda=1
- learned behavior correlates with actual drift

Compare:
- fixed lambda values
- learned lambda
- oracle change-point forgetting

Experiment 3:
Jointly learn beta and lambda only after the separate experiments work.

Record whether their interpretations remain identifiable or whether the network uses them in unintended ways.

Do not force the model to behave like the theory merely because the parameters were named beta and lambda.

Report what gradient descent actually does.

PASS GATE:
Learned gating must produce a reproducible benefit over fixed gates on tasks where evidence quality or latent drift genuinely varies.

Write:
results/phase7_gating.md
