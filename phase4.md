This is Phase 4.

Goal:
Test whether CSM behaves like the predicted Bayesian/ridge estimator under noisy repeated evidence, and whether its confidence output contains useful calibrated information.

Generate latent linear operators W*.

Sample:

v_i = W* k_i + sigma * noise

Test:
- varying observation noise
- repeated observations of the same association
- conflicting observations
- heteroscedastic evidence
- missing directions in key space
- out-of-distribution query directions

Use beta_i to represent observation precision.

Experiments:

A. noisy duplicates
Compare CSM prediction against:
- simple averaging
- Hebbian memory
- softmax memory
- oracle ridge regression

B. beta precision
Provide observations with different known noise variances.
Set beta according to true precision.
Check whether weighting reduces prediction risk.

C. confidence
For every query record:

c(q) = q^T (S + epsilon I)^-1 q

and actual squared prediction error.

Evaluate:
- Spearman correlation between confidence/uncertainty and error
- calibration plots
- risk conditioned on confidence quantile
- AUROC for detecting high-error queries
- selective prediction: accuracy when abstaining on highest-confidence-risk samples

D. unseen directions
Query directions progressively farther from the observed key span.
Check whether uncertainty increases appropriately.

The manuscript distinguishes in-model calibration from model-free guarantees. Respect that distinction.

Run both:
1. exactly linear-Gaussian data
2. deliberately misspecified data:
   - Laplace noise
   - Student-like heavy-tailed noise
   - nonlinear latent functions

Do not claim Bayesian optimality outside the declared model.

PASS GATE:
Within the linear-Gaussian setting, CSM should match the expected ridge/Bayesian behavior and confidence should meaningfully track predictive uncertainty.

Outside the model, characterize degradation rather than hiding it.

Write:
results/phase4_uncertainty_and_noise.md
