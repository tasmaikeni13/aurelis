# Phase 2 — Hybrid mechanism separation and matched baselines

Start only after Phase 1 PASS. Read the paper, literature review, all prior
evidence, and `phases/AUTONOMY_PROTOCOL.md`. Execute the failure-repair loop
until every gate passes.

The purpose is not to force AURELIS to win. It is to identify regimes in which
each defining component changes the answer under fair state and compute
budgets.

## Baselines and ablations

Implement equation-tested versions of:

- local softmax attention only;
- remote Bayesian ridge only;
- global positive-feature linear attention;
- delta-rule/Gated-Delta-style memory;
- cumulative least-squares/Mesa-style remote memory;
- simple learned sum or concatenation of local and remote outputs;
- inverse-variance fusion that incorrectly assumes endpoint independence;
- full-residual output with fixed `g=1`;
- AURELIS-B and AURELIS-E; and
- if feasible, a Native-Hybrid-like recurrent-slot plus recent-token softmax.

Verify each baseline equation on tiny hand-computable cases. Compare in four
views: same feature dimension, same parameter count, same live-state bytes,
and approximately same measured FLOPs. Do not collapse them into one ranking.

## Falsification suites

Include at minimum:

- exact linear maps with diffuse attention;
- nonlinear maps where remote linear transport should fail;
- recent exceptions and remote exceptions;
- correlated keys and convex-hull/first-moment failures;
- denoising with known and corrupted evidence;
- cache-boundary queries;
- arbitrary associative recall below/above rank and window capacity;
- adversarial distractors and confidently wrong local matches; and
- context-length/state-byte/latency sweeps.

For every row retain endpoint errors, routed error, gate, attention entropy,
selected-key margin, `||q-kbar||`, posterior quadratic forms, rank,
conditioning, state bytes, FLOPs, and synchronized prepared latency.

## Research repair requirement

When a baseline or AURELIS regime contradicts the paper, research the closest
primary literature and derive the mechanism. If the residual formula requires
a new premise, add the premise and a counterexample to the paper and Lean where
formalizable. If a baseline is stronger than expected, preserve that result and
promote it to later phases.

## PASS gates

- Every baseline passes its own equation tests and receives a fair budget.
- Linear reproduction benefit is isolated from temperature, parameter, and
  state-size confounds.
- AURELIS-B's advantage in matched Gaussian regimes survives across every seed
  and agrees with its predicted variance within uncertainty.
- AURELIS-E's recent exception benefit is isolated from the Bayes objective;
  neither is marketed as universal.
- At least one nonlinear/misspecified regime with no AURELIS advantage is
  retained and explained.
- Capacity failures remain visible and agree with rank/window limits.
- The full covariance gate outperforms or equals the independence heuristic on
  a constructed correlated-endpoint suite.
- All inherited tests and Lean proofs pass, and
  `results/phase2/PASS.md` satisfies the shared PASS record.
