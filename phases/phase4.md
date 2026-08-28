# Phase 4 — Nonstationarity, compositional access, and capacity limits

Start only after Phase 3 PASS. Read all prior artifacts and
`phases/AUTONOMY_PROTOCOL.md`. Apply the failure-repair loop until PASS.

This phase tests the principal ways an undiscounted remote relation can become
wrong. Extensions are allowed only after deriving their changed semantics.

## Required suites

- abrupt and gradual operator drift with observable and unobservable changes;
- heterogeneous write precision, corrupted precision labels, outliers,
  Student-like noise, and nonlinear misspecification;
- repeated updates, overrides, many-to-one mappings, and state pollution;
- pointer chasing and multi-hop composition at hops `{1,2,4,8,16}`;
- cache/remote mixed-hop chains in every order;
- rank, state-byte, and adversarial-association capacity sweeps; and
- sequences at least 16 times longer than training length.

Compare the undiscounted state with researched extensions such as tempered
evidence, observable-cue decay, run-length beliefs, protected memory, or sparse
fallback. Do not call a learned forget action a changepoint posterior unless an
explicit probabilistic state supports that claim. Any decay changes the scan,
posterior, variance, and gate equations; update theory, numerical oracle, Lean
coverage, and tests before using its results.

For multi-hop reads, distinguish adaptive round count, vector error, decoded
success, operator norm, confidence, and actual latency. One attention read and
`H` adaptive reads are different computational budgets.

## PASS gates

- The stationary method retains its Phase 3 behavior on stationary controls.
- A drift-aware variant improves post-change risk on every paired seed when an
  observable signal exists, and unobservable-change limitations are retained.
- Evidence weighting improves heteroscedastic risk when precision is valid and
  degrades transparently when precision is corrupted.
- Mixed cache/remote multi-hop chains meet preregistered vector and decoded
  gates through the declared hop count, with error-propagation diagnostics.
- Rank/state lower-bound failures remain present; no fixed-state unlimited
  recall claim appears.
- Every extension has updated mathematics, a faithful formalization where
  feasible, a regression test, and an ablation against the base head.
- All inherited gates and Lean proofs pass, and
  `results/phase4/PASS.md` satisfies the shared PASS record.
