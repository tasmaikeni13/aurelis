This is Phase 2.

Goal:
Empirically test the interpolation and finite-epsilon behavior of the Gauss–Markov CSM memory.

Use the validated Phase 1 implementation.

Construct associative-memory datasets with K key-value pairs.

Sweep:

d_k in {8,16,32,64,128}
K/d_k in approximately {0.125,0.25,0.5,0.75,1.0,1.25,1.5,2.0}
epsilon over a logarithmic range
multiple random seeds

Generate several key regimes:

A. orthogonal / approximately orthogonal keys
B. random normalized Gaussian keys
C. correlated keys
D. deliberately near-collinear keys
E. duplicate keys

Values should include:
- random Gaussian vectors
- one-hot values
- random binary-like vectors

Measure:
- relative recall error
- exact-recall rate under a tolerance
- condition number / minimum Gram eigenvalue
- confidence c(q)
- error versus epsilon
- error versus K/d_k
- error versus key conditioning

Explicitly test the manuscript prediction:
for linearly independent keys and K <= d_k,
recall error should tend toward zero as epsilon -> 0.

Compare measured error against the finite-epsilon theoretical upper bound wherever applicable.

Plot:
1. recall error vs epsilon
2. recall error vs K/d_k
3. recall error vs minimum Gram singular/eigenvalue
4. confidence vs actual error
5. error heatmap over capacity and conditioning

IMPORTANT:
Do not hide cases where the bound is loose.
Do not remove pathological seeds.
Report median, quantiles, and worst cases.

PHASE 2 PASS GATE:
The qualitative theorem-predicted interpolation behavior must appear clearly for independent keys under capacity.
Finite-epsilon behavior must move in the theoretically predicted direction.
Breakdown around dependent/over-capacity regimes must be measurable and explainable.

If these do not hold, stop and diagnose the theory/implementation discrepancy.

Write:
results/phase2_interpolation_report.md

Do not proceed automatically.
