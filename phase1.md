This is Phase 1 of the CSM research program.

Goal:
Implement the smallest possible mathematically transparent Gauss–Markov CSM memory and verify that the implementation exactly represents the equations in the manuscript.

Do NOT:
- train a language model
- optimize kernels
- implement the dyadic cascade
- introduce learned encoders
- use approximate inverses

Implement a clean reference module in PyTorch.

For a sequence of writes (k_t, v_t, beta_t, lambda_t):

S_t = lambda_t * S_{t-1} + beta_t * k_t k_t^T
C_t = lambda_t * C_{t-1} + beta_t * v_t k_t^T

A_t = S_t + epsilon I

read(q) = C_t A_t^{-1} q

confidence(q) = q^T A_t^{-1} q

Requirements:

1. Implement an explicit fp64 reference version.
2. Never form a matrix inverse directly in production-style paths; use torch.linalg.solve or Cholesky solves.
3. Also implement a direct-inverse oracle only for tiny matrices as a testing reference.
4. Unit test d_k in {2,4,8,16,32}.
5. Test random beta, lambda, epsilon.
6. Compare sequential recurrence against a naive recomputation from all historical sufficient statistics.
7. Numerically verify positive semidefiniteness of S.
8. Verify differentiability with torch autograd.
9. Use torch.autograd.gradcheck where practical.
10. Test pathological cases:
   - repeated keys
   - nearly collinear keys
   - beta=0
   - lambda=1
   - lambda<1
   - tiny epsilon
   - very large beta
   - zero values
   - single observation

Produce plots/errors rather than only pass/fail tests.

Required artifact:
results/phase1_report.md

PHASE 1 PASS GATE:
For well-conditioned fp64 cases, sequential and recomputed states/reads must agree to numerical precision. Gradients must agree with finite-difference or gradcheck references.

If the gate fails, investigate until the discrepancy is understood. Do not change the mathematical definition to make tests pass.

Do not proceed beyond Phase 1.
