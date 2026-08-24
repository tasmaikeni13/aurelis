This is Phase 6.

Goal:
Determine whether learned neural feature maps can discover key/query/value representations that make CSM useful.

This is the first phase where learned encoders are introduced.

Do NOT train a general language model yet.

Build small differentiable sequence models around the CSM memory.

Start with synthetic tasks:

1. associative recall
2. copy/selective-copy
3. key-value lookup
4. correlated-key lookup
5. in-context linear regression
6. noisy in-context regression
7. contextual associative recall where raw input keys are not already convenient vectors

Learn:

k_t = normalize(f_k(x_t))
q_t = f_q(x_t)
v_t = f_v(x_t)

Initially fix:
beta = 1
lambda = 1

Only after this works, make beta learnable.

Track representation geometry:

- Gram eigenvalue spectrum
- pairwise cosine similarity
- effective rank
- minimum singular value
- condition number of S + epsilon I
- fraction of capacity used
- gradient norms
- epsilon
- retrieval error

Compare:
A. learned CSM
B. fixed random CSM features
C. learned Hebbian memory
D. small attention baseline

Critical scientific question:
Does gradient descent naturally produce useful, sufficiently separated CSM key geometry?

Do NOT add an orthogonality regularizer initially.
We need to know the unassisted result.

Only after documenting the natural behavior may you test regularizers as explicit ablations.

Run multiple seeds.

PASS GATE:
The learned CSM must reliably outperform its untrained/random representation and learn the target tasks across seeds.

A representation that works only after aggressive hand-crafted geometry regularization should be reported as a limitation.

Write:
results/phase6_learnability.md
