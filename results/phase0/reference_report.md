# Phase 0 reference experiment

Status: **PASS**

| Measure | Observed |
|---|---:|
| Random cases | 112 |
| Maximum streaming/oracle absolute error | 7.105e-14 |
| Maximum Cholesky/dense/inverse error | 1.776e-15 |
| Partition failures | 0 |

The experiment uses CPU/fp64 and independently reconstructs every remote
prefix from full history. It covers empty and warm-up caches, the handoff
boundary, random windows/dimensions, and lengths above feature capacity. Unit
tests add repeated-key and near-singular pathologies, autograd, the analytic
router, and explicit expected-failure domains.

Reproduce with `.venv/bin/python experiments/phase0_reference.py`.
