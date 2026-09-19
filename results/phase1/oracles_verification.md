# Phase 1: Mathematics & Independent Oracle Verification (fp64)

Evaluation of pure Python ScalarOracle against PyTorch float64 TensorOracle.

| Equation | Operation | Tested Dimensions | Max Abs Error | Tolerance (atol) | Status |
|---|---|---|---|---|---|
| Eq. (2) | Gated delta recurrent update | [2, 2], [4, 4], [6, 8], [12, 16], [16, 32] | 1.48e-15 | 1.0e-10 | **PASS** |
| Eq. (2) Corollary | Unit-key exact write (beta=1, alpha=1) | [2, 2], [4, 4], [6, 8], [12, 16], [16, 32] | 1.25e-15 | 1.0e-10 | **PASS** |
| Eq. (3) | Bounded read with local attention & transport | [2, 2], [4, 4], [6, 8], [12, 16], [16, 32] | 1.41e-14 | 1.0e-10 | **PASS** |
| Eq. (4) | Transport error identity & linear reproduction | [2, 2], [4, 4], [6, 8], [12, 16], [16, 32] | 1.24e-14 | 1.0e-10 | **PASS** |
| Eq. (5) | Rank-one perturbation energy identity & nonexpansion | d_k=2, d_k=4, d_k=8, d_k=16, d_k=32 | 3.55e-15 | 1.0e-10 | **PASS** |
| Eq. (7) | Explicit full softmax reference y_* | [2, 2], [4, 4], [6, 8], [12, 16], [16, 32] | 8.58e-16 | 1.0e-10 | **PASS** |
| Eq. (8) | Mass-consistent normalized archive completion | d_v=2, d_v=4, d_v=6, d_v=12, d_v=16 | 0.00e+00 | 1.0e-10 | **PASS** |
| Eq. (9) | Two-term vector error decomposition | d_v=2, d_v=4, d_v=6, d_v=12, d_v=16 | 7.27e-16 | 1.0e-10 | **PASS** |
| Eq. (10) | Deterministic residual certificate bound E_A | d_v=2, d_v=4, d_v=6, d_v=12, d_v=16 | 4.44e-16 | 1.0e-10 | **PASS** |
| Eqs. (11, 12) | Page coordinate score and value residual envelopes | [2, 2], [4, 4], [6, 8], [12, 16], [16, 32] | 2.00e-11 | 1.0e-10 | **PASS** |
| Eq. (13) | Per-page grouped comparator formula & certificate | d_v=2, d_v=4, d_v=6, d_v=12, d_v=16 | 0.00e+00 | 1.0e-10 | **PASS** |

All tests satisfied atol=1.0e-10, rtol=1.0e-09 with zero circular shared helpers.
