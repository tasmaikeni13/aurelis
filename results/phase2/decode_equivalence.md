# Phase 2: Populated-Cache Decode & Continuation Equivalence

Verification that token-by-token execution, multi-token prefill, and continuation agree exactly.

| Evaluation Track | Test Comparison | Max Abs Error | Tolerance | Status |
|---|---|---|---|---|
| Bounded Mode | Token-by-Token vs Multi-Token Prefill | 0.00e+00 | 1.0e-12 | **PASS** |
| Bounded Mode | Token-by-Token vs Continuation | 0.00e+00 | 1.0e-12 | **PASS** |
| Archive Mode | Token-by-Token vs Multi-Token Prefill | 0.00e+00 | 1.0e-12 | **PASS** |
| Archive Mode | Token-by-Token vs Continuation | 0.00e+00 | 1.0e-12 | **PASS** |
| Archive Mode | Step Outputs vs Full-History Softmax | 1.61e-15 | 1.0e-12 | **PASS** |
