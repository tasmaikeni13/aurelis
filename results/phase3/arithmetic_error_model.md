# AURELIS Phase 3: Documented Arithmetic Error Model

Deterministic bounds accounting for dot products, exponentiation, reductions, division, and rounding:

| Dtype | Unit Roundoff $u$ | Inner Product Bound $\Delta_{\text{dot}}$ | Completion Allowance $\delta_{\text{num}}$ |
|---|---|---|---|
| `float64` | 1.1102e-16 | 2.3365e-14 | 4.9086e-15 |
| `float32` | 5.9605e-08 | 1.3400e-05 | 2.8780e-06 |
