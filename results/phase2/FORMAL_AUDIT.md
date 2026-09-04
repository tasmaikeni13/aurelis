# Phase 2 formal-requirement audit

This audit compares the deterministic theorems used by the Phase 2
experiment with their actual Lean statement.

| Experiment claim | Lean statement | Faithfulness / boundary |
|---|---|---|
| Optimality of Bayes gate over independent inverse-variance fusion | `clippedGate_le_clippedIndependentGate`, `clippedGate_le_independentGate` in `Aurelis/Router.lean` | Full real-algebra theorems proving $V(g_B) \le V(g_{\text{indep}})$ whenever denominator is positive; no statistical independence assumption is masked. |
| General-gate and full-residual error identity | `gated_error_identity`, `gatedRead_one`, `corrected_error_identity` in `Aurelis/ResidualCorrection.lean` | Full linear-map algebraic identities; verified across all 10 baseline comparisons. |
| Linear reproduction under exact remote operator | `corrected_reproduces_linear` | Full linear-map theorem; verified empirically across temperatures in fp64. |
| Exact one-hot cached hit | `corrected_exact_hit` | Full linear-map theorem; verified for certified episodic override $e_t=1.0$. |
| Convex clipped gate optimality | `clippedGate_optimal` in `Aurelis/Router.lean` | Full real-algebra theorem over arbitrary convex gates $g \in [0, 1]$. |

Remaining empirical boundaries:
- The advantage in the linear-Gaussian regime across multiple seeds is a statistical property of the Gaussian distribution and is verified via Monte Carlo and multi-seed sweeps.
- Capacity limits (window $w$ and rank $d_k$) are structural properties of finite state representation and are demonstrated via sequence recall sweeps.
- Kernel execution latency and GPU timing are hardware-measured on AMD Instinct MI300X.
