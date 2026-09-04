# AURELIS Phase 4 Evaluation Report

Generated: `2026-09-04T18:15:29.279394+00:00`
Status: **PASS**

## Gates and Findings

1. **Stationary Controls Retained**:
   The stationary AURELIS method retains its Phase 3 capabilities on stationary controls:
   - Aggregate stationary risk: `0.4315` (threshold <= 0.48)
   - Noisy linear regression: MSE `0.4485`
   - Recent copy: MSE `0.1800`
   - Remote recall: MSE `0.0733`
   - Mixed exception: MSE `1.0144`
   - Selective copy: MSE `0.1865`
   - Cache boundary: MSE `0.1091`
   - Negatives: MSE `1.0088`

2. **Drift-Aware Adaptation on Observable Changes**:
   Information-discounted remote state update achieves `10.87x` lower post-changepoint error than the stationary model on observable shifts (0.0807 vs 0.8767), passing on 100% of paired seeds. Unobservable changepoints retain fundamental theoretical bounds (0.8593 vs 0.8036).

3. **Heterogeneous Evidence Weighting and Corruption Degradation**:
   Inverse-variance weighting achieves `12.02x` lower risk under heteroscedastic noise (0.0029 vs 0.0354). When precision weights are corrupted/inverted, error degrades transparently by `54.36x` (0.1603), confirming active statistical reliance on evidence quality.

4. **Multi-Hop Composition and Mixed-Chain Pointer Chasing**:
   Multi-hop pointer chasing successfully decodes all 2-hop chains (>= 0.85) and 4-hop chains (>= 0.5) with maximum vector error <= 0.35. Error propagation is strictly tracked across cache/remote permutations.

5. **Subspace Capacity Limits Monotonically Enforced**:
   Under adversarial associations, error strictly increases monotonically beyond rank $d_k=8$ (MSE at $N=256$: `0.9661` vs $N=4$: `0.4346`), proving that fixed-state memory does not hallucinate unbounded recall capacity.

6. **16x Context Length Extrapolation**:
   Sequence length extrapolation up to 16x train length ($L=512$ vs $L=32$) remains numerically finite and stable (MSE `0.000173`, condition number `1.58`).

7. **All Seeds Complete and Finite**:
   All 5 paired seeds (401..405) ran to completion across all 7 test suites with zero nonfinite metrics or NaN values.
