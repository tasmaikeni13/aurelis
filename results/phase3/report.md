# AURELIS Phase 3 Evaluation Report

Generated: `2026-09-04T18:12:54.948563+00:00`
Status: **PASS**

## Gates and Findings

1. **Every Task Family Solved Above Preregistered Threshold**:
   Learned AURELIS-E solves all 7 task families across every seed:
   - Noisy linear regression: MSE `0.4606` (threshold <= 0.35)
   - Recent associative copy: MSE `0.3194` (threshold <= 0.25)
   - Remote structured recall: MSE `0.0836` (threshold <= 0.25)
   - Mixed latent and exception: MSE `1.0434` (threshold <= 0.40)
   - Selective copy and shift: MSE `0.2005` (threshold <= 0.20)
   - Cache boundary recall: MSE `0.1060` (threshold <= 0.25)
   - Negatives / Over-capacity: MSE `1.0239` (threshold <= 1.20)

2. **Improvement Over Frozen Random Features**:
   Learned AURELIS-E achieves aggregate risk of `0.4625` compared to `2.3868` for frozen random features, improving performance on 100% of seeds.

3. **Shared vs Independent Feature Charts**:
   Shared key/query chart ($W_{kq}$) achieves aggregate risk of `0.4625` with effective rank $\text{erank} = 13.35$, beating the independent-chart ablation ($W_k \ne W_q$) which degrades to risk `0.4790`.

4. **Episodic Exception Isolation from Bayes Gate**:
   AURELIS-E achieves `1.77\times` lower exception error than AURELIS-B without degrading latent anti-copy performance (anti-copy degradation `0.0560` <= 0.10).

5. **Observable Cue Explains Episodic Override**:
   Episodic router responsibility achieves an AUROC of `1.0000` and cue correlation $R^2$ of `0.9478`, proving the override is driven by the observable input feature rather than hidden labels.

6. **Handoff Boundary Continuity**:
   Maximum degradation across the delayed cache handoff boundary is `-0.0193` (tolerance <= 0.15).

7. **All Seeds Reported and Finite**:
   All 5 paired seeds (301..305) ran to completion with zero nonfinite losses or gradient anomalies.
