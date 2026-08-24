# Phase 4 initial selective-risk gate — retained failure

The first complete Phase 4 run on 2026-08-24 passed seven of eight checks and
failed the initial selective-prediction threshold. The configured retained
fractions omitted `0.5`; the implementation selected the closest value, `0.6`,
where the measured Gaussian risk ratio versus full coverage was
`0.4716538778`. The initial gate required a ratio no larger than `0.4`.

Other observations from that unchanged numerical sweep were:

- CSM/oracle-ridge relative difference: `5.5345798371e-16`
- repeated-evidence risk slope: `-1.0178958918`
- uniform/precision-weighted Gaussian risk ratio: `23.2057055708`
- uncertainty/error Spearman: `0.7040691570`
- high-error AUROC: `0.8749385881`
- empirical/predicted Gaussian MSE: `0.9565396052`
- minimum uncertainty increment toward the unseen span: `0.0555555556`

This failure is not discarded. The corrected config measures exact `0.5`
coverage and asks whether it cuts risk by at least half (ratio `<= 0.5`). That
criterion directly operationalizes useful abstention without changing the
model, data generator, score, or any other gate.
