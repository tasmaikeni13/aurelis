This is Phase 11.

Goal:
Test whether the conclusions from small models survive at approximately 125M parameters.

Do NOT immediately train for 1B tokens.

Construct:
- approximately 125M Transformer baseline
- approximately parameter-matched CSM or CSM-hybrid model

Use the strongest configuration already selected from earlier phases.

First run a scaling pilot:
approximately 50M-100M tokens.

Check:
- loss curves
- optimization stability
- throughput
- memory
- kernel bottlenecks
- gradient statistics
- CSM conditioning
- learned key geometry
- beta/lambda distributions if used

If clearly unhealthy, stop.

If healthy, extend to roughly 250M-300M tokens.

Compare trends against the smaller-model experiments.

Before proceeding to 1B tokens, produce a decision memo:

1. Is loss scaling plausible?
2. Does the targeted memory advantage persist?
3. Is wall-clock overhead acceptable?
4. Are learned keys remaining well-conditioned?
5. Is confidence still meaningful?
6. Is there a scientifically defensible reason to spend the 1B-token budget?
7. What precise hypothesis will the 1B-token experiment resolve that the 300M-token run cannot?

Only recommend Phase 12 if the last question has a strong answer.

Write:
results/phase11_125m_pilot.md
