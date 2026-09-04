# Phase 2 failed iteration: Softmax temperature precision in certified exception test

- Command: `.venv/bin/python experiments/phase2_baselines.py --config configs/phase2_baselines.json --device cuda`
- Seed/config: `20260904`, `configs/phase2_baselines.json`
- Failure: `gate_4_aurelis_e_exception_isolated` evaluated to False because `max_e_episodic_error` was $3.72 \times 10^{-9}$, exceeding tolerance $10^{-12}$.
- Classification: numerical conditioning / evaluator parameterization.
- Mechanism: In Section 5.4 of `aurelis.md`, Corollary 5.4 specifies: "Suppose attention is one-hot on cached index $j$, $q = k_j$, and $g = 1$. Then $y_g(q) = v_j$, independently of $M$." When attention is computed via finite softmax at $\tau = 64.0$, off-target weights sum to $e^{-64 \Delta} \approx 10^{-9}$, leaving a residual error of $3.72 \times 10^{-9}$.
- Repair: Use certified one-hot attention ($w_j = 1, w_{i \ne j} = 0$) or temperature $\tau \ge 1000.0$ for the certified episodic hit test, matching the mathematical premise of Corollary 5.4 and Section 4.5 ("A certified one-hot hit sets $e_t = 1$").
- Scientific rows retained/viewed: recorded in `results/phase2/raw/falsification_rows.jsonl`.
