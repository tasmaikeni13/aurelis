# Phase 2 failed iteration: Finite ridge shrinkage in linear reproduction suite

- Command: `.venv/bin/python experiments/phase2_baselines.py --config configs/phase2_baselines.json --device cuda`
- Seed/config: `20260904`, `configs/phase2_baselines.json`
- Failure: `gate_2_linear_reproduction_isolated` evaluated to False because `max_linear_error_aurelis` was ~2.26.
- Classification: statistical-model / evaluator misspecification.
- Mechanism: Theorem 5.2 and Corollary 5.3 state: "If local values obey $v_s = W k_s$ and $M = W$, then $y_H(q) = W q$ for every query and every normalized set of attention weights." In the test suite, $M$ was formed with a default prior of $\alpha = 1.0$ over only 32 tokens, introducing finite-ridge regularization bias $M - W = -\alpha W P^{-1} \ne 0$.
- Repair: In the exact linear reproduction evaluation, set the ridge prior to numerical floor ($\alpha = 10^{-12}$) or use full-rank noise-free history so that $M = W$ to machine precision ($< 10^{-12}$), isolating the residual correction mechanism from remote estimation bias as intended by Corollary 5.3.
- Scientific rows retained/viewed: recorded in `results/phase2/raw/falsification_rows.jsonl`.
