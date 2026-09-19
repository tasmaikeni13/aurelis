# AURELIS Claim Registry

This registry tracks the formal claims and empirical research hypotheses of the AURELIS architecture. Analytic means a mathematical derivation in [the specification](aurelis.md); Lean-checked means the theorem compiles in Lean 4 under namespace `Aurelis` with its stated premises.

| ID | Claim | Evidence and Scope | Status |
|---|---|---|---|
| R-CAPACITY | Exact arbitrary finite-address recall requires at least $m^n$ states | `exact_recall_capacity` (namespace `Aurelis`); deterministic finite-state premise | Lean-checked; bit corollary analytic |
| R-WRITE | Gated delta evaluation is linear in query; unit-key $\beta=1$ writes the supplied value at that key | `deltaRead_add`, `deltaRead_exact_write` | Lean-checked |
| R-STABILITY | Rank-one row transition is nonexpansive when $\beta \ge 0$ and $\beta \|k\|^2 \le 2$ | `deltaTransition_energy`, `deltaTransition_nonexpansive`, `decayed_delta_nonexpansive` | Lean-checked; fixed inputs |
| R-HANDOFF | Recent/remote occurrence lists partition history | `handoff_partition`, `recent_length_le_window` | Lean-checked |
| R-TRANSPORT | Transport error decomposition, conditional linear reproduction and one-hot hit | `corrected_error_identity`, `corrected_reproduces_linear`, `corrected_exact_hit` | Lean-checked for any linear map |
| R-COMPLETE | Completed output normalizes selected values plus predicted unread mass | `completedRead_balance` | Lean-checked |
| R-ERROR | Vector error has residual and denominator-uncertainty terms | `completion_error_identity` | Lean-checked in real normed spaces |
| R-CERT | Midpoint completion error obeys Eq. (10) | `completedRead_certificate`, `residual_certificate`, `midpoint_error` | Lean-checked conditional on valid envelopes |
| R-ENVELOPE | Score intervals and weighted residual bounds support page envelopes | `exp_score_interval`, `weighted_residual_bound`, `coordinate_product_interval`, `dot_box_interval`, `page_mass_interval`, `page_residual_ball`; Eqs. (11)–(12) | Lean-checked arithmetic |
| R-FULL | Zero unread mass reduces completion to full selected attention | `completedRead_full` | Lean-checked algebra; real arithmetic |
| R-GROUPED | Per-page predictor comparator has bound (13) | `groupedCompletedRead_balance`, `grouped_residual_certificate` (namespace `Aurelis`); Eq. (13) | Lean-checked |
| R-COST | Bounded mode eliminates per-token matrix inversions | Equations (2)–(3), cost model (15) | Design accounting |
| R-ARCHIVE | Exact archive and metadata grow with history; worst-case full read is linear per query | State definition and cost model | Analytic |
| H-RECURRENCE | Recurrent completion improves cost at fixed error vs strongest cheap completion | Phases 4–8 | Hypothesis |
| H-LM | Architecture preserves competitive trained LM/recall quality | Phases 6–8 | Hypothesis |
| H-DEPLOY | Meets registered quality/SLO/cost and failure-handling criteria | Phases 7–8 | Hypothesis |
| H-NOVELTY | The precise coupling is a distinct useful contribution beyond prior art | Literature review and ablations | Candidate |
| H-FP | Production arithmetic returns sound enclosures | Phase 3 numerical correspondence | Hypothesis |
