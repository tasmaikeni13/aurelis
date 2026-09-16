# AURELIS-R v2 claim registry

Updated 2026-09-16. This registry supersedes v1 publication claims. Preserved v1
artifacts are historical, not v2 evidence. Analytic means a derivation in the
paper; Lean-checked means the stated theorem under its explicit premises
compiles. Neither status establishes a numerical implementation or performance.

| ID | Claim | Evidence and scope | Status |
|---|---|---|---|
| R-CAPACITY | Exact arbitrary finite-address recall requires at least m^n states | Capacity.exact_recall_capacity (namespace Aurelis.V2, theorem exact_recall_capacity); deterministic finite-state premise | Lean-checked; bit corollary analytic |
| R-WRITE | Gated delta evaluation is linear in query; unit-key beta=1 writes the supplied value at that key | deltaRead_add, deltaRead_exact_write | Lean-checked; known update, not novelty |
| R-STABILITY | Rank-one row transition is nonexpansive when beta≥0 and beta·norm(key)²≤2 | deltaTransition_energy/nonexpansive, decayed_delta_nonexpansive | Lean-checked; fixed inputs, not global training stability |
| R-HANDOFF | Recent/remote occurrence lists partition history | handoff_partition, recent_length_le_window | Lean-checked; implementation pending |
| R-TRANSPORT | Transport error decomposition, conditional linear reproduction and one-hot hit | corrected_error_identity/reproduces_linear/exact_hit | Lean-checked for any linear map; bounded predictor only |
| R-COMPLETE | Completed output normalizes selected values plus predicted unread mass | completedRead_balance | Lean-checked |
| R-ERROR | Vector error has residual and denominator-uncertainty terms | completion_error_identity | Lean-checked in real normed spaces |
| R-CERT | Midpoint completion error obeys paper (10) | completedRead_certificate, residual_certificate, midpoint_error | Lean-checked conditional on valid envelopes |
| R-ENVELOPE | Score intervals and weighted residual bounds support page envelopes | exp_score_interval, weighted_residual_bound, coordinate_product_interval, dot_box_interval, page_mass_interval, page_residual_ball; paper (11)–(12) | Lean-checked arithmetic; full page construction/correspondence pending |
| R-FULL | Zero unread mass reduces completion to full selected attention | completedRead_full | Lean-checked algebra; real arithmetic, same Q/K/V |
| R-GROUPED | Per-page predictor comparator has bound (13) | Paper §6.3 | Analytic; not yet Lean-checked |
| R-COST | Bounded mode removes the precision matrix and per-token ridge factorization | Equations (2)–(3), cost model (15) | Design accounting; no measured speedup |
| R-ARCHIVE | Exact archive and metadata grow with history; worst-case full read is linear per query | State definition and cost model | Analytic; implementation pending |
| H-RECURRENCE | Recurrent completion improves cost at fixed error vs strongest cheap completion | Required phases 4–8 | Hypothesis; no result |
| H-LM | New architecture preserves useful trained LM/recall quality | Required phases 6–8 | Pending |
| H-DEPLOY | Meets registered quality/SLO/cost and failure-handling criteria | Required phases 7–8 | Pending |
| H-NOVELTY | The precise coupling is a distinct useful contribution beyond prior art | Literature ledger plus required ablations | Candidate; priority and value unestablished |
| H-FP | Production arithmetic returns sound enclosures | Required phase 3 numerical proof/correspondence | Pending; fp64 tests alone are insufficient |

## Retired publication evidence

The old 8× “measured” state-memory reduction was based on formula accounting
in the audited evaluator. Its recall and exception MSE values were assigned,
and its decode timing did not carry a populated prefix cache. These are not
accepted as evidence of trained recall, measured cached decode performance,
or large-scale LM parity. See [the source audit](research/V1_AUDIT.md).

Legacy ridge definiteness/router algebra remains under its own assumptions,
but v2 has no Bayesian posterior or Bayes-optimal gate. No old empirical PASS
is inherited. Standard Lean foundational axioms are not “zero assumptions.”
