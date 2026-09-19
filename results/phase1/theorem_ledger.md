# Phase 1: Theorem-to-Equation Ledger & Formal Correspondence

Machine-checked formal correspondence under namespace `Aurelis` (Lean 4 / Mathlib 4.19.0):

| Equation | Paper Concept | Formal Lean Theorem | Exact Formal Scope |
|---|---|---|---|
| Eq. (1) | Finite-state capacity lower bound | `exact_recall_capacity` | At least $m^n$ states for injective decoding |
| Eq. (2) | Query linearity of delta update | `deltaRead_add` | Linear memory map under query addition |
| Eq. (2) | Unit-key exact write | `deltaRead_exact_write` | $\langle k, k \rangle = 1, \beta = 1$ writes $v$ at $k$ |
| Eq. (3) | Local residual transport | `correctedRead` | Linear memory transport from barycenter |
| Eq. (4) | Transport error decomposition | `corrected_error_identity` | Split into local residual & slope error |
| Eq. (4) | Linear reproduction | `corrected_reproduces_linear` | Exact when memory equals target map |
| Eq. (4) | Conditional exact hit | `corrected_exact_hit` | Exact value returned at one-hot query |
| Eq. (5) | Perturbation energy identity | `deltaTransition_energy` | Exact rank-one energy identity in real inner-product spaces |
| Eq. (5) | Contraction stability | `deltaTransition_nonexpansive`, `decayed_delta_nonexpansive` | Nonexpansion when $\beta \ge 0, \beta \|k\|^2 \le 2$ |
| Eq. (7) | Ground truth attention | `softmaxWeight_sum`, `softmaxWeight_pos` | Normalized positive finite softmax mixture |
| Eq. (8) | Normalized completion balance | `completedRead_balance` | $(Z_A + \widehat Z_O) \widehat y_A = N_A + \widehat Z_O r$ |
| Eq. (8) | All-pages recovery endpoint | `completedRead_full` | Recovers exact selected attention when $O = \emptyset$ |
| Eq. (9) | Two-term error identity | `completion_error_identity` | $(Z_A + Z_O)(y_* - \widehat y_A) = R_O + (Z_O - \widehat Z_O)(r - \widehat y_A)$ |
| Eq. (10) | Midpoint mass error | `midpoint_error` | $|Z_O - \widehat Z_O| \le \eta$ for $L_O \le Z_O \le U_O$ |
| Eq. (10) | Deterministic residual certificate | `residual_certificate`, `completedRead_certificate` | $\|y_* - \widehat y_A\| \le \mathcal E_A$ conditional on valid envelopes |
| Eqs. (11, 12) | Coordinate box score interval | `coordinate_product_interval`, `dot_box_interval` | Bounds on $\kappa q^\top k$ via coordinate boxes |
| Eqs. (11, 12) | Page mass interval | `page_mass_interval`, `exp_score_interval` | Monotonic exponential mass interval |
| Eqs. (11, 12) | Page residual envelope | `page_residual_ball`, `weighted_residual_bound` | Residual norm bounded by $U_j(\|c_j - r\| + \rho_j)$ |
| Eq. (13) | Grouped comparator balance | `groupedCompletedRead_balance` | Balance identity across finite index collection |
| Eq. (13) | Grouped certificate bound | `grouped_residual_certificate` | Upper bound on error under per-page envelopes |
| Causal Handoff | Disjoint cache/remote partition | `handoff_partition`, `recent_length_le_window` | List partition and bounded cache invariants |
