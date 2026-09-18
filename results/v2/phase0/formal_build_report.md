# AURELIS-R v2 Phase 0: Formal Build Report & Theorem Mapping

**Generated UTC:** `2026-09-18T06:45:55.319360+00:00`  
**Theory Revision:** `v2.0`  
**Lean Version:** `Lean 4.19.0 (commit 6caaee842e94)`  
**Mathlib Version:** `v4.19.0 (commit c44e0c8ee63ca166450922a373c7409c5d26b00b)`  
**Build Status:** `SUCCESS` (completed in 1.22s)  
**Axioms / Sorry / Admit:** `NONE (Clean Build)`

---

## 1. Theorem-to-Paper Mapping Catalog

All formal results in `lean/Aurelis/` compile with zero errors, zero warnings, zero admitted goals (`sorry`/`admit`), and zero project axioms. Standard Lean 4 foundation axioms (propext, Classical.choice, Quot.sound) remain part of the trusted computing base.

| Theorem Name | File | Paper Reference | Mathematical Claim | Exact Formal Scope |
|---|---|---|---|---|
| `exact_recall_injective` | `lean/Aurelis/Capacity.lean` | aurelis.md §1.1 | Exact address recall across all arbitrary history assignments forces injective encoder. | Arbitrary address, value, and state types; deterministic correct decoder. |
| `exact_recall_capacity` | `lean/Aurelis/Capacity.lean` | aurelis.md §1.1, Eq. (1) | Finite-state memory lower bound |S| ≥ m^n (b ≥ n log2 m bits). | Finite types with arbitrary discrete assignments; bit interpretation is analytic corollary. |
| `deltaRead_add` | `lean/Aurelis/DeltaMemory.lean` | aurelis.md §4, Eq. (2) | Evaluation of S' = α S + β (v - α S k) kᵀ is linear in the query vector. | Inner-product space X, vector space Y over reals; arbitrary linear map memory. |
| `deltaRead_exact_write` | `lean/Aurelis/DeltaMemory.lean` | aurelis.md §4, Eq. (2) | Unit-norm key with rate β=1 perfectly reproduces the written value at the write key. | Inner-product space X; single unit-norm write key; does not preserve unrelated keys. |
| `deltaTransition_energy` | `lean/Aurelis/DeltaMemory.lean` | aurelis.md §4.1, Eq. (5) | Exact rank-one row perturbation energy identity for delta recurrence update. | Real inner-product space; arbitrary real error vector, key, and rate β. |
| `deltaTransition_nonexpansive` | `lean/Aurelis/DeltaMemory.lean` | aurelis.md §4.1, Eq. (5) | Row transition is non-expansive when β ≥ 0 and β ‖k‖² ≤ 2. | Row error vector; fixed identical inputs to both states. |
| `decayed_delta_nonexpansive` | `lean/Aurelis/DeltaMemory.lean` | aurelis.md §4.1, Eq. (6) | State forgetting factor α ≥ 0 contracts row perturbations: ‖D+‖ ≤ α ‖D‖. | Row transition; matrix Frobenius norm corollary follows analytically by row summation. |
| `completedRead_balance` | `lean/Aurelis/CertifiedRead.lean` | aurelis.md §5, Eq. (8) | Completed read balance identity: denominator multiplies normalized estimate to reconstruct numerator. | Real normed space V; non-zero total estimated mass zs + zh. |
| `completedRead_full` | `lean/Aurelis/CertifiedRead.lean` | aurelis.md §5, Eq. (7)/(8) | Empty unread mass (zh = 0) exactly reduces completed read to standard attention. | Real arithmetic; exact equality over identical queries, keys, and values. |
| `completion_error_identity` | `lean/Aurelis/CertifiedRead.lean` | aurelis.md §6, Eq. (9) | Exact algebraic split of completion error into value residual and normalizer mass uncertainty. | Real normed vector space; both true mass and estimated mass non-zero. |
| `residual_certificate` | `lean/Aurelis/CertifiedRead.lean` | aurelis.md §6, Eq. (10) | Deterministic norm upper bound accounting for residual error and normalizer uncertainty. | Real normed space; positive lower bound floor on denominator mass. |
| `midpoint_error` | `lean/Aurelis/CertifiedRead.lean` | aurelis.md §5, Eq. (8) | Midpoint mass estimate error bounded by half-width η = (U_O - L_O)/2. | Exact real interval arithmetic. |
| `exp_score_interval` | `lean/Aurelis/CertifiedRead.lean` | aurelis.md §6.1, Eq. (11)-(12) | Monotonicity of real exponential yields score intervals to softmax mass intervals. | Scalar real exponential. |
| `weighted_residual_bound` | `lean/Aurelis/CertifiedRead.lean` | aurelis.md §6.1, Eq. (12) | Weighted sum of residual vectors bounded by sum of mass upper bounds times value radii. | Finite index set, nonnegative weights bounded by upper, values within radius of prior. |
| `completedRead_certificate` | `lean/Aurelis/CertifiedRead.lean` | aurelis.md §6, Eq. (10) | Full end-to-end conditional certificate for midpoint completed read. | Exact reals; positive selected mass zs > 0; true unread mass zo in [lo, hi]; valid residual bound radius. |
| `coordinate_product_interval` | `lean/Aurelis/PageEnvelope.lean` | aurelis.md §6.1, Eq. (11) | 1D coordinate product interval bounding q_d * k_d. | Exact real numbers; handles sign of q_d automatically. |
| `dot_box_interval` | `lean/Aurelis/PageEnvelope.lean` | aurelis.md §6.1, Eq. (11) | Multidimensional coordinate bounding box implies dot-product score intervals [ℓ_j, u_j]. | Finite dimension set D, non-negative scale factor κ. |
| `page_mass_interval` | `lean/Aurelis/PageEnvelope.lean` | aurelis.md §6.1, Eq. (12) | Page mass bound L_j = n_j exp(ℓ_j) ≤ Z_j ≤ n_j exp(u_j) = U_j. | Finite page of size n_j, uniform score bounds [lower, upper]. |
| `page_residual_ball` | `lean/Aurelis/PageEnvelope.lean` | aurelis.md §6.1, Eq. (12) | Page residual norm envelope b_j(r) = U_j (‖c_j - r‖ + ρ_j). | Exact reals; value ball around center with radius ρ_j; prior predictor r. |
| `handoff_partition` | `lean/Aurelis/Handoff.lean` | aurelis.md §3 | Exact sequence partition: recent cache and remote suffix reconstruct history exactly once. | Finite lists; occurrence-level identity; no double counting. |
| `recent_length_le_window` | `lean/Aurelis/Handoff.lean` | aurelis.md §3 | Recent attention cache is bounded by sliding window size w. | List length arithmetic. |
| `remote_empty_before_window` | `lean/Aurelis/Handoff.lean` | aurelis.md §3 | Remote set is empty while sequence length is within local window. | List length arithmetic. |
| `cache_overlap_redundancy` | `lean/Aurelis/Handoff.lean` | aurelis.md §3 | Overlap redundancy lemma: double counting cache tokens inflates sequence length. | List length arithmetic. |
| `gated_error_identity` | `lean/Aurelis/ResidualCorrection.lean` | aurelis.md §4 | General-gate error identity for arbitrary linear memory. | Arbitrary linear maps memory, truth over real vector spaces. |
| `gatedRead_one` | `lean/Aurelis/ResidualCorrection.lean` | aurelis.md §4, Eq. (3) | Full residual gate (gate=1) recovers residual corrected read r(q) = v_L + S(q - k_L). | Real vector spaces. |
| `corrected_error_identity` | `lean/Aurelis/ResidualCorrection.lean` | aurelis.md §4, Eq. (4) | Residual correction decomposes error into local residual plus remote slope error on query residual. | Real vector spaces, arbitrary linear maps. |
| `corrected_reproduces_linear` | `lean/Aurelis/ResidualCorrection.lean` | aurelis.md §4 | Exact slope truth and linear consistency reproduces linear truth exactly. | Arbitrary linear maps. |
| `corrected_exact_hit` | `lean/Aurelis/ResidualCorrection.lean` | aurelis.md §4 | One-hot local attention hit with query = localKey reproduces target value exactly. | Arbitrary linear maps; independent of remote memory state. |
| `map_weightedMean` | `lean/Aurelis/ResidualCorrection.lean` | aurelis.md §4 | Linear operator commutes with finite attention barycenters. | Real vector spaces, finite index sets. |
| `weighted_residual_identity` | `lean/Aurelis/ResidualCorrection.lean` | aurelis.md §4 | Barycentric attention error is the barycenter of pointwise association residuals. | Real vector spaces, finite index sets. |

---

## 2. Explicit Non-Coverage and Boundaries

In accordance with AUTONOMY_PROTOCOL.md and `PROOF_COVERAGE.md`, the formal Lean build strictly supports only its stated theorem conclusions. It explicitly **DOES NOT** prove:

1. **Floating-point safety:** Theorems are proved over exact real numbers $\mathbb{R}$. Floating-point interval rounding, outward directed rounding, and underflow/overflow handling remain Phase 3 implementation obligations.
2. **Accelerator kernel equivalence:** Matrix delta chunk scans and hardware implementations are not proved equivalent by Lake build.
3. **Model task quality:** Lean proofs do not establish neural learning dynamics, gradient descent convergence, or recall benchmark accuracy.
4. **Systems latency/memory:** Algorithmic time complexity and device memory footprints remain empirical engineering benchmarks.
5. **Global answer safety:** The certificate bounds local attention error on the current query/head/layer; it does not certify end-to-end factuality or multi-layer logit drift without Lipschitz premises.

---

## 3. Formal Gate Verdict

- Lake build executed and passed without changing pins: **PASS**
- Theorem statements mapped to paper equations: **PASS**
- Zero sorry, admit, or undeclared axioms: **PASS**
