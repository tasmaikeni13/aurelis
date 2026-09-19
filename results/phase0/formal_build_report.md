# AURELIS Phase 0: Formal Build Report & Theorem Mapping

**Generated UTC:** `2026-09-19T07:02:05.355409+00:00`  
**Build Tool:** `/home/tas_ken_rt25/.elan/bin/lake`  
**Build Status:** **PASS** (Elapsed: 1.302s)  
**Axioms / Sorry Check:** `None found (Clean)`

---

## 1. Theorem-to-Specification Mapping

| Formal Theorem | Namespace | Equation | Mathematical Claim | Exact Scope |
|---|---|---|---|---|
| `exact_recall_injective` | `Aurelis` | §1.1 | Exact address recall across all arbitrary history assignments forces injective encoder. | Arbitrary address, value, and state types; deterministic correct decoder. |
| `exact_recall_capacity` | `Aurelis` | (1) | Finite-state memory lower bound |S| ≥ m^n (b ≥ n log2 m bits). | Finite types with arbitrary discrete assignments; bit interpretation is analytic corollary. |
| `deltaRead_add` | `Aurelis` | (2) | Gated delta evaluation is linear under query addition. | Linear map memory over normed inner product space. |
| `deltaRead_exact_write` | `Aurelis` | (2) | Unit-norm key with beta=1 immediately writes the value at key. | Inner product ⟨k,k⟩=1; does not guarantee persistence across subsequent writes. |
| `deltaTransition_energy` | `Aurelis` | (5) | Rank-one row transition energy identity. | Real inner-product space, arbitrary real beta. |
| `deltaTransition_nonexpansive` | `Aurelis` | (5) | Gated delta transition is non-expansive when beta ≥ 0 and beta ||k||^2 ≤ 2. | Row-level contraction under matched inputs. |
| `decayed_delta_nonexpansive` | `Aurelis` | (5)-(6) | Decay factor alpha scales the perturbation bound; ||D+||_F ≤ alpha ||D||_F. | Fixed input perturbation; matrix Frobenius corollary is analytic. |
| `completedRead_balance` | `Aurelis` | (8) | Normalized completion satisfies mass-consistent linear balance. | Vectors in real normed space; nonzero total mass. |
| `completedRead_full` | `Aurelis` | (7)-(8) | Empty-unread endpoint recovers exact full softmax over selected tokens. | Real arithmetic, identical Q/K/V. |
| `completion_error_identity` | `Aurelis` | (9) | Error decomposition into numerator residual and normalizer mass discrepancy. | Algebraic vector identity in real normed spaces. |
| `residual_certificate` | `Aurelis` | (10) | Deterministic residual certificate norm bound. | Positive denominator floor; valid residual and mass bounds. |
| `midpoint_error` | `Aurelis` | (10) | Midpoint mass estimation error is bounded by half the interval width. | Real intervals. |
| `completedRead_certificate` | `Aurelis` | (10) | Complete midpoint certificate tied directly to completedRead output. | Positive selected mass, nonnegative lower unread bound. |
| `envelope_dot_upper` | `Aurelis` | (11) | Coordinate key-box implies score upper bound. | Exact finite sum; sign decomposition. |
| `envelope_dot_lower` | `Aurelis` | (11) | Coordinate key-box implies score lower bound. | Exact finite sum; sign decomposition. |
| `exp_envelope_dot_upper` | `Aurelis` | (11)-(12) | Exponential monotonicity preserves score bounds. | Real exponential. |
| `exp_envelope_dot_lower` | `Aurelis` | (11)-(12) | Exponential monotonicity preserves lower score bounds. | Real exponential. |
| `page_residual_ball` | `Aurelis` | (12) | Page value ball and score upper bound yield aggregate residual radius. | Finite weighted sum; requires verified page ball metadata. |
| `handoff_partition` | `Aurelis` | §3 | Recent and remote occurrence lists partition history disjointly. | List partitioning. |
| `recent_length_le_window` | `Aurelis` | §3 | Recent cache length is strictly bounded by sliding window size w. | List length bounded by w. |
| `corrected_error_identity` | `Aurelis` | (4) | Linear transport error decomposes into local bias and memory error. | Arbitrary linear maps. |
| `corrected_reproduces_linear` | `Aurelis` | (4) | Exact linear state and consistent local values reproduce linear ground truth. | Exact linear relations. |
| `corrected_exact_hit` | `Aurelis` | §4 | One-hot local hit reproduces verbatim value at queried key. | Local key match. |


---

## 2. Axioms & Admitted Proofs Check

Grep for `sorry`, `admit`, `axiom` within formal source:
```
Zero occurrences found.
```

Verdict: **PASS** (Zero project axioms, zero admitted proofs).
