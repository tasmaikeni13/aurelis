import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Data.Real.StarOrdered
import CSM.AffineScan

/-!
# Matrix invariants of the Gauss–Markov sufficient statistic

This file formalizes the actual finite-dimensional `S` update from Definition
5.1 over real matrices.
-/

namespace CSM

open Matrix

variable {n : Type*} [Fintype n]

/-- The pair of sufficient statistics `(S, C)` from Definition 5.1. -/
abbrev Statistics (n m : Type*) := Matrix n n ℝ × Matrix m n ℝ

/-- A per-token affine element for both sufficient statistics simultaneously. -/
def statisticsElement {m : Type*}
    (decay evidence : ℝ) (key : n → ℝ) (value : m → ℝ) :
    Affine ℝ (Statistics n m) :=
  ⟨decay,
    (evidence • Matrix.vecMulVec key key,
     evidence • Matrix.vecMulVec value key)⟩

omit [Fintype n] in
/-- The generic affine action is exactly the manuscript's simultaneous `(S,C)` write. -/
theorem statisticsElement_action {m : Type*}
    (decay evidence : ℝ) (key : n → ℝ) (value : m → ℝ)
    (state : Statistics n m) :
    Affine.act (statisticsElement decay evidence key value) state =
      (decay • state.1 + evidence • Matrix.vecMulVec key key,
       decay • state.2 + evidence • Matrix.vecMulVec value key) := by
  rfl

/-- A real rank-one outer product `k kᵀ` is positive semidefinite. -/
theorem outer_self_posSemidef (key : n → ℝ) :
    (Matrix.vecMulVec key key).PosSemidef := by
  constructor
  · apply Matrix.IsHermitian.ext
    intro i j
    simp [Matrix.vecMulVec_apply, mul_comm]
  · intro query
    simp only [dotProduct, Matrix.mulVec, Matrix.vecMulVec_apply, star_trivial]
    calc
      ∑ i, query i * ∑ j, key i * key j * query j =
          ∑ i, (query i * key i) * (∑ j, key j * query j) := by
            apply Finset.sum_congr rfl
            intro i _
            rw [Finset.mul_sum]
            rw [Finset.mul_sum]
            apply Finset.sum_congr rfl
            intro j _
            ring
      _ =
          (∑ i, query i * key i) * (∑ j, key j * query j) := by
            rw [Finset.sum_mul]
      _ = (∑ i, key i * query i) ^ 2 := by
            have hCommute : (∑ i, query i * key i) = ∑ i, key i * query i := by
              apply Finset.sum_congr rfl
              intro i _
              ring
            rw [hCommute, pow_two]
      _ ≥ 0 := sq_nonneg _

/-- Nonnegative real scaling preserves positive semidefiniteness. -/
theorem posSemidef_nonneg_smul
    {matrix : Matrix n n ℝ} (hMatrix : matrix.PosSemidef)
    {weight : ℝ} (hWeight : 0 ≤ weight) :
    (weight • matrix).PosSemidef := by
  constructor
  · apply Matrix.IsHermitian.ext
    intro i j
    have hSymmetric : matrix j i = matrix i j := by
      simpa using hMatrix.1.apply i j
    simp [hSymmetric]
  · intro query
    rw [Matrix.smul_mulVec_assoc, dotProduct_smul]
    exact mul_nonneg hWeight (hMatrix.2 query)

/-- The exact `S' = λS + βkkᵀ` update preserves PSD. -/
theorem memory_S_update_posSemidef
    {state : Matrix n n ℝ} (hState : state.PosSemidef)
    (key : n → ℝ) {decay evidence : ℝ}
    (hDecay : 0 ≤ decay) (hEvidence : 0 ≤ evidence) :
    (decay • state + evidence • Matrix.vecMulVec key key).PosSemidef := by
  exact (posSemidef_nonneg_smul hState hDecay).add
    (posSemidef_nonneg_smul (outer_self_posSemidef key) hEvidence)

/-- Adding a strictly positive diagonal `εI` makes a PSD state positive definite. -/
theorem regularized_system_posDef
    [DecidableEq n] {state : Matrix n n ℝ} (hState : state.PosSemidef)
    {ε : ℝ} (hε : 0 < ε) :
    (state + Matrix.diagonal (fun _ ↦ ε)).PosDef := by
  exact Matrix.PosDef.posSemidef_add hState
    (Matrix.PosDef.diagonal (fun _ ↦ hε))

/-- Therefore the exact regularized read system has a unique linear solve. -/
theorem regularized_system_isUnit
    [DecidableEq n] {state : Matrix n n ℝ} (hState : state.PosSemidef)
    {ε : ℝ} (hε : 0 < ε) :
    IsUnit (state + Matrix.diagonal (fun _ ↦ ε)) := by
  exact (regularized_system_posDef hState hε).isUnit

end CSM
