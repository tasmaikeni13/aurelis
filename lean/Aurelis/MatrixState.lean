import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Data.Real.StarOrdered
import Aurelis.AffineScan

/-!
# Positive-definite remote inference state
-/

namespace Aurelis

open Matrix

variable {n : Type*} [Fintype n]

abbrev Statistics (n m : Type*) := Matrix n n ℝ × Matrix m n ℝ

def statisticsElement {m : Type*}
    (decay evidence : ℝ) (key : n → ℝ) (value : m → ℝ) :
    Affine ℝ (Statistics n m) :=
  ⟨decay,
    (evidence • Matrix.vecMulVec key key,
     evidence • Matrix.vecMulVec value key)⟩

omit [Fintype n] in
theorem statisticsElement_action {m : Type*}
    (decay evidence : ℝ) (key : n → ℝ) (value : m → ℝ)
    (state : Statistics n m) :
    Affine.act (statisticsElement decay evidence key value) state =
      (decay • state.1 + evidence • Matrix.vecMulVec key key,
       decay • state.2 + evidence • Matrix.vecMulVec value key) := by
  rfl

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
      _ = (∑ i, query i * key i) * (∑ j, key j * query j) := by
            rw [Finset.sum_mul]
      _ = (∑ i, key i * query i) ^ 2 := by
            have hCommute : (∑ i, query i * key i) =
                ∑ i, key i * query i := by
              apply Finset.sum_congr rfl
              intro i _
              ring
            rw [hCommute, pow_two]
      _ ≥ 0 := sq_nonneg _

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

theorem precision_update_posSemidef
    {state : Matrix n n ℝ} (hState : state.PosSemidef)
    (key : n → ℝ) {decay evidence : ℝ}
    (hDecay : 0 ≤ decay) (hEvidence : 0 ≤ evidence) :
    (decay • state + evidence • Matrix.vecMulVec key key).PosSemidef := by
  exact (posSemidef_nonneg_smul hState hDecay).add
    (posSemidef_nonneg_smul (outer_self_posSemidef key) hEvidence)

theorem regularized_precision_posDef
    [DecidableEq n] {state : Matrix n n ℝ} (hState : state.PosSemidef)
    {prior : ℝ} (hPrior : 0 < prior) :
    (state + Matrix.diagonal (fun _ ↦ prior)).PosDef := by
  exact Matrix.PosDef.posSemidef_add hState
    (Matrix.PosDef.diagonal (fun _ ↦ hPrior))

theorem regularized_precision_isUnit
    [DecidableEq n] {state : Matrix n n ℝ} (hState : state.PosSemidef)
    {prior : ℝ} (hPrior : 0 < prior) :
    IsUnit (state + Matrix.diagonal (fun _ ↦ prior)) := by
  exact (regularized_precision_posDef hState hPrior).isUnit

/-- Leaky convex regularization under exponential decay preserves positive definiteness. -/
theorem leaky_precision_update_posDef
    [DecidableEq n] {state : Matrix n n ℝ} (hState : state.PosSemidef)
    (key : n → ℝ) {decay prior evidence : ℝ}
    (hDecayNonneg : 0 ≤ decay) (hDecayLt : decay < 1)
    (hPrior : 0 < prior) (hEvidence : 0 ≤ evidence) :
    (decay • state + evidence • Matrix.vecMulVec key key +
      Matrix.diagonal (fun _ ↦ (1 - decay) * prior)).PosDef := by
  have hPos : 0 < (1 - decay) * prior := mul_pos (sub_pos.mpr hDecayLt) hPrior
  exact Matrix.PosDef.posSemidef_add
    (precision_update_posSemidef hState key hDecayNonneg hEvidence)
    (Matrix.PosDef.diagonal (fun _ ↦ hPos))

end Aurelis
