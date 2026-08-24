import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

/-!
# Convexity certificates for normalized softmax memory

These theorems formalize the structural part of the Phase 3 separation. A
softmax read over standard-basis values returns its normalized nonnegative
weights, so targets with negative coordinates, coordinates above one, or a
sum other than one cannot be reproduced exactly.
-/

namespace CSM

open scoped BigOperators

variable {ι : Type*} [Fintype ι] [Nonempty ι]

/-- One coordinate of a finite softmax distribution. -/
noncomputable def softmaxWeight (score : ι → ℝ) (index : ι) : ℝ :=
  Real.exp (score index) / ∑ item, Real.exp (score item)

theorem softmaxDenominator_pos (score : ι → ℝ) :
    0 < ∑ item, Real.exp (score item) := by
  positivity

/-- Every finite softmax coordinate is nonnegative. -/
theorem softmaxWeight_nonneg (score : ι → ℝ) (index : ι) :
    0 ≤ softmaxWeight score index := by
  unfold softmaxWeight
  positivity

/-- Finite softmax coordinates sum exactly to one over the reals. -/
theorem softmaxWeight_sum (score : ι → ℝ) :
    ∑ index, softmaxWeight score index = 1 := by
  unfold softmaxWeight
  simp_rw [div_eq_mul_inv]
  rw [← Finset.sum_mul]
  exact mul_inv_cancel₀ (ne_of_gt (softmaxDenominator_pos score))

omit [Nonempty ι] in
/-- A nonnegative normalized coordinate cannot exceed one. -/
theorem normalizedWeight_le_one
    (weight : ι → ℝ) (hNonneg : ∀ index, 0 ≤ weight index)
    (hSum : ∑ index, weight index = 1) (index : ι) :
    weight index ≤ 1 := by
  rw [← hSum]
  exact Finset.single_le_sum (fun item _ ↦ hNonneg item) (Finset.mem_univ index)

omit [Fintype ι] [Nonempty ι] in
/-- A simplex-valued read cannot equal a target having a negative coordinate. -/
theorem negative_coordinate_not_normalized
    (weight target : ι → ℝ) (hNonneg : ∀ index, 0 ≤ weight index)
    {index : ι} (hTarget : target index < 0) :
    weight ≠ target := by
  intro hEqual
  have := hNonneg index
  rw [hEqual] at this
  linarith

omit [Nonempty ι] in
/-- A simplex-valued read cannot equal a target having a coordinate above one. -/
theorem above_one_coordinate_not_normalized
    (weight target : ι → ℝ) (hNonneg : ∀ index, 0 ≤ weight index)
    (hSum : ∑ index, weight index = 1)
    {index : ι} (hTarget : 1 < target index) :
    weight ≠ target := by
  intro hEqual
  have hBound := normalizedWeight_le_one weight hNonneg hSum index
  rw [hEqual] at hBound
  linarith

omit [Nonempty ι] in
/-- A normalized read cannot equal a target whose coordinates do not sum to one. -/
theorem nonunit_sum_not_normalized
    (weight target : ι → ℝ) (hSum : ∑ index, weight index = 1)
    (hTarget : (∑ index, target index) ≠ 1) :
    weight ≠ target := by
  intro hEqual
  rw [hEqual] at hSum
  exact hTarget hSum

end CSM
