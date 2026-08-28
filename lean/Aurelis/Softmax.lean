import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic.Positivity

/-!
# Finite softmax barycentric weights
-/

namespace Aurelis

open scoped BigOperators

variable {ι : Type*} [Fintype ι] [Nonempty ι]

noncomputable def softmaxWeight (score : ι → ℝ) (index : ι) : ℝ :=
  Real.exp (score index) / ∑ item, Real.exp (score item)

theorem softmaxDenominator_pos (score : ι → ℝ) :
    0 < ∑ item, Real.exp (score item) := by
  positivity

theorem softmaxWeight_pos (score : ι → ℝ) (index : ι) :
    0 < softmaxWeight score index := by
  unfold softmaxWeight
  positivity

theorem softmaxWeight_nonneg (score : ι → ℝ) (index : ι) :
    0 ≤ softmaxWeight score index :=
  (softmaxWeight_pos score index).le

theorem softmaxWeight_sum (score : ι → ℝ) :
    ∑ index, softmaxWeight score index = 1 := by
  unfold softmaxWeight
  simp_rw [div_eq_mul_inv]
  rw [← Finset.sum_mul]
  exact mul_inv_cancel₀ (ne_of_gt (softmaxDenominator_pos score))

end Aurelis
