import Mathlib.Analysis.Normed.Module.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Module

/-!
# Residual completion and a conditional attention certificate

All statements are over exact reals. The implementation must establish the
page envelopes and account for rounding before using the word certified.
-/

namespace Aurelis

open scoped BigOperators

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

noncomputable def completedRead (selectedMass estimatedMass : ℝ) (numerator prior : V) : V :=
  (selectedMass + estimatedMass)⁻¹ • (numerator + estimatedMass • prior)

theorem completedRead_balance (zs zh : ℝ) (ns prior : V) (h : zs + zh ≠ 0) :
    (zs + zh) • completedRead zs zh ns prior = ns + zh • prior := by
  simp [completedRead, smul_smul, h]

theorem completedRead_full (zs : ℝ) (ns prior : V) :
    completedRead zs 0 ns prior = zs⁻¹ • ns := by
  simp [completedRead]

/-- Numerator residual AND normalizer error are both required. -/
theorem completion_error_identity (zs zo zh : ℝ) (ns no prior : V)
    (ht : zs + zo ≠ 0) (ha : zs + zh ≠ 0) :
    (zs + zo) • ((zs + zo)⁻¹ • (ns + no) - completedRead zs zh ns prior) =
      (no - zo • prior) + (zo - zh) • (prior - completedRead zs zh ns prior) := by
  have htruth : (zs + zo) • ((zs + zo)⁻¹ • (ns + no)) = ns + no := by
    simp [smul_smul, ht]
  have happ := completedRead_balance zs zh ns prior ha
  generalize happrox : completedRead zs zh ns prior = approx at *
  rw [smul_sub, htruth]
  have hns : ns = (zs + zh) • approx - zh • prior := by
    rw [happ]
    abel
  conv_lhs => rw [hns]
  module

/-- Norm consequence of the vector error identity, with a positive denominator floor. -/
theorem residual_certificate (truth approx residual prior : V)
    (mass floor delta radius gap : ℝ)
    (hfloor : 0 < floor) (hmass : floor ≤ mass)
    (hid : mass • (truth - approx) = residual + delta • (prior - approx))
    (hres : ‖residual‖ ≤ radius) (hgap : |delta| ≤ gap) :
    ‖truth - approx‖ ≤ (radius + gap * ‖prior - approx‖) / floor := by
  have hpos : 0 < mass := lt_of_lt_of_le hfloor hmass
  have hbound : mass * ‖truth - approx‖ ≤ radius + gap * ‖prior - approx‖ := by
    calc
      mass * ‖truth - approx‖ = ‖mass • (truth - approx)‖ := by
        rw [norm_smul, Real.norm_eq_abs, abs_of_pos hpos]
      _ = ‖residual + delta • (prior - approx)‖ := congrArg norm hid
      _ ≤ ‖residual‖ + ‖delta • (prior - approx)‖ := norm_add_le _ _
      _ = ‖residual‖ + |delta| * ‖prior - approx‖ := by rw [norm_smul, Real.norm_eq_abs]
      _ ≤ radius + gap * ‖prior - approx‖ :=
        add_le_add hres (mul_le_mul_of_nonneg_right hgap (norm_nonneg _))
  apply (le_div_iff₀ hfloor).mpr
  calc
    ‖truth - approx‖ * floor ≤ ‖truth - approx‖ * mass :=
      mul_le_mul_of_nonneg_left hmass (norm_nonneg _)
    _ ≤ radius + gap * ‖prior - approx‖ := by simpa [mul_comm] using hbound

theorem midpoint_error (lo actual hi : ℝ) (hl : lo ≤ actual) (hu : actual ≤ hi) :
    |actual - (lo + hi) / 2| ≤ (hi - lo) / 2 := by
  apply abs_le.mpr
  constructor <;> linarith

/-- A scalar score interval implies a positive softmax-mass interval. -/
theorem exp_score_interval (lower score upper : ℝ)
    (hl : lower ≤ score) (hu : score ≤ upper) :
    Real.exp lower ≤ Real.exp score ∧ Real.exp score ≤ Real.exp upper :=
  ⟨Real.exp_le_exp.mpr hl, Real.exp_le_exp.mpr hu⟩

/-- Page residual sums are bounded by mass upper bounds times value radii. -/
theorem weighted_residual_bound {ι : Type*} (indices : Finset ι)
    (weight upper radius : ι → ℝ) (value : ι → V) (prior : V)
    (hw : ∀ i ∈ indices, 0 ≤ weight i)
    (hu : ∀ i ∈ indices, weight i ≤ upper i)
    (hr : ∀ i ∈ indices, ‖value i - prior‖ ≤ radius i) :
    ‖∑ i ∈ indices, weight i • (value i - prior)‖ ≤
      ∑ i ∈ indices, upper i * radius i := by
  apply (norm_sum_le _ _).trans
  apply Finset.sum_le_sum
  intro i hi
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (hw i hi)]
  exact mul_le_mul (hu i hi) (hr i hi) (norm_nonneg _) (le_trans (hw i hi) (hu i hi))

/-- The complete certificate with actual numerator/mass definitions. -/
theorem completedRead_certificate (zs zo lo hi : ℝ) (ns no prior : V) (radius : ℝ)
    (hselected : 0 < zs) (hlo : 0 ≤ lo) (hl : lo ≤ zo) (hu : zo ≤ hi)
    (hres : ‖no - zo • prior‖ ≤ radius) :
    ‖(zs + zo)⁻¹ • (ns + no) - completedRead zs ((lo + hi) / 2) ns prior‖ ≤
      (radius + ((hi - lo) / 2) *
        ‖prior - completedRead zs ((lo + hi) / 2) ns prior‖) / (zs + lo) := by
  have ht : zs + zo ≠ 0 := ne_of_gt (by linarith)
  have ha : zs + (lo + hi) / 2 ≠ 0 := ne_of_gt (by linarith)
  exact residual_certificate _ _ _ _ _ _ _ _ _ (by linarith) (by linarith)
    (completion_error_identity zs zo ((lo + hi) / 2) ns no prior ht ha)
    hres (midpoint_error lo zo hi hl hu)

end Aurelis
