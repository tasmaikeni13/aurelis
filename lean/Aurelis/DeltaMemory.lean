import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Module
import Mathlib.Tactic.Ring

/-!
# Solve-free delta memory

The update is the existing gated delta rule, not a new recurrence. The energy
theorem concerns differences of two states receiving the SAME key/value/gates.
It does not prove bounded driven states or learned-model quality.
-/

namespace Aurelis.V2

variable {X Y : Type*} [NormedAddCommGroup X] [InnerProductSpace ℝ X]
variable [AddCommGroup Y] [Module ℝ Y]

def deltaRead (memory : X →ₗ[ℝ] Y) (key query : X) (value : Y)
    (decay rate : ℝ) : Y :=
  decay • memory query +
    (rate * inner (𝕜 := ℝ) key query) • (value - decay • memory key)

/-- Evaluation of S' = α S + β (v - α S k) kᵀ is linear in the query. -/
theorem deltaRead_add (memory : X →ₗ[ℝ] Y) (key q₁ q₂ : X)
    (value : Y) (decay rate : ℝ) :
    deltaRead memory key (q₁ + q₂) value decay rate =
      deltaRead memory key q₁ value decay rate +
        deltaRead memory key q₂ value decay rate := by
  simp [deltaRead, inner_add_right, mul_add, add_smul, smul_add]
  module

theorem deltaRead_exact_write (memory : X →ₗ[ℝ] Y) (key : X)
    (value : Y) (decay : ℝ) (hkey : inner (𝕜 := ℝ) key key = 1) :
    deltaRead memory key key value decay 1 = value := by
  simp [deltaRead, hkey]

def deltaTransition (key error : X) (rate : ℝ) : X :=
  error - (rate * inner (𝕜 := ℝ) key error) • key

/-- Exact rowwise energy identity; summing rows gives the Frobenius identity. -/
theorem deltaTransition_energy (key error : X) (rate : ℝ) :
    ‖deltaTransition key error rate‖ ^ 2 = ‖error‖ ^ 2 -
      rate * (2 - rate * ‖key‖ ^ 2) * (inner (𝕜 := ℝ) key error) ^ 2 := by
  rw [deltaTransition, norm_sub_sq_real, real_inner_smul_right, norm_smul,
    Real.norm_eq_abs, mul_pow, sq_abs, real_inner_comm error key]
  ring

theorem deltaTransition_nonexpansive (key error : X) (rate : ℝ)
    (hrate : 0 ≤ rate) (hstep : rate * ‖key‖ ^ 2 ≤ 2) :
    ‖deltaTransition key error rate‖ ≤ ‖error‖ := by
  have hdrop : 0 ≤ rate * (2 - rate * ‖key‖ ^ 2) *
      (inner (𝕜 := ℝ) key error) ^ 2 :=
    mul_nonneg (mul_nonneg hrate (sub_nonneg.mpr hstep)) (sq_nonneg _)
  have henergy := deltaTransition_energy key error rate
  nlinarith [norm_nonneg (deltaTransition key error rate), norm_nonneg error]

/-- Forgetting contracts perturbations for fixed inputs; α=1 is allowed. -/
theorem decayed_delta_nonexpansive (key error : X) (decay rate : ℝ)
    (hdecay : 0 ≤ decay) (hrate : 0 ≤ rate)
    (hstep : rate * ‖key‖ ^ 2 ≤ 2) :
    ‖decay • deltaTransition key error rate‖ ≤ decay * ‖error‖ := by
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hdecay]
  exact mul_le_mul_of_nonneg_left
    (deltaTransition_nonexpansive key error rate hrate hstep) hdecay

end Aurelis.V2
