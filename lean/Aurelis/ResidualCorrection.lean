import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic.Module

/-!
# Attention residual correction

The local attention branch supplies a value barycenter and key barycenter.
The remote linear map predicts only the query residual.  These statements are
the deterministic algebraic core of the AURELIS head.
-/

namespace Aurelis

open scoped BigOperators

variable {X Y : Type*}
variable [AddCommGroup X] [Module ℝ X]
variable [AddCommGroup Y] [Module ℝ Y]

def correctedRead (memory : X →ₗ[ℝ] Y)
    (localValue : Y) (localKey query : X) : Y :=
  localValue + memory (query - localKey)

/-- Exact error decomposition into a local residual and remote slope error. -/
theorem corrected_error_identity (memory truth : X →ₗ[ℝ] Y)
    (localValue : Y) (localKey query : X) :
    correctedRead memory localValue localKey query - truth query =
      (localValue - truth localKey) +
        (memory - truth) (query - localKey) := by
  simp [correctedRead, map_sub]
  module

/-- If the remote map is correct, correction reproduces every linear query. -/
theorem corrected_reproduces_linear (truth : X →ₗ[ℝ] Y)
    (localKey query : X) :
    correctedRead truth (truth localKey) localKey query = truth query := by
  simp [correctedRead, map_sub]

/-- A one-hot attention hit is exact, independently of the remote state. -/
theorem corrected_exact_hit (memory : X →ₗ[ℝ] Y)
    (target : Y) (query : X) :
    correctedRead memory target query query = target := by
  simp [correctedRead]

variable {ι : Type*} [Fintype ι]

def weightedMean (weight : ι → ℝ) (value : ι → X) : X :=
  ∑ index, weight index • value index

/-- Linear maps commute with attention barycenters. -/
theorem map_weightedMean (truth : X →ₗ[ℝ] Y)
    (weight : ι → ℝ) (value : ι → X) :
    truth (weightedMean weight value) =
      ∑ index, weight index • truth (value index) := by
  simp [weightedMean]

/-- The barycentric local error is the barycenter of pointwise residuals. -/
theorem weighted_residual_identity (truth : X →ₗ[ℝ] Y)
    (weight : ι → ℝ) (key : ι → X) (value : ι → Y) :
    (∑ index, weight index • value index) -
        truth (weightedMean weight key) =
      ∑ index, weight index • (value index - truth (key index)) := by
  simp [weightedMean, smul_sub]

end Aurelis
