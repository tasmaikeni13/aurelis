import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Module
import Mathlib.Tactic.Ring

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

/-- The general gated read, before restricting the scalar gate to `[0,1]`. -/
def gatedRead (memory : X →ₗ[ℝ] Y) (gate : ℝ)
    (localValue : Y) (localKey query : X) : Y :=
  memory query + gate • (localValue - memory localKey)

/-- Equation (5.3): general-gate error with no probabilistic assumptions. -/
theorem gated_error_identity (memory truth : X →ₗ[ℝ] Y) (gate : ℝ)
    (localValue : Y) (localKey query : X) :
    gatedRead memory gate localValue localKey query - truth query =
      (memory - truth) query +
        gate • ((localValue - truth localKey) -
          (memory - truth) localKey) := by
  simp [gatedRead]
  module

/-- The full-residual endpoint of the gated expression is `correctedRead`. -/
theorem gatedRead_one (memory : X →ₗ[ℝ] Y)
    (localValue : Y) (localKey query : X) :
    gatedRead memory 1 localValue localKey query =
      correctedRead memory localValue localKey query := by
  simp [gatedRead, correctedRead, map_sub]
  module

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

/-- Error decomposition for multi-hop composition of two linear maps. -/
theorem composition_error_identity (memory2 truth2 memory1 truth1 : X →ₗ[ℝ] X) (query : X) :
    (memory2.comp memory1) query - (truth2.comp truth1) query =
      (memory2 - truth2) (memory1 query) + truth2 ((memory1 - truth1) query) := by
  simp only [LinearMap.comp_apply, LinearMap.sub_apply, map_sub]
  module

/-- Exact composition reproduces the true composite linear operator. -/
theorem composition_reproduces_linear (truth2 truth1 : X →ₗ[ℝ] X) (query : X) :
    (truth2.comp truth1) query = truth2 (truth1 query) := by
  rfl

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

/-- Scalar specialization of the finite-ridge slope identity (5.4). -/
theorem scalar_ridge_slope_error (signal prior truth : ℝ)
    (hDenominator : signal + prior ≠ 0) :
    truth * signal / (signal + prior) - truth =
      -(prior * truth) / (signal + prior) := by
  field_simp
  ring

/-- Faithful one-dimensional finite-ridge residual bound (indeed equality). -/
theorem scalar_ridge_residual_bound (signal prior truth residual : ℝ)
    (hPrior : 0 ≤ prior) (hDenominator : 0 < signal + prior) :
    |(truth * signal / (signal + prior) - truth) * residual| =
      prior * |truth| / (signal + prior) * |residual| := by
  rw [scalar_ridge_slope_error signal prior truth (ne_of_gt hDenominator)]
  rw [abs_mul, abs_div, abs_neg, abs_mul]
  rw [abs_of_nonneg hPrior, abs_of_pos hDenominator]

end Aurelis
