import CSM.AffineScan

/-!
# Scalar certificates for the Phase 1 matrix recurrence

After evaluating `S' = λS + βkkᵀ` on a query `x`, the quadratic form is
`λ (xᵀ S x) + β (xᵀ k)^2`. The first theorems certify the sign argument used
for PSD preservation and the epsilon floor. The last theorems certify the exact
one-normalized-key ridge formula.
-/

namespace CSM

/-- Quadratic-form value after one CSM write. -/
def quadraticUpdate (decay β previous projection : ℝ) : ℝ :=
  decay * previous + β * projection ^ 2

theorem quadraticUpdate_nonneg
    {decay β previous projection : ℝ}
    (hDecay : 0 ≤ decay) (hβ : 0 ≤ β) (hPrevious : 0 ≤ previous) :
    0 ≤ quadraticUpdate decay β previous projection := by
  unfold quadraticUpdate
  positivity

/-- Adding `ε I` makes every nonzero-query quadratic form strictly positive. -/
theorem regularizedQuadratic_positive
    {quadratic normSquared ε : ℝ}
    (hQuadratic : 0 ≤ quadratic) (hNorm : 0 < normSquared) (hε : 0 < ε) :
    0 < quadratic + ε * normSquared := by
  positivity

/-- Closed-form shrinkage error for one normalized key. -/
theorem oneKey_read_error
    {β ε value : ℝ} (hDenominator : β + ε ≠ 0) :
    β * value / (β + ε) - value = -(ε * value) / (β + ε) := by
  field_simp
  ring

/-- The one-key posterior epistemic variance is strictly positive. -/
theorem oneKey_variance_positive
    {β ε : ℝ} (hβ : 0 ≤ β) (hε : 0 < ε) :
    0 < 1 / (β + ε) := by
  positivity

/-- More evidence reduces the one-key epistemic variance. -/
theorem oneKey_variance_antitone
    {β₁ β₂ ε : ℝ}
    (hβ₁ : 0 ≤ β₁) (hOrder : β₁ ≤ β₂) (hε : 0 < ε) :
    1 / (β₂ + ε) ≤ 1 / (β₁ + ε) := by
  exact one_div_le_one_div_of_le (by positivity) (by linarith)

/-- The spectral shrinkage factor in the finite-epsilon interpolation bound. -/
noncomputable def ridgeFactor (ε eigenvalue : ℝ) : ℝ :=
  ε / (eigenvalue + ε)

/-- A nonnegative regularizer and positive eigenvalue give nonnegative shrinkage. -/
theorem ridgeFactor_nonneg
    {ε eigenvalue : ℝ} (hε : 0 ≤ ε) (hEigenvalue : 0 < eigenvalue) :
    0 ≤ ridgeFactor ε eigenvalue := by
  unfold ridgeFactor
  positivity

/-- Every spectral error multiplier is at most one in the theorem's domain. -/
theorem ridgeFactor_le_one
    {ε eigenvalue : ℝ} (hε : 0 ≤ ε) (hEigenvalue : 0 < eigenvalue) :
    ridgeFactor ε eigenvalue ≤ 1 := by
  unfold ridgeFactor
  apply (div_le_one (by linarith)).2
  linarith

/-- Increasing a positive Gram eigenvalue cannot increase ridge shrinkage error. -/
theorem ridgeFactor_antitone_eigenvalue
    {ε eigenvalue₁ eigenvalue₂ : ℝ}
    (hε : 0 ≤ ε) (hEigenvalue₁ : 0 < eigenvalue₁)
    (hOrder : eigenvalue₁ ≤ eigenvalue₂) :
    ridgeFactor ε eigenvalue₂ ≤ ridgeFactor ε eigenvalue₁ := by
  unfold ridgeFactor
  exact div_le_div_of_nonneg_left hε (by linarith) (by linarith)

end CSM
