import Mathlib.Data.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# The analytic uncertainty router

`routeVariance vRemote vResidual covariance gate` is the conditional variance
of a convex combination of the remote and full-residual estimators.
-/

namespace Aurelis

def routeVariance
    (vRemote vResidual covariance gate : ℝ) : ℝ :=
  (1 - gate) ^ 2 * vRemote + gate ^ 2 * vResidual +
    2 * gate * (1 - gate) * covariance

noncomputable def rawGate (vRemote vResidual covariance : ℝ) : ℝ :=
  (vRemote - covariance) /
    (vRemote + vResidual - 2 * covariance)

/-- Euclidean projection of the analytic gate onto the admissible interval. -/
noncomputable def clippedGate (vRemote vResidual covariance : ℝ) : ℝ :=
  if rawGate vRemote vResidual covariance < 0 then 0
  else if 1 < rawGate vRemote vResidual covariance then 1
  else rawGate vRemote vResidual covariance

@[simp] theorem routeVariance_zero
    (vRemote vResidual covariance : ℝ) :
    routeVariance vRemote vResidual covariance 0 = vRemote := by
  simp [routeVariance]

@[simp] theorem routeVariance_one
    (vRemote vResidual covariance : ℝ) :
    routeVariance vRemote vResidual covariance 1 = vResidual := by
  simp [routeVariance]

/-- Completing the square exposes the unique unconstrained minimizer. -/
theorem routeVariance_completion
    (vRemote vResidual covariance gate : ℝ)
    (hDenominator : vRemote + vResidual - 2 * covariance ≠ 0) :
    routeVariance vRemote vResidual covariance gate =
      routeVariance vRemote vResidual covariance
        (rawGate vRemote vResidual covariance) +
      (vRemote + vResidual - 2 * covariance) *
        (gate - rawGate vRemote vResidual covariance) ^ 2 := by
  unfold routeVariance rawGate
  field_simp
  ring

/-- An interior analytic gate is no worse than the remote endpoint. -/
theorem rawGate_le_remote
    (vRemote vResidual covariance : ℝ)
    (hPositive : 0 < vRemote + vResidual - 2 * covariance) :
    routeVariance vRemote vResidual covariance
        (rawGate vRemote vResidual covariance) ≤ vRemote := by
  have hNe : vRemote + vResidual - 2 * covariance ≠ 0 := ne_of_gt hPositive
  have hCompletion := routeVariance_completion
    vRemote vResidual covariance 0 hNe
  rw [routeVariance_zero] at hCompletion
  nlinarith [sq_nonneg (rawGate vRemote vResidual covariance)]

/-- An interior analytic gate is no worse than the full-residual endpoint. -/
theorem rawGate_le_residual
    (vRemote vResidual covariance : ℝ)
    (hPositive : 0 < vRemote + vResidual - 2 * covariance) :
    routeVariance vRemote vResidual covariance
        (rawGate vRemote vResidual covariance) ≤ vResidual := by
  have hNe : vRemote + vResidual - 2 * covariance ≠ 0 := ne_of_gt hPositive
  have hCompletion := routeVariance_completion
    vRemote vResidual covariance 1 hNe
  rw [routeVariance_one] at hCompletion
  nlinarith [sq_nonneg (1 - rawGate vRemote vResidual covariance)]

/-- The clipped analytic gate minimizes variance over every convex gate. -/
theorem clippedGate_optimal
    (vRemote vResidual covariance gate : ℝ)
    (hPositive : 0 < vRemote + vResidual - 2 * covariance)
    (hGateLower : 0 ≤ gate) (hGateUpper : gate ≤ 1) :
    routeVariance vRemote vResidual covariance
        (clippedGate vRemote vResidual covariance) ≤
      routeVariance vRemote vResidual covariance gate := by
  let raw := rawGate vRemote vResidual covariance
  let denominator := vRemote + vResidual - 2 * covariance
  have hNe : denominator ≠ 0 := ne_of_gt hPositive
  have hGateCompletion := routeVariance_completion
    vRemote vResidual covariance gate hNe
  have hClipCompletion := routeVariance_completion
    vRemote vResidual covariance
      (clippedGate vRemote vResidual covariance) hNe
  have hDistance :
      (clippedGate vRemote vResidual covariance - raw) ^ 2 ≤
        (gate - raw) ^ 2 := by
    by_cases hRawLower : raw < 0
    · have hClip : clippedGate vRemote vResidual covariance = 0 := by
        simp [clippedGate, raw, hRawLower]
      rw [hClip]
      have hProduct : 0 ≤ gate * (-raw) :=
        mul_nonneg hGateLower (neg_nonneg.mpr (le_of_lt hRawLower))
      nlinarith [sq_nonneg gate]
    · have hRawNonneg : 0 ≤ raw := le_of_not_gt hRawLower
      by_cases hRawUpper : 1 < raw
      · have hClip : clippedGate vRemote vResidual covariance = 1 := by
          simp [clippedGate, raw, hRawLower, hRawUpper]
        rw [hClip]
        have hFirst : 0 ≤ 1 - gate := sub_nonneg.mpr hGateUpper
        have hSecond : 0 ≤ 2 * raw - 1 - gate := by
          linarith
        have hProduct : 0 ≤ (1 - gate) * (2 * raw - 1 - gate) :=
          mul_nonneg hFirst hSecond
        nlinarith
      · have hClip : clippedGate vRemote vResidual covariance = raw := by
          simp [clippedGate, raw, hRawLower, hRawUpper]
        rw [hClip]
        simpa using sq_nonneg (gate - raw)
  dsimp [raw] at hDistance
  dsimp [denominator] at hNe
  nlinarith [mul_nonneg hPositive.le (sub_nonneg.mpr hDistance)]

/-- Inverse-variance gate that incorrectly assumes endpoint independence (covariance = 0). -/
noncomputable def independentGate (vRemote vResidual : ℝ) : ℝ :=
  vRemote / (vRemote + vResidual)

/-- The clipped independence gate projects the independence heuristic to `[0, 1]`. -/
noncomputable def clippedIndependentGate (vRemote vResidual : ℝ) : ℝ :=
  if independentGate vRemote vResidual < 0 then 0
  else if 1 < independentGate vRemote vResidual then 1
  else independentGate vRemote vResidual

/-- The clipped independent gate is always in `[0, 1]`. -/
theorem clippedIndependentGate_bounds (vRemote vResidual : ℝ) :
    0 ≤ clippedIndependentGate vRemote vResidual ∧
      clippedIndependentGate vRemote vResidual ≤ 1 := by
  unfold clippedIndependentGate
  split_ifs with h1 h2
  · exact ⟨le_refl 0, by linarith⟩
  · exact ⟨by linarith, le_refl 1⟩
  · push_neg at h1 h2
    exact ⟨h1, h2⟩

/-- The full covariance Bayes gate outperforms or equals the independence heuristic for any endpoints. -/
theorem clippedGate_le_clippedIndependentGate
    (vRemote vResidual covariance : ℝ)
    (hPositive : 0 < vRemote + vResidual - 2 * covariance) :
    routeVariance vRemote vResidual covariance
        (clippedGate vRemote vResidual covariance) ≤
      routeVariance vRemote vResidual covariance
        (clippedIndependentGate vRemote vResidual) := by
  have hBounds := clippedIndependentGate_bounds vRemote vResidual
  exact clippedGate_optimal vRemote vResidual covariance
    (clippedIndependentGate vRemote vResidual)
    hPositive hBounds.1 hBounds.2

/-- When the independent gate is already within `[0, 1]`, the unclipped formula is also non-inferior to Bayes. -/
theorem clippedGate_le_independentGate
    (vRemote vResidual covariance : ℝ)
    (hPositive : 0 < vRemote + vResidual - 2 * covariance)
    (hIndepLower : 0 ≤ independentGate vRemote vResidual)
    (hIndepUpper : independentGate vRemote vResidual ≤ 1) :
    routeVariance vRemote vResidual covariance
        (clippedGate vRemote vResidual covariance) ≤
      routeVariance vRemote vResidual covariance
        (independentGate vRemote vResidual) :=
  clippedGate_optimal vRemote vResidual covariance
    (independentGate vRemote vResidual) hPositive hIndepLower hIndepUpper

/-- Under the posterior formulas, the quadratic denominator simplifies. -/
theorem posterior_denominator_identity
    (localNoise queryVariance keyVariance cross : ℝ) :
    queryVariance +
          (localNoise + queryVariance - 2 * cross + keyVariance) -
          2 * (queryVariance - cross) =
      localNoise + keyVariance := by
  ring

/-- The routing numerator is the posterior query/key-barycenter covariance. -/
theorem posterior_numerator_identity
    (queryVariance cross : ℝ) :
    queryVariance - (queryVariance - cross) = cross := by
  ring

/-- The episodic gate combines the Bayes gate with an explicit episodic responsibility. -/
def episodicGate (bayesGate episodicResponsibility : ℝ) : ℝ :=
  max bayesGate episodicResponsibility

@[simp] theorem episodicGate_ge_bayes (gB e : ℝ) :
    gB ≤ episodicGate gB e :=
  le_max_left gB e

@[simp] theorem episodicGate_ge_episodic (gB e : ℝ) :
    e ≤ episodicGate gB e :=
  le_max_right gB e

theorem episodicGate_bounds {gB e : ℝ} (hB0 : 0 ≤ gB) (hB1 : gB ≤ 1) (_he0 : 0 ≤ e) (he1 : e ≤ 1) :
    0 ≤ episodicGate gB e ∧ episodicGate gB e ≤ 1 := by
  constructor
  · exact le_trans hB0 (le_max_left gB e)
  · exact max_le hB1 he1

end Aurelis

