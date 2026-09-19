import Aurelis.CertifiedRead

/-! # Exact-real page envelopes

These theorems bridge coordinate containment, page mass bounds, and value-ball
residual bounds. They do not verify how a mutable implementation builds or
rounds the summaries, nor that an index covers every unread occurrence.
-/

namespace Aurelis

open scoped BigOperators

theorem coordinate_product_interval (q lower key upper : ℝ)
    (hl : lower ≤ key) (hu : key ≤ upper) :
    min (q * lower) (q * upper) ≤ q * key ∧
      q * key ≤ max (q * lower) (q * upper) := by
  rcases le_total 0 q with hq | hq
  · exact ⟨(min_le_left _ _).trans (mul_le_mul_of_nonneg_left hl hq),
      (mul_le_mul_of_nonneg_left hu hq).trans (le_max_right _ _)⟩
  · exact ⟨(min_le_right _ _).trans (mul_le_mul_of_nonpos_left hu hq),
      (mul_le_mul_of_nonpos_left hl hq).trans (le_max_left _ _)⟩

theorem dot_box_interval {D : Type*} (dims : Finset D)
    (query lower key upper : D → ℝ) (scale : ℝ) (hs : 0 ≤ scale)
    (hl : ∀ d ∈ dims, lower d ≤ key d) (hu : ∀ d ∈ dims, key d ≤ upper d) :
    scale * (∑ d ∈ dims, min (query d * lower d) (query d * upper d)) ≤
      scale * (∑ d ∈ dims, query d * key d) ∧
    scale * (∑ d ∈ dims, query d * key d) ≤
      scale * (∑ d ∈ dims, max (query d * lower d) (query d * upper d)) := by
  constructor
  · apply mul_le_mul_of_nonneg_left _ hs
    exact Finset.sum_le_sum fun d hd =>
      (coordinate_product_interval _ _ _ _ (hl d hd) (hu d hd)).1
  · apply mul_le_mul_of_nonneg_left _ hs
    exact Finset.sum_le_sum fun d hd =>
      (coordinate_product_interval _ _ _ _ (hl d hd) (hu d hd)).2

theorem page_mass_interval {I : Type*} (page : Finset I) (score : I → ℝ)
    (lower upper : ℝ) (hl : ∀ i ∈ page, lower ≤ score i)
    (hu : ∀ i ∈ page, score i ≤ upper) :
    (page.card : ℝ) * Real.exp lower ≤ (∑ i ∈ page, Real.exp (score i)) ∧
      (∑ i ∈ page, Real.exp (score i)) ≤ (page.card : ℝ) * Real.exp upper := by
  constructor
  · have h := Finset.sum_le_sum fun i hi => Real.exp_le_exp.mpr (hl i hi)
    simpa using h
  · have h := Finset.sum_le_sum fun i hi => Real.exp_le_exp.mpr (hu i hi)
    simpa using h

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

theorem page_residual_ball {I : Type*} (page : Finset I) (score : I → ℝ)
    (value : I → V) (center prior : V) (upper radius : ℝ)
    (hu : ∀ i ∈ page, score i ≤ upper)
    (hr : ∀ i ∈ page, ‖value i - center‖ ≤ radius) :
    ‖∑ i ∈ page, Real.exp (score i) • (value i - prior)‖ ≤
      ((page.card : ℝ) * Real.exp upper) * (‖center - prior‖ + radius) := by
  have hb : ∀ i ∈ page, ‖value i - prior‖ ≤ ‖center - prior‖ + radius := by
    intro i hi
    calc
      ‖value i - prior‖ = ‖(value i - center) + (center - prior)‖ := by (congr 1; abel)
      _ ≤ ‖value i - center‖ + ‖center - prior‖ := norm_add_le _ _
      _ ≤ ‖center - prior‖ + radius := by linarith [hr i hi]
  have h := weighted_residual_bound page (fun i => Real.exp (score i))
    (fun _ => Real.exp upper) (fun _ => ‖center - prior‖ + radius) value prior
    (fun i _ => (Real.exp_pos (score i)).le)
    (fun i hi => Real.exp_le_exp.mpr (hu i hi)) hb
  simpa [mul_assoc] using h

end Aurelis
