import Mathlib.Algebra.Module.Basic

/-!
# Affine scan algebra for AURELIS remote sufficient statistics

An element `(decay, increment)` represents `z ↦ decay • z + increment`.
`combine newer older` applies `older` first and `newer` second.
-/

namespace Aurelis

structure Affine (R X : Type*) where
  decay : R
  increment : X
deriving Repr

@[ext] theorem Affine.extensionality {R X : Type*} (left right : Affine R X)
    (hDecay : left.decay = right.decay)
    (hIncrement : left.increment = right.increment) : left = right := by
  cases left
  cases right
  simp_all

namespace Affine

variable {R X : Type*}
variable [CommSemiring R] [AddCommMonoid X] [Module R X]

def identity : Affine R X := ⟨1, 0⟩

def combine (newer older : Affine R X) : Affine R X :=
  ⟨newer.decay * older.decay,
   newer.increment + newer.decay • older.increment⟩

def act (element : Affine R X) (state : X) : X :=
  element.decay • state + element.increment

@[simp] theorem act_identity (state : X) :
    act (identity : Affine R X) state = state := by
  simp [act, identity]

theorem combine_identity_left (element : Affine R X) :
    combine identity element = element := by
  ext <;> simp [combine, identity]

theorem combine_identity_right (element : Affine R X) :
    combine element identity = element := by
  ext <;> simp [combine, identity]

theorem combine_assoc (newest middle oldest : Affine R X) :
    combine newest (combine middle oldest) =
      combine (combine newest middle) oldest := by
  ext <;> simp [combine, mul_assoc, smul_add, mul_smul, add_assoc]

theorem act_combine (newer older : Affine R X) (state : X) :
    act (combine newer older) state = act newer (act older state) := by
  simp [act, combine, mul_smul, smul_add, add_assoc, add_comm, add_left_comm]

def run : List (Affine R X) → X → X
  | [], state => state
  | element :: later, state => run later (act element state)

def aggregate : List (Affine R X) → Affine R X
  | [] => identity
  | element :: later => combine (aggregate later) element

theorem aggregate_correct (elements : List (Affine R X)) (state : X) :
    act (aggregate elements) state = run elements state := by
  induction elements generalizing state with
  | nil => simp [aggregate, run]
  | cons element later inductionHypothesis =>
      simp only [aggregate, run]
      rw [act_combine]
      exact inductionHypothesis (act element state)

end Affine
end Aurelis
