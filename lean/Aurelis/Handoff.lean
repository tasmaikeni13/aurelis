import Mathlib.Data.List.Basic

/-!
# Delayed handoff

Histories are represented newest first. `recent window history` is the exact
attention cache and `remote window history` is the disjoint suffix summarized
by the inference state.
-/

namespace Aurelis

def recent {α : Type*} (window : Nat) (history : List α) : List α :=
  history.take window

def remote {α : Type*} (window : Nat) (history : List α) : List α :=
  history.drop window

/-- The cache and remote suffix reconstruct every token occurrence exactly once. -/
theorem handoff_partition {α : Type*} (window : Nat) (history : List α) :
    recent window history ++ remote window history = history := by
  exact List.take_append_drop window history

theorem recent_length_le_window {α : Type*} (window : Nat) (history : List α) :
    (recent window history).length ≤ window := by
  simpa [recent] using Nat.min_le_left window history.length

theorem remote_empty_before_window {α : Type*} {window : Nat} {history : List α}
    (h : history.length ≤ window) : remote window history = [] := by
  simp [remote, List.drop_eq_nil_iff.mpr h]

end Aurelis
