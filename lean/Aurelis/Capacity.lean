import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.BigOperators

/-! # Finite-state exact-recall lower bound

This is a deterministic finite-state statement. It assumes arbitrary values
at fixed distinct addresses and exact answers to every address query.
-/

namespace Aurelis

theorem exact_recall_injective {Address Value State : Type*}
    (encode : (Address → Value) → State) (decode : State → Address → Value)
    (correct : ∀ history address, decode (encode history) address = history address) :
    Function.Injective encode := by
  intro left right heq
  funext address
  calc
    left address = decode (encode left) address := (correct left address).symm
    _ = decode (encode right) address := by rw [heq]
    _ = right address := correct right address

theorem exact_recall_capacity {Address Value State : Type*}
    [Fintype Address] [Fintype Value] [Fintype State]
    (encode : (Address → Value) → State) (decode : State → Address → Value)
    (correct : ∀ history address, decode (encode history) address = history address) :
    Fintype.card Value ^ Fintype.card Address ≤ Fintype.card State := by
  classical
  simpa using Fintype.card_le_of_injective encode
    (exact_recall_injective encode decode correct)

end Aurelis
