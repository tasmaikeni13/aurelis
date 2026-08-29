# Phase 1 failed iteration: Lean toolchain manager absent

- Observation: the repository bootstrap completed but reported both `lake`
  and `${HOME}/.elan/bin/lake` absent.
- Classification: external environment / formal-verification substrate.
- Disposition: install Ubuntu's `elan` package, then let the repository's
  pinned `lean-toolchain` select Lean 4.19.0. No theorem or numerical gate is
  changed.
- Formal build attempted before failure: no.

