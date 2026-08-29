# Phase 0 failed iteration: Lean toolchain absent

- Status: repaired in a subsequent iteration
- UTC date: 2026-08-29
- Classification: external environment / formal toolchain
- Command: `./scripts/run_phase0.sh`
- Base commit: `efe860b154ccd7003a5660b17bdc35193694e153`
- Preceding gate: 43 Python tests passed
- Failed gate: pinned Lean build

## Frozen output

The raw log is retained as `lean_toolchain_missing_20260829.txt`:

```text
./scripts/run_phase0.sh: line 17: lake: command not found
```

This is not a theorem or proof-script failure: no Lean source was compiled.
The project already pins `leanprover/lean4:v4.19.0` in `lean/lean-toolchain`
and mathlib `v4.19.0` in the manifest.

## Research and repair

The official elan documentation states that elan installs `lean` and `lake`
proxies which select and download the version named by a project's
`lean-toolchain` file. Source consulted 2026-08-29:
https://github.com/leanprover/elan/blob/master/README.md

The repair installs elan in the user account, leaves the pinned project files
unchanged, and invokes the resulting `lake` proxy. This predicts selection of
Lean 4.19.0 and a normal `lake build`; it cannot turn a failed proof into a
pass, so all proof errors remain visible to the fail-fast gate.
