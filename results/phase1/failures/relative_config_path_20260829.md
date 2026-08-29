# Phase 1 failed iteration: relative config metadata path

- Command/config/seed: pinned Phase 1 command, config, and seed `20260829`.
- Failure: the completed in-memory experiment could not serialize metadata
  because `Path.relative_to` was called on a relative path and an absolute
  repository path.
- Classification: experiment reporting implementation.
- Repair: resolve the config path before making it repository-relative.
- Gate/tolerance disposition: unchanged. No metrics file was emitted, so this
  iteration is not cited as evidence.

