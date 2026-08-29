#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [ -x "${HOME}/.elan/bin/lake" ]; then
  export PATH="${HOME}/.elan/bin:${PATH}"
fi

mkdir -p results/phase0/raw

.venv/bin/python scripts/audit_environment.py \
  2>&1 | tee results/phase0/raw/environment_audit.log

.venv/bin/python -m pytest \
  2>&1 | tee results/phase0/raw/pytest.log

(
  cd lean
  lake build
) 2>&1 | tee results/phase0/raw/lean_build.log

.venv/bin/python experiments/phase0_reference.py \
  2>&1 | tee results/phase0/raw/reference_experiment.log

.venv/bin/python benchmarks/phase0_components.py \
  2>&1 | tee results/phase0/raw/benchmark.log

.venv/bin/python scripts/verify_phase0.py
