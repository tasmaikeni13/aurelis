#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [ -x "${HOME}/.elan/bin/lake" ]; then
  export PATH="${HOME}/.elan/bin:${PATH}"
fi

mkdir -p results/phase1/raw

./scripts/run_phase0.sh \
  2>&1 | tee results/phase1/raw/inherited_phase0.log

.venv/bin/python -m pytest \
  2>&1 | tee results/phase1/raw/pytest.log

(
  cd lean
  lake build
) 2>&1 | tee results/phase1/raw/lean_build.log

.venv/bin/python experiments/phase1_oracle.py \
  --config configs/phase1_oracle.json \
  2>&1 | tee results/phase1/raw/experiment.log

.venv/bin/python scripts/verify_phase1.py

