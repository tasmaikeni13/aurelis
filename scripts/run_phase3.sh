#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [ -x "${HOME}/.elan/bin/lake" ]; then
  export PATH="${HOME}/.elan/bin:${PATH}"
fi

mkdir -p results/phase3/raw

echo "=== Running Inherited Phase 0 ==="
./scripts/run_phase0.sh \
  2>&1 | tee results/phase3/raw/inherited_phase0.log

echo "=== Running Inherited Phase 1 ==="
./scripts/run_phase1.sh \
  2>&1 | tee results/phase3/raw/inherited_phase1.log

echo "=== Running Inherited Phase 2 ==="
./scripts/run_phase2.sh \
  2>&1 | tee results/phase3/raw/inherited_phase2.log

echo "=== Running Pytest Suite ==="
.venv/bin/python -m pytest \
  2>&1 | tee results/phase3/raw/pytest.log

echo "=== Building Lean Formal Proofs ==="
(
  cd lean
  lake build
) 2>&1 | tee results/phase3/raw/lean_build.log

echo "=== Checking for Lean Placeholders ==="
if rg -n --glob "*.lean" "\b(sorry|admit|axiom)\b" lean; then
  echo "Found Lean placeholder or axiom!" >&2
  exit 1
fi

echo "=== Running Phase 3 Experiment ==="
.venv/bin/python -u experiments/phase3_learned.py \
  --config configs/phase3_learned.json \
  2>&1 | tee results/phase3/raw/experiment.log

echo "=== Verifying Phase 3 Gates ==="
.venv/bin/python -u scripts/verify_phase3.py
