#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

.venv/bin/python scripts/audit_environment.py \
  --output environment.txt \
  --json-output results/environment.json
.venv/bin/python -m pytest
(
  cd lean
  lake build
)
.venv/bin/python experiments/phase1_numerics.py \
  --config configs/phase1_reference.json

