#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

.venv/bin/python analysis/aurelis_numerical.py
(
  cd lean
  lake build
)
.venv/bin/python analysis/audit_artifacts.py
