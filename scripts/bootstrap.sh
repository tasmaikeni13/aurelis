#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! python3 -m venv .venv; then
  echo "Python venv bootstrap failed. On Ubuntu, install the matching python3.12-venv package." >&2
  exit 1
fi
.venv/bin/python -m pip install 'pip==26.2.1' 'wheel==0.48.0'
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e '.[dev]'
.venv/bin/python -m pip check

if ! command -v lake >/dev/null 2>&1 && [ ! -x "${HOME}/.elan/bin/lake" ]; then
  echo "Lean/Lake is absent. Install elan from https://github.com/leanprover/elan before running Phase 0." >&2
fi
