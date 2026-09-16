#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! python3 -m venv --system-site-packages .venv; then
  echo "Python venv bootstrap failed." >&2
  exit 1
fi

.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e '.[dev]'

if ! command -v lake >/dev/null 2>&1 && [ ! -x "${HOME}/.elan/bin/lake" ]; then
  echo "Lean/Lake is absent. Install elan from https://github.com/leanprover/elan before running Phase 0." >&2
fi

echo "Bootstrap complete for Cloud TPU v4 pod substrate."
