#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

echo "=== Running Phase 6 Benchmark Experiment on AMD Instinct MI300X ==="
.venv/bin/python experiments/phase6_benchmarks.py

echo "=== Verifying Phase 6 PASS Gates ==="
.venv/bin/python scripts/verify_phase6.py

echo "=== Phase 6 Complete ==="
