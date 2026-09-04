# Phase 2 failed iteration: Evaluation report generation order in verification

- Command: `./scripts/run_phase2.sh`
- Seed/config: `20260904`, `configs/phase2_baselines.json`
- Failure: `AssertionError: Phase 2 verification failed: {'required_artifacts_present': False}` because `results/phase2/report.md` had not yet been written when `verify_phase2.py` checked required artifacts.
- Classification: implementation / evaluator sequence order.
- Mechanism: In `verify_phase2.py`, `required_artifacts_present` checked for `results/phase2/report.md` before `report.md` was generated, causing a circular ordering dependency.
- Repair: Move `report.md` generation to `experiments/phase2_baselines.py` (matching Phase 1 design where the experiment emits `report.md` alongside `metrics.json`), ensuring all artifacts exist before `verify_phase2.py` validates them.
- Scientific rows retained/viewed: all experiment outputs and gate checks passed prior to this verifier assertion.
