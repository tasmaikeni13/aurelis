# Phase 1 failed iteration: pinned virtual environment absent

- Observed command: `.venv/bin/python -m py_compile experiments/phase1_oracle.py`
- Observed result: `/bin/bash: .venv/bin/python: No such file or directory`
- Classification: external environment / reproducibility substrate
- Disposition: rebuild the repository's pinned virtual environment with
  `./scripts/bootstrap.sh`; do not change experiment tolerances or equations.
- Experiment rows viewed before failure: none

The repository bootstrap then reproduced the earlier Phase 0 host failure:
Python 3.12 has no `ensurepip` because the matching Ubuntu venv package is
absent. The repair is the distribution-provided `python3.12-venv` package,
followed by an unchanged bootstrap rerun.
