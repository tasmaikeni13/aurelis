# Phase 1 failed iteration: nonexistent Torch API

- Command: `.venv/bin/python experiments/phase1_oracle.py --config configs/phase1_oracle.json`
- Seed/config: `20260829`, `configs/phase1_oracle.json`
- Failure: `AttributeError: module 'torch' has no attribute 'geomspace'` in
  the handoff evidence constructor.
- Classification: experiment implementation.
- Smallest repair: use the supported `torch.logspace(-1, 1, length)` for the
  same preregistered endpoints and log spacing.
- Scientific rows retained/viewed: no aggregate was emitted before failure;
  no tolerance or equation changed.

