# Phase 2 failed iteration: Torch randn_like generator API mismatch

- Command: `.venv/bin/python experiments/phase2_baselines.py --config configs/phase2_baselines.json --device cuda`
- Seed/config: `20260904`, `configs/phase2_baselines.json`
- Failure: `TypeError: randn_like() got an unexpected keyword argument 'generator'` at line 358.
- Classification: implementation / framework API mismatch.
- Mechanism: In PyTorch 2.8 on ROCm, `torch.randn_like` does not accept a `generator` keyword argument.
- Repair: Replace `torch.randn_like(tensor, generator=gen)` with `torch.randn(tensor.shape, dtype=tensor.dtype, generator=gen).to(device)`.
- Scientific rows retained/viewed: no aggregate was emitted before failure; no equation or gate was altered.
