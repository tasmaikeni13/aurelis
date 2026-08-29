"""Shared deterministic test configuration."""

from __future__ import annotations

import random

import torch


def pytest_sessionstart() -> None:
    random.seed(20260829)
    torch.manual_seed(20260829)
    torch.set_default_dtype(torch.float64)

