from __future__ import annotations

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_test_seed() -> None:
    random.seed(1729)
    np.random.seed(1729)
    torch.manual_seed(1729)

