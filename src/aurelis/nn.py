"""Minimal learned projection wrapper around the exact AURELIS reference."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .training import vectorized_reference
from .types import SequenceOutput


class AurelisProjectionBlock(nn.Module):
    """Project hidden states into a multi-head exact AURELIS reference block."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        d_key: int,
        d_value: int,
        window: int,
        *,
        prior: float = 1.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.d_key = d_key
        self.d_value = d_value
        self.window = window
        self.prior = prior
        self.query = nn.Linear(d_model, heads * d_key, bias=False)
        self.key = nn.Linear(d_model, heads * d_key, bias=False)
        self.value = nn.Linear(d_model, heads * d_value, bias=False)
        self.evidence = nn.Linear(d_model, heads, bias=True)
        self.episodic = nn.Linear(d_model, heads, bias=True)
        self.log_temperature = nn.Parameter(torch.zeros(heads))
        self.output = nn.Linear(heads * d_value, d_model, bias=False)

    def _heads(self, tensor: Tensor, dimension: int) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.heads, dimension).transpose(1, 2)

    def forward(self, hidden: Tensor) -> tuple[Tensor, SequenceOutput]:
        keys = self._heads(self.key(hidden), self.d_key)
        queries = self._heads(self.query(hidden), self.d_key)
        values = self._heads(self.value(hidden), self.d_value)
        evidence = F.softplus(self.evidence(hidden)).transpose(1, 2) + 1e-6
        responsibility = torch.sigmoid(self.episodic(hidden)).transpose(1, 2)
        sequence = vectorized_reference(
            keys,
            values,
            evidence,
            queries,
            window=self.window,
            prior=self.prior,
            temperature=self.log_temperature.exp().view(1, self.heads, 1),
            episodic_responsibility=responsibility,
        )
        joined = sequence.episodic.transpose(1, 2).reshape(
            hidden.shape[0], hidden.shape[1], self.heads * self.d_value
        )
        return self.output(joined), sequence
