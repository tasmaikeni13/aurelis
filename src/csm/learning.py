"""Small differentiable memory models used by the learned CSM phases.

These are deliberately reference-scale modules.  They expose the encoded
geometry and sufficient statistics so experiments can audit what optimization
actually learned instead of treating the memory as an opaque layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

LearnedMemoryKind = Literal["csm", "hebbian", "attention"]


@dataclass(frozen=True)
class BatchedCSMState:
    """Batched sufficient statistics with shapes ``[B,d_k,d_k]`` and ``[B,d_v,d_k]``."""

    S: Tensor
    C: Tensor


@dataclass(frozen=True)
class MemoryForward:
    prediction: Tensor
    keys: Tensor
    queries: Tensor
    values: Tensor
    state: BatchedCSMState | None
    uncertainty: Tensor | None


def _feature_mlp(input_dimension: int, hidden_dimension: int, output_dimension: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dimension, hidden_dimension),
        nn.GELU(),
        nn.Linear(hidden_dimension, output_dimension),
    )


class EpisodicMemoryModel(nn.Module):
    """Learned feature maps around CSM, Hebbian, or explicit attention reads."""

    def __init__(
        self,
        raw_key_dimension: int,
        raw_value_dimension: int,
        key_dimension: int,
        value_dimension: int,
        hidden_dimension: int,
        *,
        epsilon: float,
        kind: LearnedMemoryKind = "csm",
        fixed_random_features: bool = False,
        shared_key_query: bool = True,
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if kind not in ("csm", "hebbian", "attention"):
            raise ValueError(f"unknown memory kind: {kind}")
        self.raw_key_dimension = raw_key_dimension
        self.raw_value_dimension = raw_value_dimension
        self.key_dimension = key_dimension
        self.value_dimension = value_dimension
        self.epsilon = float(epsilon)
        self.kind = kind
        self.fixed_random_features = fixed_random_features
        self.shared_key_query = shared_key_query

        self.key_encoder = _feature_mlp(
            raw_key_dimension, hidden_dimension, key_dimension
        )
        if shared_key_query or fixed_random_features:
            # A ridge operator and its query must inhabit one coordinate chart.
            # An independent query map remains available as an explicit ablation.
            self.query_encoder = self.key_encoder
        else:
            import copy

            self.query_encoder = copy.deepcopy(self.key_encoder)
        self.log_query_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.value_encoder = nn.Linear(
            raw_value_dimension, value_dimension, bias=False
        )
        self.output_decoder = nn.Linear(
            value_dimension, raw_value_dimension, bias=True
        )
        self.log_attention_scale = nn.Parameter(
            torch.tensor(math.log(10.0), dtype=torch.float32)
        )
        if fixed_random_features:
            for module in (self.key_encoder, self.value_encoder):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            self.log_query_scale.requires_grad_(False)
            self.log_attention_scale.requires_grad_(False)

    def encode_keys(self, raw_keys: Tensor) -> Tensor:
        return F.normalize(self.key_encoder(raw_keys), dim=-1)

    def encode_queries(self, raw_queries: Tensor) -> Tensor:
        # A scalar can calibrate uniform finite-epsilon shrinkage without
        # introducing a second feature coordinate system.
        scale = torch.exp(self.log_query_scale).clamp(max=10.0)
        return scale * F.normalize(self.query_encoder(raw_queries), dim=-1)

    def encode_values(self, raw_values: Tensor) -> Tensor:
        return self.value_encoder(raw_values)

    def decode(self, latent_values: Tensor) -> Tensor:
        return self.output_decoder(latent_values)

    def initial_state(self, batch_size: int, reference: Tensor) -> BatchedCSMState:
        return BatchedCSMState(
            S=reference.new_zeros(
                (batch_size, self.key_dimension, self.key_dimension)
            ),
            C=reference.new_zeros(
                (batch_size, self.value_dimension, self.key_dimension)
            ),
        )

    def write_state(
        self,
        state: BatchedCSMState,
        key: Tensor,
        value: Tensor,
        beta: Tensor,
        decay: Tensor,
        *,
        value_weight: Tensor | None = None,
    ) -> BatchedCSMState:
        """Batched recurrence; ``value_weight`` enables the generic-gate control."""

        if value_weight is None:
            value_weight = beta
        key_outer = key.unsqueeze(-1) * key.unsqueeze(-2)
        value_key = value.unsqueeze(-1) * key.unsqueeze(-2)
        return BatchedCSMState(
            S=decay[:, None, None] * state.S + beta[:, None, None] * key_outer,
            C=(
                decay[:, None, None] * state.C
                + value_weight[:, None, None] * value_key
            ),
        )

    def read_state(
        self, state: BatchedCSMState, queries: Tensor
    ) -> tuple[Tensor, Tensor]:
        identity = torch.eye(
            self.key_dimension, dtype=state.S.dtype, device=state.S.device
        )
        system = state.S + self.epsilon * identity
        solved = torch.linalg.solve(system, queries.mT).mT
        latent = solved @ state.C.mT
        uncertainty = torch.einsum("bqd,bqd->bq", queries, solved)
        return latent, uncertainty

    def forward(
        self,
        support_keys: Tensor,
        support_values: Tensor,
        queries: Tensor,
        *,
        beta: Tensor | None = None,
        value_weight: Tensor | None = None,
    ) -> MemoryForward:
        keys = self.encode_keys(support_keys)
        encoded_queries = self.encode_queries(queries)
        values = self.encode_values(support_values)
        associations = keys.shape[1]
        if beta is None:
            beta = keys.new_ones(keys.shape[:2])

        state: BatchedCSMState | None = None
        uncertainty: Tensor | None = None
        if self.kind == "csm":
            weighted_keys = beta.unsqueeze(-1) * keys
            S = keys.mT @ weighted_keys
            cross_weights = beta if value_weight is None else value_weight
            C = values.mT @ (cross_weights.unsqueeze(-1) * keys)
            state = BatchedCSMState(S=S, C=C)
            latent, uncertainty = self.read_state(state, encoded_queries)
        elif self.kind == "hebbian":
            C = values.mT @ keys
            latent = (encoded_queries @ C.mT) / associations
        else:
            logits = (
                encoded_queries @ keys.mT
                * torch.exp(self.log_attention_scale).clamp(max=100.0)
            )
            weights = torch.softmax(logits, dim=-1)
            latent = weights @ values
        return MemoryForward(
            prediction=self.decode(latent),
            keys=keys,
            queries=encoded_queries,
            values=values,
            state=state,
            uncertainty=uncertainty,
        )


class BoundedScalarGate(nn.Module):
    """Small observable-cue gate with explicit probabilistic support bounds."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        *,
        minimum: float,
        maximum: float,
        initial_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= minimum < maximum:
            raise ValueError("gate bounds must satisfy 0 <= minimum < maximum")
        if not 0.0 < initial_fraction < 1.0:
            raise ValueError("initial_fraction must be in (0, 1)")
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(
            self.network[-1].bias,
            math.log(initial_fraction / (1.0 - initial_fraction)),
        )

    def forward(self, cues: Tensor) -> Tensor:
        fraction = torch.sigmoid(self.network(cues)).squeeze(-1)
        return self.minimum + (self.maximum - self.minimum) * fraction


def orthogonality_penalty(keys: Tensor) -> Tensor:
    """Mean squared off-diagonal Gram entry, used only as an explicit ablation."""

    associations = keys.shape[1]
    gram = keys @ keys.mT
    identity = torch.eye(associations, dtype=keys.dtype, device=keys.device)
    return ((gram - identity) ** 2).sum(dim=(-2, -1)).mean() / max(
        associations * (associations - 1), 1
    )


def geometry_metrics(keys: Tensor, epsilon: float) -> dict[str, float | list[float]]:
    """Summarize learned key geometry across an episodic batch."""

    with torch.no_grad():
        gram = keys @ keys.mT
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        normalized = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(keys.dtype).tiny
        )
        entropy = -(
            normalized
            * torch.log(normalized.clamp_min(torch.finfo(keys.dtype).tiny))
        ).sum(dim=-1)
        effective_rank = torch.exp(entropy)
        singular = torch.linalg.svdvals(keys)
        identity = torch.eye(
            keys.shape[-1], dtype=keys.dtype, device=keys.device
        )
        system = keys.mT @ keys + epsilon * identity
        system_condition = torch.linalg.cond(system)
        associations = keys.shape[1]
        off_diagonal = gram - torch.eye(
            associations, dtype=keys.dtype, device=keys.device
        )
        mask = ~torch.eye(
            associations, dtype=torch.bool, device=keys.device
        )
        pairwise = off_diagonal[:, mask]
        mean_spectrum = eigenvalues.mean(dim=0).cpu().tolist()
        return {
            "gram_eigenvalue_spectrum": [float(value) for value in mean_spectrum],
            "minimum_gram_eigenvalue": float(eigenvalues[:, 0].mean().item()),
            "maximum_gram_eigenvalue": float(eigenvalues[:, -1].mean().item()),
            "mean_pairwise_cosine": float(pairwise.mean().item()),
            "mean_absolute_pairwise_cosine": float(pairwise.abs().mean().item()),
            "maximum_absolute_pairwise_cosine": float(pairwise.abs().max().item()),
            "effective_rank": float(effective_rank.mean().item()),
            "minimum_singular_value": float(singular[:, -1].mean().item()),
            "system_condition_number": float(system_condition.mean().item()),
            "fraction_capacity_used": float(
                associations / keys.shape[-1]
            ),
            # rank(K) <= min(number of associations, feature dimension)
            "effective_capacity_fraction": float(
                effective_rank.mean().item()
                / min(associations, keys.shape[-1])
            ),
        }
