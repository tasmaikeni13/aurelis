"""Small decoder-only language models for controlled CSM comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .systems import AffineSummary, read_prefix_states, token_summaries

ArchitectureKind = Literal["transformer", "csm", "hybrid", "recurrent"]


@dataclass(frozen=True)
class CSMDecodeState:
    S: Tensor
    C: Tensor


@dataclass(frozen=True)
class AttentionDecodeState:
    keys: Tensor
    values: Tensor


LayerDecodeState = CSMDecodeState | AttentionDecodeState | Tensor | None


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, *, window: int | None = None) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.window = window
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def _split(self, tensor: Tensor) -> Tensor:
        batch, steps, _ = tensor.shape
        return tensor.view(batch, steps, self.heads, self.head_width).transpose(1, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)
        q, k, v = self._split(query), self._split(key), self._split(value)
        if self.window is None:
            mixed = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            steps = inputs.shape[1]
            position = torch.arange(steps, device=inputs.device)
            distance = position[:, None] - position[None, :]
            allowed = (distance >= 0) & (distance < self.window)
            mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.output(mixed.transpose(1, 2).contiguous().view_as(inputs))

    def step(
        self, inputs: Tensor, state: AttentionDecodeState | None
    ) -> tuple[Tensor, AttentionDecodeState]:
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)
        q, k, v = self._split(query), self._split(key), self._split(value)
        if state is not None:
            k = torch.cat((state.keys, k), dim=-2)
            v = torch.cat((state.values, v), dim=-2)
        if self.window is not None and k.shape[-2] > self.window:
            k = k[..., -self.window :, :]
            v = v[..., -self.window :, :]
        mixed = F.scaled_dot_product_attention(q, k, v)
        output = self.output(mixed.transpose(1, 2).contiguous().view_as(inputs))
        return output, AttentionDecodeState(k, v)

    def recurrent_state_bytes(self, batch_size: int, context_length: int) -> int:
        retained = context_length if self.window is None else min(context_length, self.window)
        return batch_size * retained * 2 * self.width * 2


class CSMMixer(nn.Module):
    """Multi-head exact ridge-memory mixer with fp32 state and solves."""

    def __init__(
        self,
        width: int,
        heads: int,
        key_dimension: int,
        *,
        epsilon: float,
        beta_minimum: float = 0.05,
        beta_maximum: float = 1.0,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.width = width
        self.heads = heads
        self.key_dimension = key_dimension
        self.value_dimension = width // heads
        self.epsilon = float(epsilon)
        self.beta_minimum = float(beta_minimum)
        self.beta_maximum = float(beta_maximum)
        self.key = nn.Linear(width, heads * key_dimension, bias=False)
        self.query = nn.Linear(width, heads * key_dimension, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.beta = nn.Linear(width, heads, bias=True)
        self.output = nn.Linear(width, width, bias=False)
        self.log_query_scale = nn.Parameter(torch.zeros(heads))
        nn.init.zeros_(self.beta.weight)
        nn.init.constant_(self.beta.bias, 2.0)

    def _features(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, steps, _ = inputs.shape
        keys = self.key(inputs).view(batch, steps, self.heads, self.key_dimension).transpose(1, 2)
        queries = self.query(inputs).view(batch, steps, self.heads, self.key_dimension).transpose(1, 2)
        values = self.value(inputs).view(batch, steps, self.heads, self.value_dimension).transpose(1, 2)
        keys = F.normalize(keys.float(), dim=-1).to(inputs.dtype)
        scale = self.log_query_scale.exp().clamp(max=10)[None, :, None, None]
        queries = (scale * F.normalize(queries.float(), dim=-1)).to(inputs.dtype)
        fraction = torch.sigmoid(self.beta(inputs)).transpose(1, 2)
        beta = self.beta_minimum + (self.beta_maximum - self.beta_minimum) * fraction
        return keys, queries, values, beta

    def forward(self, inputs: Tensor) -> Tensor:
        keys, queries, values, beta = self._features(inputs)
        decay = torch.ones_like(beta)
        tokens = token_summaries(keys, values, beta, decay, accumulation_dtype=torch.float32)
        prefixes = AffineSummary(
            decay=torch.ones_like(tokens.decay),
            S=torch.cumsum(tokens.S, dim=-3),
            C=torch.cumsum(tokens.C, dim=-3),
        )
        reads, _ = read_prefix_states(
            prefixes, queries, self.epsilon, output_dtype=inputs.dtype
        )
        joined = reads.transpose(1, 2).contiguous().view_as(inputs)
        return self.output(joined)

    def step(
        self, inputs: Tensor, state: CSMDecodeState | None
    ) -> tuple[Tensor, CSMDecodeState]:
        if inputs.shape[1] != 1:
            raise ValueError("incremental CSM step expects exactly one token")
        keys, queries, values, beta = self._features(inputs)
        key = keys[:, :, 0].float()
        query = queries[:, :, 0].float()
        value = values[:, :, 0].float()
        weight = beta[:, :, 0].float()
        if state is None:
            S = key.new_zeros(
                (inputs.shape[0], self.heads, self.key_dimension, self.key_dimension)
            )
            C = key.new_zeros(
                (inputs.shape[0], self.heads, self.value_dimension, self.key_dimension)
            )
        else:
            S, C = state.S, state.C
        S = S + weight[..., None, None] * key[..., :, None] * key[..., None, :]
        C = C + weight[..., None, None] * value[..., :, None] * key[..., None, :]
        identity = torch.eye(self.key_dimension, dtype=S.dtype, device=S.device)
        factor = torch.linalg.cholesky(S + self.epsilon * identity)
        solved = torch.cholesky_solve(query.unsqueeze(-1), factor).squeeze(-1)
        read = torch.einsum("bhvk,bhk->bhv", C, solved).to(inputs.dtype)
        joined = read.reshape(inputs.shape[0], 1, self.width)
        return self.output(joined), CSMDecodeState(S, C)

    def recurrent_state_bytes(self, batch_size: int, _: int) -> int:
        return (
            batch_size
            * self.heads
            * self.key_dimension
            * (self.key_dimension + self.value_dimension)
            * 4
        )


class RecurrentMixer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.gru = nn.GRU(width, width, batch_first=True)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output(self.gru(inputs)[0])

    def step(self, inputs: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor]:
        output, hidden = self.gru(inputs, state)
        return self.output(output), hidden

    def recurrent_state_bytes(self, batch_size: int, _: int) -> int:
        return batch_size * self.width * 4


class DecoderBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        feedforward_width: int,
        mixer_kind: Literal["attention", "local_attention", "csm", "recurrent"],
        *,
        csm_key_dimension: int,
        csm_epsilon: float,
        local_window: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(width)
        self.norm2 = nn.RMSNorm(width)
        if mixer_kind == "attention":
            self.mixer: CausalSelfAttention | CSMMixer | RecurrentMixer = CausalSelfAttention(width, heads)
        elif mixer_kind == "local_attention":
            self.mixer = CausalSelfAttention(width, heads, window=local_window)
        elif mixer_kind == "csm":
            self.mixer = CSMMixer(width, heads, csm_key_dimension, epsilon=csm_epsilon)
        elif mixer_kind == "recurrent":
            self.mixer = RecurrentMixer(width)
        else:
            raise ValueError(f"unknown mixer kind: {mixer_kind}")
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_width, bias=False),
            nn.GELU(),
            nn.Linear(feedforward_width, width, bias=False),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs + self.mixer(self.norm1(inputs))
        return hidden + self.feedforward(self.norm2(hidden))

    def step(
        self, inputs: Tensor, state: LayerDecodeState
    ) -> tuple[Tensor, LayerDecodeState]:
        mixed, next_state = self.mixer.step(self.norm1(inputs), state)  # type: ignore[arg-type]
        hidden = inputs + mixed
        return hidden + self.feedforward(self.norm2(hidden)), next_state


class TinyDecoderLM(nn.Module):
    """A matched decoder family with training and true incremental paths."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        width: int,
        layers: int,
        heads: int,
        feedforward_width: int,
        architecture: ArchitectureKind,
        csm_key_dimension: int = 16,
        csm_epsilon: float = 0.1,
        local_window: int = 64,
    ) -> None:
        super().__init__()
        if architecture not in ("transformer", "csm", "hybrid", "recurrent"):
            raise ValueError(f"unknown architecture: {architecture}")
        self.vocabulary_size = vocabulary_size
        self.width = width
        self.architecture = architecture
        self.embedding = nn.Embedding(vocabulary_size, width)
        self.blocks = nn.ModuleList()
        for index in range(layers):
            if architecture == "transformer":
                mixer = "attention"
            elif architecture == "csm":
                mixer = "csm"
            elif architecture == "hybrid":
                mixer = "local_attention" if index % 2 == 0 else "csm"
            else:
                mixer = "recurrent"
            self.blocks.append(
                DecoderBlock(
                    width,
                    heads,
                    feedforward_width,
                    mixer,  # type: ignore[arg-type]
                    csm_key_dimension=csm_key_dimension,
                    csm_epsilon=csm_epsilon,
                    local_window=local_window,
                )
            )
        self.norm = nn.RMSNorm(width)
        self.output = nn.Linear(width, vocabulary_size, bias=False)
        self.output.weight = self.embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _position(self, length: int, offset: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        positions = torch.arange(offset, offset + length, device=device, dtype=torch.float32)
        frequencies = torch.exp(
            torch.arange(0, self.width, 2, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / self.width)
        )
        angles = positions[:, None] * frequencies[None, :]
        embedding = torch.empty(length, self.width, device=device, dtype=torch.float32)
        embedding[:, 0::2] = torch.sin(angles)
        embedding[:, 1::2] = torch.cos(angles)
        return embedding.to(dtype)

    def forward(self, tokens: Tensor) -> Tensor:
        hidden = self.embedding(tokens)
        hidden = hidden + self._position(tokens.shape[1], 0, tokens.device, hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.norm(hidden))

    def step(
        self,
        token: Tensor,
        states: list[LayerDecodeState] | None,
        position: int,
    ) -> tuple[Tensor, list[LayerDecodeState]]:
        if token.ndim == 1:
            token = token[:, None]
        if token.shape[1] != 1:
            raise ValueError("incremental decode expects one token per batch")
        if states is None:
            states = [None] * len(self.blocks)
        hidden = self.embedding(token)
        hidden = hidden + self._position(1, position, token.device, hidden.dtype)
        next_states: list[LayerDecodeState] = []
        for block, state in zip(self.blocks, states, strict=True):
            hidden, next_state = block.step(hidden, state)
            next_states.append(next_state)
        return self.output(self.norm(hidden)), next_states

    @torch.no_grad()
    def generate(self, prompt: Tensor, new_tokens: int) -> Tensor:
        if prompt.ndim != 2:
            raise ValueError("prompt must have shape [batch, time]")
        states: list[LayerDecodeState] | None = None
        logits = None
        for position in range(prompt.shape[1]):
            logits, states = self.step(prompt[:, position], states, position)
        generated = [prompt]
        current = prompt[:, -1]
        for offset in range(new_tokens):
            assert logits is not None
            current = logits[:, -1].argmax(dim=-1)
            generated.append(current[:, None])
            if offset + 1 < new_tokens:
                logits, states = self.step(current, states, prompt.shape[1] + offset)
        return torch.cat(generated, dim=1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def recurrent_state_bytes(self, batch_size: int, context_length: int) -> int:
        return sum(
            block.mixer.recurrent_state_bytes(batch_size, context_length)
            for block in self.blocks
        )
