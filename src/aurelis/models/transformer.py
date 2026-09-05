"""Modern Causal Transformer baseline with RoPE, Pre-RMSNorm, and SwiGLU."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import LMConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Applies RoPE to query and key tensors."""
    # cos, sin: [L, D] -> [1, 1, L, D]
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLUMLP(nn.Module):
    """SwiGLU Feedforward Network."""

    def __init__(self, d_model: int, d_ffn: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=bias)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and KV caching."""

    def __init__(self, config: LMConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=config.bias)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=-2)
            v = torch.cat([prev_v, v], dim=-2)
        new_kv_cache = (k, v)

        # Causal scaled dot-product attention
        is_causal = (kv_cache is None) or (L > 1)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, dropout_p=0.0, scale=self.scale
        )
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out), new_kv_cache


class TransformerBlock(nn.Module):
    """Transformer decoder layer with pre-RMSNorm."""

    def __init__(self, config: LMConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.post_attention_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config.d_model, config.d_ffn, bias=config.bias)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        normed_x = self.input_norm(x)
        attn_out, new_kv = self.attn(normed_x, cos, sin, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_norm(x))
        return x, new_kv


class TransformerLM(nn.Module):
    """Full causal Transformer language model baseline."""

    def __init__(self, config: LMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(
            config.head_dim, max_seq_len=config.max_position_embeddings, theta=config.rope_theta
        )
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        kv_caches: Optional[list[Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, list[Tuple[Tensor, Tensor]]]:
        B, L = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self.rope(L, x.device)

        new_caches = []
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, new_kv = layer(x, cos, sin, kv_cache=cache_i)
            new_caches.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_caches

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
