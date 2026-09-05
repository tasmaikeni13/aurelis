"""AURELIS Language Model with Uncertainty-Routed Residual Attention over Delayed Bayesian State."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import LMConfig
from .hip_kernels import hip_fused_residual_gate
from .transformer import RMSNorm, RotaryEmbedding, SwiGLUMLP, apply_rotary_pos_emb


@dataclass
class AurelisDecodeCache:
    """Bounded, constant-size O(1) state per layer during autoregressive inference."""

    precision: Tensor  # [B, H, d_k, d_k]
    cross: Tensor      # [B, H, d_v, d_k]
    buffer_k: Tensor   # [B, H, window, d_k]
    buffer_v: Tensor   # [B, H, window, d_v]
    buffer_b: Tensor   # [B, H, window]
    count: int = 0


class AurelisAttentionBlock(nn.Module):
    """AURELIS same-head dual-memory attention layer."""

    def __init__(self, config: LMConfig, gate_mode: str = "aurelis_e") -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_key = config.d_key
        self.d_value = config.d_value
        self.window = config.window_size
        self.prior = config.prior
        self.gate_mode = gate_mode

        # Shared key/query chart preserves positive semi-definite RKHS kernel geometry
        self.key_query = nn.Linear(config.d_model, config.n_heads * config.d_key, bias=False)
        self.value = nn.Linear(config.d_model, config.n_heads * config.d_value, bias=False)
        self.evidence = nn.Linear(config.d_model, config.n_heads, bias=True)

        if gate_mode == "aurelis_e":
            self.episodic = nn.Linear(config.d_model, config.n_heads, bias=True)
            self.episodic.bias.data.fill_(-2.0)
        else:
            self.episodic = None

        self.log_temperature = nn.Parameter(torch.zeros(config.n_heads))
        self.out_proj = nn.Linear(config.n_heads * config.d_value, config.d_model, bias=config.bias)

    def _split_heads(self, x: Tensor, dim: int) -> Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, dim).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        decode_cache: Optional[AurelisDecodeCache] = None,
    ) -> Tuple[Tensor, AurelisDecodeCache]:
        B, L, _ = x.shape
        keys = self._split_heads(self.key_query(x), self.d_key)
        queries = keys.clone()
        values = self._split_heads(self.value(x), self.d_value)
        evidence = F.softplus(self.evidence(x)).transpose(1, 2) + 1e-4

        if self.episodic is not None:
            responsibility = torch.sigmoid(self.episodic(x)).transpose(1, 2)
        else:
            responsibility = None

        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

        if decode_cache is not None and L == 1:
            return self._forward_decode_step(
                queries, keys, values, evidence, responsibility, decode_cache
            )

        return self._forward_sequence(
            queries, keys, values, evidence, responsibility, x.device, x.dtype
        )

    def _forward_sequence(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        evidence: Tensor,
        responsibility: Optional[Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, AurelisDecodeCache]:
        B, H, L, D_k = keys.shape
        D_v = values.shape[-1]
        w = self.window

        # 1. Prefix cumulative state for the remote Bayesian regression
        # outer_precision: [B, H, L, D_k, D_k]
        outer_p = torch.einsum("bhli,bhlj,bhl->bhlij", keys, keys, evidence)
        outer_c = torch.einsum("bhlv,bhli,bhl->bhlvi", values, keys, evidence)

        prefix_p = torch.cumsum(outer_p, dim=2)
        prefix_c = torch.cumsum(outer_c, dim=2)

        # Delayed handoff: tokens stay in the window for w steps before entering remote state
        if w >= L:
            remote_p = torch.zeros_like(prefix_p)
            remote_c = torch.zeros_like(prefix_c)
        else:
            pad_p = torch.zeros(B, H, w, D_k, D_k, device=device, dtype=dtype)
            pad_c = torch.zeros(B, H, w, D_v, D_k, device=device, dtype=dtype)
            remote_p = torch.cat([pad_p, prefix_p[:, :, :-w]], dim=2)
            remote_c = torch.cat([pad_c, prefix_c[:, :, :-w]], dim=2)

        eye = torch.eye(D_k, device=device, dtype=dtype).view(1, 1, 1, D_k, D_k)
        precision = remote_p + self.prior * eye
        cross = remote_c

        # 2. Local causal softmax attention over the sliding window
        temp = self.log_temperature.exp().view(1, H, 1, 1)
        scores = torch.einsum("bhid,bhjd->bhij", queries, keys) * temp

        pos = torch.arange(L, device=device)
        q_pos = pos[:, None]
        k_pos = pos[None, :]
        causal_window_mask = (k_pos <= q_pos) & (k_pos > q_pos - w)
        scores = scores.masked_fill(~causal_window_mask.view(1, 1, L, L), -1e9)
        attn_weights = torch.softmax(scores, dim=-1)

        kbar = torch.einsum("bhij,bhjd->bhid", attn_weights, keys)
        vbar = torch.einsum("bhij,bhjv->bhiv", attn_weights, values)
        h = torch.sum(attn_weights.square() / evidence.unsqueeze(-2), dim=-1)

        # 3. Remote solve and innovation residual
        # Solve precision * P^{-1} [q, kbar]
        rhs = torch.stack((queries, kbar), dim=-1)  # [B, H, L, D_k, 2]
        factors = torch.linalg.cholesky(precision)
        solved = torch.cholesky_solve(rhs, factors)
        p_q, p_k = solved.unbind(dim=-1)

        remote = torch.einsum("bhlvi,bhli->bhlv", cross, p_q)
        mapped_kbar = torch.einsum("bhlvi,bhli->bhlv", cross, p_k)
        innovation = vbar - mapped_kbar

        # 4. Uncertainty Bayes gate g_B and episodic override g_E
        denom = (h + torch.sum(kbar * p_k, dim=-1)).clamp_min(1e-6)
        g_raw = torch.sum(queries * p_k, dim=-1) / denom
        g_B = torch.clamp(g_raw, 0.0, 1.0)

        if self.gate_mode == "aurelis_e" and responsibility is not None:
            # Straight-Through Estimator (STE)
            g_E_hard = torch.maximum(g_B, responsibility)
            g_E_soft = g_B + (1.0 - g_B) * responsibility
            g = g_E_hard + (g_E_soft - g_E_soft.detach())
        else:
            g = g_B

        # Fused residual gate
        y = hip_fused_residual_gate(remote, vbar, mapped_kbar, g)
        out = y.transpose(1, 2).contiguous().view(B, L, H * D_v)

        # Save decode cache from end of sequence
        new_cache = AurelisDecodeCache(
            precision=precision[:, :, -1, :, :].clone(),
            cross=cross[:, :, -1, :, :].clone(),
            buffer_k=keys[:, :, -w:, :].clone() if L >= w else keys.clone(),
            buffer_v=values[:, :, -w:, :].clone() if L >= w else values.clone(),
            buffer_b=evidence[:, :, -w:].clone() if L >= w else evidence.clone(),
            count=L,
        )
        return self.out_proj(out), new_cache

    def _forward_decode_step(
        self,
        query: Tensor,  # [B, H, 1, D_k]
        key: Tensor,    # [B, H, 1, D_k]
        value: Tensor,  # [B, H, 1, D_v]
        evidence: Tensor,  # [B, H, 1]
        responsibility: Optional[Tensor],  # [B, H, 1]
        cache: AurelisDecodeCache,
    ) -> Tuple[Tensor, AurelisDecodeCache]:
        B, H, _, D_k = key.shape
        D_v = value.shape[-1]
        w = self.window

        # Append new key/value to buffer
        buf_k = torch.cat([cache.buffer_k, key], dim=2)
        buf_v = torch.cat([cache.buffer_v, value], dim=2)
        buf_b = torch.cat([cache.buffer_b, evidence], dim=2)

        prec = cache.precision.clone()
        cross = cache.cross.clone()

        # Evict oldest if buffer exceeds window
        if buf_k.shape[2] > w:
            k_evict = buf_k[:, :, 0, :]  # [B, H, D_k]
            v_evict = buf_v[:, :, 0, :]  # [B, H, D_v]
            b_evict = buf_b[:, :, 0]     # [B, H]
            # Rank-1 update
            prec = prec + torch.einsum("bhi,bhj,bh->bhij", k_evict, k_evict, b_evict)
            cross = cross + torch.einsum("bhv,bhi,bh->bhvi", v_evict, k_evict, b_evict)
            buf_k = buf_k[:, :, 1:, :]
            buf_v = buf_v[:, :, 1:, :]
            buf_b = buf_b[:, :, 1:]

        # Attention against buffer
        temp = self.log_temperature.exp().view(1, H, 1, 1)
        scores = torch.einsum("bhid,bhjd->bhij", query, buf_k) * temp
        weights = torch.softmax(scores, dim=-1)

        kbar = torch.einsum("bhij,bhjd->bhid", weights, buf_k)
        vbar = torch.einsum("bhij,bhjv->bhiv", weights, buf_v)
        h = torch.sum(weights.square() / buf_b.unsqueeze(-2), dim=-1).squeeze(2)

        # Cholesky solve with 4D batch [B, H, D_k]
        q_2d = query.squeeze(2)
        k_2d = kbar.squeeze(2)
        rhs = torch.stack((q_2d, k_2d), dim=-1)  # [B, H, D_k, 2]
        factors = torch.linalg.cholesky(prec)
        solved = torch.cholesky_solve(rhs, factors)
        p_q, p_k = solved.unbind(dim=-1)

        remote = torch.einsum("bhvi,bhi->bhv", cross, p_q).unsqueeze(2)
        mapped_kbar = torch.einsum("bhvi,bhi->bhv", cross, p_k).unsqueeze(2)
        innovation = vbar - mapped_kbar

        denom = (h + torch.sum(k_2d * p_k, dim=-1)).clamp_min(1e-6)
        g_raw = torch.sum(q_2d * p_k, dim=-1) / denom
        g_B = torch.clamp(g_raw, 0.0, 1.0)

        if self.gate_mode == "aurelis_e" and responsibility is not None:
            resp = responsibility.squeeze(2)
            g_E_hard = torch.maximum(g_B, resp)
            g_E_soft = g_B + (1.0 - g_B) * resp
            g = g_E_hard + (g_E_soft - g_E_soft.detach())
        else:
            g = g_B

        y = hip_fused_residual_gate(remote, vbar, mapped_kbar, g)
        out = y.transpose(1, 2).contiguous().view(B, 1, H * D_v)

        new_cache = AurelisDecodeCache(
            precision=prec,
            cross=cross,
            buffer_k=buf_k,
            buffer_v=buf_v,
            buffer_b=buf_b,
            count=cache.count + 1,
        )
        return self.out_proj(out), new_cache


class AurelisBlock(nn.Module):
    """Full decoder layer combining Aurelis attention and SwiGLU MLP."""

    def __init__(self, config: LMConfig, gate_mode: str = "aurelis_e") -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = AurelisAttentionBlock(config, gate_mode=gate_mode)
        self.post_attention_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config.d_model, config.d_ffn, bias=config.bias)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        decode_cache: Optional[AurelisDecodeCache] = None,
    ) -> Tuple[Tensor, AurelisDecodeCache]:
        normed = self.input_norm(x)
        attn_out, new_cache = self.attn(normed, cos, sin, decode_cache=decode_cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_norm(x))
        return x, new_cache


class AurelisLM(nn.Module):
    """Full AURELIS Language Model (Candidate 1)."""

    def __init__(self, config: LMConfig, gate_mode: str = "aurelis_e") -> None:
        super().__init__()
        self.config = config
        self.gate_mode = gate_mode
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(
            config.d_key, max_seq_len=config.max_position_embeddings, theta=config.rope_theta
        )
        self.layers = nn.ModuleList(
            [AurelisBlock(config, gate_mode=gate_mode) for _ in range(config.n_layers)]
        )
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
        decode_caches: Optional[list[AurelisDecodeCache]] = None,
    ) -> Tuple[Tensor, list[AurelisDecodeCache]]:
        B, L = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self.rope(L, x.device)

        new_caches = []
        for i, layer in enumerate(self.layers):
            cache_i = decode_caches[i] if decode_caches is not None else None
            x, new_c = layer(x, cos, sin, decode_cache=cache_i)
            new_caches.append(new_c)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_caches

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
