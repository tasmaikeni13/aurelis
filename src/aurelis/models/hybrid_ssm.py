"""Strong SSM + Attention Hybrid candidate (Mamba/SSM + Multi-Head Causal Attention)."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import LMConfig
from .hip_kernels import hip_recurrent_scan
from .transformer import CausalSelfAttention, RMSNorm, RotaryEmbedding, SwiGLUMLP


class SelectiveSSMBlock(nn.Module):
    """Selective State-Space Model block (Mamba-2 style selective scan)."""

    def __init__(self, config: LMConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.ssm_state_dim
        self.d_inner = config.d_model
        self.conv_kernel = config.ssm_conv_kernel

        # In-projection for input and gating branch
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=config.bias)

        # 1D causal depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.conv_kernel,
            groups=self.d_inner,
            padding=self.conv_kernel - 1,
            bias=True,
        )

        # Discretization projections: Delta, B, C
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # State transition parameter A: log-initialized
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)

    def forward(
        self,
        x: Tensor,
        ssm_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B, L, _ = x.shape

        # In-project: [B, L, 2 * d_inner] -> split into branch and gate
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # 1D Causal convolution
        if ssm_cache is not None:
            prev_conv, prev_h = ssm_cache
            conv_input = torch.cat([prev_conv, x_branch.transpose(1, 2)], dim=-1)
            new_prev_conv = conv_input[:, :, -self.conv_kernel + 1 :]
            conv_out = self.conv1d(conv_input)[..., -L:]
        else:
            conv_out = self.conv1d(x_branch.transpose(1, 2))[..., :L]
            new_prev_conv = x_branch.transpose(1, 2)[:, :, -self.conv_kernel + 1 :]
            prev_h = None

        x_conv = F.silu(conv_out.transpose(1, 2))

        # Delta, B, C input-dependent projections
        ssm_params = self.x_proj(x_conv)  # [B, L, 2 * d_state + 1]
        dt = ssm_params[..., :1]
        B_proj = ssm_params[..., 1 : 1 + self.d_state]  # [B, L, d_state]
        C_proj = ssm_params[..., 1 + self.d_state :]    # [B, L, d_state]

        dt = F.softplus(self.dt_proj(dt))  # [B, L, d_inner]
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        # Selective scan: h_t = exp(dt_t * A) * h_{t-1} + (dt_t * B_t) * x_t
        # Discretized decay per dim: [B, L, d_inner, d_state]
        dA = torch.exp(torch.einsum("bld,dn->bldn", dt, A))
        dBx = torch.einsum("bld,bln,bld->bldn", dt, B_proj, x_conv)

        # Recurrence using optimized scan or step loop
        if ssm_cache is None and L > 1:
            # Full sequence forward
            # Reshape for scan: [B, d_inner * d_state, L]
            dBx_flat = dBx.view(B, L, -1).transpose(1, 2).unsqueeze(1)  # [B, 1, D_flat, L]
            dA_flat = dA.view(B, L, -1).transpose(1, 2).unsqueeze(1)
            # Scan along L: transpose back to [B, 1, L, D_flat]
            dBx_scan = dBx_flat.transpose(2, 3)
            dA_scan = dA_flat.transpose(2, 3)

            h_scan = hip_recurrent_scan(dBx_scan, dA_scan)
            h_all = h_scan.squeeze(1).view(B, L, self.d_inner, self.d_state)
            new_prev_h = h_all[:, -1, :, :]
            y = torch.einsum("bldn,bln->bld", h_all, C_proj)
        else:
            # Step by step
            h_curr = prev_h if prev_h is not None else torch.zeros(
                B, self.d_inner, self.d_state, dtype=x.dtype, device=x.device
            )
            y_list = []
            for t in range(L):
                h_curr = dA[:, t, :, :] * h_curr + dBx[:, t, :, :]
                y_t = torch.einsum("bdn,bn->bd", h_curr, C_proj[:, t, :])
                y_list.append(y_t)
            new_prev_h = h_curr
            y = torch.stack(y_list, dim=1)

        # Multiplicative SiLU gate
        y = y * F.silu(z_branch)
        new_cache = (new_prev_conv, new_prev_h)
        return self.out_proj(y), new_cache


class HybridBlock(nn.Module):
    """Interleaved layer containing either an SSM block or an Attention block, plus SwiGLU MLP."""

    def __init__(self, config: LMConfig, is_attention_layer: bool) -> None:
        super().__init__()
        self.is_attention_layer = is_attention_layer
        self.input_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        if is_attention_layer:
            self.mixer = CausalSelfAttention(config)
        else:
            self.mixer = SelectiveSSMBlock(config)

        self.post_mixer_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config.d_model, config.d_ffn, bias=config.bias)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        layer_cache: Optional[Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        normed = self.input_norm(x)
        if self.is_attention_layer:
            mixer_out, new_cache = self.mixer(normed, cos, sin, kv_cache=layer_cache)
        else:
            mixer_out, new_cache = self.mixer(normed, ssm_cache=layer_cache)

        x = x + mixer_out
        x = x + self.mlp(self.post_mixer_norm(x))
        return x, new_cache


class HybridSSMLM(nn.Module):
    """Full SSM + Attention Hybrid Language Model (Candidate 3)."""

    def __init__(self, config: LMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(
            config.head_dim, max_seq_len=config.max_position_embeddings, theta=config.rope_theta
        )

        # Interleave SSM and Attention: every 2nd or 3rd layer is Attention
        # Standard Samba / Jamba pattern: alternating or 1 attention every 2 SSM layers
        layers = []
        for i in range(config.n_layers):
            is_attn = (i % 2 == 1)  # 1:1 alternating SSM and Attention
            layers.append(HybridBlock(config, is_attention_layer=is_attn))
        self.layers = nn.ModuleList(layers)

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
        caches: Optional[list[Optional[Tuple[Tensor, Tensor]]]] = None,
    ) -> Tuple[Tensor, list[Optional[Tuple[Tensor, Tensor]]]]:
        B, L = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self.rope(L, x.device)

        new_caches = []
        for i, layer in enumerate(self.layers):
            cache_i = caches[i] if caches is not None else None
            x, new_c = layer(x, cos, sin, layer_cache=cache_i)
            new_caches.append(new_c)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_caches

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
