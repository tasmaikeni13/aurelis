"""Native JAX / XLA / HLO implementation of AURELIS for Google Cloud TPU v4 Pod (16 v4 TPUs)."""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from .config import LMConfig
from .tpu_kernels import jax_fused_residual_gate, jax_rmsnorm, jax_swiglu


class JaxAurelisDecodeCache(NamedTuple):
    """Constant O(1) state per layer during autoregressive inference on Cloud TPU v4."""
    precision: jax.Array   # [B, H, D_k, D_k]
    cross: jax.Array       # [B, H, D_v, D_k]
    buffer_k: jax.Array    # [B, H, window, D_k]
    buffer_v: jax.Array    # [B, H, window, D_v]
    buffer_b: jax.Array    # [B, H, window]
    count: int = 0


@partial(jax.jit, static_argnames=("window", "prior"))
def jax_aurelis_attention_sequence(
    queries: jax.Array,       # [B, H, L, D_k]
    keys: jax.Array,          # [B, H, L, D_k]
    values: jax.Array,        # [B, H, L, D_v]
    evidence: jax.Array,      # [B, H, L]
    temperature: jax.Array,   # [H]
    responsibility: Optional[jax.Array] = None, # [B, H, L] or None
    window: int = 128,
    prior: float = 1.0,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Execute full sequence AURELIS attention compiled directly to XLA/HLO for TPU v4."""
    B, H, L, D_k = keys.shape
    D_v = values.shape[-1]

    # 1. Prefix cumulative state for the remote Bayesian regression
    # outer_p: [B, H, L, D_k, D_k]
    outer_p = jnp.einsum("bhli,bhlj,bhl->bhlij", keys, keys, evidence)
    outer_c = jnp.einsum("bhlv,bhli,bhl->bhlvi", values, keys, evidence)

    prefix_p = jnp.cumsum(outer_p, axis=2)
    prefix_c = jnp.cumsum(outer_c, axis=2)

    # Delayed handoff
    if window >= L:
        remote_p = jnp.zeros_like(prefix_p)
        remote_c = jnp.zeros_like(prefix_c)
    else:
        pad_p = jnp.zeros((B, H, window, D_k, D_k), dtype=prefix_p.dtype)
        pad_c = jnp.zeros((B, H, window, D_v, D_k), dtype=prefix_c.dtype)
        remote_p = jnp.concatenate([pad_p, prefix_p[:, :, :-window]], axis=2)
        remote_c = jnp.concatenate([pad_c, prefix_c[:, :, :-window]], axis=2)

    eye = jnp.eye(D_k, dtype=keys.dtype).reshape(1, 1, 1, D_k, D_k)
    precision = remote_p + prior * eye
    cross = remote_c

    # 2. Local causal softmax attention over the sliding window
    temp = jnp.exp(temperature).reshape(1, H, 1, 1)
    scores = jnp.einsum("bhid,bhjd->bhij", queries, keys) * temp

    pos = jnp.arange(L)
    q_pos = pos[:, None]
    k_pos = pos[None, :]
    causal_window_mask = (k_pos <= q_pos) & (k_pos > q_pos - window)
    scores = jnp.where(causal_window_mask.reshape(1, 1, L, L), scores, -1e9)
    attn_weights = jax.nn.softmax(scores, axis=-1)

    kbar = jnp.einsum("bhij,bhjd->bhid", attn_weights, keys)
    vbar = jnp.einsum("bhij,bhjv->bhiv", attn_weights, values)
    h = jnp.sum(jnp.square(attn_weights) / jnp.expand_dims(evidence, -2), axis=-1)

    # 3. Remote solve using Cholesky factor / solve
    rhs = jnp.stack((queries, kbar), axis=-1)  # [B, H, L, D_k, 2]
    # Solve precision * P^{-1} [queries, kbar]
    solved = jax.scipy.linalg.solve(precision, rhs, assume_a="pos")
    p_q = solved[..., 0]
    p_k = solved[..., 1]

    remote = jnp.einsum("bhlvi,bhli->bhlv", cross, p_q)
    mapped_kbar = jnp.einsum("bhlvi,bhli->bhlv", cross, p_k)

    # 4. Uncertainty Bayes gate g_B and episodic override g_E
    denom = jnp.maximum(h + jnp.sum(kbar * p_k, axis=-1), 1e-6)
    g_raw = jnp.sum(queries * p_k, axis=-1) / denom
    g_B = jnp.clip(g_raw, 0.0, 1.0)

    if responsibility is not None:
        g_E = jnp.maximum(g_B, responsibility)
        g = g_E
    else:
        g = g_B

    # 5. Fused residual gate
    y = jax_fused_residual_gate(remote, vbar, mapped_kbar, g)
    # Transpose to [B, L, H, D_v] and flatten heads: [B, L, H * D_v]
    out = jnp.reshape(jnp.swapaxes(y, 1, 2), (B, L, H * D_v))
    return out, precision[:, :, -1, :, :], cross[:, :, -1, :, :]
