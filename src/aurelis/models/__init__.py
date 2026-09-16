"""Aurelis language model candidate architectures and configurations targeting Cloud TPU v4."""

from __future__ import annotations

from .aurelis_lm import AurelisDecodeCache, AurelisLM
from .config import LMConfig, get_125m_config, get_350m_config
from .hybrid_ssm import HybridSSMLM, JambaHybridLM
from .jax_aurelis import JaxAurelisDecodeCache, jax_aurelis_attention_sequence
from .tpu_kernels import (
    get_tpu_pod_info,
    init_tpu_pod,
    jax_fused_residual_gate,
    jax_recurrent_scan,
    jax_rmsnorm,
    jax_swiglu,
    tpu_fused_residual_gate,
    tpu_recurrent_scan,
    tpu_rmsnorm,
    tpu_swiglu,
)
from .transformer import CausalSelfAttention, RMSNorm, RotaryEmbedding, SwiGLUMLP, TransformerLM

__all__ = [
    "AurelisDecodeCache",
    "AurelisLM",
    "CausalSelfAttention",
    "HybridSSMLM",
    "JambaHybridLM",
    "JaxAurelisDecodeCache",
    "LMConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLUMLP",
    "TransformerLM",
    "get_125m_config",
    "get_350m_config",
    "get_tpu_pod_info",
    "init_tpu_pod",
    "jax_aurelis_attention_sequence",
    "jax_fused_residual_gate",
    "jax_recurrent_scan",
    "jax_rmsnorm",
    "jax_swiglu",
    "tpu_fused_residual_gate",
    "tpu_recurrent_scan",
    "tpu_rmsnorm",
    "tpu_swiglu",
]
