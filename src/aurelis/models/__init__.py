"""Aurelis language model candidate architectures and configurations."""

from __future__ import annotations

from .aurelis_lm import AurelisDecodeCache, AurelisLM
from .config import LMConfig, get_125m_config, get_350m_config
from .hip_kernels import (
    hip_fused_residual_gate,
    hip_recurrent_scan,
    hip_rmsnorm,
    hip_swiglu,
)
from .hybrid_ssm import HybridSSMLM, JambaHybridLM
from .transformer import CausalSelfAttention, RMSNorm, RotaryEmbedding, SwiGLUMLP, TransformerLM

__all__ = [
    "AurelisDecodeCache",
    "AurelisLM",
    "CausalSelfAttention",
    "HybridSSMLM",
    "JambaHybridLM",
    "LMConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLUMLP",
    "TransformerLM",
    "get_125m_config",
    "get_350m_config",
    "hip_fused_residual_gate",
    "hip_recurrent_scan",
    "hip_rmsnorm",
    "hip_swiglu",
]

