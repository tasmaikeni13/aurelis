"""Model configurations for 125M and 350M language model candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class LMConfig:
    """Unified configuration for language model architectures."""

    model_type: Literal["transformer", "ssm_hybrid", "aurelis_b", "aurelis_e"] = "aurelis_e"
    vocab_size: int = 50257
    max_position_embeddings: int = 4096
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_key: int = 64
    d_value: int = 64
    d_ffn: int = 2048
    window_size: int = 128
    prior: float = 1.0
    dropout: float = 0.0
    bias: bool = False
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    ssm_state_dim: int = 16  # For SSM block in hybrid
    ssm_conv_kernel: int = 4  # 1D causal convolution in SSM block
    use_hip_kernels: bool = True  # Utilize accelerated HIP kernels when on ROCm

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def get_125m_config(
    model_type: Literal["transformer", "ssm_hybrid", "aurelis_b", "aurelis_e"] = "aurelis_e",
    **overrides,
) -> LMConfig:
    """Standard 125M parameter scale configuration (d_model=768, heads=12, layers=12)."""
    base_kwargs = dict(
        model_type=model_type,
        vocab_size=50257,
        max_position_embeddings=4096,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_key=64,
        d_value=64,
        d_ffn=2048,
        window_size=128,
        prior=1.0,
        tie_word_embeddings=True,
    )
    base_kwargs.update(overrides)
    return LMConfig(**base_kwargs)


def get_350m_config(
    model_type: Literal["transformer", "ssm_hybrid", "aurelis_b", "aurelis_e"] = "aurelis_e",
    **overrides,
) -> LMConfig:
    """Standard 350M parameter scale configuration (d_model=1024, heads=16, layers=24)."""
    base_kwargs = dict(
        model_type=model_type,
        vocab_size=50257,
        max_position_embeddings=4096,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        d_key=64,
        d_value=64,
        d_ffn=2730,
        window_size=128,
        prior=1.0,
        tie_word_embeddings=True,
    )
    base_kwargs.update(overrides)
    return LMConfig(**base_kwargs)
