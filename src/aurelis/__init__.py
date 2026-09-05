"""AURELIS transparent reference and ROCm research substrate."""

from .baselines import (
    baseline_flops,
    baseline_parameter_count,
    baseline_state_bytes,
    cumulative_least_squares_mesa,
    delta_rule_memory,
    full_residual_fixed_gate,
    global_linear_attention,
    independent_inverse_variance_fusion,
    learned_local_remote_concat,
    learned_local_remote_sum,
    local_softmax_attention,
    native_hybrid_attention,
    remote_bayes_ridge,
)
from .functional import (
    aurelis_read,
    explicit_inverse_read,
    local_causal_softmax,
    prepared_aurelis_head,
)
from .nn import AurelisProjectionBlock
from .oracle import historical_oracle
from .streaming import (
    chronological_cache,
    consume,
    initial_state,
    occurrence_partition,
    read,
)
from .training import vectorized_reference
from .types import ReadDiagnostics, ReadOutput, SequenceOutput, StreamingState

from .models import (
    AurelisDecodeCache,
    AurelisLM,
    HybridSSMLM,
    LMConfig,
    TransformerLM,
    get_125m_config,
    get_350m_config,
    hip_fused_residual_gate,
    hip_recurrent_scan,
)

__all__ = [
    "AurelisDecodeCache",
    "AurelisLM",
    "AurelisProjectionBlock",
    "HybridSSMLM",
    "LMConfig",
    "ReadDiagnostics",
    "ReadOutput",
    "SequenceOutput",
    "StreamingState",
    "TransformerLM",
    "aurelis_read",
    "baseline_flops",
    "baseline_parameter_count",
    "baseline_state_bytes",
    "chronological_cache",
    "consume",
    "cumulative_least_squares_mesa",
    "delta_rule_memory",
    "explicit_inverse_read",
    "full_residual_fixed_gate",
    "get_125m_config",
    "get_350m_config",
    "global_linear_attention",
    "hip_fused_residual_gate",
    "hip_recurrent_scan",
    "historical_oracle",
    "independent_inverse_variance_fusion",
    "initial_state",
    "learned_local_remote_concat",
    "learned_local_remote_sum",
    "local_causal_softmax",
    "local_softmax_attention",
    "native_hybrid_attention",
    "occurrence_partition",
    "prepared_aurelis_head",
    "read",
    "remote_bayes_ridge",
    "vectorized_reference",
]

