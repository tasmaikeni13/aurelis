"""Reference CSM memory, baselines, synthetic data, and chained reads."""

from .memory import (
    CSMState,
    FP64GaussMarkovMemory,
    GaussMarkovMemory,
    direct_inverse_oracle,
    recompute_state,
)
from .baselines import (
    estimated_flops_per_query,
    explicit_pair_state,
    explicit_pair_state_bytes,
    hebbian_read_many,
    hebbian_state,
    hebbian_state_bytes,
    least_squares_read_many,
    least_squares_state,
    linear_attention_read_many,
    linear_attention_state,
    linear_attention_state_bytes,
    maximum_pairs_for_budget,
    softmax_read_many,
)
from .synthetic import KEY_REGIMES, VALUE_REGIMES, make_keys, make_values
from .multihop import (
    ChainedReads,
    chase_indices,
    csm_chained_reads,
    nearest_code,
    prepared_read_operator,
    softmax_chained_reads,
)

__all__ = [
    "CSMState",
    "FP64GaussMarkovMemory",
    "GaussMarkovMemory",
    "direct_inverse_oracle",
    "recompute_state",
    "KEY_REGIMES",
    "VALUE_REGIMES",
    "make_keys",
    "make_values",
    "estimated_flops_per_query",
    "explicit_pair_state",
    "explicit_pair_state_bytes",
    "hebbian_read_many",
    "hebbian_state",
    "hebbian_state_bytes",
    "least_squares_read_many",
    "least_squares_state",
    "linear_attention_read_many",
    "linear_attention_state",
    "linear_attention_state_bytes",
    "maximum_pairs_for_budget",
    "softmax_read_many",
    "ChainedReads",
    "chase_indices",
    "csm_chained_reads",
    "nearest_code",
    "prepared_read_operator",
    "softmax_chained_reads",
]
