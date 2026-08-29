"""AURELIS transparent reference and ROCm research substrate."""

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

__all__ = [
    "AurelisProjectionBlock",
    "ReadDiagnostics",
    "ReadOutput",
    "SequenceOutput",
    "StreamingState",
    "aurelis_read",
    "chronological_cache",
    "consume",
    "explicit_inverse_read",
    "historical_oracle",
    "initial_state",
    "local_causal_softmax",
    "occurrence_partition",
    "prepared_aurelis_head",
    "read",
    "vectorized_reference",
]
