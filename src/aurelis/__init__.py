"""AURELIS: Solve-Free Recurrent Memory with Residual-Certified Retrieval."""

from .functional import (
    bounded_read,
    completed_read,
    gated_delta_update,
    local_attention,
    pointwise_envelope,
    residual_certificate,
)
from .streaming import (
    consume,
    initial_state,
    occurrence_partition,
    read,
)
from .types import (
    CertificateBound,
    DeltaState,
    PageDescriptor,
    ReadResult,
    ReadStatus,
)
from .oracles import (
    ScalarOracle,
    TensorOracle,
    ScalarStreamingOracle,
    TensorStreamingOracle,
)


__all__ = [
    "CertificateBound",
    "DeltaState",
    "PageDescriptor",
    "ReadResult",
    "ReadStatus",
    "ScalarOracle",
    "TensorOracle",
    "ScalarStreamingOracle",
    "TensorStreamingOracle",
    "bounded_read",
    "completed_read",
    "consume",
    "gated_delta_update",
    "initial_state",
    "local_attention",
    "occurrence_partition",
    "pointwise_envelope",
    "read",
    "residual_certificate",
]
