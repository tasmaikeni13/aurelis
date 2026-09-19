"""AURELIS: Solve-Free Recurrent Memory with Residual-Certified Retrieval."""

from .archive import (
    Archive,
    archive_reference_read,
)
from .functional import (
    bounded_read,
    completed_read,
    compute_page_envelope,
    full_history_softmax,
    gated_delta_update,
    local_attention,
    pointwise_envelope,
    residual_certificate,
)
from .oracles import (
    ScalarOracle,
    ScalarStreamingOracle,
    TensorOracle,
    TensorStreamingOracle,
)
from .session import AurelisSession
from .streaming import (
    consume,
    create_snapshot,
    initial_state,
    occurrence_partition,
    read,
    restore_from_snapshot,
)
from .types import (
    ArchiveEntry,
    CertificateBound,
    DeltaState,
    MemoryProfile,
    PageDescriptor,
    ReadResult,
    ReadStatus,
    StateSnapshot,
)

__all__ = [
    "Archive",
    "ArchiveEntry",
    "AurelisSession",
    "CertificateBound",
    "DeltaState",
    "MemoryProfile",
    "PageDescriptor",
    "ReadResult",
    "ReadStatus",
    "ScalarOracle",
    "ScalarStreamingOracle",
    "StateSnapshot",
    "TensorOracle",
    "TensorStreamingOracle",
    "archive_reference_read",
    "bounded_read",
    "completed_read",
    "compute_page_envelope",
    "consume",
    "create_snapshot",
    "full_history_softmax",
    "gated_delta_update",
    "initial_state",
    "local_attention",
    "occurrence_partition",
    "pointwise_envelope",
    "read",
    "residual_certificate",
    "restore_from_snapshot",
]
