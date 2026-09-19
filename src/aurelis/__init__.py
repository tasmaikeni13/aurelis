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
from .certificate import (
    compute_numerical_allowance,
    compute_outward_page_envelope,
    compute_outward_score_interval,
    evaluate_certified_bound,
    gamma,
    get_unit_roundoff,
    validate_intervals,
    validate_page_summary,
    validate_unread_cover,
)
from .policy import (
    CandidateState,
    RetrievalPolicy,
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
    "CandidateState",
    "CertificateBound",
    "DeltaState",
    "MemoryProfile",
    "PageDescriptor",
    "ReadResult",
    "ReadStatus",
    "RetrievalPolicy",
    "ScalarOracle",
    "ScalarStreamingOracle",
    "StateSnapshot",
    "TensorOracle",
    "TensorStreamingOracle",
    "archive_reference_read",
    "bounded_read",
    "completed_read",
    "compute_numerical_allowance",
    "compute_outward_page_envelope",
    "compute_outward_score_interval",
    "compute_page_envelope",
    "consume",
    "create_snapshot",
    "evaluate_certified_bound",
    "full_history_softmax",
    "gamma",
    "gated_delta_update",
    "get_unit_roundoff",
    "initial_state",
    "local_attention",
    "occurrence_partition",
    "pointwise_envelope",
    "read",
    "residual_certificate",
    "restore_from_snapshot",
    "validate_intervals",
    "validate_page_summary",
    "validate_unread_cover",
]
