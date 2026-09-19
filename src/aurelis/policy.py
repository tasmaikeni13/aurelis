"""Retrieval policy, priority heuristic, refinement, and budget accounting for AURELIS (Paper §6.2, Phase 3).

Implements:
1. Flat unread page scan with independent summary containment & unread partition validation.
2. Common shifted exponentials for exact selected sums and unread envelopes.
3. Paper §6.2 priority heuristic: (b_j + eta_j * ||r - y_hat||) / estimated_service_cost_j.
4. Non-monotone bound candidate tracking: always preserves the best valid candidate.
5. Distinct lifecycle statuses: certified, full_read, budget_exhausted, invalid_interval, invalid_state, archive_unavailable, archive_error.
6. Full accounting: summary visits, pages fetched, bytes transferred, selection ops.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence
import torch
from torch import Tensor

from .certificate import (
    compute_outward_page_envelope,
    compute_outward_score_interval,
    evaluate_certified_bound,
    get_unit_roundoff,
    validate_page_summary,
    validate_unread_cover,
)
from .functional import completed_read, full_history_softmax, local_attention
from .types import ArchiveEntry, CertificateBound, PageDescriptor, ReadResult, ReadStatus


@dataclass(frozen=True)
class CandidateState:
    """Snapshot of a valid intermediate retrieval candidate."""

    output: Tensor
    certificate: CertificateBound
    pages_read: int
    bytes_read: int
    total_bound: float
    step_index: int


class RetrievalPolicy:
    """Retrieval policy executing certified archive search with non-monotone candidate tracking."""

    def __init__(
        self,
        cost_fn: Optional[Callable[[PageDescriptor], float]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.cost_fn = cost_fn or (lambda p: max(1.0, float(p.count)))
        self.timeout_seconds = timeout_seconds

    def execute_retrieval(
        self,
        query: Tensor,
        recent_keys: Tensor,
        recent_values: Tensor,
        S: Tensor,
        archive_entries: list[ArchiveEntry],
        page_descriptors: list[PageDescriptor],
        get_page_entries_fn: Callable[[int], list[ArchiveEntry]],
        causal_position: Optional[int] = None,
        kappa: float = 1.0,
        epsilon: float = 1e-3,
        max_pages: Optional[int] = None,
        max_bytes: Optional[int] = None,
        read_all_pages: bool = False,
        dtype: Optional[torch.dtype] = None,
        inject_corrupt_bounds: bool = False,
        inject_missing_page: bool = False,
        inject_unsound_scale: float = 1.0,
        simulate_timeout: bool = False,
    ) -> ReadResult:
        """Execute deterministic certified retrieval according to Phase 3 contract."""
        start_time = time.monotonic()
        dt = dtype or query.dtype
        d_v = S.shape[0]
        d_k = query.shape[-1]

        q = query.to(dtype=dt)
        rk = recent_keys.to(dtype=dt)
        rv = recent_values.to(dtype=dt)
        S_dt = S.to(dtype=dt)

        summary_visits = 0
        selection_cost_ops = 0
        pages_read = 0
        bytes_read = 0

        # 1. Empty history check
        if rk.shape[0] == 0:
            return ReadResult(
                output=torch.zeros(d_v, dtype=dt, device=query.device),
                mode="archive",
                status="full_read",
                certificate=None,
                pages_read=0,
                bytes_read=0,
                details={"summary_visits": 0, "selection_cost_ops": 0},
            )

        # 2. Compute prior r(q) = vbar_L + S_t(q - kbar_L) (Eq. 3)
        # S is strictly read, never mutated!
        _, kbar, vbar = local_attention(rk, rv, q, kappa=kappa)
        diff = q - kbar
        prior = vbar + torch.einsum("vk,k->v", S_dt, diff)

        # 3. Handle empty remote archive (all history is in the recent window)
        if not page_descriptors:
            exact_out = full_history_softmax(rk, rv, q, kappa=kappa)
            return ReadResult(
                output=exact_out,
                mode="archive",
                status="full_read",
                certificate=None,
                pages_read=0,
                bytes_read=0,
                details={
                    "summary_visits": 0,
                    "selection_cost_ops": 0,
                    "approximation_bound": 0.0,
                    "delta_num": 0.0,
                },
            )

        # 4. Validate summary containment and disjoint cover
        all_remote_ids = [e.occurrence_id for e in archive_entries]
        cover_valid, cover_err = validate_unread_cover(page_descriptors, all_remote_ids)
        if not cover_valid:
            return ReadResult(
                output=torch.zeros(d_v, dtype=dt, device=query.device),
                mode="archive",
                status="invalid_state",
                certificate=None,
                pages_read=0,
                bytes_read=0,
                details={"error": cover_err},
            )

        # Check each page summary against actual entries
        for p in page_descriptors:
            summary_visits += 1
            p_entries = get_page_entries_fn(p.page_id)
            sum_valid, sum_err = validate_page_summary(p, p_entries, dtype=dt)
            if not sum_valid:
                return ReadResult(
                    output=torch.zeros(d_v, dtype=dt, device=query.device),
                    mode="archive",
                    status="invalid_interval",
                    certificate=None,
                    pages_read=0,
                    bytes_read=0,
                    details={"error": f"Summary validation failed: {sum_err}"},
                )

        # 5. Common log-sum-exp shift
        recent_scores = kappa * torch.einsum("d,kd->k", q, rk)
        max_recent_s = float(torch.max(recent_scores).item()) if len(recent_scores) > 0 else -1e9

        # Preliminary scan for max upper score across unread pages (linear algebra only, no exp!)
        max_u = -1e9
        for p in page_descriptors:
            _, u_out, _ = compute_outward_score_interval(
                query=q,
                key_min=p.key_min,
                key_max=p.key_max,
                kappa=kappa,
                dtype=dt,
            )
            selection_cost_ops += d_k * 3 + 10
            max_u = max(max_u, u_out)

        m_shift = max(max_recent_s, max_u)

        # 6. Apply common shift to recent and unread mass
        shifted_recent_scores = recent_scores - m_shift
        recent_weights = torch.exp(shifted_recent_scores)
        Z_A = float(torch.sum(recent_weights).item())
        N_A = torch.sum(recent_weights[:, None] * rv, dim=0)

        # Shift unread envelopes
        page_envelopes = []
        L_O = 0.0
        U_O = 0.0
        B_O = 0.0

        for p in page_descriptors:
            env = compute_outward_page_envelope(
                page=p,
                query=q,
                prior=prior,
                kappa=kappa,
                m_shift=m_shift,
                dtype=dt,
            )
            # Support deliberate injection of unsound bounds for testing gate
            if inject_corrupt_bounds:
                # Invert interval: set L_j strictly greater than U_j
                env["Lj"], env["Uj"] = env["Uj"] + 10.0, env["Lj"]
            elif inject_unsound_scale != 1.0:
                env["bj"] = env["bj"] * inject_unsound_scale

            page_envelopes.append(env)
            L_O += env["Lj"]
            U_O += env["Uj"]
            B_O += env["bj"]

        Z_hat_O = (L_O + U_O) / 2.0

        # Eq. (8) Normalized midpoint completion
        y_hat_A = completed_read(Z_A, Z_hat_O, N_A, prior)

        # Eq. (10) Deterministic certificate with arithmetic error allowance
        is_valid, cert, reason = evaluate_certified_bound(
            selected_mass=Z_A,
            unread_lower=L_O,
            unread_upper=U_O,
            residual_bound=B_O,
            prior=prior,
            completed=y_hat_A,
            N_A=N_A,
            count_A=rk.shape[0],
            count_O=sum(p.count for p in page_descriptors),
            d_k=d_k,
            d_v=d_v,
            dtype=dt,
        )

        if not is_valid:
            return ReadResult(
                output=y_hat_A,
                mode="archive",
                status="invalid_interval",
                certificate=cert,
                pages_read=0,
                bytes_read=0,
                details={"error": reason},
            )

        # Initialize best candidate tracker
        best_candidate: CandidateState = CandidateState(
            output=y_hat_A.clone(),
            certificate=cert,
            pages_read=0,
            bytes_read=0,
            total_bound=cert.total_bound,
            step_index=0,
        )

        # 7. Check if stop condition met immediately without reading any remote page
        if not read_all_pages and cert.total_bound <= epsilon:
            return ReadResult(
                output=y_hat_A,
                mode="archive",
                status="certified",
                certificate=cert,
                pages_read=0,
                bytes_read=0,
                details={
                    "summary_visits": summary_visits,
                    "selection_cost_ops": selection_cost_ops,
                    "approximation_bound": cert.approximation_bound,
                    "delta_num": cert.delta_num,
                    "total_bound": cert.total_bound,
                    "best_candidate_retained": False,
                },
            )

        # 8. Refinement loop with Paper §6.2 priority heuristic
        unread_envelopes = list(page_envelopes)
        step_index = 0

        while unread_envelopes:
            step_index += 1

            # Check timeout
            elapsed = time.monotonic() - start_time
            if simulate_timeout or (self.timeout_seconds is not None and elapsed > self.timeout_seconds):
                return ReadResult(
                    output=best_candidate.output,
                    mode="archive",
                    status="archive_unavailable",
                    certificate=best_candidate.certificate,
                    pages_read=pages_read,
                    bytes_read=bytes_read,
                    details={
                        "error": "I/O timeout during archive retrieval",
                        "elapsed_seconds": elapsed,
                        "best_candidate_bound": best_candidate.total_bound,
                        "best_candidate_retained": True,
                    },
                )

            # Check page budget before fetching
            if max_pages is not None and pages_read >= max_pages:
                return ReadResult(
                    output=best_candidate.output,
                    mode="archive",
                    status="budget_exhausted",
                    certificate=best_candidate.certificate,
                    pages_read=best_candidate.pages_read,
                    bytes_read=best_candidate.bytes_read,
                    details={
                        "reason": f"Page budget exhausted ({max_pages} pages)",
                        "best_candidate_step": best_candidate.step_index,
                        "best_candidate_retained": True,
                        "summary_visits": summary_visits,
                        "selection_cost_ops": selection_cost_ops,
                    },
                )

            # Paper §6.2 priority heuristic:
            # priority_j = [b_j + ((U_j - L_j)/2) * ||r - y_hat_A||] / cost_j
            prior_diff = float(torch.linalg.vector_norm(prior - y_hat_A).item())

            def eval_priority(env: dict[str, Any]) -> float:
                eta_j = (env["Uj"] - env["Lj"]) / 2.0
                cost = self.cost_fn(env["page"])
                return (env["bj"] + eta_j * prior_diff) / cost

            unread_envelopes.sort(key=eval_priority, reverse=True)
            selection_cost_ops += len(unread_envelopes) * 10 + int(math.log2(max(2, len(unread_envelopes))))

            target_env = unread_envelopes.pop(0)
            target_page = target_env["page"]

            # Simulated missing page failure
            if inject_missing_page and target_page.page_id == page_descriptors[-1].page_id:
                return ReadResult(
                    output=best_candidate.output,
                    mode="archive",
                    status="archive_error",
                    certificate=best_candidate.certificate,
                    pages_read=pages_read,
                    bytes_read=bytes_read,
                    details={"error": f"Page {target_page.page_id} missing from storage"},
                )

            # Fetch page entries (reading NEVER writes S)
            summary_visits += 1
            p_entries = get_page_entries_fn(target_page.page_id)
            if not p_entries:
                continue

            # Check byte budget
            page_bytes = len(p_entries) * (d_k + d_v) * target_page.key_min.element_size()
            if max_bytes is not None and (bytes_read + page_bytes) > max_bytes:
                return ReadResult(
                    output=best_candidate.output,
                    mode="archive",
                    status="budget_exhausted",
                    certificate=best_candidate.certificate,
                    pages_read=best_candidate.pages_read,
                    bytes_read=best_candidate.bytes_read,
                    details={
                        "reason": f"Byte budget exhausted ({max_bytes} bytes limit, needed {bytes_read + page_bytes})",
                        "best_candidate_step": best_candidate.step_index,
                        "best_candidate_retained": True,
                        "summary_visits": summary_visits,
                        "selection_cost_ops": selection_cost_ops,
                    },
                )

            p_keys = torch.stack([e.key.to(dtype=dt) for e in p_entries])
            p_vals = torch.stack([e.value.to(dtype=dt) for e in p_entries])

            p_scores = kappa * torch.einsum("d,kd->k", q, p_keys)
            shifted_p_scores = p_scores - m_shift
            p_exp = torch.exp(shifted_p_scores)

            # Update selected sums
            Z_A += float(torch.sum(p_exp).item())
            N_A = N_A + torch.sum(p_exp[:, None] * p_vals, dim=0)

            # Recompute unread mass and residual bounds from remaining unread envelopes
            L_O = sum(env["Lj"] for env in unread_envelopes)
            U_O = max(L_O, sum(env["Uj"] for env in unread_envelopes))
            B_O = sum(env["bj"] for env in unread_envelopes)
            Z_hat_O = (L_O + U_O) / 2.0

            pages_read += 1
            bytes_read += page_bytes

            # Check if all pages have been fetched (exact full softmax recovery!)
            if not unread_envelopes or (L_O == 0.0 and U_O == 0.0 and B_O == 0.0):
                exact_out = N_A / Z_A
                return ReadResult(
                    output=exact_out,
                    mode="archive",
                    status="full_read",
                    certificate=None,
                    pages_read=pages_read,
                    bytes_read=bytes_read,
                    details={
                        "summary_visits": summary_visits,
                        "selection_cost_ops": selection_cost_ops,
                        "full_read_recovered": True,
                        "steps": step_index,
                    },
                )

            # Recompute candidate completion and certificate
            y_hat_A = completed_read(Z_A, Z_hat_O, N_A, prior)
            is_valid, cert, reason = evaluate_certified_bound(
                selected_mass=Z_A,
                unread_lower=L_O,
                unread_upper=U_O,
                residual_bound=B_O,
                prior=prior,
                completed=y_hat_A,
                N_A=N_A,
                count_A=rk.shape[0] + pages_read * target_page.count,
                count_O=sum(env["count"] for env in unread_envelopes),
                d_k=d_k,
                d_v=d_v,
                dtype=dt,
            )

            if not is_valid:
                return ReadResult(
                    output=y_hat_A,
                    mode="archive",
                    status="invalid_interval",
                    certificate=cert,
                    pages_read=pages_read,
                    bytes_read=bytes_read,
                    details={"error": reason},
                )

            # Non-monotone tracking: retain best valid candidate if current is worse
            if cert.total_bound < best_candidate.total_bound:
                best_candidate = CandidateState(
                    output=y_hat_A.clone(),
                    certificate=cert,
                    pages_read=pages_read,
                    bytes_read=bytes_read,
                    total_bound=cert.total_bound,
                    step_index=step_index,
                )

            # Check if certified stopping met
            if not read_all_pages and cert.total_bound <= epsilon:
                return ReadResult(
                    output=y_hat_A,
                    mode="archive",
                    status="certified",
                    certificate=cert,
                    pages_read=pages_read,
                    bytes_read=bytes_read,
                    details={
                        "summary_visits": summary_visits,
                        "selection_cost_ops": selection_cost_ops,
                        "approximation_bound": cert.approximation_bound,
                        "delta_num": cert.delta_num,
                        "total_bound": cert.total_bound,
                        "steps": step_index,
                        "best_candidate_retained": (best_candidate.step_index != step_index),
                    },
                )

        # Unread pages exhausted: full read
        return ReadResult(
            output=N_A / Z_A,
            mode="archive",
            status="full_read",
            certificate=None,
            pages_read=pages_read,
            bytes_read=bytes_read,
            details={
                "summary_visits": summary_visits,
                "selection_cost_ops": selection_cost_ops,
                "steps": step_index,
            },
        )
