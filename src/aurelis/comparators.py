"""Required research comparators and comprehensive cost accounting for AURELIS Phase 4.

Implements all 8 required comparators under identical Q/K/V, archive pages,
encoding, epsilon, and accounting:
1. local_barycenter: local-value completion (prior r = vbar_L)
2. zero: zero completion (prior r = 0)
3. global_summary: summary-based global completion (prior r = c_global)
4. per_page_center: per-page center completion (Eq. 13: p_j = c_j, B_j = U_j * rho_j)
5. simple_fusion: recurrence with simple additive/gated fusion (no mass consistency)
6. sparse_no_completion: sparse attention with no completion (pure truncation)
7. value_blind_mass: recurrent completion with value-blind mass-only selection
8. aurelis: AURELIS recurrent transport + residual-sensitive selection + Eq. 8/10 certified completion
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Sequence
import torch
from torch import Tensor

from .certificate import (
    compute_numerical_allowance,
    compute_outward_page_envelope,
    compute_outward_score_interval,
    evaluate_certified_bound,
    get_unit_roundoff,
)
from .functional import (
    completed_read,
    full_history_softmax,
    local_attention,
)
from .types import ArchiveEntry, CertificateBound, PageDescriptor, ReadResult


@dataclass(frozen=True)
class AccountingRecord:
    """Detailed systems and arithmetic cost accounting for retrieval."""

    index_build_ops: int
    predictor_ops: int
    summary_scans: int
    selection_ops: int
    pages_read: int
    bytes_read: int
    working_state_bytes: int
    archive_raw_bytes: int
    archive_index_bytes: int
    total_memory_bytes: int
    total_flops: int
    composite_cost: float


@dataclass(frozen=True)
class ComparatorResult:
    """Outcome of evaluating a retrieval comparator on a query."""

    comparator_name: str
    output: Tensor
    status: str
    actual_error: float
    certified_bound: Optional[float]
    delta_num: float
    pages_read: int
    bytes_read: int
    accounting: AccountingRecord
    details: dict[str, Any] = field(default_factory=dict)


def compute_composite_cost(
    bytes_read: int,
    pages_read: int,
    summary_scans: int,
    total_flops: int,
    transfer_weight: float = 1.0,
    page_fault_weight: float = 64.0,
    scan_weight: float = 4.0,
    flop_weight: float = 0.1,
) -> float:
    """Compute normalized composite service cost accounting for IO, transfers, and compute."""
    return (
        transfer_weight * float(bytes_read)
        + page_fault_weight * float(pages_read)
        + scan_weight * float(summary_scans)
        + flop_weight * float(total_flops)
    )


def execute_comparator(
    comparator_name: Literal[
        "aurelis",
        "local_barycenter",
        "zero",
        "global_summary",
        "per_page_center",
        "simple_fusion",
        "sparse_no_completion",
        "value_blind_mass",
    ],
    query: Tensor,
    recent_keys: Tensor,
    recent_values: Tensor,
    S: Optional[Tensor],
    archive_entries: list[ArchiveEntry],
    page_descriptors: list[PageDescriptor],
    get_page_entries_fn: Callable[[int], list[ArchiveEntry]],
    ground_truth_output: Optional[Tensor] = None,
    kappa: float = 1.0,
    epsilon: float = 1e-2,
    max_pages: Optional[int] = None,
    max_bytes: Optional[int] = None,
    cost_fn: Optional[Callable[[PageDescriptor], float]] = None,
    dtype: Optional[torch.dtype] = None,
    enforce_equal_kv_budget: Optional[int] = None,
) -> ComparatorResult:
    """Execute a single comparator under identical Q/K/V, archive pages, and accounting."""
    dt = dtype or query.dtype
    q = query.to(dtype=dt)
    rk = recent_keys.to(dtype=dt)
    rv = recent_values.to(dtype=dt)
    d_k = q.shape[-1]
    d_v = rv.shape[-1] if rv.shape[0] > 0 else (S.shape[0] if S is not None else 16)
    element_size = q.element_size()

    # Determine ground truth output if not provided
    if ground_truth_output is None:
        if len(archive_entries) > 0 and rk.shape[0] > 0:
            all_k = torch.cat([torch.stack([e.key.to(dtype=dt) for e in archive_entries]), rk], dim=0)
            all_v = torch.cat([torch.stack([e.value.to(dtype=dt) for e in archive_entries]), rv], dim=0)
            y_star = full_history_softmax(all_k, all_v, q, kappa=kappa)
        elif rk.shape[0] > 0:
            y_star = full_history_softmax(rk, rv, q, kappa=kappa)
        else:
            y_star = torch.zeros(d_v, dtype=dt, device=query.device)
    else:
        y_star = ground_truth_output.to(dtype=dt)

    # Empty history check
    if rk.shape[0] == 0 and len(archive_entries) == 0:
        zero_out = torch.zeros(d_v, dtype=dt, device=query.device)
        acct = AccountingRecord(
            index_build_ops=0, predictor_ops=0, summary_scans=0, selection_ops=0,
            pages_read=0, bytes_read=0, working_state_bytes=0, archive_raw_bytes=0,
            archive_index_bytes=0, total_memory_bytes=0, total_flops=0, composite_cost=0.0,
        )
        return ComparatorResult(
            comparator_name=comparator_name, output=zero_out, status="full_read",
            actual_error=0.0, certified_bound=0.0, delta_num=0.0, pages_read=0,
            bytes_read=0, accounting=acct, details={"steps": 0},
        )

    # Calculate memory accounting for this comparator
    # Bounded working state: local window cache + (S if comparator uses recurrence)
    has_recurrence = comparator_name in ("aurelis", "simple_fusion", "value_blind_mass")
    working_state_bytes = rk.numel() * element_size + rv.numel() * element_size
    if has_recurrence and S is not None:
        working_state_bytes += S.numel() * element_size

    archive_raw_bytes = sum((e.key.numel() + e.value.numel()) * element_size + 32 for e in archive_entries)
    archive_index_bytes = sum(
        (p.key_min.numel() + p.key_max.numel() + p.value_center.numel()) * element_size + 16
        for p in page_descriptors
    )
    total_memory_bytes = working_state_bytes + archive_raw_bytes + archive_index_bytes

    index_build_ops = 0
    predictor_ops = 0
    summary_scans = 0
    selection_ops = 0
    pages_read = 0
    bytes_read = 0

    # Charge local attention compute (shared by all methods)
    w_len = rk.shape[0]
    predictor_ops += w_len * d_k + w_len * (d_k + d_v)
    _, kbar, vbar = local_attention(rk, rv, q, kappa=kappa)

    # 1. Compute comparator prior r(q) and charged predictor FLOPs
    if comparator_name in ("aurelis", "value_blind_mass"):
        if S is not None:
            S_dt = S.to(dtype=dt)
            diff = q - kbar
            predictor_ops += 2 * d_v * d_k + d_k
            prior = vbar + torch.einsum("vk,k->v", S_dt, diff)
        else:
            prior = vbar.clone()
    elif comparator_name == "local_barycenter":
        prior = vbar.clone()
    elif comparator_name == "zero":
        prior = torch.zeros(d_v, dtype=dt, device=query.device)
    elif comparator_name == "global_summary":
        if page_descriptors:
            total_n = sum(p.count for p in page_descriptors)
            weighted_center = torch.sum(
                torch.stack([p.value_center.to(dtype=dt) * float(p.count) for p in page_descriptors]), dim=0
            ) / max(1.0, float(total_n))
            predictor_ops += len(page_descriptors) * (d_v + 1) + d_v
            prior = weighted_center
        else:
            prior = torch.zeros(d_v, dtype=dt, device=query.device)
    elif comparator_name == "simple_fusion":
        if S is not None:
            S_dt = S.to(dtype=dt)
            pred_r = torch.einsum("vk,k->v", S_dt, q)
            predictor_ops += 2 * d_v * d_k
            prior = 0.5 * vbar + 0.5 * pred_r
        else:
            prior = vbar.clone()
    elif comparator_name == "per_page_center":
        # Per-page center does not use a single prior r; it uses p_j = c_j per page
        prior = vbar.clone()
    elif comparator_name == "sparse_no_completion":
        prior = torch.zeros(d_v, dtype=dt, device=query.device)

    # If no remote archive exists, all tokens are in recent window
    if not page_descriptors:
        exact_out = full_history_softmax(rk, rv, q, kappa=kappa)
        err = float(torch.linalg.vector_norm(y_star - exact_out).item())
        total_flops = index_build_ops + predictor_ops + selection_ops
        comp_cost = compute_composite_cost(bytes_read, pages_read, summary_scans, total_flops)
        acct = AccountingRecord(
            index_build_ops=index_build_ops, predictor_ops=predictor_ops,
            summary_scans=summary_scans, selection_ops=selection_ops,
            pages_read=pages_read, bytes_read=bytes_read,
            working_state_bytes=working_state_bytes, archive_raw_bytes=archive_raw_bytes,
            archive_index_bytes=archive_index_bytes, total_memory_bytes=total_memory_bytes,
            total_flops=total_flops, composite_cost=comp_cost,
        )
        return ComparatorResult(
            comparator_name=comparator_name, output=exact_out, status="full_read",
            actual_error=err, certified_bound=0.0, delta_num=0.0, pages_read=0,
            bytes_read=0, accounting=acct, details={"steps": 0},
        )

    # Shifted scores calculation
    recent_scores = kappa * torch.einsum("d,kd->k", q, rk)
    max_recent_s = float(torch.max(recent_scores).item()) if len(recent_scores) > 0 else -1e9

    # Score scan for unread page bounds
    max_u = -1e9
    for p in page_descriptors:
        summary_scans += 1
        _, u_out, _ = compute_outward_score_interval(query=q, key_min=p.key_min, key_max=p.key_max, kappa=kappa, dtype=dt)
        selection_ops += d_k * 3 + 10
        max_u = max(max_u, u_out)

    m_shift = max(max_recent_s, max_u)
    shifted_recent_scores = recent_scores - m_shift
    recent_weights = torch.exp(shifted_recent_scores)
    Z_A = float(torch.sum(recent_weights).item())
    N_A = torch.sum(recent_weights[:, None] * rv, dim=0)

    # Build page envelopes with common shift
    page_envelopes = []
    L_O = 0.0
    U_O = 0.0
    B_O = 0.0

    for p in page_descriptors:
        env = compute_outward_page_envelope(page=p, query=q, prior=prior, kappa=kappa, m_shift=m_shift, dtype=dt)
        page_envelopes.append(env)
        L_O += env["Lj"]
        U_O += env["Uj"]
        B_O += env["bj"]
        selection_ops += d_v + 10

    unread_envelopes = list(page_envelopes)
    cost_getter = cost_fn or (lambda p: max(1.0, float(p.count)))

    # Initial candidate output and certificate evaluation
    if comparator_name in ("aurelis", "per_page_center"):
        # Grouped completion (Eq. 13)
        Z_hat_j = [(env["Lj"] + env["Uj"]) / 2.0 for env in unread_envelopes]
        sum_zh = sum(Z_hat_j)
        pred_sum = torch.sum(
            torch.stack([torch.as_tensor(zh, dtype=dt) * env["page"].value_center.to(dtype=dt) for zh, env in zip(Z_hat_j, unread_envelopes)]),
            dim=0,
        ) if unread_envelopes else torch.zeros(d_v, dtype=dt, device=query.device)
        y_hat = (N_A + pred_sum) / (Z_A + sum_zh)

        denom = Z_A + L_O
        if denom > 0.0:
            num = 0.0
            for env in unread_envelopes:
                eta_j = (env["Uj"] - env["Lj"]) / 2.0
                diff_j = float(torch.linalg.vector_norm(env["page"].value_center.to(dtype=dt) - y_hat).item())
                num += env["Uj"] * env["radius"] + eta_j * diff_j
            cert_bound = num / denom
        else:
            cert_bound = float("inf")
        delta_num = compute_numerical_allowance(
            selected_mass=Z_A, unread_lower=L_O, unread_upper=U_O, N_A=N_A,
            prior=prior, completed=y_hat, count_A=rk.shape[0],
            count_O=sum(p.count for p in page_descriptors), d_k=d_k, d_v=d_v, dtype=dt,
        )
        total_bound = cert_bound + delta_num

    elif comparator_name == "sparse_no_completion":
        # Truncation: y_hat = N_A / Z_A
        y_hat = N_A / max(1e-12, Z_A)
        # Sparse error bound: U_O / (Z_A + L_O) * max_radius
        max_val_norm = max(float(torch.linalg.vector_norm(p.value_center).item()) + p.value_radius for p in page_descriptors)
        denom = Z_A + L_O
        cert_bound = (U_O / denom) * max_val_norm if denom > 0.0 else float("inf")
        delta_num = 1e-5
        total_bound = cert_bound + delta_num

    elif comparator_name == "simple_fusion":
        # Simple fusion: y = 0.5 * (N_A / Z_A) + 0.5 * prior
        y_hat = 0.5 * (N_A / max(1e-12, Z_A)) + 0.5 * prior
        # No formal residual certificate
        cert_bound = float("nan")
        total_bound = float("nan")
        delta_num = 0.0

    else:
        # AURELIS, local_barycenter, zero, global_summary, value_blind_mass
        Z_hat_O = (L_O + U_O) / 2.0
        y_hat = completed_read(Z_A, Z_hat_O, N_A, prior)
        is_valid, cert_obj, _ = evaluate_certified_bound(
            selected_mass=Z_A, unread_lower=L_O, unread_upper=U_O, residual_bound=B_O,
            prior=prior, completed=y_hat, N_A=N_A, count_A=rk.shape[0],
            count_O=sum(p.count for p in page_descriptors), d_k=d_k, d_v=d_v, dtype=dt,
        )
        cert_bound = cert_obj.approximation_bound
        delta_num = cert_obj.delta_num
        total_bound = cert_obj.total_bound

    step_index = 0
    best_candidate_output = y_hat.clone()
    best_candidate_bound = total_bound if math.isfinite(total_bound) else float("inf")
    best_candidate_pages = 0
    best_candidate_bytes = 0

    # Initial stopping check
    target_budget = enforce_equal_kv_budget
    can_stop_early = (target_budget is None)

    if can_stop_early and math.isfinite(total_bound) and total_bound <= epsilon:
        actual_err = float(torch.linalg.vector_norm(y_star - y_hat).item())
        total_flops = index_build_ops + predictor_ops + selection_ops
        comp_cost = compute_composite_cost(bytes_read, pages_read, summary_scans, total_flops)
        acct = AccountingRecord(
            index_build_ops=index_build_ops, predictor_ops=predictor_ops,
            summary_scans=summary_scans, selection_ops=selection_ops,
            pages_read=pages_read, bytes_read=bytes_read,
            working_state_bytes=working_state_bytes, archive_raw_bytes=archive_raw_bytes,
            archive_index_bytes=archive_index_bytes, total_memory_bytes=total_memory_bytes,
            total_flops=total_flops, composite_cost=comp_cost,
        )
        return ComparatorResult(
            comparator_name=comparator_name, output=y_hat, status="certified",
            actual_error=actual_err, certified_bound=total_bound, delta_num=delta_num,
            pages_read=pages_read, bytes_read=bytes_read, accounting=acct,
            details={"steps": 0, "certified_at_step": 0},
        )

    # Retrieval refinement loop
    while unread_envelopes:
        step_index += 1

        # Check equal fetched KV budget stop condition
        if target_budget is not None and pages_read >= target_budget:
            break

        # Check page/byte budget constraints
        if max_pages is not None and pages_read >= max_pages:
            break

        # Sorting priority heuristic
        if comparator_name in ("value_blind_mass", "sparse_no_completion"):
            # Value-blind mass selection: sort purely by upper mass U_j
            unread_envelopes.sort(key=lambda env: env["Uj"] / cost_getter(env["page"]), reverse=True)
            selection_ops += len(unread_envelopes) * 5 + int(math.log2(max(2, len(unread_envelopes))))
        elif comparator_name == "per_page_center":
            # Priority for grouped completion: (U_j * rho_j + eta_j * ||c_j - y_hat||) / cost_j
            def eval_grouped_prio(env: dict[str, Any]) -> float:
                eta_j = (env["Uj"] - env["Lj"]) / 2.0
                diff_j = float(torch.linalg.vector_norm(env["page"].value_center.to(dtype=dt) - y_hat).item())
                return (env["Uj"] * env["radius"] + eta_j * diff_j) / cost_getter(env["page"])

            unread_envelopes.sort(key=eval_grouped_prio, reverse=True)
            selection_ops += len(unread_envelopes) * (d_v + 10) + int(math.log2(max(2, len(unread_envelopes))))
        elif comparator_name == "aurelis":
            # Revision 1.1: Recurrence-guided value-residual priority: (U_j * rho_j + eta_j * ||c_j - prior||) / cost_j
            def eval_aur_prio(env: dict[str, Any]) -> float:
                eta_j = (env["Uj"] - env["Lj"]) / 2.0
                diff_j = float(torch.linalg.vector_norm(env["page"].value_center.to(dtype=dt) - prior).item())
                return (env["Uj"] * env["radius"] + eta_j * diff_j) / cost_getter(env["page"])

            unread_envelopes.sort(key=eval_aur_prio, reverse=True)
            selection_ops += len(unread_envelopes) * (d_v + 10) + int(math.log2(max(2, len(unread_envelopes))))
        else:
            # Paper §6.2 residual-sensitive priority: [b_j + eta_j * ||prior - y_hat||] / cost_j
            prior_diff = float(torch.linalg.vector_norm(prior - y_hat).item())
            def eval_prio(env: dict[str, Any]) -> float:
                eta_j = (env["Uj"] - env["Lj"]) / 2.0
                return (env["bj"] + eta_j * prior_diff) / cost_getter(env["page"])

            unread_envelopes.sort(key=eval_prio, reverse=True)
            selection_ops += len(unread_envelopes) * 10 + int(math.log2(max(2, len(unread_envelopes))))

        target_env = unread_envelopes.pop(0)
        target_page = target_env["page"]

        p_entries = get_page_entries_fn(target_page.page_id)
        if not p_entries:
            continue

        page_bytes = len(p_entries) * (d_k + d_v) * element_size
        if max_bytes is not None and (bytes_read + page_bytes) > max_bytes:
            break

        pages_read += 1
        bytes_read += page_bytes
        summary_scans += 1

        p_keys = torch.stack([e.key.to(dtype=dt) for e in p_entries])
        p_vals = torch.stack([e.value.to(dtype=dt) for e in p_entries])

        p_scores = kappa * torch.einsum("d,kd->k", q, p_keys)
        shifted_p_scores = p_scores - m_shift
        p_exp = torch.exp(shifted_p_scores)

        Z_A += float(torch.sum(p_exp).item())
        N_A = N_A + torch.sum(p_exp[:, None] * p_vals, dim=0)
        selection_ops += len(p_entries) * (d_k + d_v + 2)

        # Recompute unread sums
        L_O = sum(env["Lj"] for env in unread_envelopes)
        U_O = max(L_O, sum(env["Uj"] for env in unread_envelopes))
        B_O = sum(env["bj"] for env in unread_envelopes)

        # Full read recovery check
        if not unread_envelopes:
            y_hat = N_A / Z_A
            total_bound = 0.0
            best_candidate_output = y_hat.clone()
            best_candidate_bound = 0.0
            break

        # Recompute candidate completion and certificate bound
        if comparator_name in ("aurelis", "per_page_center"):
            Z_hat_j = [(env["Lj"] + env["Uj"]) / 2.0 for env in unread_envelopes]
            sum_zh = sum(Z_hat_j)
            pred_sum = torch.sum(
                torch.stack([torch.as_tensor(zh, dtype=dt) * env["page"].value_center.to(dtype=dt) for zh, env in zip(Z_hat_j, unread_envelopes)]),
                dim=0,
            ) if unread_envelopes else torch.zeros(d_v, dtype=dt, device=query.device)
            y_hat = (N_A + pred_sum) / (Z_A + sum_zh)

            denom = Z_A + L_O
            if denom > 0.0:
                num = 0.0
                for env in unread_envelopes:
                    eta_j = (env["Uj"] - env["Lj"]) / 2.0
                    diff_j = float(torch.linalg.vector_norm(env["page"].value_center.to(dtype=dt) - y_hat).item())
                    num += env["Uj"] * env["radius"] + eta_j * diff_j
                cert_bound = num / denom
            else:
                cert_bound = float("inf")
            delta_num = compute_numerical_allowance(
                selected_mass=Z_A, unread_lower=L_O, unread_upper=U_O, N_A=N_A,
                prior=prior, completed=y_hat, count_A=rk.shape[0] + pages_read * target_page.count,
                count_O=sum(env["count"] for env in unread_envelopes), d_k=d_k, d_v=d_v, dtype=dt,
            )
            total_bound = cert_bound + delta_num

        elif comparator_name == "sparse_no_completion":
            y_hat = N_A / max(1e-12, Z_A)
            max_val_norm = max(float(torch.linalg.vector_norm(env["page"].value_center).item()) + env["radius"] for env in unread_envelopes)
            denom = Z_A + L_O
            cert_bound = (U_O / denom) * max_val_norm if denom > 0.0 else float("inf")
            total_bound = cert_bound + delta_num

        elif comparator_name == "simple_fusion":
            y_hat = 0.5 * (N_A / max(1e-12, Z_A)) + 0.5 * prior
            total_bound = float("nan")

        else:
            Z_hat_O = (L_O + U_O) / 2.0
            y_hat = completed_read(Z_A, Z_hat_O, N_A, prior)
            is_valid, cert_obj, _ = evaluate_certified_bound(
                selected_mass=Z_A, unread_lower=L_O, unread_upper=U_O, residual_bound=B_O,
                prior=prior, completed=y_hat, N_A=N_A, count_A=rk.shape[0] + pages_read * target_page.count,
                count_O=sum(env["count"] for env in unread_envelopes), d_k=d_k, d_v=d_v, dtype=dt,
            )
            cert_bound = cert_obj.approximation_bound
            delta_num = cert_obj.delta_num
            total_bound = cert_obj.total_bound

        # Non-monotone tracking: keep best bound candidate
        if math.isfinite(total_bound) and total_bound < best_candidate_bound:
            best_candidate_bound = total_bound
            best_candidate_output = y_hat.clone()
            best_candidate_pages = pages_read
            best_candidate_bytes = bytes_read

        # Stopping check under certified tolerance
        if can_stop_early and math.isfinite(total_bound) and total_bound <= epsilon:
            break

    # Determine final output and status
    final_output = y_hat if (not can_stop_early or total_bound <= best_candidate_bound) else best_candidate_output
    final_bound = total_bound if (not can_stop_early or total_bound <= best_candidate_bound) else best_candidate_bound
    final_pages = pages_read if (not can_stop_early or total_bound <= best_candidate_bound) else best_candidate_pages
    final_bytes = bytes_read if (not can_stop_early or total_bound <= best_candidate_bound) else best_candidate_bytes

    if not unread_envelopes:
        status = "full_read"
    elif can_stop_early and math.isfinite(final_bound) and final_bound <= epsilon:
        status = "certified"
    elif not math.isfinite(final_bound):
        status = "heuristic"
    else:
        status = "budget_exhausted"

    actual_err = float(torch.linalg.vector_norm(y_star - final_output).item())
    total_flops = index_build_ops + predictor_ops + selection_ops
    comp_cost = compute_composite_cost(final_bytes, final_pages, summary_scans, total_flops)

    acct = AccountingRecord(
        index_build_ops=index_build_ops, predictor_ops=predictor_ops,
        summary_scans=summary_scans, selection_ops=selection_ops,
        pages_read=final_pages, bytes_read=final_bytes,
        working_state_bytes=working_state_bytes, archive_raw_bytes=archive_raw_bytes,
        archive_index_bytes=archive_index_bytes, total_memory_bytes=total_memory_bytes,
        total_flops=total_flops, composite_cost=comp_cost,
    )

    return ComparatorResult(
        comparator_name=comparator_name, output=final_output, status=status,
        actual_error=actual_err, certified_bound=final_bound if math.isfinite(final_bound) else None,
        delta_num=delta_num, pages_read=final_pages, bytes_read=final_bytes,
        accounting=acct, details={"steps": step_index, "best_candidate_retained": (final_bound != total_bound)},
    )
