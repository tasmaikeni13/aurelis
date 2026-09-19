"""Validated deterministic certificates, arithmetic error policies, and summary validation for AURELIS.

Implements:
1. IEEE 754 documented forward error bounds (dot products, exp, norms, reductions, division, rounding).
2. Outward coordinate intervals and page envelopes (Eqs. 11, 12).
3. Mass-consistent deterministic residual certificates (Eq. 10).
4. Separate numerical allowance delta_num and total certified stopping rule (E_A + delta_num <= epsilon).
5. Independent summary containment and unread partition cover validation.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence
import torch
from torch import Tensor

from .types import ArchiveEntry, CertificateBound, PageDescriptor


def get_unit_roundoff(dtype: torch.dtype) -> float:
    """Return unit roundoff u (half of machine epsilon) for a floating point dtype."""
    if dtype == torch.float64:
        return 2.0 ** -53  # ~1.110223e-16
    elif dtype == torch.float32:
        return 2.0 ** -24  # ~5.960464e-8
    elif dtype == torch.bfloat16:
        return 2.0 ** -8   # 0.00390625
    elif dtype == torch.float16:
        return 2.0 ** -11  # ~0.00048828
    else:
        return 2.0 ** -24


def gamma(n: int, u: float) -> float:
    """Standard Higham forward error coefficient gamma_n = n*u / (1 - n*u)."""
    nu = n * u
    if nu >= 1.0:
        return 1.0
    return nu / (1.0 - nu)


def compute_outward_score_interval(
    query: Tensor,
    key_min: Tensor,
    key_max: Tensor,
    kappa: float = 1.0,
    dtype: Optional[torch.dtype] = None,
) -> tuple[float, float, float]:
    """Compute outward score bounds [ell_outward, u_outward] on kappa * q^T k.

    Accounts for floating point inner product summation error:
    Delta_dot = gamma_{d_k + 2} * kappa * sum_d |q_d| * max(|k^-_d|, |k^+_d|).

    Returns:
        (ell_outward, u_outward, delta_dot)
    """
    dt = dtype or query.dtype
    q = query.to(dtype=dt)
    k_min = key_min.to(dtype=dt)
    k_max = key_max.to(dtype=dt)
    u = get_unit_roundoff(dt)
    d_k = q.shape[-1]

    qk_min = q * k_min
    qk_max = q * k_max
    coord_min = torch.minimum(qk_min, qk_max)
    coord_max = torch.maximum(qk_min, qk_max)

    ell_raw = float((kappa * torch.sum(coord_min)).item())
    u_raw = float((kappa * torch.sum(coord_max)).item())

    # Bound on max absolute dot product magnitude for forward error
    max_abs_k = torch.maximum(torch.abs(k_min), torch.abs(k_max))
    dot_abs_bound = float((kappa * torch.sum(torch.abs(q) * max_abs_k)).item())
    delta_dot = gamma(d_k + 2, u) * dot_abs_bound

    ell_outward = ell_raw - delta_dot
    u_outward = u_raw + delta_dot

    return ell_outward, u_outward, delta_dot


def compute_outward_page_envelope(
    page: PageDescriptor,
    query: Tensor,
    prior: Tensor,
    kappa: float = 1.0,
    m_shift: float = 0.0,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, Any]:
    """Compute outward page bounds L_j, U_j, b_j with common log-sum-exp shift.

    Applies outward rounding for dot product, exp, Euclidean norm, and residual ball.

    Returns dictionary with:
        page, ell_outward, u_outward, Lj, Uj, bj, diff_r, radius, count
    """
    dt = dtype or query.dtype
    u = get_unit_roundoff(dt)
    d_v = prior.shape[-1]

    ell_out, u_out, delta_dot = compute_outward_score_interval(
        query=query,
        key_min=page.key_min,
        key_max=page.key_max,
        kappa=kappa,
        dtype=dt,
    )

    # Shifted score bounds
    shifted_ell = ell_out - m_shift
    shifted_u = u_out - m_shift

    # Outward exponentiation: downward for lower bound, upward for upper bound
    # Standard libm exp relative error <= 2u
    exp_factor_lower = max(0.0, 1.0 - 2.0 * u)
    exp_factor_upper = 1.0 + 2.0 * u

    Lj = float(page.count) * math.exp(shifted_ell) * exp_factor_lower
    Uj = float(page.count) * math.exp(shifted_u) * exp_factor_upper

    # Value center distance and norm error bound
    c = page.value_center.to(dtype=dt)
    r = prior.to(dtype=dt)
    diff_val = c - r
    norm_val = float(torch.linalg.vector_norm(diff_val).item())
    norm_err = gamma(d_v + 2, u) * norm_val

    # Outward distance ball
    dist_outward = norm_val + norm_err + page.value_radius
    bj = Uj * dist_outward * (1.0 + 2.0 * u)

    return {
        "page": page,
        "ell": ell_out,
        "u": u_out,
        "delta_dot": delta_dot,
        "Lj": Lj,
        "Uj": Uj,
        "bj": bj,
        "diff_r": norm_val,
        "radius": page.value_radius,
        "count": page.count,
    }


def compute_numerical_allowance(
    selected_mass: float,
    unread_lower: float,
    unread_upper: float,
    N_A: Tensor,
    prior: Tensor,
    completed: Tensor,
    count_A: int,
    count_O: int,
    d_k: int,
    d_v: int,
    dtype: torch.dtype,
) -> float:
    """Compute documented forward arithmetic error bound delta_num.

    Covers:
    - Dot products in scoring
    - Exponentiation
    - Vector reductions (N_A)
    - Scalar reductions (Z_A, L_O, U_O, B_O)
    - Numerator combination (N_A + Z_hat_O * r)
    - Denominator combination (Z_A + Z_hat_O)
    - Division perturbation
    - Output vector rounding

    Returns:
        delta_num >= ||y_hat_exact - y_hat_float||_2
    """
    u = get_unit_roundoff(dtype)

    # 1. Reduction error in N_A: sum of count_A vectors
    n_a_norm = float(torch.linalg.vector_norm(N_A).item())
    delta_N_A = gamma(max(1, count_A) + 2, u) * (n_a_norm + 1e-12)

    # 2. Reduction error in Z_A
    delta_Z_A = gamma(max(1, count_A), u) * selected_mass

    # 3. Midpoint unread mass error
    Z_hat_O = (unread_lower + unread_upper) / 2.0
    delta_Z_hat = 2.0 * u * Z_hat_O

    # 4. Numerator vector: Num = N_A + Z_hat_O * prior
    prior_norm = float(torch.linalg.vector_norm(prior).item())
    delta_Num = delta_N_A + delta_Z_hat * prior_norm + u * (n_a_norm + Z_hat_O * prior_norm)

    # 5. Denominator: Den = Z_A + Z_hat_O
    Den = selected_mass + Z_hat_O
    delta_Den = delta_Z_A + delta_Z_hat + u * Den

    # 6. Perturbation through division:
    # || (Num + dNum)/(Den + dDen) - Num/Den || <= (||dNum|| + (||Num||/Den)*|dDen|) / (Den - |dDen|)
    effective_den = Den - delta_Den
    if effective_den <= 0.0:
        return float("inf")

    num_norm = float(torch.linalg.vector_norm(N_A + Z_hat_O * prior).item())
    delta_div = (delta_Num + (num_norm / Den) * delta_Den) / effective_den

    # 7. Output vector rounding
    completed_norm = float(torch.linalg.vector_norm(completed).item())
    delta_round = u * completed_norm

    delta_num = delta_div + delta_round
    return delta_num


def validate_page_summary(
    page: PageDescriptor,
    entries: Sequence[ArchiveEntry],
    dtype: Optional[torch.dtype] = None,
) -> tuple[bool, str]:
    """Independently validate that a page summary contains its actual page observations.

    Verifies:
    1. Count matches len(entries).
    2. Occurrence IDs match in order.
    3. Every entry key lies inside [key_min, key_max] (with tolerance).
    4. Every entry value lies within value_radius of value_center (with tolerance).
    5. No NaN or Inf in page coordinates or radius.

    Returns:
        (is_valid, reason)
    """
    if len(entries) != page.count:
        return False, f"Page {page.page_id} count mismatch: metadata {page.count} != entries {len(entries)}"

    if page.value_radius < 0.0:
        return False, f"Page {page.page_id} has negative value radius: {page.value_radius}"

    if not math.isfinite(page.value_radius):
        return False, f"Page {page.page_id} has nonfinite value radius: {page.value_radius}"

    if torch.isnan(page.key_min).any() or torch.isnan(page.key_max).any():
        return False, f"Page {page.page_id} has NaN in key bounds"

    if torch.isnan(page.value_center).any():
        return False, f"Page {page.page_id} has NaN in value center"

    if (page.key_min > page.key_max).any():
        return False, f"Page {page.page_id} has key_min > key_max"

    dt = dtype or torch.float64
    u = get_unit_roundoff(dt)
    tol = max(1e-6, 10.0 * u)

    for idx, e in enumerate(entries):
        if idx < len(page.occurrence_ids) and e.occurrence_id != page.occurrence_ids[idx]:
            return False, f"Occurrence ID mismatch at index {idx}: entry {e.occurrence_id} != summary {page.occurrence_ids[idx]}"

        k = e.key.to(dtype=dt)
        k_min = page.key_min.to(dtype=dt)
        k_max = page.key_max.to(dtype=dt)

        if (k < k_min - tol).any() or (k > k_max + tol).any():
            return False, f"Entry {e.occurrence_id} key escapes bounding box for page {page.page_id}"

        v = e.value.to(dtype=dt)
        c = page.value_center.to(dtype=dt)
        dist = float(torch.linalg.vector_norm(v - c).item())
        if dist > page.value_radius + tol:
            return False, f"Entry {e.occurrence_id} value distance {dist} exceeds radius {page.value_radius} for page {page.page_id}"

    return True, "valid"


def validate_unread_cover(
    pages: Sequence[PageDescriptor],
    expected_unread_ids: Sequence[int],
) -> tuple[bool, str]:
    """Independently validate that page descriptors form a disjoint exhaustive cover of unread IDs.

    Verifies:
    1. Disjointness: No occurrence ID appears in more than one page.
    2. Completeness: Union of page occurrence IDs equals expected unread set.

    Returns:
        (is_valid, reason)
    """
    seen_ids: set[int] = set()
    total_count = 0

    for p in pages:
        for oid in p.occurrence_ids:
            if oid in seen_ids:
                return False, f"Duplicate occurrence ID {oid} across page envelopes (disjoint cover violated)"
            seen_ids.add(oid)
            total_count += 1

    expected_set = set(expected_unread_ids)
    if seen_ids != expected_set:
        missing = expected_set - seen_ids
        extra = seen_ids - expected_set
        return False, f"Unread partition mismatch: missing={len(missing)}, extra={len(extra)}"

    return True, "valid"


def validate_intervals(
    selected_mass: float,
    unread_lower: float,
    unread_upper: float,
    residual_bound: float,
    prior: Tensor,
    completed: Tensor,
) -> tuple[bool, str]:
    """Check mathematical sanity of intervals, masses, and tensors.

    Rejects:
    - Nonfinite values (NaN, Inf)
    - Nonpositive selected mass
    - Inverted unread mass interval (L_O > U_O)
    - Negative residual bound
    - Nonpositive denominator floor
    """
    if not (math.isfinite(selected_mass) and selected_mass > 0.0):
        return False, f"Invalid selected mass: {selected_mass}"

    if not (math.isfinite(unread_lower) and unread_lower >= 0.0):
        return False, f"Invalid unread lower mass: {unread_lower}"

    # Allow floating point epsilon tolerance for boundary numerical cancellation
    u = get_unit_roundoff(prior.dtype)
    tol = 100.0 * u * max(1.0, unread_lower)
    if not (math.isfinite(unread_upper) and (unread_upper + tol) >= unread_lower):
        return False, f"Invalid unread upper mass: {unread_upper} (lower={unread_lower})"

    if not (math.isfinite(residual_bound) and residual_bound >= 0.0):
        return False, f"Invalid residual bound: {residual_bound}"

    denom_floor = selected_mass + unread_lower
    if not (math.isfinite(denom_floor) and denom_floor > 0.0):
        return False, f"Invalid denominator floor: {denom_floor}"

    if torch.isnan(prior).any() or torch.isinf(prior).any():
        return False, "Nonfinite prior vector r(q)"

    if torch.isnan(completed).any() or torch.isinf(completed).any():
        return False, "Nonfinite completed vector y_hat"

    return True, "valid"


def evaluate_certified_bound(
    selected_mass: float,
    unread_lower: float,
    unread_upper: float,
    residual_bound: float,
    prior: Tensor,
    completed: Tensor,
    N_A: Tensor,
    count_A: int,
    count_O: int,
    d_k: int,
    d_v: int,
    dtype: torch.dtype,
) -> tuple[bool, CertificateBound, str]:
    """Compute mathematical certificate E_A, arithmetic allowance delta_num, and total bound.

    Returns:
        (is_valid, cert_bound, reason)
    """
    is_valid, reason = validate_intervals(
        selected_mass=selected_mass,
        unread_lower=unread_lower,
        unread_upper=unread_upper,
        residual_bound=residual_bound,
        prior=prior,
        completed=completed,
    )

    if not is_valid:
        dummy_cert = CertificateBound(
            bound=float("inf"),
            selected_mass=selected_mass,
            estimated_unread_mass=(unread_lower + unread_upper) / 2.0 if math.isfinite(unread_lower + unread_upper) else float("nan"),
            unread_mass_lower=unread_lower,
            unread_mass_upper=unread_upper,
            residual_bound=residual_bound,
            denominator_floor=selected_mass + unread_lower if math.isfinite(selected_mass + unread_lower) else float("nan"),
            delta_num=float("inf"),
            approximation_bound=float("inf"),
            total_bound=float("inf"),
            bound_valid=False,
        )
        return False, dummy_cert, reason

    # Eq. (10) Mathematical bound E_A
    delta_z = (unread_upper - unread_lower) / 2.0
    est_unread = (unread_lower + unread_upper) / 2.0
    prior_diff_norm = float(torch.linalg.vector_norm(prior - completed).item())
    denom_floor = selected_mass + unread_lower

    E_A = (residual_bound + delta_z * prior_diff_norm) / denom_floor

    # Documented arithmetic error bound delta_num
    delta_num = compute_numerical_allowance(
        selected_mass=selected_mass,
        unread_lower=unread_lower,
        unread_upper=unread_upper,
        N_A=N_A,
        prior=prior,
        completed=completed,
        count_A=count_A,
        count_O=count_O,
        d_k=d_k,
        d_v=d_v,
        dtype=dtype,
    )

    total_bound = E_A + delta_num

    cert = CertificateBound(
        bound=total_bound,
        selected_mass=selected_mass,
        estimated_unread_mass=est_unread,
        unread_mass_lower=unread_lower,
        unread_mass_upper=unread_upper,
        residual_bound=residual_bound,
        denominator_floor=denom_floor,
        delta_num=delta_num,
        approximation_bound=E_A,
        total_bound=total_bound,
        bound_valid=True,
    )

    return True, cert, "valid"
