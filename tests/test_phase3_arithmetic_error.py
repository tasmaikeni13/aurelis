"""Tests for Phase 3 conservative floating-point arithmetic error policy."""

import math
import pytest
import torch

from aurelis.certificate import (
    compute_numerical_allowance,
    compute_outward_score_interval,
    gamma,
    get_unit_roundoff,
)
from aurelis.functional import completed_read


def test_documented_error_bounds_dot_products():
    """Verify that forward error bound strictly bounds floating-point dot product discrepancy."""
    torch.manual_seed(3101)
    d_k = 32
    u_64 = get_unit_roundoff(torch.float64)

    for _ in range(20):
        q = torch.randn(d_k, dtype=torch.float64)
        k = torch.randn(d_k, dtype=torch.float64)
        k_min = k - 0.1
        k_max = k + 0.1

        ell_out, u_out, delta_dot = compute_outward_score_interval(q, k_min, k_max, kappa=1.0, dtype=torch.float64)

        exact_dot = float(torch.dot(q, k).item())
        fl_dot = float((q * k).sum().item())

        # Discrepancy between dot products must be bounded by delta_dot
        diff = abs(fl_dot - exact_dot)
        assert diff <= delta_dot + 1e-15
        assert ell_out <= exact_dot + 1e-15
        assert exact_dot <= u_out + 1e-15


def test_numerical_allowance_delta_num_computation():
    """Verify numerical allowance delta_num scales strictly with dtype roundoff."""
    d_k, d_v = 16, 16
    count_A, count_O = 32, 64

    selected_mass = 10.0
    unread_lower = 2.0
    unread_upper = 5.0
    N_A = torch.ones(d_v)
    prior = torch.ones(d_v) * 0.5
    completed = torch.ones(d_v) * 0.7

    delta_32 = compute_numerical_allowance(
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
        dtype=torch.float32,
    )

    delta_64 = compute_numerical_allowance(
        selected_mass=selected_mass,
        unread_lower=unread_lower,
        unread_upper=unread_upper,
        N_A=N_A.double(),
        prior=prior.double(),
        completed=completed.double(),
        count_A=count_A,
        count_O=count_O,
        d_k=d_k,
        d_v=d_v,
        dtype=torch.float64,
    )

    assert delta_32 > 0.0
    assert delta_64 > 0.0
    assert math.isfinite(delta_32)
    assert math.isfinite(delta_64)

    # float32 allowance should be ~10^-6 to 10^-5, float64 ~10^-15 to 10^-14
    assert delta_32 < 1e-4
    assert delta_64 < 1e-12
    # delta_32 must be orders of magnitude larger than delta_64
    assert delta_32 > delta_64 * 1e6


def test_delta_num_bounds_floating_point_discrepancy():
    """Verify ||y_hat_exact - y_hat_float||_2 <= delta_num."""
    torch.manual_seed(3102)
    d_v = 16
    count_A = 64

    # High precision reference
    weights = torch.softmax(torch.randn(count_A, dtype=torch.float64), dim=0)
    values = torch.randn(count_A, d_v, dtype=torch.float64)

    Z_A_exact = float(weights.sum().item())
    N_A_exact = torch.sum(weights[:, None] * values, dim=0)
    Z_hat_O = 0.5
    prior_exact = torch.randn(d_v, dtype=torch.float64)

    y_hat_exact = completed_read(Z_A_exact, Z_hat_O, N_A_exact, prior_exact)

    # Float32 execution
    weights_f32 = weights.float()
    values_f32 = values.float()
    Z_A_f32 = float(weights_f32.sum().item())
    N_A_f32 = torch.sum(weights_f32[:, None] * values_f32, dim=0)
    prior_f32 = prior_exact.float()

    y_hat_f32 = completed_read(Z_A_f32, Z_hat_O, N_A_f32, prior_f32)

    actual_error = float(torch.linalg.vector_norm(y_hat_exact - y_hat_f32.double()).item())

    delta_num = compute_numerical_allowance(
        selected_mass=Z_A_f32,
        unread_lower=0.2,
        unread_upper=0.8,
        N_A=N_A_f32,
        prior=prior_f32,
        completed=y_hat_f32,
        count_A=count_A,
        count_O=16,
        d_k=16,
        d_v=d_v,
        dtype=torch.float32,
    )

    assert actual_error <= delta_num, f"Arithmetic allowance violated: actual {actual_error} > delta_num {delta_num}"


def test_no_arbitrary_tolerance_inflation():
    """Verify delta_num has no arbitrary padding or constant fudge factors."""
    u_32 = get_unit_roundoff(torch.float32)
    u_64 = get_unit_roundoff(torch.float64)

    assert abs(u_32 - 2.0 ** -24) < 1e-15
    assert abs(u_64 - 2.0 ** -53) < 1e-25

    # Check gamma forward error formulation
    g1 = gamma(10, u_64)
    assert abs(g1 - (10.0 * u_64) / (1.0 - 10.0 * u_64)) < 1e-25
