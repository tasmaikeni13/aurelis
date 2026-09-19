"""Phase 1 Counterexamples and Impossibility Demonstrations.

Implements the four mandatory mathematical counterexamples and demonstrations:
1. Normalizer Omission Counterexample:
   Dropping the normalizer term (eta * ||r - y_hat_A||_2) can underestimate actual error:
   ||y_* - y_hat_A||_2 > B_O / (Z_A + L_O).
2. Finite-State Recall Capacity Lower Bound Demonstration:
   Bounded recurrent state S in R^{d_v x d_k} cannot achieve arbitrary exact recall past capacity.
3. Finite Softmax is Not a Hard Lookup Demonstration:
   Finite temperature / score scale kappa leaves residual probability mass on non-target keys,
   mixing values rather than retrieving an exact token.
4. Non-Monotonic Error Reduction Demonstration:
   Fetching an unread page can cause the certificate bound to increase, proving that monotonic
   error reduction on every fetch is a false property.
"""

from __future__ import annotations

import math
from typing import Any
import pytest
import torch

from aurelis.oracles import ScalarOracle, TensorOracle

# Registry to collect demonstration data for json export
COUNTEREXAMPLE_RECORDS: dict[str, dict[str, Any]] = {}


def test_counterexample_normalizer_omission():
    """Counterexample 1: Dropping normalizer term underestimates actual error.

    Shows an explicit instance where ||y_* - y_hat_A||_2 > B_O / (Z_A + L_O).
    """
    d_v = 2
    # Prior estimate r
    r = torch.tensor([0.0, 1.0], dtype=torch.float64)

    # Selected set A: mass Z_A = 1.0, value barycenter [1.0, 0.0]
    Z_A = 1.0
    N_A = torch.tensor([1.0, 0.0], dtype=torch.float64)

    # Unread set O: all unread values equal r exactly!
    # Therefore unread residual R_O = sum_O exp(s_i)(v_i - r) = 0
    # and residual upper bound B_O = 0.
    B_O = 0.0

    # But unread mass Z_O is uncertain: interval [L_O, U_O] = [0.2, 10.0]
    L_O = 0.2
    U_O = 10.0
    Z_hat_O = (L_O + U_O) / 2.0  # 5.1
    eta = (U_O - L_O) / 2.0      # 4.9

    # Suppose true unread mass is at the high end: Z_O_true = 9.5
    Z_O_true = 9.5
    N_O_true = Z_O_true * r

    # True full softmax attention output
    y_star = (N_A + N_O_true) / (Z_A + Z_O_true)

    # Completed read output using midpoint mass estimate Z_hat_O
    y_hat_A = TensorOracle.completed_read(Z_A, Z_hat_O, N_A, r)

    # Actual approximation error
    actual_error = float(torch.linalg.vector_norm(y_star - y_hat_A).item())

    # Naive bound without normalizer term: B_O / (Z_A + L_O) = 0 / 1.2 = 0.0
    naive_bound = B_O / (Z_A + L_O)

    # Full certified bound with normalizer term (Eq. 10)
    cert = TensorOracle.residual_certificate(
        selected_mass=Z_A,
        unread_lower=L_O,
        unread_upper=U_O,
        residual_bound=B_O,
        prior=r,
        completed=y_hat_A,
    )
    full_bound = cert.bound

    # Verification:
    # 1. Naive bound claims 0.0 error
    assert naive_bound == 0.0
    # 2. Actual error is strictly positive (> 0.05)
    assert actual_error > 0.05
    # 3. Naive bound UNDERESTIMATES actual error: naive_bound < actual_error
    assert naive_bound < actual_error
    # 4. Full certified bound remains valid and sound: full_bound >= actual_error
    assert full_bound >= actual_error

    COUNTEREXAMPLE_RECORDS["normalizer_omission"] = {
        "title": "Dropping normalizer term underestimates actual error",
        "naive_bound": naive_bound,
        "actual_error": actual_error,
        "full_certified_bound": full_bound,
        "underestimation_factor": float("inf"),
        "soundness_verified": full_bound >= actual_error,
        "parameters": {
            "Z_A": Z_A,
            "L_O": L_O,
            "U_O": U_O,
            "Z_hat_O": Z_hat_O,
            "Z_O_true": Z_O_true,
            "B_O": B_O,
            "r": r.tolist(),
            "y_star": y_star.tolist(),
            "y_hat_A": y_hat_A.tolist(),
        },
    }


def test_demonstration_finite_state_recall_capacity():
    """Demonstration 2: Bounded recurrent state cannot meet arbitrary exact recall past capacity.

    Shows that for d_k = 2, d_v = 1 (a 1x2 state S with 2 parameters),
    storing N = 6 distinct associations with independent binary values
    forces recall errors on past associations.
    """
    d_k = 2
    d_v = 1
    N = 6

    # 6 distinct unit key directions in R^2
    angles = [i * (math.pi / N) for i in range(N)]
    keys = torch.tensor([[math.cos(a), math.sin(a)] for a in angles], dtype=torch.float64)

    # 6 arbitrary binary values: alternating +1, -1
    values = torch.tensor([[1.0], [-1.0], [1.0], [-1.0], [1.0], [-1.0]], dtype=torch.float64)

    # Sequentially feed all N associations through gated delta recurrence
    S = TensorOracle.zero_matrix(d_v, d_k)
    for i in range(N):
        S = TensorOracle.gated_delta_update(S, keys[i], values[i], alpha=1.0, beta=1.0)

    # Test recall of each past association: pred_i = S @ k_i
    recall_errors = []
    for i in range(N):
        pred_i = float(torch.mv(S, keys[i]).item())
        true_i = float(values[i].item())
        err_i = abs(pred_i - true_i)
        recall_errors.append(err_i)

    max_recall_error = max(recall_errors)
    mean_recall_error = sum(recall_errors) / len(recall_errors)

    # Capacity impossibility: max recall error MUST be significantly non-zero (> 0.2)
    assert max_recall_error > 0.2
    assert mean_recall_error > 0.1

    # Also test optimal least-squares matrix: S_opt = values^T @ pinv(keys^T)
    # Even the best possible static matrix in R^{1x2} cannot fit 6 arbitrary binary values on 6 directions
    S_opt = torch.linalg.lstsq(keys, values).solution.T
    opt_errors = [abs(float(torch.mv(S_opt, keys[i]).item()) - float(values[i].item())) for i in range(N)]
    max_opt_error = max(opt_errors)
    assert max_opt_error > 0.2

    COUNTEREXAMPLE_RECORDS["finite_state_capacity"] = {
        "title": "Bounded recurrent state cannot achieve arbitrary exact recall past capacity",
        "d_k": d_k,
        "d_v": d_v,
        "num_associations": N,
        "max_recall_error_sequential": max_recall_error,
        "mean_recall_error_sequential": mean_recall_error,
        "max_recall_error_optimal_least_squares": max_opt_error,
        "recalled_errors_per_token": recall_errors,
        "conclusion": "With dim 2, state has at most 2 degrees of freedom; N=6 arbitrary associations force collision and recall loss.",
    }


def test_demonstration_finite_softmax_not_hard_lookup():
    """Demonstration 3: Finite softmax is not a hard lookup.

    Shows that for any finite score scale kappa < inf, finite softmax mixes distractors,
    placing strictly less than 1.0 probability mass on the target.
    """
    d_k = 4
    d_v = 1
    N = 10  # 1 target + 9 distractors

    # Target key at index 0
    target_key = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    target_val = torch.tensor([1.0], dtype=torch.float64)

    # Distractor keys orthogonal to query (q^T k_i = 0) with zero values
    keys = torch.zeros(N, d_k, dtype=torch.float64)
    keys[0] = target_key
    values = torch.zeros(N, d_v, dtype=torch.float64)
    values[0] = target_val  # target is 1.0, all distractors are 0.0

    query = target_key.clone()  # Query matches target key exactly: q^T k_0 = 1.0

    kappa_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    leakage_records = []

    for kappa in kappa_values:
        y_out = TensorOracle.full_softmax(keys, values, query, kappa=kappa)
        target_prob = float(y_out[0].item())
        distractor_leakage = 1.0 - target_prob

        # Theoretical weight: exp(kappa) / (exp(kappa) + (N - 1) * exp(0))
        expected_weight = math.exp(kappa) / (math.exp(kappa) + (N - 1))
        assert math.isclose(target_prob, expected_weight, abs_tol=1e-12)

        # Non-lookup proof: target probability is STRICTLY less than 1.0
        assert target_prob < 1.0
        assert distractor_leakage > 0.0

        leakage_records.append({
            "kappa": kappa,
            "target_probability": target_prob,
            "distractor_leakage": distractor_leakage,
        })

    COUNTEREXAMPLE_RECORDS["finite_softmax_non_lookup"] = {
        "title": "Finite softmax is an interpolator, not a hard addressable lookup",
        "num_items": N,
        "kappa_sweeps": leakage_records,
        "conclusion": "Finite softmax always leaks probability mass to non-matching keys; retrieval mixes values for all finite kappa.",
    }


def test_demonstration_non_monotonic_error_reduction():
    """Demonstration 4: Error reduction is not monotonic on each fetch.

    Constructs a concrete instance where fetching an unread page increases the
    certificate error bound E_A.
    """
    d_v = 1
    # Prior estimate r = 0.0
    r = torch.tensor([0.0], dtype=torch.float64)

    # Initial selected set A: small mass Z_A = 1.0, N_A = [0.0]
    Z_A = 1.0
    N_A = torch.tensor([0.0], dtype=torch.float64)

    # Two unread pages: Page 1 and Page 2
    # Page 1: Values all close to r=0. Center c1 = 0.0, rho1 = 0.05.
    # Page 1 mass interval [L1, U1] = [1.0, 2.0]. b1 = U1 * 0.05 = 0.10.
    # Page 2: Large value c2 = 10.0, rho2 = 0.5.
    # Page 2 mass interval [L2, U2] = [2.0, 4.0]. b2 = U2 * (10.0 + 0.5) = 4.0 * 10.5 = 42.0.

    L1, U1, b1 = 1.0, 2.0, 0.10
    L2, U2, b2 = 2.0, 4.0, 42.0

    # Step 1: Initial state before any fetch (Pages 1 & 2 both unread)
    L_O_1 = L1 + L2      # 3.0
    U_O_1 = U1 + U2      # 6.0
    Z_hat_O_1 = (L_O_1 + U_O_1) / 2.0  # 4.5
    B_O_1 = b1 + b2      # 42.10

    # Completed read at step 1:
    # y_hat_1 = (N_A + Z_hat_O_1 * r) / (Z_A + Z_hat_O_1) = 0.0
    y_hat_1 = TensorOracle.completed_read(Z_A, Z_hat_O_1, N_A, r)
    cert_1 = TensorOracle.residual_certificate(
        selected_mass=Z_A,
        unread_lower=L_O_1,
        unread_upper=U_O_1,
        residual_bound=B_O_1,
        prior=r,
        completed=y_hat_1,
    )
    bound_1 = cert_1.bound
    # bound_1 = (42.10 + 1.5 * ||0 - 0||) / (1.0 + 3.0) = 42.10 / 4.0 = 10.525

    # Step 2: Fetch Page 1!
    # Page 1 has actual values near 5.0 (outlier inside page 1)
    # Moving Page 1 into selected set A:
    # Suppose Page 1 actual mass Z_1 = 1.5, actual N_1 = [7.5]
    Z_A_new = Z_A + 1.5   # 2.5
    N_A_new = N_A + torch.tensor([7.5], dtype=torch.float64)  # [7.5]

    # Remaining unread set is now only Page 2:
    L_O_2 = L2   # 2.0
    U_O_2 = U2   # 4.0
    Z_hat_O_2 = (L_O_2 + U_O_2) / 2.0  # 3.0
    B_O_2 = b2   # 42.0

    # Completed read at step 2:
    # y_hat_2 = (7.5 + 3.0 * 0) / (2.5 + 3.0) = 7.5 / 5.5 = 1.3636
    y_hat_2 = TensorOracle.completed_read(Z_A_new, Z_hat_O_2, N_A_new, r)
    # Now ||r - y_hat_2|| = 1.3636 > 0!
    cert_2 = TensorOracle.residual_certificate(
        selected_mass=Z_A_new,
        unread_lower=L_O_2,
        unread_upper=U_O_2,
        residual_bound=B_O_2,
        prior=r,
        completed=y_hat_2,
    )
    bound_2 = cert_2.bound
    # eta_2 = (4 - 2)/2 = 1.0
    # prior_diff = 1.3636
    # bound_2 = (42.0 + 1.0 * 1.3636) / (2.5 + 2.0) = 43.3636 / 4.5 = 9.636...
    # Let's adjust parameters to make bound_2 > bound_1 strictly:
    # If Z_A is very small, say Z_A = 0.1, and L_O_1 = 3.0:
    # Let's verify if bound_2 > bound_1 can occur:
    # Notice: denom at step 1: Z_A + L1 + L2 = 1.0 + 1.0 + 2.0 = 4.0
    # denom at step 2: Z_A + Z1 + L2 = 1.0 + 1.5 + 2.0 = 4.5.
    # What if Page 1 had a very large lower bound L1 = 10.0, but when fetched its actual mass was only 1.0?
    # Then denom drops from (Z_A + 10.0 + 2.0) = 13.0 down to (Z_A + 1.0 + 2.0) = 4.0!
    # Let's use valid mass interval: L1 <= Z1 <= U1. If L1 = 1.0, Z1 = 1.0, but shift in ||r - y_hat|| increases numerator!

    # Let's construct explicit parameters where bound increases:
    # Step 1:
    # Z_A = 0.5, r = [0.0], N_A = [0.0]
    # Page 1: L1 = 5.0, U1 = 5.5, b1 = 0.05
    # Page 2: L2 = 0.5, U2 = 4.5, b2 = 50.0
    # Before fetch:
    # Denom = 0.5 + 5.0 + 0.5 = 6.0
    # eta = (5.5 + 4.5 - 5.5)/2 = 2.25
    # y_hat_1 = 0.0, so prior_diff = 0.0
    # bound_1 = (50.05 + 0) / 6.0 = 8.3416
    # Fetch Page 1: Z_1 = 5.0, N_1 = [25.0] (value 5.0)
    # After fetch:
    # Z_A_new = 0.5 + 5.0 = 5.5, N_A_new = [25.0]
    # Remaining Page 2: L2 = 0.5, U2 = 4.5, b2 = 50.0, eta_2 = 2.0
    # Z_hat_O_2 = 2.5
    # y_hat_2 = 25.0 / (5.5 + 2.5) = 25.0 / 8.0 = 3.125
    # prior_diff = ||0 - 3.125|| = 3.125
    # numerator = 50.0 + 2.0 * 3.125 = 56.25
    # denominator = 5.5 + 0.5 = 6.0
    # bound_2 = 56.25 / 6.0 = 9.375!
    # 9.375 > 8.3416! The certificate bound increased from 8.34 to 9.38!

    # Let's check this exact calculation:
    Z_A_case = 0.5
    N_A_case = torch.tensor([0.0], dtype=torch.float64)
    r_case = torch.tensor([0.0], dtype=torch.float64)

    L1_c, U1_c, b1_c = 5.0, 5.5, 0.05
    L2_c, U2_c, b2_c = 0.5, 4.5, 50.0

    # Step 1: Before fetch
    L_O_init = L1_c + L2_c   # 5.5
    U_O_init = U1_c + U2_c   # 10.0
    Z_hat_init = (L_O_init + U_O_init) / 2.0
    B_O_init = b1_c + b2_c   # 50.05

    y_hat_init = TensorOracle.completed_read(Z_A_case, Z_hat_init, N_A_case, r_case)
    cert_init = TensorOracle.residual_certificate(
        selected_mass=Z_A_case,
        unread_lower=L_O_init,
        unread_upper=U_O_init,
        residual_bound=B_O_init,
        prior=r_case,
        completed=y_hat_init,
    )

    # Step 2: Fetch Page 1 (actual mass 5.0, numerator 25.0)
    Z_A_after = Z_A_case + 5.0
    N_A_after = N_A_case + torch.tensor([25.0], dtype=torch.float64)

    L_O_after = L2_c
    U_O_after = U2_c
    Z_hat_after = (L_O_after + U_O_after) / 2.0
    B_O_after = b2_c

    y_hat_after = TensorOracle.completed_read(Z_A_after, Z_hat_after, N_A_after, r_case)
    cert_after = TensorOracle.residual_certificate(
        selected_mass=Z_A_after,
        unread_lower=L_O_after,
        unread_upper=U_O_after,
        residual_bound=B_O_after,
        prior=r_case,
        completed=y_hat_after,
    )

    # Verify that bound increased:
    assert cert_after.bound > cert_init.bound
    increase = cert_after.bound - cert_init.bound

    COUNTEREXAMPLE_RECORDS["non_monotonic_error_reduction"] = {
        "title": "Error reduction is not monotonic on each fetch",
        "initial_bound_before_fetch": cert_init.bound,
        "bound_after_fetching_page_1": cert_after.bound,
        "bound_increase": increase,
        "monotonicity_violated": True,
        "mechanism": "Fetching page 1 with non-prior values shifted y_hat away from r, increasing normalizer uncertainty penalty faster than unread residual reduced.",
    }
