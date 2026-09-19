# Phase 1: Pathology & Numerical Condition Verification

| Pathology Family | Value Scale | Max Abs Error | Max Rel Error | Status | Diagnostic Notes |
|---|---|---|---|---|---|
| `empty_remote_sets` | 2.43e+00 | 1.67e-16 | 8.22e-17 | **PASS** | Unread set empty; completed read equals full softmax exactly; certificate bound is 0. |
| `singleton_pages` | 1.76e+00 | 0.00e+00 | 0.00e+00 | **PASS** | Page radius is 0; mass upper and lower bounds coincide with exact single-item exponential. |
| `partial_pages` | 2.48e+00 | 0.00e+00 | 0.00e+00 | **PASS** | Correctly scales mass and bounds by actual count n=3 rather than nominal page capacity. |
| `negative_query_coordinates` | 5.89e+00 | 0.00e+00 | 0.00e+00 | **PASS** | Coordinate box min(q*k-, q*k+) correctly inverts coordinate orientation when q_d < 0. |
| `zero_keys` | 0.00e+00 | 0.00e+00 | 0.00e+00 | **PASS** | Key norm floor handles k=0 with zero division avoided; S^+ simplifies to alpha * S exactly. |
| `beta_endpoints` | 1.00e+00 | 0.00e+00 | 0.00e+00 | **PASS** | Tested beta=0 (identity), beta=1 (full unit write), and beta=2 (isometry reflection). |
| `alpha_endpoints` | 1.98e+00 | 0.00e+00 | 0.00e+00 | **PASS** | alpha=0 completely resets prior state; alpha=1 preserves conservative recurrence. |
| `repeated_keys` | 1.32e+01 | 0.00e+00 | 0.00e+00 | **PASS** | Successive writes at identical key overwrite the linear association cleanly with S^+ k = v_new. |
| `huge_value_outliers` | 1.00e+06 | 0.00e+00 | 0.00e+00 | **PASS** | Stable under 10^6 value scaling; certificate bound scales linearly without overflow or cancellation. |
| `uniform_scores` | 1.62e+00 | 0.00e+00 | 0.00e+00 | **PASS** | Completely diffuse attention weights equal 1/t; barycenter is arithmetic mean. |
| `concentrated_scores` | 1.64e+00 | 0.00e+00 | 0.00e+00 | **PASS** | Numerical stability under large score difference (kappa*q^T k = 50); no exp overflow via max subtraction. |
| `stale_state_residuals` | 3.87e+00 | 3.13e+00 | 7.52e-01 | **PASS** | Proves stored residuals v - S_write k are invalid under evolved S_t; dynamic envelope ||c - r_t|| is required. |
